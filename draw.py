import numpy as np
import utils
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from quickdraw_bin_parser import unpack_drawings,vector_to_raster
import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader,random_split

class QuickDraw():

	def __init__(self,root_dir,side=128,num_imgs_per_class=12000):
		self.root_dir = Path(root_dir)
		self.dataset  = self.root_dir / 'dataset'
		self.settings = self.dataset / 'settings.json'
		self.num_imgs_per_class = num_imgs_per_class
		self.img_side = side
		self.bin_files = list(self.root_dir.glob('*.bin'))
		self.num_classes = len(self.bin_files)
		if self._settings_changed():
			self._extract_dataset()


	def read_settings(self):
		if self.settings.exists():
			with open(self.settings,'r') as f:
				settings = json.loads(f.read())
				return settings
		else:
			return {}

	def _save_settings(self):
		settings = {}
		settings['num_classes'] = self.num_classes
		settings['num_imgs_per_class'] = self.num_imgs_per_class
		settings['img_side'] = self.img_side
		settings['bin_files'] = [str(f) for f in self.bin_files]
		with open(self.settings,'w') as f:
			f.write(json.dumps(settings))

	def _settings_changed(self):
		prev_settings = self.read_settings()
		settings_changed = True
		if len(prev_settings) > 0:
			settings_changed = (prev_settings['num_classes'] != self.num_classes) or \
				(prev_settings['num_imgs_per_class'] != self.num_imgs_per_class) or \
				(prev_settings['img_side'] != self.img_side) or \
				(prev_settings['bin_files'] != [str(f) for f in self.bin_files])
		return settings_changed


	def _extract_to_npy(self,fname,dir,num_imgs,side):
		vector_images = []
		for drawing in unpack_drawings(fname):
			vector_images.append(drawing['image'])

		rand_idx = np.random.randint(0,len(vector_images),size=num_imgs)
		for i in tqdm(range(num_imgs)):
			img = vector_to_raster([vector_images[rand_idx[i]]],side=side)[0]
			img = img.reshape(side,side,1)
			# img = utils.replicate_channels(img)
			np.save(dir/('%d.npy' % i),img)


	def _extract_dataset(self):		
		if self.dataset.exists():
			shutil.rmtree(self.dataset)
		
		self.dataset.mkdir(exist_ok=False)
		
		for i in range(self.num_classes):
			class_name = str(self.bin_files[i]).split('.')[0].split('_')[-1]
			print('Extracting %s\n' % class_name)
			class_dir = self.dataset / class_name				
			class_dir.mkdir(exist_ok=False)
			self._extract_to_npy(self.bin_files[i],
				class_dir,
				self.num_imgs_per_class,
				self.img_side)
			print('')
		self._save_settings()
	
	#static method
	def npy_loader(path):
		img = torch.from_numpy(np.load(path))
		return img

	def get_loaders(self,train_batch_size,validation_batch_size,
		train_set_proportion=0.8,
		train_set_shuffle=True,validation_set_shuffle=False,
		transforms=None):
		
		if transforms is None:
			dataset = DatasetFolder(root=str(self.dataset),
				loader=QuickDraw.npy_loader,
				extensions='.npy')
		else:
			dataset = DatasetFolder(root=str(self.dataset),
				loader=QuickDraw.npy_loader,
				extensions='.npy',
				transform=transforms)

		total_num_imgs = self.num_classes*self.num_imgs_per_class
		train_size = int(train_set_proportion * total_num_imgs)
		val_size   = total_num_imgs - train_size
		train_ds,val_ds = random_split(dataset,[train_size,val_size])

		train_loader = DataLoader(dataset=train_ds,
			batch_size=train_batch_size,
			shuffle=train_set_shuffle)
		
		val_loader = DataLoader(dataset=val_ds,
			batch_size=validation_batch_size,
			shuffle=validation_set_shuffle)
		
		return train_loader,val_loader

	#static batch
	def plot_batch(tensor_batch,labels,num_rows,savename):
		np_batch = tensor_batch.numpy()
		num_cols = tensor_batch.shape[0] // num_rows
		side  	 = tensor_batch.shape[1]
		channels = tensor_batch.shape[3]
		if channels == 1:
			np_batch = np_batch.reshape(num_rows,num_cols,side,side)
		else:
			np_batch = np_batch.reshape(num_rows,num_cols,side,side,channels)
		labels   = labels.reshape(num_rows,num_cols,1)
		utils.plot_examples(np_batch,savename,labels=labels)


class SpecDraw:
	def __init__(self,root_dir,side=128,num_imgs_per_class=60000):
		self.root_dir = Path(root_dir)
		self.root_dir.mkdir(exist_ok=True)
		self.dataset  = self.root_dir / 'dataset'
		self.settings = self.dataset / 'settings.json'
		self.num_imgs_per_class = num_imgs_per_class
		self.img_side = side
		self.class_names  = ['circle', 'square']
		self.num_classes = len(self.class_names)
		self.draw_limits = {
			'sz_hi' : (self.img_side/2) - 10,
			'sz_lo' : (self.img_side/2) // 2 - 10,
			'br_lo' : 100,
			'br_hi' : 255,
			'th' 	: 5
		}

	def _circle_opencv(self,xm,ym,r,br,t,side):
		canvas = np.zeros(shape=(side,side))	
		img = cv2.circle(canvas,(xm,ym),r,int(br),t)
		return img

	def _square_opencv(self,x0,y0,r,br,t,side):
		canvas = np.zeros(shape=(side,side))	
		img    = cv2.rectangle(canvas,(x0,y0),(x0+r,y0+r),int(br),t)
		return img

	def draw_shape(self,shape,x,y,r,br,t,side):
		switcher = {
			'circle': self._circle_opencv,
			'square': self._square_opencv
		}
		return switcher[shape](x,y,r,br,t,side)

	def random_labels(self):			
		sz_lo = self.draw_limits['sz_lo']
		sz_hi = self.draw_limits['sz_hi']
		br_lo = self.draw_limits['br_lo']
		br_hi = self.draw_limits['br_hi']
		th_co = self.draw_limits['th']			

		labels = []
		for i in range(self.num_classes):
			cl = np.full(self.num_imgs_per_class,i)
			th = np.full(self.num_imgs_per_class,th_co)
			br = np.random.randint(low=br_lo,high=br_hi,size=self.num_imgs_per_class)
			x0 = np.zeros(self.num_imgs_per_class)
			y0 = np.zeros(self.num_imgs_per_class)
			if self.class_names[i] == 'circle':
				#sz for circle is radius
				sz = np.random.randint(low=sz_lo,high=sz_hi,size=self.num_imgs_per_class)
				for s in np.arange(sz_lo,sz_hi):
					ind = np.where(sz == s)[0]
					lo  = s
					hi  = self.img_side - s
					x0[ind] = np.random.randint(low=lo,high=hi,size=ind.size)
					y0[ind] = np.random.randint(low=lo,high=hi,size=ind.size)
			else:
				#sz for square is side length
				sz = np.random.randint(low=sz_lo,high=2*sz_hi,size=self.num_imgs_per_class)
				for s in np.arange(sz_lo,2*sz_hi):
					ind = np.where(sz == s)[0]
					lo = 0
					hi = self.img_side-s
					x0[ind] = np.random.randint(low=lo,high=hi,size=ind.size)
					y0[ind] = np.random.randint(low=lo,high=hi,size=ind.size)
			labels.append(np.stack([cl,x0,y0,sz,br,th],axis=1))	
		labels = np.vstack(labels)
		return labels.astype(int)


	def draw_dataset(self):
		if self.dataset.exists():
			shutil.rmtree(self.dataset)
		self.dataset.mkdir(exist_ok=False)

		y_draw = self.random_labels()
		utils.plot_label_spread(self.dataset/'label_spread.png',y_draw,self.img_side)
		for i in range(self.num_classes):
			class_name = self.class_names[i]
			print('Drawing %s\n' % class_name)
			class_dir = self.dataset / class_name				
			class_dir.mkdir(exist_ok=False)
			y_class = y_draw[np.where(y_draw[:,0] == i)]
			for j in tqdm(range(y_class.shape[0])):
				y = y_class[j]
				shape = self.draw_shape(class_name,y[1],y[2],y[3],y[4],y[5],self.img_side)
				shape = shape.reshape(self.img_side,self.img_side,1)
				np.save(class_dir/('%d.npy' % j),shape)
			np.save(self.dataset/'labels.npy',y_draw)

	#static method
	def npy_loader(path):
		img = torch.from_numpy(np.load(path))
		return img

	def get_loaders(self,train_batch_size,validation_batch_size,
		train_set_proportion=0.8,
		train_set_shuffle=True,validation_set_shuffle=False,
		transforms=None):
		
		if transforms is None:
			dataset = DatasetFolder(root=str(self.dataset),
				loader=QuickDraw.npy_loader,
				extensions='.npy')
		else:
			dataset = DatasetFolder(root=str(self.dataset),
				loader=QuickDraw.npy_loader,
				extensions='.npy',
				transform=transforms)

		total_num_imgs = self.num_classes*self.num_imgs_per_class
		train_size = int(train_set_proportion * total_num_imgs)
		val_size   = total_num_imgs - train_size
		train_ds,val_ds = random_split(dataset,[train_size,val_size])

		train_loader = DataLoader(dataset=train_ds,
			batch_size=train_batch_size,
			shuffle=train_set_shuffle)
		
		val_loader = DataLoader(dataset=val_ds,
			batch_size=validation_batch_size,
			shuffle=validation_set_shuffle)
		
		return train_loader,val_loader




if __name__ == '__main__':

	side = 128

	# qd10 = QuickDraw('./qd_10',side=side,num_imgs_per_class=1200)	
	# train_loader,val_loader = qd10.get_loaders(16,16)

	# for batch in train_loader:
	# 	x,y = batch
	# 	labels = np.array([train_loader.dataset.dataset.classes[l] for l in y])
	# 	QuickDraw.plot_batch(x,labels,2,qd10.root_dir/'train_examples.png')
	# 	break

	# for batch in val_loader:
	# 	x,y = batch
	# 	labels = np.array([val_loader.dataset.dataset.classes[l] for l in y])
	# 	QuickDraw.plot_batch(x,labels,2,qd10.root_dir/'validation_examples.png')
	# 	break

	# qdshapes = QuickDraw('./qd_shapes',side=side,num_imgs_per_class=60000)
	# train_loader,val_loader = qdshapes.get_loaders(16,16)
	# for batch in train_loader:
	# 	x,y = batch
	# 	labels = np.array([train_loader.dataset.dataset.classes[l] for l in y])
	# 	QuickDraw.plot_batch(x,labels,2,qdshapes.root_dir/'train_examples.png')
	# 	break

	# for batch in val_loader:
	# 	x,y = batch
	# 	labels = np.array([val_loader.dataset.dataset.classes[l] for l in y])
	# 	QuickDraw.plot_batch(x,labels,2,qdshapes.root_dir/'validation_examples.png')
	# 	break

	sdshapes = SpecDraw('./sd_shapes',side=side)
	sdshapes.draw_dataset()
	train_loader,val_loader = sdshapes.get_loaders(16,16)
	for batch in train_loader:
		x,y = batch
		labels = np.array([train_loader.dataset.dataset.classes[l] for l in y])
		QuickDraw.plot_batch(x,labels,2,sdshapes.root_dir/'train_examples.png')
		break

	for batch in val_loader:
		x,y = batch
		labels = np.array([val_loader.dataset.dataset.classes[l] for l in y])
		QuickDraw.plot_batch(x,labels,2,sdshapes.root_dir/'validation_examples.png')
		break

