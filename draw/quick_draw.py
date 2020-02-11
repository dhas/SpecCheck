import numpy as np
import pickle
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from .quickdraw_bin_parser import unpack_drawings,vector_to_raster

class QuickDraw():

	def __init__(self,root_dir,side,num_imgs_per_class,min_sz):
		self.root_dir = Path(root_dir)
		self.dataset  = self.root_dir / 'dataset'
		self.settings = self.dataset / 'settings.json'
		self.est_labels = self.dataset / 'est_labels.pkl'
		self.num_imgs_per_class = num_imgs_per_class
		self.img_side = side
		self.bin_files = list(self.root_dir.glob('*.bin'))
		self.num_classes = len(self.bin_files)
		self.min_sz = min_sz
		self.pop_mean = 0
		self.pop_var  = 0
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
		settings['mean'] = self.pop_mean
		settings['variance'] = self.pop_var
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


	def estimate_labels(self,x,shape,min_sz):		
		nz  = np.transpose(np.nonzero(x))
		x_min = np.min(nz[:,0])
		x_max = np.max(nz[:,0])
		y_min = np.min(nz[:,1])
		y_max = np.max(nz[:,1])
		br = int(np.mean(x[x>0]))
		x_sp = (x_max - x_min)
		y_sp = (y_max - y_min)

		invalid_img = False

		if shape == 'circle':
			if x_sp < 2*min_sz or y_sp < 2*min_sz:
				invalid_img = True
			else:
				x0 = x_min + (x_sp//2)
				y0 = y_min + (y_sp//2)
				sz = int(np.mean([x_sp,y_sp])/2)
		else:
			if x_sp < min_sz or y_sp < min_sz:
				invalid_img = True
			else:
				x0 = x_min
				y0 = y_min
				sz = int(np.mean([x_sp, y_sp]))

		if invalid_img:
			return [0,0,0,0]
		else:
			return [x0,y0,sz,br]

	def _extract_to_npy(self,class_name,fname,dir,num_imgs,side):
		vector_images = []
		for drawing in unpack_drawings(fname):
			vector_images.append(drawing['image'])

		y_class = []
		mu_class = 0
		var_class = 0
		rand_idx = np.random.randint(0,len(vector_images),size=num_imgs)
		for i in tqdm(range(num_imgs)):
			img = vector_to_raster([vector_images[rand_idx[i]]],side=side)[0]
			img = img.reshape(side,side,1)
			np.save(dir/('%d.npy' % i),img)
			y = self.estimate_labels(img,class_name,self.min_sz)
			y_class.append(y)
			mu_class  += img.mean()
			var_class += img.var()
		return y_class,(mu_class/self.num_imgs_per_class),(var_class/self.num_imgs_per_class)


	def _extract_dataset(self):
		if self.dataset.exists():			
			shutil.rmtree(self.dataset)		
		self.dataset.mkdir(exist_ok=False)
		y_est = {}
		for i in range(self.num_classes):
			class_name = str(self.bin_files[i]).split('.')[0].split('_')[-1]
			print('Extracting %s\n' % class_name)
			class_dir = self.dataset / class_name				
			class_dir.mkdir(exist_ok=False)
			y_class, mean, var = self._extract_to_npy(class_name,
				self.bin_files[i],
				class_dir,
				self.num_imgs_per_class,
				self.img_side)
			self.pop_mean += mean
			self.pop_var  += var
			y_est[class_name] = np.array(y_class)			
			print('')
		
		self.pop_mean = self.pop_mean/self.num_classes
		self.pop_var  = self.pop_var/self.num_classes
		pickle.dump(y_est,open(self.est_labels,'wb'))
		self._save_settings()