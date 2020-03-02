import numpy as np
from pathlib import Path
import shutil
import json
import pickle
from tqdm import tqdm
from .load import get_npy_dataloader
from .annotations import classes, annotations_to_sample

class SpecDraw:
	def __init__(self,root_dir,side,num_imgs_per_class,draw_limits):
		self.dataset  = root_dir
		self.settings = self.dataset / 'settings.json'
		self.labels = self.dataset / 'labels.npy'
		self.num_imgs_per_class = num_imgs_per_class
		self.img_side = side
		self.class_names  = classes
		self.num_classes = len(self.class_names)
		self.draw_limits = draw_limits
		self.pop_mean = 0
		self.pop_var  = 0

		if self._settings_changed():
			self._draw_dataset()

	def read_settings(self):
		if self.settings.exists():
			with open(self.settings,'r') as f:
				settings = json.loads(f.read())
				return settings
		else:
			return {}

	def _save_settings(self):
		settings = {}		
		settings['num_classes'] 		= self.num_classes
		settings['class_names'] 		= self.class_names
		settings['num_imgs_per_class'] 	= self.num_imgs_per_class
		settings['img_side'] 			= self.img_side
		settings['draw_limits'] 		= self.draw_limits
		settings['mean'] 				= self.pop_mean
		settings['variance'] 			= self.pop_var
		with open(self.settings,'w') as f:
			f.write(json.dumps(settings))


	def _settings_changed(self):
		prev_settings = self.read_settings()
		settings_changed = True
		if len(prev_settings) > 0:
			settings_changed = (prev_settings['num_classes'] != self.num_classes) or \
				(prev_settings['class_names'] != self.class_names) or \
				(prev_settings['num_imgs_per_class'] != self.num_imgs_per_class) or \
				(prev_settings['img_side'] != self.img_side) or \
				(prev_settings['draw_limits'] != self.draw_limits)
		return settings_changed


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
				#sz for circle is diameter
				sz = np.random.randint(low=sz_lo//2,high=sz_hi//2,size=self.num_imgs_per_class)
				for s in sz:
					ind = np.where(sz == s)[0]
					lo  = s
					hi  = self.img_side - s
					x0[ind] = np.random.randint(low=lo,high=hi,size=ind.size)
					y0[ind] = np.random.randint(low=lo,high=hi,size=ind.size)
			else:
				#sz for square is side length
				sz = np.random.randint(low=sz_lo,high=sz_hi,size=self.num_imgs_per_class)
				for s in sz:
					ind = np.where(sz == s)[0]
					lo = 0
					hi = self.img_side - s
					x0[ind] = np.random.randint(low=lo,high=hi,size=ind.size)
					y0[ind] = np.random.randint(low=lo,high=hi,size=ind.size)
			labels.append(np.stack([cl,x0,y0,sz,br,th],axis=1))	
		labels = np.stack(labels)
		return labels.astype(int)


	def _draw_dataset(self):
		if self.dataset.exists():
			shutil.rmtree(self.dataset)
		self.dataset.mkdir(exist_ok=False)

		y_draw = self.random_labels()
		labels = []
		for i in range(self.num_classes):
			class_name = self.class_names[i]
			print('Drawing %s\n' % class_name)
			class_dir = self.dataset / class_name				
			class_dir.mkdir(exist_ok=False)
			y_class = y_draw[i]
			for j in tqdm(range(y_class.shape[0])):
				y = y_class[j]
				shape, ann = annotations_to_sample(y,self.img_side)
				shape = shape.reshape(self.img_side,self.img_side,1)
				np.save(class_dir/('%d.npy' % j),shape)
				self.pop_mean += shape.mean()
				self.pop_var  += shape.var()
				labels.append(ann) 
		self.pop_mean = self.pop_mean/(self.num_classes*self.num_imgs_per_class)
		self.pop_var  = self.pop_var/(self.num_classes*self.num_imgs_per_class)
		np.save(self.labels,np.vstack(labels))
		self._save_settings()

	def get_annotations(self):
		ANN = np.load(self.labels)
		return ANN

	def get_dataset_loader(self, batch_size, transforms=None, shuffle=False):
		return get_npy_dataloader(self.dataset, batch_size, transforms=transforms, shuffle=shuffle)
