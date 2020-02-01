import numpy as np
import cv2
from pathlib import Path
import shutil
import json
from tqdm import tqdm


class SpecDraw:
	def __init__(self,root_dir,side,num_imgs_per_class):
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



	#static method
	def draw_shape(shape,x,y,r,br,t,side):

		def _circle_opencv(xm,ym,r,br,t,side):
			canvas = np.zeros(shape=(side,side))	
			img = cv2.circle(canvas,(xm,ym),r,int(br),t)
			return img

		def _square_opencv(x0,y0,r,br,t,side):
			canvas = np.zeros(shape=(side,side))	
			img    = cv2.rectangle(canvas,(x0,y0),(x0+r,y0+r),int(br),t)
			return img

		switcher = {
			'circle': _circle_opencv,
			'square': _square_opencv
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


	def _draw_dataset(self):
		if self.dataset.exists():
			shutil.rmtree(self.dataset)
		self.dataset.mkdir(exist_ok=False)

		y_draw = self.random_labels()
		# utils.plot_label_spread(self.dataset/'label_spread.png',y_draw,self.img_side)
		for i in range(self.num_classes):
			class_name = self.class_names[i]
			print('Drawing %s\n' % class_name)
			class_dir = self.dataset / class_name				
			class_dir.mkdir(exist_ok=False)
			y_class = y_draw[np.where(y_draw[:,0] == i)]
			for j in tqdm(range(y_class.shape[0])):
				y = y_class[j]
				shape = SpecDraw.draw_shape(class_name,y[1],y[2],y[3],y[4],y[5],self.img_side)
				shape = shape.reshape(self.img_side,self.img_side,1)
				np.save(class_dir/('%d.npy' % j),shape)
		np.save(self.dataset/'labels.npy',y_draw)
		self._save_settings()