import json
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
from .quickdraw_bin_parser import unpack_drawings,vector_to_raster

class QuickDraw():

	def __init__(self,root_dir,side,num_imgs_per_class):
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