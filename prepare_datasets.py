import numpy as np
import pickle
from pathlib import Path
from draw.quick_draw import QuickDraw
from draw.spec_draw import SpecDraw
from draw.annotations import plot_samples, plot_annotation_distribution, get_annotations_for_batch, class_to_label_dict
from draw import load
from utils import other_utils
from config import config


def sample_dataset(loader, savename, annotations=None):
	def _get_one_batch(loader):
		for batch in loader:
			x,y,fnames = batch			
			break
		return x,y.numpy(),fnames

	x,y,fnames = _get_one_batch(loader)
	r,c = 2,8
	batch_imgs   = other_utils.as_rows_and_cols(x,r,c)

	if annotations is None:
		plot_samples(batch_imgs,
				savename, labels=y.reshape(r,c,1))
	else:
		ann = get_annotations_for_batch(fnames,annotations)
		plot_samples(batch_imgs,
				savename, labels=ann.reshape(r,c,-1))		

def sources_available(root,sources,sources_url):
	prefix_shown = False	
	bins_present = True
	
	for src in sources:
		src_path = root/src
		if not src_path.exists():
			srcs_present = False
			if not prefix_shown:
				print('From %s download:' % sources_url)
				prefix_shown = True
			src_name = src.split('_')[-1]
			print('%s to %s' %(src_name,src_path))
	return bins_present


def main(project_root):	
	draw_cfg 	= config.get_draw_config(project_root)	
	main_root 	= project_root/draw_cfg['main_root']
	main_root.mkdir(exist_ok=True)
	sources_root = main_root/draw_cfg['sources']
	sources_root.mkdir(exist_ok=True)
	
	sample_batch_size = 16

	datasets 	= draw_cfg['datasets']	
	
	for ds_name in datasets:
		print('Preparing %s' % ds_name)
		ds 		= datasets[ds_name]
		ds_root = main_root/ds['root']
		ds_root.mkdir(exist_ok=True)
		ds_plots = ds_root/'plots'
		ds_plots.mkdir(exist_ok=True)

		#QD
		qd_ds   = ds['qd']
		qd_root = ds_root/qd_ds['root']
		qd_root.mkdir(exist_ok=True)		
		if not sources_available(sources_root, 
			qd_ds['sources'], qd_ds['sources_url']):
			return

		source_paths = [sources_root/src for src in qd_ds['sources']]

		qd_shapes = QuickDraw(qd_root,ds['dim'],
			source_paths,qd_ds['imgs_per_class'],
			qd_ds['min_sz'])

		qd_loader = qd_shapes.get_dataset_loader(sample_batch_size,
			shuffle=True)
		sample_dataset(qd_loader, ds_plots/'1_qd_examples.png')

		qd_ANN = qd_shapes.get_annotations()		
		plot_annotation_distribution(qd_ANN, ds['dim'], ds_plots/'2_qd_annotations.png')

		#SD
		sd_ds  	= ds['sd']
		sd_root = ds_root/sd_ds['root']
		sd_root.mkdir(exist_ok=True)

		sd_shapes = SpecDraw(sd_root,
		ds['dim'],
		sd_ds['imgs_per_class'],
		sd_ds['draw_limits'])

		sd_ANN    = sd_shapes.get_annotations()
		sd_loader = sd_shapes.get_dataset_loader(sample_batch_size,
			shuffle=True)
		sample_dataset(sd_loader, ds_plots/'3_sd_examples.png',
			annotations=sd_ANN)
		plot_annotation_distribution(sd_ANN, ds['dim'], ds_plots/'4_sd_annotations.png')

if __name__ == '__main__':
	main(Path('./'))