import numpy as np
import pickle
from pathlib import Path
from draw.quick_draw import QuickDraw
from draw.spec_draw import SpecDraw
from utils import train_utils
from utils import other_utils

def sample_dataset(dataset_dir, savename, annotation_src=None):
	def _get_one_batch(loader):
		for batch in loader:
			x,y,fnames = batch			
			break
		return x,y.numpy(),fnames

	
	num_samples = 16
	loader = train_utils.get_npy_dataloader(dataset_dir,num_samples)

	x,y,fnames = _get_one_batch(loader)
	r,c = 2,8
	batch_imgs   = other_utils.as_rows_and_cols(x,r,c)
	title 	= '%s' % loader.dataset.class_to_idx

	if annotation_src is None:
		other_utils.plot_samples(batch_imgs,
				savename,
				title=title,
				labels=y.reshape(r,c,1))
	else:
		ann 	= other_utils.get_annotations_for_batch(fnames,annotation_src,loader.dataset.class_to_idx)		
		other_utils.plot_samples(batch_imgs,
				savename,
				title=title,
				labels=ann.reshape(r,c,-1))		

def check_qd_bins(root,bins):
	prefix_shown = False
	prefix_url   = 'https://console.cloud.google.com/storage/quickdraw_dataset/full/binary/'
	bins_present = True
	
	for bin in bins:
		bin_path = root/bin
		if not bin_path.exists():
			bins_present = False
			if not prefix_shown:
				print('From %s download:' % prefix_url)
				prefix_shown = True
			shape_bin = bin.split('_')[-1]
			print('%s to %s' %(shape_bin,root/bin))
	return bins_present


if __name__ == '__main__':
	dataset_settings = {
		'img_side' 			: 28,
		'qd_imgs_per_class'	: 25000,
		'sd_imgs_per_class'	: 5000,

	}

	draw_limits = {
		'sz_lo' : 5,
		'sz_hi' : 10,
		'br_lo' : 100,
		'br_hi' : 255,
		'th' 	: 1
	}

	plots_dir = Path('./plots')
	plots_dir.mkdir(exist_ok=True)

	qd_shapes_root = Path('./datasets/qd_shapes')
	qd_shapes_root.mkdir(parents=True,exist_ok=True)
	qd_shapes_bins = ['full_binary_circle.bin','full_binary_square.bin']
	if check_qd_bins(qd_shapes_root,qd_shapes_bins):
		print('All bins present, extracting qd-shapes')
		
		qd_shapes = QuickDraw(qd_shapes_root,
		dataset_settings['img_side'],
		dataset_settings['qd_imgs_per_class'],
		draw_limits['sz_lo']-1)
		print('Done')
		sample_dataset(qd_shapes.dataset,
		plots_dir/'1_qd_examples.png')

		y_est = pickle.load(open(qd_shapes.est_labels,'rb'))
		other_utils.plot_label_spread(plots_dir/'2_qd_est_label_spread.png',
			y_est,
			dataset_settings['img_side'],
			step=1)

	print('Drawing sd-shapes')
	sd_shapes_root = Path('./datasets/sd_shapes')
	sd_shapes_root.mkdir(exist_ok=True)
	
	sd_shapes = SpecDraw(sd_shapes_root,
	dataset_settings['img_side'],
	dataset_settings['sd_imgs_per_class'],
	draw_limits)

	sample_dataset(sd_shapes.dataset,
	('%s/3_sd_examples.png' % plots_dir),
	annotation_src=sd_shapes.labels)

	y_draw = pickle.load(open(sd_shapes.labels,'rb'))
	other_utils.plot_label_spread(plots_dir/'4_sd_label_spread.png',
			y_draw,
			dataset_settings['img_side'],
			step=1)

	# print('Done')

	# from torch.utils.data import DataLoader

	# ds = train_utils.DrawDataset(root=sd_shapes.dataset,
	# 	loader=train_utils.npy_loader,
	# 	extensions='.npy')
	# dl = DataLoader(dataset=ds,
	# 	batch_size=128,
	# 	shuffle=True)

	# x,y,p = next(iter(dl))