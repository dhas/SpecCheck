import numpy as np
import pickle
from pathlib import Path
from draw.quick_draw import QuickDraw
from draw.spec_draw import SpecDraw
from draw import load
from utils import other_utils
from config import config

def sample_dataset(dataset_dir, savename, annotations=None):
	def _get_one_batch(loader):
		for batch in loader:
			x,y,fnames = batch			
			break
		return x,y.numpy(),fnames

	
	num_samples = 16
	loader = load.get_npy_dataloader(dataset_dir,num_samples)

	x,y,fnames = _get_one_batch(loader)
	r,c = 2,8
	batch_imgs   = other_utils.as_rows_and_cols(x,r,c)
	title 	= '%s' % loader.dataset.class_to_idx

	if annotations is None:
		other_utils.plot_samples(batch_imgs,
				savename,
				title=title,
				labels=y.reshape(r,c,1))
	else:
		ann 	= other_utils.get_annotations_for_batch(fnames,annotations,loader.dataset.class_to_idx)		
		other_utils.plot_samples(batch_imgs,
				savename,
				title=title,
				labels=ann.reshape(r,c,-1))		

def check_qd_bins(root,bins,prefix_url):
	prefix_shown = False	
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


def main(project_root, plots_dir):
	plots_dir = plots_dir/'dataset'
	plots_dir.mkdir(exist_ok=True)
	
	draw_cfg = config.get_draw_config(project_root)
	qd_shapes_root = Path(draw_cfg['root'])/draw_cfg['qd']['root']
	qd_shapes_root.mkdir(parents=True,exist_ok=True)
	
	qd_shapes_bins = draw_cfg['qd']['bins']
	if check_qd_bins(qd_shapes_root,qd_shapes_bins,draw_cfg['qd']['bins_url']):
		print('All bins present, extracting qd-shapes')
		
		qd_shapes = QuickDraw(qd_shapes_root,
		draw_cfg['img_side'],
		draw_cfg['qd']['imgs_per_class'],
		draw_cfg['qd']['min_sz'])
		print('Done')
		sample_dataset(qd_shapes.dataset,
		plots_dir/'1_qd_examples.png')

		y_est = pickle.load(open(qd_shapes.est_labels,'rb'))
		other_utils.plot_label_spread(plots_dir/'2_qd_est_label_spread.png',
			y_est,
			draw_cfg['img_side'],
			step=1)

	print('Drawing sd-shapes')
	sd_shapes_root = Path(draw_cfg['root'])/draw_cfg['sd']['root']
	sd_shapes_root.mkdir(exist_ok=True)
	
	sd_shapes = SpecDraw(sd_shapes_root,
	draw_cfg['img_side'],
	draw_cfg['sd']['imgs_per_class'],
	draw_cfg['sd']['draw_limits'])

	sample_dataset(sd_shapes.dataset,
	('%s/3_sd_examples.png' % plots_dir),
	annotations=pickle.load(open(sd_shapes.labels,'rb')))

	y_draw = pickle.load(open(sd_shapes.labels,'rb'))
	other_utils.plot_label_spread(plots_dir/'4_sd_label_spread.png',
			y_draw,
			draw_cfg['img_side'],
			step=1)	

if __name__ == '__main__':
	main(Path('./'), Path('./_outputs'))