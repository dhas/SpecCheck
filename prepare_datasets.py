import numpy as np
from pathlib import Path
from draw.quick_draw import QuickDraw
from draw.spec_draw import SpecDraw
from utils import train_utils
from utils import other_utils

def sample_dataset(dataset_dir, train_savename, val_savename):
	def _get_one_batch(loader):
		for batch in train_loader:
			x,y = batch
			labels = np.array([train_loader.dataset.dataset.classes[l] for l in y])
			break
		return x,labels

	
	num_samples = 16
	train_loader,val_loader = train_utils.get_npy_dataloader(dataset_dir,
	num_samples,
	num_samples)

	x,labels = _get_one_batch(train_loader)
	other_utils.plot_samples(other_utils.as_rows_and_cols(x,2,8),
			train_savename,
			labels=labels.reshape(2,8,1))

	x,labels = _get_one_batch(val_loader)
	other_utils.plot_samples(other_utils.as_rows_and_cols(x,2,8),
			val_savename,
			labels=labels.reshape(2,8,1))


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
		'img_side' 			: 128,
		'num_imgs_per_class': 60000
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
		dataset_settings['num_imgs_per_class'])
		print('Done')
		sample_dataset(qd_shapes.dataset,
		plots_dir/'1_qd_trn_examples.png',
		plots_dir/'2_qd_val_examples.png')

	print('Drawing sd-shapes')
	sd_shapes_root = Path('./datasets/sd_shapes')
	sd_shapes_root.mkdir(exist_ok=True)
	sd_shapes = SpecDraw(sd_shapes_root,
	dataset_settings['img_side'],
	dataset_settings['num_imgs_per_class'])
	sample_dataset(sd_shapes.dataset,
	('%s/3_sd_trn_examples.png' % plots_dir),
	('%s/4_sd_val_examples.png' % plots_dir))
	print('Done')