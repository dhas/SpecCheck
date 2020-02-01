import numpy as np
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


dataset_settings = {
	'img_side' 			: 128,
	'num_imgs_per_class': 60000
}


plots_dir = './plots'
qd_shapes_root = './datasets/qd_shapes'
qd_shapes = QuickDraw(qd_shapes_root,
	dataset_settings['img_side'],
	dataset_settings['num_imgs_per_class'])

sample_dataset(qd_shapes.dataset,
	('%s/1_qd_trn_examples.png' % plots_dir),
	('%s/2_qd_val_examples.png' % plots_dir))

sd_shapes_root = './datasets/sd_shapes'
sd_shapes = SpecDraw(sd_shapes_root,
	dataset_settings['img_side'],
	dataset_settings['num_imgs_per_class'])

sample_dataset(sd_shapes.dataset,
	('%s/3_sd_trn_examples.png' % plots_dir),
	('%s/4_sd_val_examples.png' % plots_dir))


# qd_train_loader, qd_val_loader = train_utils.get_npy_dataloader(qd_shapes.dataset,
# 	train_settings['batch_size'],
# 	train_settings['batch_size'])

# for batch in qd_train_loader:
# 		x,y = batch		
# 		x,y = other_utils.sample_from_batch(x,y,16)
# 		labels = np.array([qd_train_loader.dataset.dataset.classes[l] for l in y]).reshape(2,8,1)
# 		other_utils.plot_samples(other_utils.as_rows_and_cols(x,2,8),
# 			('%s/1_qd_examples.png' % plots_dir),
# 			labels=labels)
# 		break