import numpy as np
import torch
import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader,random_split

def npy_loader(path):
	img = torch.from_numpy(np.load(path))
	return img


def get_npy_dataloader(dataset_root,
		train_batch_size,
		validation_batch_size,
		train_set_proportion=0.8,
		train_set_shuffle=True,
		validation_set_shuffle=False,
		transforms=None):

	if transforms is None:
		dataset = DatasetFolder(root=str(dataset_root),
			loader=npy_loader,
			extensions='.npy')
	else:
		dataset = DatasetFolder(root=str(dataset_root),
			loader=npy_loader,
			extensions='.npy',
			transform=transforms)

	total_num_imgs 	= len(dataset)
	train_size 		= int(train_set_proportion * total_num_imgs)
	val_size   		= total_num_imgs - train_size
	train_ds,val_ds = random_split(dataset,[train_size,val_size])

	train_loader = DataLoader(dataset=train_ds,
		batch_size=train_batch_size,
		shuffle=train_set_shuffle)
	
	val_loader = DataLoader(dataset=val_ds,
		batch_size=validation_batch_size,
		shuffle=validation_set_shuffle)
	
	return train_loader,val_loader