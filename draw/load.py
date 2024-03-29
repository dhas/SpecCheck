import numpy as np
import json
import torch
import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

def get_transformations(mean,var):
	trans = transforms.Compose([
				transforms.Lambda(lambda x:x.permute(2,0,1).type(torch.FloatTensor)),
				transforms.Normalize(mean=[mean], std=[np.sqrt(var)])
		])
	return trans


class my_subset(Dataset):
	"""r
	Subset of a dataset at specified indices.

	Arguments:
		dataset (Dataset): The whole Dataset
		indices (sequence): Indices in the whole set selected for subset
		labels(sequence) : targets as required for the indices. will be the same length as indices
	"""
	def __init__(self, dataset, indices,labels):
		self.dataset = dataset
		self.indices = indices
		labels_hold = torch.zeros(len(dataset)).type(torch.float) #( some number not present in the #labels just to make sure
		labels_hold[self.indices] = labels 
		self.labels = labels_hold
	def __getitem__(self, idx):
		image = self.dataset[self.indices[idx]][0]
		label = self.labels[self.indices[idx]]
		return (image, label)

	def __len__(self):
		return len(self.indices)


class DrawDataset(DatasetFolder):	

	def __getitem__(self,index):		
		path, target = self.samples[index]
		sample = self.loader(path)
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target, path



def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}


def npy_loader(path):
	img = torch.from_numpy(np.load(path))
	return img


def get_npy_dataloader(dataset_root,
		batch_size,
		shuffle=True,
		transforms=None):
	
	if transforms is None:
		dataset = DrawDataset(root=dataset_root,
			loader=npy_loader,
			extensions='.npy')
	else:
		dataset = DrawDataset(root=dataset_root,
			loader=npy_loader,
			extensions='.npy',
			transform=transforms)		

	dataloader = DataLoader(dataset=dataset,
		batch_size=batch_size,
		shuffle=shuffle)

	return dataloader


def get_npy_train_loader(dataset_root,
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