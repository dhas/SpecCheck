import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

class Encoder(nn.Module):	
	def __init__(self, net_name, input_dim, net_cfg, num_classes, dropout_rate, FC_size, norm_type='bn'):
		super(Encoder, self).__init__()
		self.net_name = net_name		
		self.features = self._make_layers(net_cfg,norm_type)
		
		net_cfg = np.array(net_cfg)
		num_pools = len(np.where(net_cfg == 'M')[0])		
		_f_dims = input_dim // (2**num_pools)		
		_f_channels = int(net_cfg[net_cfg != 'M'][-1])
		
		if dropout_rate > 0:
			self.classifier = nn.Sequential(
				nn.Linear(_f_channels*_f_dims*_f_dims, FC_size),
				nn.ReLU(True),
				nn.Dropout(p=dropout_rate),
				nn.Linear(FC_size, FC_size),
				nn.ReLU(True),
				nn.Dropout(p=dropout_rate),
				nn.Linear(FC_size, num_classes))
		else:
			self.classifier = nn.Sequential(
				nn.Linear(_f_channels*_f_dims*_f_dims, FC_size),
				nn.ReLU(True),				
				nn.Linear(FC_size, FC_size),
				nn.ReLU(True),				
				nn.Linear(FC_size, num_classes))


	def _make_layers(self, cfg, norm_type):
		layers = []
		in_channels = 1
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				if 'bn' == 'norm_type':
					layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
							nn.BatchNorm2d(x),
							nn.ReLU(inplace=True)]
				else:
					layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),							
							nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.features(x)        
		out = torch.flatten(out,1)
		out = self.classifier(out)
		return out

	def evaluate(self,x):
		return F.softmax(self.forward(x))