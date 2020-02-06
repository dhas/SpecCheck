import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

class Encoder(nn.Module):	
	def __init__(self, net_name, input_dim, net_cfg, num_classes):
		super(Encoder, self).__init__()
		self.net_name = net_name		
		self.head = self._make_head(net_cfg['head'])
		
		net_cfg_arr = np.array(net_cfg['head'])
		num_pools = len(np.where(net_cfg_arr == 'M')[0])		
		_f_dims = input_dim // (2**num_pools)		
		_f_channels = int(net_cfg_arr[net_cfg_arr != 'M'][-1].split('#')[0])

		self.tail = self._make_tail(net_cfg['tail'],
				_f_channels*_f_dims*_f_dims,
				num_classes)

	def _make_tail(self, cfg, in_channels, num_classes):
		layers = []
		for x in cfg:
			if x.startswith('D#'):
				dropout_rate = float(x.split('#')[-1])
				layers += [nn.Dropout(p=dropout_rate)]
			else:				
				layers += [nn.Linear(in_channels, int(x)),
						   nn.ReLU(True)]
				in_channels = int(x)
		layers += [nn.Linear(in_channels, num_classes)]
		return nn.Sequential(*layers)


	def _make_head(self, cfg):
		layers = []
		in_channels = 1
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				block_cfg = x.split('#')
				out_channels = int(block_cfg[0])
				block_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
				
				if len(block_cfg) > 1:
					if block_cfg[1] == 'BN':
						block_layers += [nn.BatchNorm2d(out_channels)]
					elif block_cfg[1] == 'IN':
						block_layers += [nn.InstanceNorm2d(out_channels)]					
				block_layers += [nn.ReLU(inplace=True)]
				layers += block_layers					
				in_channels = out_channels
						
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.head(x)        
		out = torch.flatten(out,1)
		out = self.tail(out)
		return out

	def evaluate(self,x):
		return F.softmax(self.forward(x))