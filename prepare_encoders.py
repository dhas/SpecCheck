from pathlib import Path
import pickle
import json
import torch
from torch import nn, optim
from torchsummary import summary
from draw import load
from ood.encoder import Encoder
from ood.train import EncoderTrainer, training_necessary
from ood.explainV2 import BaselineExplainer
from utils import other_utils
from config import config
import numpy as np


def get_optimizer(parameters,settings):
	if settings['optim'] == 'Adam':
		optimizer = optim.Adam(parameters,lr = float(settings['lr']))
	else:
		raise Exception('Unknown optimizer')
	return optimizer


def prepare_encoder_set(project_root, ENCO_YML):
	draw_cfg = config.get_draw_config(project_root)
	enc_cfg  = config.get_encoders_config(ENCO_YML)

	encoders_root = project_root/enc_cfg['root']
	encoders_root.mkdir(exist_ok=True)

	encoders_root = encoders_root/enc_cfg['dataset']
	encoders_root.mkdir(exist_ok=True)

	ds = draw_cfg['datasets'][enc_cfg['dataset']]
	qd = ds['qd']
	qd_root 	= project_root/draw_cfg['main_root']/ds['root']/qd['root']
	qd_settings = load.read_settings(qd_root/'settings.json')
	qd_trans 	= load.get_transformations(qd_settings['mean'], qd_settings['variance'])
	input_shape = (1, ds['dim'], ds['dim'])

	train_params 	= enc_cfg['train']
	criterion 		= nn.CrossEntropyLoss()
	device 			= torch.device('cuda')
	nets = enc_cfg['nets']
	for net_name in nets:
		train_params['savedir'] = encoders_root/net_name
		train_params['savedir'].mkdir(exist_ok=True)
		checkpoint_path = train_params['savedir']/('%s.tar' % net_name)
		cfg_path        = train_params['savedir']/('%s.pkl' % net_name)	
		
		encoder = Encoder(net_name,
			ds['dim'],
			nets[net_name],
			ds['num_classes'])
		encoder.to(device)
		summary(encoder,input_shape)

		train_loader,val_loader = load.get_npy_train_loader(qd_root,
			train_params['batch_size'],
			train_params['batch_size'],
			transforms=qd_trans)	

		if not training_necessary(checkpoint_path,nets[net_name],cfg_path):
			print('Encoder %s already trained' % net_name)
		else:
			print('Training encoder %s' % net_name)			
			optimizer = get_optimizer(encoder.parameters(), 
				enc_cfg['train']['optimizer'])


			trainer = EncoderTrainer(encoder,device,
					train_loader,val_loader,
					criterion,optimizer,
					train_params)
			trainer.fit(checkpoint_path)
			pickle.dump(nets[net_name],open(cfg_path,'wb'))

if __name__ == '__main__':
	prepare_encoder_set(Path('./'), 'config/encoders_draw_28.yml')