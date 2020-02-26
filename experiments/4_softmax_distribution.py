import numpy as np
from pathlib import Path
import yaml
import json
import torch
from torch import nn
import sys
sys.path.append('..')
from draw import load, annotations
from ood.encoder import Encoder
from ood.train import training_necessary
from ood.explainV2 import OdinExplainer
sys.path.append('../../temperature_scaling/')
from temperature_scaling import ModelWithTemperature

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)

def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}

CUDA_DEVICE = 0
out_dir = Path('_4_softmax_distribution')
out_dir.mkdir(exist_ok=True)
tests_root = out_dir

test_cfg = load_config('../config/test_draw_28.yml')

test_root   = tests_root/test_cfg['test_root']
# cs_root 	= test_root/test_cfg['coverage_set']['root']
# cs_settings = read_settings(cs_root/'settings.json')
# cs_trans 	= load.get_transformations(cs_settings['mean'], cs_settings['variance'])
# cs_ann_npy	= np.load(cs_root/'labels.npy')


# batch_size 	= 100
# criterion 	= nn.CrossEntropyLoss()
# cs_loader 	= load.get_npy_dataloader(cs_root,batch_size,transforms=cs_trans,shuffle=False)


enc_cfg       = test_cfg['encoders']
encoders_root = test_root/enc_cfg['root']
cal_encoders  = encoders_root/'cal_encoders'

nets = enc_cfg['nets']
temp_spread  = 6
temp_exp     = np.flip(np.arange(-temp_spread, temp_spread + 1, 2))

for net_name in nets:
	print('\nProcessing %s' % net_name)
	encoder = Encoder(net_name,
			test_cfg['dim'],
			nets[net_name],
			test_cfg['num_classes'])
	checkpoint_path = encoders_root/net_name/('%s.tar' % net_name)
	cfg_path        = encoders_root/net_name/('%s.pkl' % net_name)
	if training_necessary(checkpoint_path,nets[net_name],cfg_path):
		raise Exception('Encoder %s unavailable' % net_name)
	else:
		explainer = OdinExplainer(encoder, checkpoint_path)
		# cal_encoder = ModelWithTemperature(encoder)
		# cal_checkpoint_path = cal_encoders/('%s.tar' % net_name)
		# if not cal_checkpoint_path.exists():			
		# 		print('Calibrating')
		# 		cal_encoder.set_temperature(cs_loader)		
		# 		torch.save({'model_state_dict': cal_encoder.state_dict()},
		# 				cal_checkpoint_path)
		# else:
		# 	checkpoint = torch.load(cal_checkpoint_path)
		# 	cal_encoder.load_state_dict(checkpoint['model_state_dict'])

		# cal_T 	= cal_encoder.temperature.item()
		# T       = np.sort([cal_T**p for p in temp_exp])
		eps     = [0]
		net_npz = encoders_root/net_name/('%s.npz' % net_name)
		# state_changed = True
		# if (net_npz.exists()):
		# 	state = np.load(net_npz)
		# 	state['T']
		# 	if (len(state['T']) == len(T)) and (np.allclose(state['T'],T)):
		# 		cPRD  = state['cPRD']
		# 		cHIT  = state['cHIT']
		# 		ANN   = state['ANN']
		# 		state_changed = False
		
		# if state_changed:
		# 	cPRD, cHIT, ANN = explainer.evaluate(CUDA_DEVICE,
		# 		criterion,cs_loader,
		# 		eps, T,
		# 		var=cs_settings['variance'],
		# 		y_ann=cs_ann_npy)
		# 	np.savez(net_npz, T=T, cPRD=cPRD, cHIT=cHIT, ANN=ANN)

		if not net_npz.exists():
			raise Exception('Saved state unavailable')
		else:
			state = np.load(net_npz)
			T = state['T']			
			cPRD  = state['cPRD']
			cHIT  = state['cHIT']
			ANN   = state['ANN']


		for label in range(len(annotations.classes)):
				label_name = annotations.classes[label]

				SFM_summary = explainer.plot_softmax_by_feature(cPRD, ANN, 
					label, eps, T,
					out_dir/('%s_1_softmax_%s.png' % (net_name, label_name)))

				SHAP_summary = explainer.plot_shap_by_feature(cPRD, ANN, 
					label, eps, T,
					out_dir/('%s_2_shap_%s.png' % (net_name, label_name)),
					softmax_scale=1)