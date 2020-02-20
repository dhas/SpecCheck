import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
import sys
sys.path.append('..')
from config import config
from draw import load
from ood.encoder import Encoder
from ood.train import training_necessary
from ood.explainV2 import OdinExplainer
sys.path.append('../../temperature_scaling/')
from temperature_scaling import ModelWithTemperature

def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}


CUDA_DEVICE = 0
project_root = Path('../')
out_dir = Path('_3_cal_temp_scaling')
out_dir.mkdir(exist_ok=True)

enc_prefix = project_root/'_outputs'
draw_cfg = config.get_draw_config(project_root)
enc_cfg  = config.get_encoders_config(project_root)

sd_dir 		= project_root/draw_cfg['root']/draw_cfg['sd']['root']/'dataset'
sd_settings = read_settings(sd_dir/'settings.json')
sd_trans 	= load.get_transformations(sd_settings['mean'], sd_settings['variance'])
sd_ann_dict	= pickle.load(open(sd_dir/'labels.pkl','rb'))

batch_size 	= 100
criterion 	= nn.CrossEntropyLoss()
sd_loader 	= load.get_npy_dataloader(sd_dir,batch_size,transforms=sd_trans,shuffle=False)

cal_encoders = out_dir/'cal_encoders'
cal_encoders.mkdir(exist_ok=True)


nets = enc_cfg['nets']
for net_name in ['NET02']: #nets
	print('\nProcessing %s' % net_name)
	encoder = Encoder(net_name,
			draw_cfg['img_side'],
			nets[net_name],
			draw_cfg['num_classes'])

	checkpoint_path = enc_prefix/enc_cfg['root']/net_name/('%s.tar' % net_name)
	cfg_path        = enc_prefix/enc_cfg['root']/net_name/('%s.pkl' % net_name)
	
	if training_necessary(checkpoint_path,nets[net_name],cfg_path):
		raise Exception('Source encoder unavailable')
	else:
		explainer = OdinExplainer(encoder, checkpoint_path)
		
		cal_encoder = ModelWithTemperature(encoder)
		cal_checkpoint_path = cal_encoders/('%s.tar' % net_name)
		if not cal_checkpoint_path.exists():			
			print('Calibrating')
			cal_encoder.set_temperature(sd_loader)		
			torch.save({'model_state_dict': cal_encoder.state_dict()},
					cal_checkpoint_path)

		cal_explainer = OdinExplainer(cal_encoder, cal_checkpoint_path, is_calibrated=True)		

		eps = [0]
		T   = [1]
		xPRD, _, _ = cal_explainer.evaluate(CUDA_DEVICE,
			criterion,sd_loader,
			eps,T,
			y_ann=sd_ann_dict)

		
		cal_T = [cal_explainer.model.temperature.item(), 1, 1/cal_explainer.model.temperature.item()]

		cPRD, cHIT, ANN = explainer.evaluate(CUDA_DEVICE,
			criterion,sd_loader,
			eps, cal_T,
			var=sd_settings['variance'],
			y_ann=sd_ann_dict)

		if(False == np.allclose(xPRD[0,0], cPRD[0,0])):
			raise Exception('Equivalence check failed')

		explainer.plot_softmax_distribution(cPRD, cHIT,
			eps, cal_T,
			out_dir/('%s_1_softmax' % net_name),
			figsize=(30,10))

		for label in range(len(sd_loader.dataset.classes)):
			label_name = sd_loader.dataset.classes[label]
			
			explainer.plot_softmax_by_feature(cPRD, ANN, 
				label, eps, cal_T,
				out_dir/('%s_2_softmax_%s.png' % (net_name, label_name)))

			explainer.plot_shap_by_feature(cPRD, ANN, 
				label, eps, cal_T,
				out_dir/('%s_3_shap_%s.png' % (net_name, label_name)))