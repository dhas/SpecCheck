import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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


big_font = 20

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
fSFM, aSFM 			= plt.subplots(len(nets), 3, figsize=(30,10))
fLabSFM, aLabSFM 	= plt.subplots(2, 4, figsize=(30,10))
fLabSHAP, aLabSHAP 	= plt.subplots(2, 4, figsize=(30,10))

net_ind = 0
for net_name in nets: #['NET02']
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
		#uncomment this to run equivalence check
		# T   = [1]
		# xPRD, _, _ = cal_explainer.evaluate(CUDA_DEVICE,
		# 	criterion,sd_loader,
		# 	eps,T,
		# 	y_ann=sd_ann_dict)

		cal_T 	= cal_explainer.model.temperature.item() 
		T 		= [cal_T, 1, 1/cal_T]
		net_npz = cal_encoders/('%s.npz' % net_name)
		
		if not net_npz.exists():
			cPRD, cHIT, ANN = explainer.evaluate(CUDA_DEVICE,
				criterion,sd_loader,
				eps, T,
				var=sd_settings['variance'],
				y_ann=sd_ann_dict)
			np.savez(net_npz,cPRD=cPRD, cHIT=cHIT, ANN=ANN)
		else:
			state = np.load(net_npz)
			cPRD  = state['cPRD']
			cHIT  = state['cHIT']
			ANN   = state['ANN']

		#uncomment this to run equivalence check
		# if(False == np.allclose(xPRD[0,0], cPRD[0,0])):
		# 	raise Exception('Equivalence check failed')

		explainer.axes_softmax_distribution(cPRD[0], cHIT[0], 
			T, aSFM[net_ind], 
			net_name, fontsize=big_font)
				

		for label in range(len(sd_loader.dataset.classes)):
			label_name = sd_loader.dataset.classes[label]
			
			SFM_summary = explainer.plot_softmax_by_feature(cPRD, ANN, 
				label, eps, T,
				out_dir/('%s_1_softmax_%s.png' % (net_name, label_name)))
			
			SHAP_summary = explainer.plot_shap_by_feature(cPRD, ANN, 
				label, eps, T,
				out_dir/('%s_2_shap_%s.png' % (net_name, label_name)))

			if label == 0:
				set_title = True
			else:
				set_title = False
			explainer.summary_by_feature(SFM_summary, T, 0,
				net_name, aLabSFM[label], label_name, set_title=set_title)

			explainer.summary_by_feature(SHAP_summary, T, len(T)-1,
				net_name, aLabSHAP[label], label_name, set_title=set_title)

		
		net_ind += 1
		# break
fSFM.savefig(out_dir/'0_softmax.png')

handles, labels = aLabSFM[0,0].get_legend_handles_labels()
fLabSFM.legend(handles, labels, fontsize=big_font, loc='upper right')
fLabSFM.savefig(out_dir/'1_softmax_by_feature.png')

handles, labels = aLabSHAP[0,0].get_legend_handles_labels()
fLabSHAP.legend(handles, labels, fontsize=big_font, loc='upper right')
fLabSHAP.savefig(out_dir/'2_shap_by_feature.png')

