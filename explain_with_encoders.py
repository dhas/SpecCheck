import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch import nn
from config import config
from draw import load, annotations
from ood.encoder import Encoder
from ood.train import training_necessary
from ood.explainV2 import OdinExplainer
import sys
sys.path.append('../temperature_scaling/')
from temperature_scaling import ModelWithTemperature


def explain_with_encoder_set(project_root, ENCO_YML):
	fontsize = 20
	explain_root = project_root/'_explanations'
	explain_root.mkdir(exist_ok=True)

	CUDA_DEVICE = 0
	enc_cfg  = config.get_encoders_config(ENCO_YML)
	draw_cfg = config.get_draw_config(project_root)
	ds 		 = draw_cfg['datasets'][enc_cfg['dataset']]
	sd 		 = ds['sd']
	
	sd_root  	= project_root/draw_cfg['main_root']/ds['root']/sd['root']
	sd_settings = load.read_settings(sd_root/'settings.json')
	sd_trans 	= load.get_transformations(sd_settings['mean'], sd_settings['variance'])
	sd_ann_npy	= np.load(sd_root/'labels.npy')
	
	batch_size 	= 100
	criterion 	= nn.CrossEntropyLoss()
	sd_loader 	= load.get_npy_dataloader(sd_root,batch_size,transforms=sd_trans,shuffle=False)

	encoders_root = project_root/enc_cfg['root']/enc_cfg['dataset']
	cal_encoders  = encoders_root/'cal_encoders'
	cal_encoders.mkdir(exist_ok=True)

	nets = enc_cfg['nets']
	fSFM, aSFM 			= plt.subplots(len(nets), 3, figsize=(30,10))	
	fLabSFM, aLabSFM 	= plt.subplots(2, 4, figsize=(30,10))
	fLabSHAP, aLabSHAP 	= plt.subplots(2, 4, figsize=(30,10))

	net_ind = 0
	for net_name in nets:
		print('\nProcessing %s' % net_name)
		encoder = Encoder(net_name,
				ds['dim'],
				nets[net_name],
				ds['num_classes'])

		checkpoint_path = encoders_root/net_name/('%s.tar' % net_name)
		cfg_path        = encoders_root/net_name/('%s.pkl' % net_name)

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
			else:
				checkpoint = torch.load(cal_checkpoint_path)
				cal_encoder.load_state_dict(checkpoint['model_state_dict'])

			cal_T 	= cal_encoder.temperature.item() 
			T 		= [cal_T, 1, 1/cal_T]
			eps     = [0]
			net_npz = cal_encoders/('%s.npz' % net_name)
			if not net_npz.exists():
				cPRD, cHIT, ANN = explainer.evaluate(CUDA_DEVICE,
					criterion,sd_loader,
					eps, T,
					var=sd_settings['variance'],
					y_ann=sd_ann_npy)
				np.savez(net_npz,cPRD=cPRD, cHIT=cHIT, ANN=ANN)
			else:
				state = np.load(net_npz)
				cPRD  = state['cPRD']
				cHIT  = state['cHIT']
				ANN   = state['ANN']

			explainer.axes_softmax_distribution(cPRD[0], cHIT[0], 
				T, aSFM[net_ind], 
				net_name)

			for label in range(len(annotations.classes)):
				label_name = annotations.classes[label]

				SFM_summary = explainer.plot_softmax_by_feature(cPRD, ANN, 
					label, eps, T,
					explain_root/('%s_1_softmax_%s.png' % (net_name, label_name)))

				SHAP_summary = explainer.plot_shap_by_feature(cPRD, ANN, 
					label, eps, T,
					explain_root/('%s_2_shap_%s.png' % (net_name, label_name)))

				if label == 0:
					set_title = True
				else:
					set_title = False
				
				explainer.summary_by_feature(SFM_summary, T, 0,
					net_name, aLabSFM[label], label_name, set_title=set_title)

				explainer.summary_by_feature(SHAP_summary, T, len(T)-1,
					net_name, aLabSHAP[label], label_name, set_title=set_title)

			net_ind += 1

	fSFM.savefig(explain_root/'0_softmax.png')

	handles, labels = aLabSFM[0,0].get_legend_handles_labels()
	fLabSFM.legend(handles, labels, fontsize=fontsize, loc='upper right')
	fLabSFM.savefig(explain_root/'1_softmax_by_feature.png')

	handles, labels = aLabSHAP[0,0].get_legend_handles_labels()
	fLabSHAP.legend(handles, labels, fontsize=fontsize, loc='upper right')
	fLabSHAP.savefig(explain_root/'2_shap_by_feature.png')


	
if __name__ == '__main__':
	explain_with_encoder_set(Path('./'), 'config/encoders_draw_28.yml')