import json
import pickle
from pathlib import Path
import numpy as np
import sys
sys.path.append('..')
from config import config
from draw import load
from ood.encoder import Encoder
from ood.train import training_necessary
from ood.explain import DatasetExplainer
from torch import nn

def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}

def ann_match(sann, classes):
	qd_ann_range = {
		'circle':{
			'x_lo' : 12, 	'x_hi' : 14,
			'y_lo' : 12, 	'y_hi' : 14,
			's_lo' : 8 , 	's_hi' : 12,
			'b_lo' : 115, 	'b_hi' : 231
		},

		'square':{
			'x_lo' : 0, 	'x_hi' : 11,
			'y_lo' : 0, 	'y_hi' : 11,
			's_lo' : 15, 	's_hi' : 25,
			'b_lo' : 91, 	'b_hi' : 235
		}
	}

	spread = qd_ann_range[classes[sann[0]]]
	match  = (sann[1] >= spread['x_lo'] and sann[1] <= spread['x_hi']) and \
			 (sann[2] >= spread['y_lo'] and sann[2] <= spread['y_hi']) and \
			 (sann[3] >= spread['s_lo'] and sann[3] <= spread['s_hi']) and \
			 (sann[4] >= spread['b_lo'] and sann[4] <= spread['b_hi'])

	return (1 if match else 0)

CUDA_DEVICE = 0
project_root = Path('../')
out_dir = Path('_2_sd_shap')
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
		raise Exception('Trained encoder unavailable')
	else:		
		explainer = DatasetExplainer(encoder, checkpoint_path) 
		eps = np.linspace(0,1,5)
		T   = np.linspace(1,20,5)
		_, PRD, ANN, _ = explainer._evaluate_odin_batch(CUDA_DEVICE,criterion,sd_loader,
			T=T,eps=eps,
			var=sd_settings['variance'],
			y_ann=sd_ann_dict)
		
		YINOUT = np.array([ann_match(ann,sd_loader.dataset.classes) for ann in ANN])

		for label in range(len(sd_loader.dataset.classes)):
			explainer.shap_spread_explain(ANN,PRD,YINOUT,
				eps, T, label, 
				out_dir/('%s_%s.png' % (net_name,sd_loader.dataset.classes[label])),
				eps0_savename=out_dir/('%s_%s_eps_0.png' % (net_name,sd_loader.dataset.classes[label])))