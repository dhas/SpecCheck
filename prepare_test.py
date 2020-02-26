import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import torch
from torch import nn, optim
from torchsummary import summary
from draw.quick_draw import QuickDraw
from draw.spec_draw import SpecDraw
from draw import load, annotations
from utils import other_utils
from ood.encoder import Encoder
from ood.train import EncoderTrainer, training_necessary
from ood.explainV2 import OdinExplainer
import sys
sys.path.append('../temperature_scaling/')
from temperature_scaling import ModelWithTemperature

def sources_available(sources,sources_url):
	prefix_shown = False	
	srcs_present = True
	
	for src_path in sources:
		if not src_path.exists():
			srcs_present = False
			if not prefix_shown:
				print('From %s download:' % sources_url)
				prefix_shown = True
			src_name = str(src_path).split('_')[-1]
			print('%s to %s' %(src_name,src_path))
	return srcs_present

def sample_dataset(root, savename, ann=None):
	def _get_one_batch(loader):
		for batch in loader:
			x,y,fnames = batch			
			break
		return x,y.numpy(),fnames

	batch_size = 16
	r,c = 2,8
	
	loader = load.get_npy_dataloader(root,batch_size)
	x,y,fnames = _get_one_batch(loader)
	batch_imgs   = other_utils.as_rows_and_cols(x,r,c)

	if ann is None:
		annotations.plot_samples(batch_imgs,
				savename, labels=y.reshape(r,c,1))
	else:
		ann = annotations.get_annotations_for_batch(fnames,ann)
		annotations.plot_samples(batch_imgs,
				savename, labels=ann.reshape(r,c,-1))		

def prepare_observed_set(cfg, dim, num_classes, test_root, sources):
	qd_root = test_root/cfg['root']
	qd_root.mkdir(exist_ok=True)

	source_paths = [sources/src for src in cfg['sources']]
	if not sources_available(source_paths, cfg['sources_url']):
		return

	qd_shapes = QuickDraw(qd_root,dim,
			source_paths,cfg['imgs_per_class'],
			cfg['min_sz'])
	sample_dataset(qd_root, qd_root/'samples.png')

def prepare_coverage_set(cfg, dim, num_classes, test_root):
	sd_root = test_root/cfg['root']
	sd_root.mkdir(exist_ok=True)

	sd_shapes = SpecDraw(sd_root,
		dim,
		cfg['imgs_per_class'],
		cfg['draw_limits'])
	sd_ann = np.load(sd_root/'labels.npy')
	sample_dataset(sd_root, sd_root/'samples.png', ann=sd_ann)

def get_optimizer(parameters,settings):
	if settings['optim'] == 'Adam':
		optimizer = optim.Adam(parameters,lr = float(settings['lr']))
	else:
		raise Exception('Unknown optimizer')
	return optimizer

def prepare_encoders(cfg, dim, num_classes, ds_root, test_root):
	encoders_root = test_root/cfg['root']
	encoders_root.mkdir(exist_ok=True)
	
	ds_settings = load.read_settings(ds_root/'settings.json')
	ds_trans    = load.get_transformations(ds_settings['mean'], 
		ds_settings['variance'])
	
	input_shape 	= (1, dim, dim)
	train_params 	= cfg['train']
	criterion 		= nn.CrossEntropyLoss()
	device 			= torch.device('cuda')
	
	nets = cfg['nets']
	for net_name in nets:
		train_params['savedir'] = encoders_root/net_name
		train_params['savedir'].mkdir(exist_ok=True)
		checkpoint_path = train_params['savedir']/('%s.tar' % net_name)
		cfg_path        = train_params['savedir']/('%s.pkl' % net_name)

		encoder = Encoder(net_name,
			dim, nets[net_name],
			num_classes)
		encoder.to(device)
		summary(encoder,input_shape)

		train_loader,val_loader = load.get_npy_train_loader(ds_root,
			train_params['batch_size'],
			train_params['batch_size'],
			transforms=ds_trans)

		if not training_necessary(checkpoint_path,nets[net_name],cfg_path):
			print('Encoder %s already trained' % net_name)
		else:
			print('Training encoder %s' % net_name)			
			optimizer = get_optimizer(encoder.parameters(), 
				cfg['train']['optimizer'])
			trainer = EncoderTrainer(encoder,device,
					train_loader,val_loader,
					criterion,optimizer,
					train_params)
			trainer.fit(checkpoint_path)
			pickle.dump(nets[net_name],open(cfg_path,'wb'))

def explain_with_encoder_set(cfg, ds_root, dim, test_root, explain_root, fontsize=20):
	explain_root = explain_root/'encoder_explanations'
	explain_root.mkdir(exist_ok=True)

	CUDA_DEVICE = 0
	ds_settings = load.read_settings(ds_root/'settings.json')
	ds_trans    = load.get_transformations(ds_settings['mean'], 
		ds_settings['variance'])
	ds_ann_npy  = np.load(ds_root/'labels.npy')

	batch_size 	= 100
	criterion 	= nn.CrossEntropyLoss()
	ds_loader 	= load.get_npy_dataloader(ds_root,batch_size,transforms=ds_trans,shuffle=False)

	encoders_root = test_root/cfg['root']
	cal_encoders  = encoders_root/'cal_encoders'
	cal_encoders.mkdir(exist_ok=True)

	nets = cfg['nets']
	num_classes  = len(annotations.classes)
	num_features = len(annotations.FEATS)
	temp_spread  = 2
	temp_exp     = np.flip(np.arange(-temp_spread, temp_spread + 1, 2))
	fSFM, aSFM 	 = plt.subplots(len(nets), len(temp_exp), figsize=(30,10))
	aSFM = aSFM.reshape(len(nets), len(temp_exp))
	fLabSFM, aLabSFM 	= plt.subplots(num_classes, num_features, figsize=(30,10))
	fLabSHAP, aLabSHAP 	= plt.subplots(num_classes, num_features, figsize=(30,10))

	net_ind = 0
	for net_name in nets:
		print('\nProcessing %s' % net_name)
		encoder = Encoder(net_name,
				dim,
				nets[net_name],
				num_classes)

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
				cal_encoder.set_temperature(ds_loader)		
				torch.save({'model_state_dict': cal_encoder.state_dict()},
						cal_checkpoint_path)
			else:
				checkpoint = torch.load(cal_checkpoint_path)
				cal_encoder.load_state_dict(checkpoint['model_state_dict'])

			cal_T 	= cal_encoder.temperature.item()
			if (abs(cal_T) > 1) and (abs(cal_T) < 2):
				cal_T = 2.0
			T       = np.sort([cal_T**p for p in temp_exp])
			# cal_T = 0.00001
			# T 		= [cal_T, 1, 1/cal_T]
			eps     = [0]
			net_npz = encoders_root/net_name/('%s.npz' % net_name)
			state_changed = True
			if (net_npz.exists()):
				state = np.load(net_npz)
				state['T']
				if (len(state['T']) == len(T)) and (np.allclose(state['T'],T)):
					cPRD  = state['cPRD']
					cHIT  = state['cHIT']
					ANN   = state['ANN']
					state_changed = False
			
			if state_changed:
				cPRD, cHIT, ANN = explainer.evaluate(CUDA_DEVICE,
					criterion,ds_loader,
					eps, T,
					var=ds_settings['variance'],
					y_ann=ds_ann_npy)
				np.savez(net_npz, T=T, cPRD=cPRD, cHIT=cHIT, ANN=ANN)			

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
					explain_root/('%s_2_shap_%s.png' % (net_name, label_name)),
					softmax_scale=1)

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