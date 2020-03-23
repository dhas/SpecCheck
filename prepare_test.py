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
import ood.explainV2 as explain
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

def explain_with_encoder_set(cfg, ds_root, os_ann, dim, test_root, explain_root, fontsize=20):
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
	figAUROC, axAUROC = plt.subplots(figsize=(30,10))
	figBaseUNC, axBaseUNC = plt.subplots(2, len(nets), figsize=(30,10))
	figCalUNC, axCalUNC = plt.subplots(2, len(nets), figsize=(30,10))
	figBaseROC, axBaseROC = plt.subplots(figsize=(30,10))
	figCalROC, axCalROC = plt.subplots(figsize=(30,10))
	figBaseSHAP, axBaseSHAP = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=(30,10))
	figCalSHAP, axCalSHAP = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=(30,10))

	
	for net_ind, net_name in enumerate(nets):
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
			eps, T = [0], [1]
			baseline_npz = encoders_root/net_name/('%s_baseline.npz' % net_name)
			if not baseline_npz.exists():
				PRD, HIT, ANN = explainer.evaluate(CUDA_DEVICE,
					criterion,ds_loader,
					eps, T,
					var=ds_settings['variance'],
					y_ann=ds_ann_npy)
				np.savez(baseline_npz, PRD=PRD, HIT=HIT, ANN=ANN)
			else:
				baseline = np.load(baseline_npz)
				PRD = baseline['PRD']
				ANN = baseline['ANN']
				HIT = baseline['HIT']
			

			_, _, idod = annotations.label_annotation_distribution(os_ann, ANN)			
			
			UNC = np.max(PRD[0,0], axis=1)
			explain.uncertainty_summary(UNC, 
				idod, net_name,
				axBaseUNC[0, net_ind], axBaseUNC[1, net_ind])			
			if net_ind == 0:
				axBaseUNC[0, net_ind].set_ylabel('ID samples', fontsize=fontsize)
				axBaseUNC[1, net_ind].set_ylabel('OOD samples', fontsize=fontsize)

			explain.roc_summary(UNC, idod, axBaseROC, net_name)
			SHAP = explain.shap_by_feature(UNC, ANN, explain_root/('%s_1_bas_shap.png' % net_name))
			explain.shap_summary(SHAP, ANN, net_name, axBaseSHAP)

			
			calibrated_npz = encoders_root/net_name/('%s_calibrated.npz' % net_name)			
			if calibrated_npz.exists():
				state = np.load(calibrated_npz)
				optT  = state['optT']
				T     = state['T']
				cPRD  = state['cPRD']
				cHIT  = state['cHIT']
			else:				
				optT    = explainer.calibrate_for_highest_auroc(idod, ds_loader)
				T = np.hstack([np.linspace(1, 10, num=10), [optT]])
				T = np.hstack([1/T[1:], T])
				T.sort()
				cPRD, cHIT, _ = explainer.evaluate(CUDA_DEVICE,
					criterion,ds_loader,
					eps, T,
					var=ds_settings['variance'],
					y_ann=ds_ann_npy)
				np.savez(calibrated_npz, optT=optT, T=T, cPRD=cPRD, cHIT=cHIT)

						
			calUNC    = np.max(cPRD[0,T == optT][0], axis=1)
			explain.uncertainty_summary(calUNC, idod, 
				net_name, axCalUNC[0, net_ind], axCalUNC[1, net_ind])
			explain.roc_summary(calUNC, idod, axCalROC, net_name)
			
			SHAP = explain.shap_by_feature(calUNC, ANN, explain_root/('%s_2_cal_shap.png' % net_name))
			explain.shap_summary(SHAP, ANN, net_name, axCalSHAP)

			aurocs = explain.roc_net(cPRD[0], idod, T, explain_root/('%s_3_roc.png' % (net_name)),
				explain_root/('%s_4_auroc.png' % (net_name)))
			# explainer.plot_inout_score_distributions(cPRD[0], idod, T, 
			# 	explain_root/('%s_3_score_inout.png' % (net_name)),
			# 	figsize=(40,20))

			axAUROC.plot(T, aurocs, label=('%s' % net_name))			

	
	figBaseUNC.savefig(explain_root/'0_bas_uncertainty_summary.png')
	figCalUNC.savefig(explain_root/'0_cal_uncertainty_summary.png')

	axBaseROC.legend(fontsize=fontsize-4, loc='lower right')
	axBaseROC.set_xlabel('False Positive Rate (FPR)', fontsize=fontsize)
	axBaseROC.set_ylabel('True Positive Rate (TPR)', fontsize=fontsize)
	figBaseROC.savefig(explain_root/'1_bas_roc_summary.png')

	axCalROC.legend(fontsize=fontsize-4, loc='lower right')
	axCalROC.set_xlabel('False Positive Rate (FPR)', fontsize=fontsize)
	axCalROC.set_ylabel('True Positive Rate (TPR)', fontsize=fontsize)
	figCalROC.savefig(explain_root/'1_cal_roc.png')
	
	axAUROC.legend(fontsize=fontsize-4, loc='lower right')
	figAUROC.savefig(explain_root/'2_auroc_summary.png')

	handles, labels = axBaseSHAP[0,0].get_legend_handles_labels()
	figBaseSHAP.legend(handles, labels, loc='upper right', fontsize=fontsize)
	figBaseSHAP.savefig(explain_root/'3_bas_shap_summary.png')

	handles, labels = axCalSHAP[0,0].get_legend_handles_labels()
	figCalSHAP.legend(handles, labels, loc='upper right', fontsize=fontsize)
	figCalSHAP.savefig(explain_root/'3_cal_shap_summary.png')