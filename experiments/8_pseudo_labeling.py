from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import torch
from torch import nn
import sys
sys.path.append('..')
from draw import load, annotations
from ood.encoder import Encoder
from ood.train import training_necessary
from ood.explainV2 import OdinExplainer
from ood.calibration import ModelWithTemperature
import pandas as pd

pos_label = annotations.id_label
neg_label = annotations.od_label

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)

def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}

def plot_distributions(complete, id, od, savename):
	fig, axs = plt.subplots(1, 3, figsize=figsize)

	ax = axs[0]
	ax.hist(complete, color='blue')

	ax = axs[1]
	ax.hist(id, color='green')

	ax = axs[2]
	ax.hist(od, color='red')

	fig.savefig(savename)

def plot_T_sampling_dist(scores, T, id_ind, od_ind, savename):
	fig, axs = plt.subplots(2, len(T), figsize=figsize)
	for i,t in enumerate(T):
		ax = axs[0,i]
		ax.hist(scores[i, id_ind], color='blue')
		ax.set_xlim([0.5, 1.0])
		mse_id = np.square(1.0 - scores[i,id_ind]).mean()
		ax.set_xlabel('MSE-%0.3f' % mse_id, fontsize=fontsize)
		ax.set_title('T-%0.3f' % t, fontsize=fontsize)

		ax = axs[1,i]
		ax.hist(scores[i, od_ind], color='red')
		ax.set_xlim([0.5, 1.0])
		mse_od = np.square(scores[i,od_ind] - 0.5).mean()
		ax.set_xlabel('MSE-%0.3f' % mse_od, fontsize=fontsize)

	fig.savefig(savename)


def _plot_distributions(scores, T, blabel, savename):
	fig, axs = plt.subplots(len(T), 3, figsize=figsize)
	for t_ind in range(len(T)):
		ax = axs[t_ind, 0]
		ax.hist(scores[t_ind], color='blue')
		ax.set_xlim([0.5, 1.0])
		ax = axs[t_ind, 1]
		ax.hist(scores[t_ind][blabel == pos_label], color='green')
		ax.set_xlim([0.5, 1.0])
		ax = axs[t_ind, 2]
		ax.hist(scores[t_ind][blabel == neg_label], color='red')
		ax.set_xlim([0.5, 1.0])
	fig.savefig(savename)


def pseudo_label(scores, p, tpr=1.0, tnr=1.0):
	pind       = math.floor(p*len(scores))
	sorted_ind = scores.argsort()

	top_pind = sorted_ind[-pind:] #np.argpartition(scores, pind)[-pind:]
	bot_pind = sorted_ind[:pind] #np.argpartition(scores, pind)[:pind]
		
	top_y    = np.full(len(top_pind), pos_label)
	if tpr < 1.0:
		fp_ind   = np.random.randint(0,len(top_pind), size=math.floor((1-tpr)*len(top_pind)))
		top_y[fp_ind] = neg_label
	
	bot_y 	 = np.full(len(bot_pind), neg_label)
	if tnr < 1.0:
		fn_ind   = np.random.randint(0,len(bot_pind), size=math.floor((1-tnr)*len(bot_pind)))
		bot_y[fn_ind] = pos_label

	return np.hstack([bot_pind,top_pind]), np.hstack([bot_y,top_y])


def thr_pseudo_label(scores, threshold):
	plabel = np.full(len(scores), 1)
	od_ind = np.where(scores < threshold)[0]
	plabel[od_ind] = 0
	return plabel

def limit_pseudo_label(scores, lo_thr, hi_thr):
	lo_ind = np.where(scores < lo_thr)[0]
	hi_ind = np.where(scores > hi_thr)[0]
	pind   = np.hstack([lo_ind, hi_ind])
	plabel = np.hstack([np.full(len(lo_ind), neg_label),
		np.full(len(hi_ind), pos_label)])
	return pind, plabel

def cmat_plot(thresholds, cmat, savename):
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(thresholds, cmat[:,0], label='tpr')
	ax.plot(thresholds, cmat[:,2], label='fpr')
	fig.savefig(savename)

CUDA_DEVICE = 0
out_dir     = Path('_8_pseudo_labeling/')
out_dir.mkdir(exist_ok=True)
figsize = (30,10)
fontsize = 20

tests_root  = Path('../_tests')
test_cfg 	= load_config('../config/test_draw_128.yml')
test_root   = tests_root/test_cfg['test_root']
cs_root 	= test_root/test_cfg['coverage_set']['root']
os_root 	= test_root/test_cfg['observed_set']['root']
draw_lims   = test_cfg['coverage_set']['draw_limits']
os_ann  	= np.load(os_root/'labels.npy')
cs_settings = read_settings(cs_root/'settings.json')
cs_trans 	= load.get_transformations(cs_settings['mean'], cs_settings['variance'])
cs_ann_npy	= np.load(cs_root/'labels.npy')

batch_size 	= 100
criterion 	= nn.CrossEntropyLoss()
cs_loader 	= load.get_npy_dataloader(cs_root,batch_size,transforms=cs_trans,shuffle=False)

enc_cfg     = test_cfg['encoders']
encoders_root = test_root/enc_cfg['root']
nets = enc_cfg['nets']

for net_name in ['VGG11']: #nets:
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
		baseline_npz = encoders_root/net_name/('%s_baseline.npz' % net_name)
		if not baseline_npz.exists():
			raise Exception('%s not available' % baseline_npz)
		baseline = np.load(baseline_npz)
		PRD = baseline['PRD']
		ANN = baseline['ANN']
		HIT = baseline['HIT']

		_, _, blabel = annotations.label_annotation_distribution(os_ann, ANN)

		# df = pd.DataFrame(ANN)
		# df.columns = ['c', 'xmin', 'ymin', 'xmax', 'ymax', 'br']
		# df['ct'] = np.max(PRD[0,0], axis=1)
		# df['blabel'] = blabel
		uncert  = np.max(PRD[0,0], axis=1)
		# # bid_ind = np.where(blabel == pos_label)[0]
		# # bod_ind = np.where(blabel == neg_label)[0]
		# # threshold = np.linspace(0.5,1, num=20)
		# # cmat = explainer._confusion_matrix(uncert, bid_ind, bod_ind, threshold)
		# # cmat_plot(threshold, cmat, out_dir/'1_cmat.png')
		
		# plabel_thr = thr_pseudo_label(uncert[0], 0.75)
		# explainer.roc_explain(PRD[0], plabel_thr, [1], 
		# 	out_dir/'1_plabel_thr_roc.png', 
		# 	out_dir/'1_plabel_thr_auroc.png')
		# p_hitrate = np.count_nonzero(blabel == plabel_thr)/len(blabel)
		# print('Benchmark hit rate before scaling %0.3f' % p_hitrate)


		# berror = np.full(len(blabel), 1.0)
		# berror[blabel == neg_label] = 0.5
		# bidod_ds = load.my_subset(cs_loader.dataset, 
		# 	np.arange(0,len(cs_loader.dataset)),
		# 	torch.tensor(berror).type(torch.float))
		# bidod_loader = torch.utils.data.DataLoader(bidod_ds, 
		# 	batch_size=100, shuffle=False)
		# cal_encoder = ModelWithTemperature(encoder)
		# cal_encoder.set_temperature(bidod_loader)


		p_subset, plabel = pseudo_label(uncert, 0.05)
		plabel_hitrate = np.count_nonzero(blabel[p_subset] == plabel)/len(p_subset)
		print('Pseudo subset hit rate %0.3f' % plabel_hitrate)

		perror = np.full(len(plabel), 1.0)
		perror[plabel == neg_label] = 0.5
		
		cs_subset = load.my_subset(cs_loader.dataset, p_subset, torch.tensor(perror).type(torch.float))
		cs_subset_loader = torch.utils.data.DataLoader(cs_subset, batch_size=100, shuffle=False)
		cal_encoder = ModelWithTemperature(encoder)
		cal_encoder.set_temperature(cs_subset_loader)

		# eps = [0]
		# T = [1, cal_encoder.temperature.item()]
		# scaled_npz = out_dir/('%s_scaled.npz' % net_name)
		# state_changed = True
		# if scaled_npz.exists():
		# 	scaled = np.load(scaled_npz)			
		# 	if (len(T) == len(scaled['T'])) and (np.allclose(T,scaled['T'])):
		# 		state_changed = False
		# 		sPRD = scaled['sPRD']
		# if state_changed:
		# 	sPRD, _, _ = explainer.evaluate(CUDA_DEVICE,
		# 			criterion,cs_loader,
		# 			eps, T,
		# 			var=cs_settings['variance'],
		# 			y_ann=cs_ann_npy)			
		# 	np.savez(scaled_npz, T=T, sPRD=sPRD)
		
		# suncert = np.max(sPRD, axis=3)
		# plabel_thr_scaled = thr_pseudo_label(suncert[0,1], 0.75)
		# sp_hitrate = np.count_nonzero(blabel == plabel_thr_scaled)/len(blabel)
		# print('Benchmark hit rate after scaling %0.3f' % sp_hitrate)

		# _plot_distributions(suncert[0], T, blabel, out_dir/'1_explore.png')
		# _plot_distributions(suncert[0], T, p_subset, plabel, out_dir/'1_explore.png')
		# for i,t in enumerate(T):
		# 	nsamples = PRD.shape[-2]
		# 	plabel_thr = np.full(nsamples, 1)
		# 	pod_ind = np.where(suncert < 0.75)[0]
		# 	plabel_thr[pod_ind] = 0			
			# print('Benchmark hit rate (threshold) ')
		# net_npz = encoders_root/net_name/('%s.npz' % net_name)


		# idod_e    = np.full(len(idod_y), 1.0)
		# idod_e[idod_y == neg_label] = 0.5
		# cs_subset = load.my_subset(cs_loader.dataset, idod_ind, torch.tensor(idod_e).type(torch.float))
		# cs_subset_loader = torch.utils.data.DataLoader(cs_subset, batch_size=100, shuffle=False)
		# cal_encoder = ModelWithTemperature(encoder)
		# cal_encoder.set_temperature(cs_subset_loader)

		# od_ind = idod_ind[idod_y == neg_label]
		# annotations.plot_annotation_distribution(ANN, draw_lims, test_cfg['dim'], 
		# 	out_dir/'2_ann_dist.png',
		# 	od_ind=od_ind)

		# annotations.plot_annotation_distribution(ANN[od_ind], draw_lims, test_cfg['dim'],
		# 	out_dir/'3_ann_od.png')

		# eps = [0]
		# T = [1, cal_encoder.temperature.item()]
		# scaled_npz = out_dir/('%s_scaled.npz' % net_name)
		# state_changed = True
		# if scaled_npz.exists():
		# 	scaled = np.load(scaled_npz)			
		# 	if (len(T) == len(scaled['T'])) and (np.allclose(T,scaled['T'])):
		# 		state_changed = False
		# 		sPRD = scaled['sPRD']
		# if state_changed:
		# 	sPRD, _, _ = explainer.evaluate(CUDA_DEVICE,
		# 			criterion,cs_loader,
		# 			eps, T,
		# 			var=cs_settings['variance'],
		# 			y_ann=cs_ann_npy)			
		# 	np.savez(scaled_npz, T=T, sPRD=sPRD)
		
		# mSFM = np.max(sPRD, axis=3)
		# _plot_distributions(mSFM[0], T, idod_ind, idod_y, out_dir/'1_explore.png')
		# net_npz = encoders_root/net_name/('%s.npz' % net_name)

		# if not net_npz.exists():
		# 	raise Exception('%s does not exist' % net_npz)

		# state = np.load(net_npz)
		# T = state['T']
		# cPRD  = state['cPRD']
		# cHIT  = state['cHIT']
		# ANN   = state['ANN']

		# PRD   = cPRD[0,np.where(T == 1)[0]][0]
		# mSFM  = np.max(PRD, axis=1)
		
		# _, _, bidod = annotations.label_annotation_distribution(os_ann, ANN)
		# bidod_e = np.full(len(bidod), 1.0)
		# bidod_e[bidod == neg_label] = 0.5
		# bidod_ds = load.my_subset(cs_loader.dataset, 
		# 	np.arange(0,len(cs_loader.dataset)),
		# 	torch.tensor(bidod_e).type(torch.float))
		# bidod_loader = torch.utils.data.DataLoader(bidod_ds, 
		# 	batch_size=100, shuffle=False)
		# cal_encoder = ModelWithTemperature(encoder)
		# cal_encoder.set_temperature(bidod_loader)
		
		# baseline_npz = out_dir/('%s_baseline.npz' % net_name)
		# scaled_npz   = out_dir/('%s_scaled.npz' % net_name)
		
		# state_changed = True
		# if baseline_npz.exists():
		# 	baseline = np.load(baseline_npz)
		# 	if (len(T) == len(baseline['T'])) and (np.allclose(T,baseline['T'])):
		# 		state_changed = False

		# if state_changed:
		# 	PRD, HIT, ANN = explainer.evaluate(CUDA_DEVICE,
		# 		criterion,cs_loader,
		# 		eps, T,
		# 		var=cs_settings['variance'],
		# 		y_ann=cs_ann_npy)
		# 	np.savez(baseline_npz, T=T,
		# 		PRD=PRD, HIT=HIT, ANN=ANN,)
		# else:
		# 	baseline = np.load(baseline_npz)
		# 	PRD = baseline['PRD']
		# 	HIT = baseline['HIT']
		# 	ANN = baseline['ANN']

		# mSFM = np.max(PRD[0],axis=2)		

		# _, _, bidod = annotations.label_annotation_distribution(os_ann, ANN)
		# bidod_e = np.full(len(bidod), 1.0)
		# bidod_e[bidod == neg_label] = 0.5		
		# id_ind = np.where(bidod == pos_label)[0]
		# od_ind = np.where(bidod == neg_label)[0]
		# mse_id = []
		# mse_od = []
		# for t_ind in range(len(T)):	
		# 	mse_id.append(np.square(1.0 - mSFM[t_ind,id_ind]).mean())
		# 	mse_od.append(np.square(mSFM[t_ind,id_ind] - 0.5).mean())
		
		# plot_T_sampling_dist(mSFM, T, id_ind, od_ind, out_dir/'1_t_sampling_dist.png')
		# explainer.auroc_explain(PRD[0], bidod, T, out_dir/'2_t_sampling_auroc.png' )

		# fig, axs = plt.subplots(1, figsize=figsize)
		# axs.scatter(mse_id, mse_od)
		# for i, t in enumerate(T):
		# 	axs.annotate('%0.3f' % t, (mse_id[i], mse_od[i]), fontsize=fontsize)
		# fig.savefig(out_dir/'1_explore.png')

			# print('T: %0.3f, MSE-id: %0.3f, MSE-od: %0.3f, MSE: %0.3f' % (T[t_ind],mse_id, mse_od, mse))

		# bidod_e = np.full(len(bidod), 1.0)
		# bidod_e[bidod == neg_label] = 0.5
		# bidod_ds = load.my_subset(cs_loader.dataset, 
		# 	np.arange(0,len(cs_loader.dataset)),
		# 	torch.tensor(bidod_e).type(torch.float))
		# bidod_loader = torch.utils.data.DataLoader(bidod_ds, 
		# 	batch_size=100, shuffle=False)
		# cal_encoder = ModelWithTemperature(encoder)
		# cal_encoder.set_temperature(bidod_loader)
		
		# idod_s, idod_ind, idod_y = pseudo_label(mSFM, 0.1)

		# plot_distributions(mSFM, 
		# 	idod_ind[idod_y == pos_label],
		# 	idod_ind[idod_y == neg_label],
		# 	out_dir/'1_score_distribution.png')

		# cs_subset = torch.utils.data.Subset(cs_loader.dataset, idod_ind)
		# idod_e    = np.full(len(idod_y), 1.0)
		# idod_e[idod_y == neg_label] = 0.5
		# cs_subset = load.my_subset(cs_loader.dataset, idod_ind, torch.tensor(idod_e).type(torch.float))
		# cs_subset_loader = torch.utils.data.DataLoader(cs_subset, batch_size=100, shuffle=False)

		# cal_encoder = ModelWithTemperature(encoder)
		# cal_encoder.set_temperature(cs_subset_loader)
		# T = [1, cal_encoder.temperature.item()]

		# # # eps, T = [0],[1, 5, 10]
		# state_changed = True
		# if scaled_npz.exists():
		# 	scaled = np.load(scaled_npz)
		# 	if (len(T) == len(scaled['T'])) and (np.allclose(T, scaled['T'])):
		# 		state_changed = False

		# if state_changed:
		# 	sPRD, _, _ = explainer.evaluate(CUDA_DEVICE,
		# 			criterion,cs_loader,
		# 			eps, T,
		# 			var=cs_settings['variance'],
		# 			y_ann=cs_ann_npy)			
		# 	np.savez(scaled_npz, T=T, sPRD=sPRD)
		# else:
		# 	sPRD = scaled['sPRD']

		# smSFM = np.max(sPRD, axis=3)
		# _plot_distributions(smSFM[0], T, idod_ind, idod_y,
		# 	out_dir/'2_score_dist_t.png')

		# explainer.roc_explain(sPRD[0], idod_y, T, out_dir/'3_roc.png')


		

		# # for batch_id, (_x, _y) in enumerate(cs_subset_loader):
		# # 	print(_y)
