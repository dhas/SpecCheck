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
from ood.explainV2 import TrainedModel
import sys
from scipy.stats import beta
from scipy.integrate import quad

def overlap_metric_example(savename):
	lower_xplot_limit_orange = -0.25
	upper_xplot_limit_orange = 7.5

	blue_plot_limits = [(-5,-1), (-2,2), (1,5), (-3,9)]
	plot_ind = [(0,0), (0,1), (1,0), (1,1)]
	V_dists = [1, 0.76, -0.48, 0.35]
	fig, ax = plt.subplots(2, 2, figsize=(12, 6))

	for idx, blue_limits in enumerate(blue_plot_limits):
		lower_xplot_limit_blue = blue_limits[0]
		upper_xplot_limit_blue = blue_limits[1]



		subplot_ind = plot_ind[idx]

		y_lim_lower = -0.01
		y_lim_upper = 0.6

		x_lim_lower = min(lower_xplot_limit_blue, lower_xplot_limit_orange) - 0.25
		x_lim_upper = max(upper_xplot_limit_blue, upper_xplot_limit_orange) + 0.25

		V_text_x_placement = x_lim_lower + 0.2
		V_text_y_placement = 0.5

		alpha_orange = 0.5
		alpha_blue = 0.5

		text_size = 20



		plot_x1 = np.linspace(lower_xplot_limit_orange, upper_xplot_limit_orange,800)
		plot_y1 = np.exp(-(plot_x1-2)**2) + 0.8*np.exp(-(plot_x1-3)**2) + 0.8*np.exp(-(plot_x1-5)**2)
		area1 = quad(lambda x: np.exp(-(x-2)**2) + 0.8*np.exp(-(x-3)**2) + 0.8*np.exp(-(x-5)**2), lower_xplot_limit_orange, upper_xplot_limit_orange)
		plot_y1 = plot_y1 / area1[0]
		area11 = np.trapz(plot_y1)
		plot_x2 = np.linspace(lower_xplot_limit_blue, upper_xplot_limit_blue,200)
		plot_y2 = beta.pdf((plot_x2-lower_xplot_limit_blue)/(upper_xplot_limit_blue - lower_xplot_limit_blue),3,2)# - 1 + np.cos(plot_x2/10)**2#0.2*np.power((plot_x2-2), 1/2) - 0.255*(plot_x2-4)
		area2 = quad(lambda x: beta.pdf((x-lower_xplot_limit_blue)/(upper_xplot_limit_blue - lower_xplot_limit_blue),3,2), lower_xplot_limit_blue, upper_xplot_limit_blue)
		plot_y2 = plot_y2 / area2[0]
		area22 = np.trapz(plot_y2)

		ax[subplot_ind].plot(plot_x1,plot_y1, 'tab:orange', label=r'$P_X$')
		ax[subplot_ind].plot([min(lower_xplot_limit_blue, lower_xplot_limit_orange) -0.5, max(upper_xplot_limit_orange,upper_xplot_limit_blue) + 0.5], 2*[0], 'k')


		ax[subplot_ind].fill_between(plot_x1, plot_x1.shape[0]*[0], plot_y1, facecolor = 'tab:orange', alpha = alpha_orange)
		ax[subplot_ind].plot(plot_x2,plot_y2, 'b', label=r'$P_Y$')
		ax[subplot_ind].fill_between(plot_x2, plot_x2.shape[0]*[0], plot_y2, facecolor = 'b', alpha = alpha_blue)

		ax[subplot_ind].set_ylim(y_lim_lower, y_lim_upper)

		ax[subplot_ind].axvline(lower_xplot_limit_orange,0.002,0.04, color = 'k')
		ax[subplot_ind].axvline(upper_xplot_limit_orange,0.002,0.04, color = 'k')
		ax[subplot_ind].axvline(lower_xplot_limit_blue,0.002,0.04, color = 'k')
		ax[subplot_ind].axvline(upper_xplot_limit_blue,0.002,0.04, color = 'k')
		ax[subplot_ind].text(V_text_x_placement, V_text_y_placement, 'V = {}'.format(V_dists[idx]), size = text_size)
		ax[subplot_ind].set_xlim(x_lim_lower, x_lim_upper)		
	#ax[subplot_ind].axis('off')

	ax[subplot_ind].legend(loc='upper right', fontsize=text_size)
	fig.savefig(savename, bbox_inches='tight')


# def overlap_metric_example(savename):
# 	lower_xplot_limit_orange = -0.25
# 	upper_xplot_limit_orange = 7.5
# 	lower_xplot_limit_blue = 1
# 	upper_xplot_limit_blue = 5

# 	y_lim_lower = 0 #-0.25
# 	y_lim_upper = 2.2

# 	alpha_orange = 0.5
# 	alpha_blue = 0.5

# 	text_size = 20


# 	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# 	plot_x1 = np.linspace(lower_xplot_limit_orange, upper_xplot_limit_orange,800)
# 	plot_y1 = 1*np.exp(-(plot_x1-2)**2) + 0.8*np.exp(-(plot_x1-3)**2) + 0.8*np.exp(-(plot_x1-5)**2)

# 	plot_x2 = np.linspace(lower_xplot_limit_blue, upper_xplot_limit_blue,200)
# 	plot_y2 = beta.pdf((plot_x2-1)/4,3,2)# - 1 + np.cos(plot_x2/10)**2#0.2*np.power((plot_x2-2), 1/2) - 0.255*(plot_x2-4)

# 	plot_x2_before = np.linspace(lower_xplot_limit_orange, lower_xplot_limit_blue, 300)
# 	plot_x2_after = np.linspace(upper_xplot_limit_blue, upper_xplot_limit_orange, 300)
# 	plot_x_orange_color = np.concatenate((plot_x2_before, plot_x2, plot_x2_after))
# 	plot_y_orange_color_lower = np.array(300*[0] + plot_y2.tolist() + 300*[0])
# 	plot_y_orange_color_upper = 1*np.exp(-(plot_x_orange_color-2)**2) + 0.8*np.exp(-(plot_x_orange_color-3)**2) + 0.8*np.exp(-(plot_x_orange_color-5)**2)


# 	ax.plot(plot_x1,plot_y1, 'tab:orange')
# 	ax.plot([lower_xplot_limit_orange -0.5, upper_xplot_limit_orange + 0.5], 2*[0], 'k')

# 	ax.fill_between(plot_x_orange_color, plot_y_orange_color_lower, plot_y_orange_color_upper, facecolor = 'tab:orange', alpha = alpha_orange)
# 	ax.plot(plot_x2,plot_y2, 'b')
# 	ax.fill_between(plot_x2, plot_x2.shape[0]*[0], plot_y2, facecolor = 'b', alpha = alpha_blue)

# 	ax.set_ylim(y_lim_lower, y_lim_upper)

# 	ax.axvline(lower_xplot_limit_orange,0,0.03, color = 'k')
# 	ax.axvline(upper_xplot_limit_orange,0,0.03, color = 'k')
# 	ax.axvline(lower_xplot_limit_blue,0,0.03, color = 'k')
# 	ax.axvline(upper_xplot_limit_blue,0,0.03, color = 'k')
# 	# ax.annotate('', (lower_xplot_limit_orange, -0.1), (lower_xplot_limit_blue, -0.1), arrowprops={'arrowstyle':'<->'})

# 	lower_text_mean = (lower_xplot_limit_orange + lower_xplot_limit_blue)/2	
# 	# ax.text(lower_text_mean, -0.2, '$l_1$', size = text_size)
# 	# ax.annotate('', (lower_xplot_limit_blue, -0.1), (upper_xplot_limit_blue, -0.1), arrowprops={'arrowstyle':'<->'})
# 	middle_text_mean = (lower_xplot_limit_blue + upper_xplot_limit_blue)/2
# 	# ax.text(middle_text_mean, -0.2, '$l_2$', size = text_size)
# 	text_off = 0.3
# 	ax.text(middle_text_mean, 0.7, '$P$', size = text_size)
# 	ax.text(lower_xplot_limit_orange-text_off, 0.1, '$min(P)$', size = text_size)
# 	ax.text(upper_xplot_limit_orange-text_off, 0.1, '$max(P)$', size = text_size)

# 	ax.text(middle_text_mean, 0.3, '$Q$', size = text_size)
# 	ax.text(lower_xplot_limit_blue-text_off, 0.1, '$min(Q)$', size = text_size)
# 	ax.text(upper_xplot_limit_blue, 0.1, '$max(Q)$', size = text_size)
# 	# ax.annotate('', (upper_xplot_limit_blue, -0.1), (upper_xplot_limit_orange, -0.1), arrowprops={'arrowstyle':'<->'})
# 	upper_text_mean = (upper_xplot_limit_blue + upper_xplot_limit_orange)/2
# 	# ax.text(upper_text_mean, -0.2, '$l_3$', size = text_size)
# 	ax.axis('off')
# 	fig.savefig(savename, bbox_inches='tight')

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

	batch_size = 10
	r,c = 2,5
	
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
	sample_dataset(qd_root, qd_root/'qd_samples.pdf')

def prepare_coverage_set(cfg, dim, num_classes, test_root):
	sd_root = test_root/cfg['root']
	sd_root.mkdir(exist_ok=True)

	sd_shapes = SpecDraw(sd_root,
		dim,
		cfg['imgs_per_class'],
		cfg['draw_limits'])
	sd_ann = np.load(sd_root/'labels.npy')
	sample_dataset(sd_root, sd_root/'sd_samples.pdf', ann=sd_ann)

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

def explain_with_encoder_set(cfg, ds_root, os_ann, dim, draw_lims, test_root, explain_root, fontsize=24):
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
	figsize = (12, 6)	
	figAUROC, axAUROC = plt.subplots(figsize=(30,10))
	figBaseUNC, axBaseUNC = plt.subplots(2, len(nets), figsize=figsize)
	figXCalUNC, axXCalUNC = plt.subplots(2, len(nets), figsize=figsize)	
	figBaseROC, axBaseROC = plt.subplots(figsize=figsize)
	figXCalROC, axXCalROC = plt.subplots(figsize=figsize)
	figSTCalROC, axSTCalROC = plt.subplots(figsize=figsize)
	figSTCalDist, axSTCalDist = plt.subplots(figsize=figsize)
	figBaseSHAP, axBaseSHAP = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=(30,10))
	figXCalSHAP, axXCalSHAP = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=(30,10))
	figMCalSHAP, axMCalSHAP = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=(30,10))

	dists_file = explain_root/'dist.txt'
	if dists_file.exists():
		dists_file.unlink()

	wdist_mu = []
	odist_mu = []
	aurocs   = []
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
			model = TrainedModel(encoder, checkpoint_path)			

			eps, T = [0], [1]
			baseline_npz = encoders_root/net_name/('%s_baseline.npz' % net_name)
			if not baseline_npz.exists():
				PRD, HIT, ANN = model.evaluate(CUDA_DEVICE,
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
			
			model_hist = pickle.load(open(encoders_root/net_name/'history.pkl', 'rb'))
			print('QD accuracy %0.2f' % (model_hist['val_history']['accuracy'][-1]))
			print('SD accuracy %0.2f' % (np.count_nonzero(HIT)/np.size(HIT)))
			_, _, idod = annotations.label_annotation_distribution(os_ann, ANN)			
			
			if net_ind == 0:
				yticks = True
			else:
				yticks = False

			# UNC = np.max(PRD[0,0], axis=1)
			# # explain.uncertainty_summary(UNC, 
			# # 	idod, net_name,
			# # 	axBaseUNC[0, net_ind], axBaseUNC[1, net_ind], fix_xlim=False)			
			# # if net_ind == 0:
			# # 	axBaseUNC[0, net_ind].set_ylabel(r'$\widetilde{X}^+$', fontsize=fontsize)
			# # 	axBaseUNC[1, net_ind].set_ylabel(r'$\widetilde{X}^-$', fontsize=fontsize)

			# explain.roc_summary(UNC, idod, axBaseROC, net_name)
			# ANNS, SHAP, IDODS = explain.contribs_by_feature(UNC, ANN, idod, explain_root/('%s_1_bas_shap.pdf' % net_name),
			# 	dtree_savename=explain_root/('%s_1_bas_dtree.pdf' % net_name))
			# wdist, odist = explain.estimate_marginals_from_shap(SHAP, ANNS, os_ann, dim,
			# 	draw_lims, explain_root/('%s_1_bas_marginals.pdf' % net_name))
			# explain.record_distances('%s, ucal' % (net_name), wdist, odist, 'a', explain_root/'dist.txt')
			# # explain.assess_pseudo_labeling('ucal', SHAP, IDODS)
			# # explain.shap_summary(SHAP, ANN, net_name, axBaseSHAP)

			
			# model.softening_summary(ds_loader, explain_root/('%s_1_softening.pdf' % net_name))			
			
			calibrated_npz = encoders_root/net_name/('%s_calibrated.npz' % net_name)
			if calibrated_npz.exists():
				state = np.load(calibrated_npz)
				xcalT  = state['xcalT']				
				stcalT = state['stcalT']
				T     = state['T']
				cPRD  = state['cPRD']
				cHIT  = state['cHIT']
			else:				
				xcalT    = model.calibrate_for_highest_auroc(idod, ds_loader)				
				stcalT   = model.calibrate_for_max_spread(ds_loader, thr=0.7)

				T = np.hstack([np.linspace(1, 10, num=10), [xcalT, stcalT]])
				T = np.hstack([1/T[1:], T])
				T.sort()
				cPRD, cHIT, _ = model.evaluate(CUDA_DEVICE,
					criterion,ds_loader,
					eps, T,
					var=ds_settings['variance'],
					y_ann=ds_ann_npy)
				np.savez(calibrated_npz, xcalT=xcalT, stcalT=stcalT, T=T, cPRD=cPRD, cHIT=cHIT)
			
			# xcalUNC = np.max(cPRD[0,T == xcalT][0], axis=1)
			# explain.uncertainty_summary(xcalUNC, idod, 
			# 	net_name, axXCalUNC[0, net_ind], axXCalUNC[1, net_ind], [r'$\widetilde{Y}^+$', r'$\widetilde{Y}^-$'],
			# 	T=xcalT, T_label=r'$T_X$', yticks=yticks)
			# explain.roc_summary(xcalUNC, idod, axXCalROC, net_name)
			# if net_ind == 0:
			# 	axXCalUNC[0, net_ind].set_ylabel(r'$\widetilde{Y}^+$', fontsize=fontsize)
			# 	axXCalUNC[1, net_ind].set_ylabel(r'$\widetilde{Y}^-$', fontsize=fontsize)

			# xcal_npz = encoders_root/net_name/('%s_xcal.npz' % net_name)			
			# if xcal_npz.exists():
			# 	state = np.load(xcal_npz)
			# 	SHAP  = state['SHAP']
			# 	ANNS  = state['ANNS']
			# else:
			# 	ANNS, SHAP, IDODS = explain.contribs_by_feature(xcalUNC, ANN, idod, explain_root/('%s_2_xcal_shap.pdf' % net_name),
			# 		dtree_savename=explain_root/('%s_2_xcal_dtree.pdf' % net_name))
			# 	np.savez(xcal_npz, SHAP=SHAP, ANNS=ANNS)
			# wdist, odist = explain.estimate_marginals_from_shap(SHAP, ANNS, os_ann, dim,
			# 	draw_lims, explain_root/('%s_2_xcal_marginals.pdf' % net_name))
			# explain.record_distances('%s, xcal' % (net_name), wdist, odist, 'a', explain_root/'dist.txt')


			stcalUNC = np.max(cPRD[0,T == stcalT][0], axis=1)
			auroc = explain.roc_summary(stcalUNC, idod, axSTCalROC, net_name)
			aurocs.append(auroc)

			stcal_npz = encoders_root/net_name/('%s_stcal.npz' % net_name)			
			if stcal_npz.exists():
				state = np.load(stcal_npz)
				SHAP  = state['SHAP']
				ANNS  = state['ANNS']
			else:
				ANNS, SHAP, IDODS = explain.contribs_by_feature(stcalUNC, ANN, idod, explain_root/('%s_4_stcal_shap.pdf' % net_name),
					dtree_savename=explain_root/('%s_4_stcal_dtree.pdf' % net_name))
				np.savez(stcal_npz, SHAP=SHAP, ANNS=ANNS)
			wdist, odist = explain.estimate_marginals_from_shap(SHAP, ANNS, os_ann, dim,
				draw_lims, explain_root/('%s_4_stcal_marginals.pdf' % net_name))
			explain.record_distances('%s, stcal' % (net_name), wdist, odist, 'a', explain_root/'dist.txt')
			wdist_mu.append(wdist.mean())
			odist_mu.append(odist.mean())

	# figBaseUNC.text(1.0, 0.5, 'Distribution of S', va='center', rotation='vertical', fontsize=fontsize)
	# figBaseUNC.savefig(explain_root/'0_bas_uncertainty_summary.pdf', bbox_inches='tight')
	axXCalUNC[0, net_ind].legend(loc='upper right', fontsize=fontsize, frameon=False)
	axXCalUNC[1, net_ind].legend(loc='upper right', fontsize=fontsize, frameon=False)
	figXCalUNC.text(0.04, 0.5, 'Distribution of S', va='center', rotation='vertical', fontsize=fontsize)
	figXCalUNC.savefig(explain_root/'0_xcal_uncertainty_summary.pdf', bbox_inches='tight')
	

	# axBaseROC.tick_params(axis='both', which='major', labelsize=fontsize-10)
	# axBaseROC.legend(fontsize=fontsize-4, loc='lower right')
	# axBaseROC.set_xlabel('False Positive Rate (FPR)', fontsize=fontsize)
	# axBaseROC.set_ylabel('True Positive Rate (TPR)', fontsize=fontsize)
	# figBaseROC.savefig(explain_root/'1_bas_roc_summary.pdf', bbox_inches='tight')

	axXCalROC.legend(fontsize=fontsize-4, loc='lower right')
	axXCalROC.set_xlabel('False Positive Rate (FPR)', labelpad=15, fontsize=fontsize)
	axXCalROC.set_ylabel('True Positive Rate (TPR)', labelpad=15, fontsize=fontsize)
	figXCalROC.savefig(explain_root/'1_xcal_roc.pdf', bbox_inches='tight')

	axSTCalROC.legend(fontsize=fontsize-4, loc='lower right')
	axSTCalROC.set_xlabel('False Positive Rate (FPR)', fontsize=fontsize)
	axSTCalROC.set_ylabel('True Positive Rate (TPR)', fontsize=fontsize)
	figSTCalROC.savefig(explain_root/'1_stcal_roc.pdf')

	summary_npz = encoders_root/'summary.npz'
	if summary_npz.exists():
		state = np.load(summary_npz)
		wdist_mu = state['wdist_mu']
		odist_mu = state['odist_mu']
		aurocs   = state['aurocs']
	else:
		np.savez(summary_npz, wdist_mu=wdist_mu, odist_mu=odist_mu, aurocs=aurocs)
	explain.distance_summary(wdist_mu, odist_mu, aurocs, nets.keys(), axSTCalDist)	
	figSTCalDist.savefig(explain_root/'2_stcal_dist.pdf', bbox_inches='tight')
	
	# axAUROC.legend(fontsize=fontsize-4, loc='lower right')
	# figAUROC.savefig(explain_root/'2_auroc_summary.pdf')

	# handles, labels = axBaseSHAP[0,0].get_legend_handles_labels()
	# figBaseSHAP.legend(handles, labels, loc='upper right', fontsize=fontsize)
	# figBaseSHAP.savefig(explain_root/'3_bas_shap_summary.pdf')

	# handles, labels = axXCalSHAP[0,0].get_legend_handles_labels()
	# figXCalSHAP.legend(handles, labels, loc='upper right', fontsize=fontsize)
	# figXCalSHAP.savefig(explain_root/'3_xcal_shap_summary.pdf')

	# handles, labels = axMCalSHAP[0,0].get_legend_handles_labels()
	# figMCalSHAP.legend(handles, labels, loc='upper right', fontsize=fontsize)
	# figMCalSHAP.savefig(explain_root/'3_mcal_shap_summary.pdf')
	# plt.close()