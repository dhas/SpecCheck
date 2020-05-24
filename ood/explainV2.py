import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm
import sys
import xgboost
import graphviz
import shap
import torch
from torch.autograd import Variable
from torch.nn import functional as F
sys.path.append('..')
from utils import other_utils
from draw import load, annotations
from ood.calibration import ModelWithTemperature
from sklearn.metrics import auc
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

def get_feature_KL(os_ann, cs_ann, eps=0.00001):
	num_classes = len(annotations.classes)
	num_feats = len(annotations.FEATS)
	fkl = np.empty((num_classes, num_feats), 
		dtype=np.float)
	bins = 50
	for clabel in range(num_classes):
		os = os_ann[os_ann[:,annotations.ANN_COL_CL] == clabel, 1:]
		cs = cs_ann[cs_ann[:,annotations.ANN_COL_CL] == clabel, 1:]
		for f_ind in range(num_feats):
			p, bins = np.histogram(os[:,f_ind], bins=50)
			p = (p/np.sum(p)) + eps
			q, bins = np.histogram(cs[:,f_ind], bins=50)
			q = (q/np.sum(q)) + eps
			fkl[clabel, f_ind] = np.sum(np.where(p != 0, p * np.log(p / q), 0))
	return fkl


def get_wdistance_by_feature(ann1, ann2):
	wdist = np.empty((len(annotations.classes), len(annotations.FEATS)),
		dtype=np.float)
	for c, cname in enumerate(annotations.classes):
		for f, feat in enumerate(annotations.FEATS):
			wdist[c, f] = wasserstein_distance(ann1[cname][:, f], ann2[cname][:, f])
	return wdist


def histogram(f):
	hist, bins = np.histogram(f, bins=15)
	freq = hist/np.sum(hist)
	return freq, bins

def record_distances(prefix, wdist, odist, mode, fname):
	with open(fname, mode) as f:
		f.write('%s\n' % prefix)
		f.write(np.array2string(wdist, formatter={'float_kind':lambda x: '%.2f' % x}, separator=' & '))
		f.write('\n')
		f.write(np.array2string(odist, formatter={'float_kind':lambda x: '%.2f' % x}, separator=' & '))
		f.write('\n')
		f.write('%0.2f & %0.2f' % (wdist.mean(), odist.mean()))
		f.write('\n')


def distance_summary(wdist_mu, odist_mu, aurocs, test_accs, net_names, axs, fontsize=24):
	
	width = 0.2 
	ticks = np.arange(len(net_names))
	axs.bar(ticks - width/2, odist_mu, width, label='Mean Overlap')
	axs.set_xticks(ticks)
	axs.set_xticklabels(net_names)
	axs.tick_params(axis='both', which='major', labelsize=fontsize-4)
	axs.set_ylabel('Overlap index', labelpad=-5, fontsize=fontsize-4)	
	axs.yaxis.set_label_coords(-0.01, 0.5)
	axs_lim = [0.2, 0.7]
	axs.set_ylim(axs_lim)
	axs.set_yticks(axs_lim)

	
	
	ax1 = axs.twinx()
	ax1.bar(ticks + width/2, wdist_mu, width, color='r', label='Mean Wasserstein')	
	ax1.tick_params(axis='both', which='major', labelsize=fontsize-4)
	ax1.set_ylabel('Wasserstein distance', labelpad=0, fontsize=fontsize-4)
	ax1.yaxis.set_label_coords(1.01, 0.5)
	ax1_lim = [5, 30]
	ax1.set_ylim(ax1_lim)
	ax1.set_yticks(ax1_lim)

	ax2 = axs.twinx()
	ax2.plot(ticks[1:], aurocs, color='k', label='AUROC')	
	ax2.tick_params(axis='both', which='major', labelsize=fontsize-4)
	ax2.spines['right'].set_position(('outward', 40))
	ax2.set_ylabel('AUROC', fontsize=fontsize-4)
	ax2.yaxis.set_label_coords(1.07, 0.5)
	ax2_lim = [0.6, 0.9]
	ax2.set_ylim(ax2_lim)
	ax2.set_yticks(ax2_lim)

	# ax3 = axs.twinx()
	# ax3.plot(ticks, test_accs, '--', color='k', label='Test acc.')
	# ax3_lim = [0.9, 1.0]
	# ax3.set_ylim(ax3_lim)
	# ax3.set_yticks(ax3_lim)
	# ax3.spines['right'].set_position(('outward', 80))
	# ax3.set_ylabel('Test acc.', labelpad=5, fontsize=fontsize-4)
	# ax3.yaxis.set_label_coords(1.13, 0.5)
	# ax3.tick_params(axis='both', which='major', labelsize=fontsize-4)

	
	
	# ax1.set_ylim([0, 1.2])
	
	# ax1.set_yticks([0, 0.5, 1.0])
	lines, labels = axs.get_legend_handles_labels()
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	# lines3, labels3 = ax3.get_legend_handles_labels()
	
	ax1.legend(lines2 + lines + lines1, labels2 + labels + labels1, fontsize=fontsize-4, loc='upper left', frameon=False, ncol=3)
	
	
	
	


def assess_pseudo_labeling(tag, SHAP, IDODS):
	for cname in annotations.classes:
		clabel 		= annotations.class_to_label_dict[cname]
		PIDODS      = np.full(IDODS[clabel].shape, annotations.od_label)
		PIDODS[np.all(SHAP[clabel] >= 0, axis=1)] = annotations.id_label
		chit        = np.count_nonzero(PIDODS == IDODS[clabel])/IDODS[clabel].size
		print('%s pseudo labeling hitrate %s - %0.3f'% (tag, cname, chit))
	
	SHAP   = SHAP.reshape(-1,SHAP.shape[-1])
	IDODS  = IDODS.reshape(-1)	
	PIDODS = np.full((IDODS.shape), annotations.od_label)
	PIDODS[np.all(SHAP >= 0, axis=1)] = annotations.id_label

	idInd   = np.where(IDODS == annotations.id_label)[0]
	idhit   = np.count_nonzero(PIDODS[idInd] == IDODS[idInd])/IDODS[idInd].size
	print('%s pseudo labeling hitrate ID - %0.3f'% (tag, idhit))
	odInd   = np.where(IDODS == annotations.od_label)[0]
	odhit   = np.count_nonzero(PIDODS[odInd] == IDODS[odInd])/IDODS[odInd].size
	print('%s pseudo labeling hitrate OD - %0.3f'% (tag, odhit))
	hit   = np.count_nonzero(PIDODS == IDODS)/IDODS.size
	print('%s pseudo labeling hitrate - %0.3f'% (tag, hit))



def estimate_marginals_from_shap(SHAPS, ANNS, os_ann, dim, draw_lims, savename, labelpad=5, fontsize=26, figsize=(12,6), overlap_ax=None):
	os_ann = os_ann[~np.all(os_ann[:,1:] == 0, axis=1)]
	os_ann = annotations.annotations_list2dict(os_ann)

	fig, axs = plt.subplots(len(annotations.class_to_label_dict), len(annotations.FEATS), figsize=figsize)
	wdist = np.zeros((len(annotations.class_to_label_dict), len(annotations.FEATS)))
	odist = np.zeros((len(annotations.class_to_label_dict), len(annotations.FEATS)))
	for cname in annotations.class_to_label_dict:
		clabel = annotations.class_to_label_dict[cname]
		oann  = os_ann[cname]
		cann  = ANNS[clabel]
		shap  = SHAPS[clabel]
		for f, feat in enumerate(annotations.FEATS):
			F, S  = cann[:, f], shap[:, f]
			f2 = F[S > 0]
			p2, bins2 = histogram(f2)
			axs[clabel, f].bar(bins2[:-1], p2, align="edge", width=np.diff(bins2), color='k', alpha=0.5, label=r'$P^+_{\mathcal{T}}$')

			f1 = oann[:, f]
			p1, bins1 = histogram(f1)
			axs[clabel, f].bar(bins1[:-1], p1, align="edge", width=np.diff(bins1), color='red', alpha=0.5, label=r'$P_{\mathcal{S}}$')
			ylim = [0, 0.3]
			axs[clabel, f].set_ylim(ylim)
			wdist[clabel, f] = wasserstein_distance(f1, f2)
			odist[clabel, f] = annotations.overlap_index(f1, f2) #annotations.distance_function(f1, f2)
			axs[clabel, f].tick_params(axis='both', which='major', labelsize=fontsize-8)
			
			#if f == 3 and clabel == 0 and (not overlap_ax is None):
			if f == 1 and clabel == 1 and (not overlap_ax is None):
				overlap_ax.bar(bins2[:-1], p2, align="edge", width=np.diff(bins2), color='k', alpha=0.5, label=r'$P^+_{\mathcal{T}}$')
				overlap_ax.bar(bins1[:-1], p1, align="edge", width=np.diff(bins1), color='red', alpha=0.5, label=r'$P_{\mathcal{S}}$')
				overlap_ax.text(30, 0.175, r'$W^%d-%0.2f$' % (f+2, wdist[clabel, f]), fontsize=fontsize)
				overlap_ax.text(30, 0.150, r'$V^%d-%0.2f$' % (f+2, odist[clabel, f]), fontsize=fontsize)
				overlap_ax.set_title(annotations.FEATS[f], fontsize=fontsize)
				overlap_ax.set_ylim(ylim)
				overlap_ax.set_yticks(ylim)
				overlap_ax.set_xlim([0, 60])
				overlap_ax.tick_params(axis='both', which='major', labelsize=fontsize-4)

			if '6' in feat: # == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				axs[clabel, f].set_xlim([100, 255])
				axs[clabel, f].set_xticks([120, 200])
				# axs[clabel, f].text(110, 0.25, r'$W^%d-%0.2f$' % (f+2, wdist[clabel, f]), fontsize=fontsize-6)
				# axs[clabel, f].text(110, 0.15, r'$V^%d-%0.2f$' % (f+2, odist[clabel, f]), fontsize=fontsize-6)
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				axs[clabel, f].set_xlim([0, 128])
				axs[clabel, f].set_xticks([0, 80])
				# axs[clabel, f].text(10, 0.25, r'$W^%d-%0.2f$' % (f+2, wdist[clabel, f]), fontsize=fontsize-6)
				# axs[clabel, f].text(10, 0.15, r'$V^%d-%0.2f$' % (f+2, odist[clabel, f]), fontsize=fontsize-6)

			if f == 0:
				axs[clabel, f].set_ylabel(r'$Y^1=%d$' % clabel, fontsize=fontsize)
				axs[clabel, f].set_yticks(ylim)
			else:
				axs[clabel, f].set_yticks([])
			if clabel == 0:
				axs[clabel, f].set_title(annotations.FEATS[f], fontsize=fontsize-4)
				axs[clabel, f].set_xticks([])
	# handles, labels = axs[clabel, f].get_legend_handles_labels()
	# fig.legend(handles, labels, loc='lower right', fontsize=fontsize-4)
	axs[clabel, f-1].legend(bbox_to_anchor=(-0.165, 0.3), loc='lower left', fontsize=fontsize, frameon=False)
	fig.savefig(savename, bbox_inches='tight')
	plt.close()
	return wdist, odist

class BaselineExplainer:
	def __init__(self,model,checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		self.model = model

	def evaluate(self, device, criterion, loader, y_ann=None):
		self.model.to(device)
		self.model.eval()

		PRD = np.empty((len(loader.dataset),
			len(loader.dataset.classes)),
		dtype=np.float)

		HIT = np.empty((len(loader.dataset),
			1),
		dtype=np.bool)

		if not y_ann is None:
			ANN = np.empty((len(loader.dataset),
			annotations.ANN_SZ),
			dtype=np.int)
			with_annotations = True
		else:
			with_annotations = False

		batch_size = loader.batch_size
		for batch_id, (_x, _y, fnames) in enumerate(tqdm(loader)):
			inputs = _x.cuda(device)
			outputs = self.model(inputs)
			
			nnOutputs = outputs.data.cpu()
			nnOutputs = nnOutputs.numpy()
			nnOutputs = nnOutputs - np.max(nnOutputs,axis=1).reshape(-1,1)								
			nnOutputs = np.array(
				[np.exp(nnOutput)/np.sum(np.exp(nnOutput)) for nnOutput in nnOutputs])
			
			chunk = slice(batch_id*batch_size,(batch_id+1)*batch_size)
			PRD[chunk] = nnOutputs
			HIT[chunk] = (np.argmax(nnOutputs, axis=1) == _y.numpy()).reshape(-1,1)
			
		if with_annotations:
			return PRD, HIT, ANN
		else:
			return PRD, HIT


def uncertainty_summary(UNC, idod, net_name, id_ax, od_ax, labels, yticks=False, fix_xlim=True, T=None, T_label=None, fontsize=25, figsize=(30,15)):
	hist, bins = np.histogram(UNC[idod==annotations.id_label], bins=15)
	freq = hist/np.sum(hist)
	id_ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins),
				color='blue', label=labels[1])
	id_ax.set_ylim([0,0.7])	
	id_ax.set_title(net_name, fontsize=fontsize)

	hist, bins = np.histogram(UNC[idod==annotations.od_label], bins=15)
	freq = hist/np.sum(hist)
	od_ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins),
		color='red', label=labels[1])
	od_ax.set_ylim([0,1])
	if fix_xlim:
		id_ax.set_xlim([0.5,1])
		od_ax.set_xlim([0.5,1])
	
	id_ax.set_xticks([])
	od_ax.set_xticks([0.5, 0.7, 0.9])
	
	if not yticks:
		id_ax.set_yticks([])
		od_ax.set_yticks([])
	else:
		ticks = [0.1, 0.5]
		id_ax.set_yticks(ticks)
		od_ax.set_yticks(ticks)

	id_ax.tick_params(axis='both', which='major', labelsize=fontsize-10)
	# id_ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
	od_ax.tick_params(axis='both', which='major', labelsize=fontsize-10)
	if not T is None:
		od_ax.set_xlabel('%s-%0.2f' % (T_label, T), labelpad=15, fontsize=fontsize)


COL_TPR = 0
COL_FNR = 1
COL_FPR = 2
COL_TNR = 3
def _confusion_matrix(scores, id_ind, od_ind, threshold):
		cmat = []
		for t in threshold:
			tpr = np.count_nonzero(scores[id_ind] > t)/len(id_ind)
			fnr = np.count_nonzero(scores[id_ind] < t)/len(id_ind)
			tnr = np.count_nonzero(scores[od_ind] < t)/len(od_ind)
			fpr = np.count_nonzero(scores[od_ind] > t)/len(od_ind)
			cmat.append([tpr, fnr, fpr, tnr])
		return np.stack(cmat)


def get_auroc(UNC, idod):
	id_ind = np.where(idod == annotations.id_label)[0]
	od_ind = np.where(idod == annotations.od_label)[0]
	threshold = np.linspace(0.5,1, num=20)
	cmat =  _confusion_matrix(UNC, id_ind, od_ind, threshold)
	auroc = auc(cmat[:,2], cmat[:,0])
	return auroc

def roc_summary(aurocs, test_accs, ax1, fontsize, width=0.2):	
	ticks = np.arange(len(aurocs))
	print(ticks)
	ax1.bar(ticks-width/2, aurocs, width, color='k', label='AUROC')
	ax1.set_ylabel('AUROC', fontsize=fontsize)
	ax1.set_xticks(ticks)
	ax1_lim = [0.5, 1]
	ax1.set_ylim(ax1_lim)
	ax1.set_yticks([0.6, 0.8, 1])
	ax1.tick_params(axis='both', which='major', labelsize=fontsize-2)
	
	ax2 = ax1.twinx()
	ax2.bar(ticks+width/2, test_accs, width, color='b', label='Test Acc.')
	ax2.set_ylabel('Test Acc.', fontsize=fontsize)
	ax2_lim = [0.9, 1]
	ax2.set_ylim([0.9, 1])
	ax2.set_yticks([0.9, 0.95, 1])
	ax2.tick_params(axis='both', which='major', labelsize=fontsize-2)
	# ax1.set_xlabel('Classifier variants')


	lines1, labels1 = ax1.get_legend_handles_labels()	
	lines2, labels2 = ax2.get_legend_handles_labels()	
	ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=fontsize-4, loc='upper left', frameon=False, ncol=2)


# def roc_summary(UNC, idod, axs, net_name, fontsize=25, figsize=(30,15)):
# 		id_ind = np.where(idod == annotations.id_label)[0]
# 		od_ind = np.where(idod == annotations.od_label)[0]
# 		threshold = np.linspace(0.5,1, num=20)
# 		cmat =  _confusion_matrix(UNC, id_ind, od_ind, threshold)
# 		auroc = auc(cmat[:,2], cmat[:,0])
# 		axs.plot(cmat[:,2], cmat[:,0],
# 				label='%s, AUROC-%0.3f' % (net_name,auroc))
# 		axs.tick_params(axis='both', which='major', labelsize=fontsize-10)
# 		return auroc

def roc_net(PRD, idod, T, roc_savename, auroc_savename, fontsize=20, figsize=(30,15)):
		id_ind = np.where(idod == annotations.id_label)[0]
		od_ind = np.where(idod == annotations.od_label)[0]
		fig, axs = plt.subplots(figsize=figsize)
		mSFM  = np.max(PRD,axis=2)		
		threshold = np.linspace(0.5,1, num=20)
		
		aurocs  = []
		mses_id = []
		mses_od = []

		for i,t in enumerate(T):
			cmat =  _confusion_matrix(mSFM[i], id_ind, od_ind, threshold)
			auroc = auc(cmat[:,2], cmat[:,0])
			axs.plot(cmat[:,2], cmat[:,0],
				label='T-%0.3f, AUROC-%0.3f' % (T[i],auroc))
			aurocs.append(auroc)
			mses_id.append(np.square(1.0 - mSFM[i,id_ind]).mean())
			mses_od.append(np.square(mSFM[i,od_ind] - 0.5).mean())

		axs.legend(fontsize=fontsize-4, loc='lower right')
		fig.savefig(roc_savename)

		fig,a0 = plt.subplots(figsize=figsize)
		a0.plot(T, aurocs, label='auroc', color='blue')
		a1 = a0.twinx()
		a1.plot(T,mses_id, label='mse_id', color='orange')
		a1.plot(T,mses_od, label='mse_od', color='green')

		h1, l1 = a0.get_legend_handles_labels()
		h2, l2 = a1.get_legend_handles_labels()
		handles = h1 + h2
		labels 	= l1 + l2		
		# fig.legend(handles, labels, bbox_to_anchor=[0.9, 0.97], loc='upper right')
		a0.legend(handles, labels,prop={'size': fontsize-4})		
		fig.savefig(auroc_savename)
		plt.close()
		return aurocs


def contribs_by_feature(UNC, ANN, idod, savename, dtree_savename=None, fontsize=26, figsize=(12,6)):
	SHAPS = []
	ANNS  = []
	IDODS = []
	fig, axs = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=figsize)
	# dtree_fig, dtree_ax = plt.subplots(1, len(annotations.classes), figsize=(30, 30))
	for cname in annotations.classes:
		clabel 		= annotations.class_to_label_dict[cname]
		cind   		= np.where(ANN[:,0] == clabel)[0]
		X   		= pd.DataFrame(ANN[cind, 1:])
		X.columns 	= annotations.FEATS
		y   		= UNC[cind]				
		dtree 		= xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=list(y), feature_names=annotations.FEATS), 500)
		# if not dtree_savename is None:
		# 	xgboost.plot_tree(dtree, rankdir='LR', ax=dtree_ax[clabel])
		# 	dtree_ax[clabel].set_title(cname, fontsize=fontsize)
		SHAP 		= dtree.predict(xgboost.DMatrix(X, feature_names=annotations.FEATS), pred_contribs=True) #shap.TreeExplainer(dtree).shap_values(X)
		axs[clabel, 0].set_ylabel(r'$Y^1=%d$' % clabel, fontsize=fontsize)
		for f, feat in enumerate(annotations.FEATS):
			x = X[feat]
			y = SHAP[:,f]
			
			# color = [str(item/255.) for item in y]
			# color = ['k' if item >= 0 else '0.75' for item in y]			
			# axs[clabel, f].scatter(x, y, c=color, cmap='Greys')
			axs[clabel, f].plot(x[y>=0], y[y>=0], '.', rasterized=True, c='k')
			axs[clabel, f].plot(x[y<0], y[y<0], '.', rasterized=True, c='0.75')			
			axs[clabel, f].tick_params(axis='both', which='major', labelsize=fontsize-8)
			axs[clabel, f].set_ylim([-0.15, 0.06])
			if f != 0:
				axs[clabel, f].set_yticks([])
			if clabel == 0:
				axs[clabel, f].set_title(feat, fontsize=fontsize-4)
				axs[clabel, f].set_xticks([])
			
			if '6' in feat: # == 'BR':
				axs[clabel, f].set_xlim([100, 255])
				x_ = np.linspace(100, 255)
				axs[clabel, f].plot(x_, np.zeros_like(x_), color='r')
				if clabel == 1:
					axs[clabel, f].set_xticks([120, 200])
				else:
					axs[clabel, f].set_xticks([])
			else:
				axs[clabel, f].set_xlim([0, 120])
				x_ = np.linspace(0, 120)
				axs[clabel, f].plot(x_, np.zeros_like(x_), color='r')
				if clabel == 1:
					axs[clabel, f].set_xticks([0, 80])
				else:
					axs[clabel, f].set_xticks([])

		ANNS.append(ANN[cind, 1:])
		SHAPS.append(SHAP)
		IDODS.append(idod[cind])		
	fig.savefig(savename, bbox_inches='tight')
	# if not dtree_savename is None:
	# 	dtree_fig.savefig(dtree_savename, dpi=300, bbox_inches='tight')
	plt.close()
	return np.stack(ANNS), np.stack(SHAPS), np.stack(IDODS)


def shap_by_feature(UNC, ANN, idod, savename, dtree_savename=None, fontsize=25, figsize=(30,15)):
	SHAPS = []
	ANNS  = []
	IDODS = []
	fig, axs = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=figsize)
	dtree_fig, dtree_ax = plt.subplots(1, len(annotations.classes), figsize=(30, 30))
	for cname in annotations.classes:
		clabel 		= annotations.class_to_label_dict[cname]
		cind   		= np.where(ANN[:,0] == clabel)[0]
		X   		= pd.DataFrame(ANN[cind, 1:])
		X.columns 	= annotations.FEATS
		y   		= UNC[cind]				
		dtree 		= xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=list(y), feature_names=annotations.FEATS), 100)
		if not dtree_savename is None:
			xgboost.plot_tree(dtree, rankdir='LR', ax=dtree_ax[clabel])
			dtree_ax[clabel].set_title(cname, fontsize=fontsize)
		SHAP 		= shap.TreeExplainer(dtree).shap_values(X)
		axs[clabel, 0].set_ylabel(r'$Y^1=%d$' % clabel, fontsize=fontsize)
		for f, feat in enumerate(annotations.FEATS):
			axs[clabel, f].scatter(X[feat], SHAP[:,f])
			axs[clabel, f].set_title(feat, fontsize=fontsize)
			axs[clabel, f].tick_params(axis='both', which='major', labelsize=fontsize-12)
		ANNS.append(ANN[cind, 1:])
		SHAPS.append(SHAP)
		IDODS.append(idod[cind])
	fig.savefig(savename, bbox_inches='tight')
	if not dtree_savename is None:
		dtree_fig.savefig(dtree_savename, dpi=300, bbox_inches='tight')
	plt.close()
	return np.stack(ANNS), np.stack(SHAPS), np.stack(IDODS)

def estimated_id_features_using_shap(SHAPS, ANNS, dim, draw_lims, id_savename, od_savename, fontsize=20, figsize=(30,15)):
	idfig, idaxs = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=figsize)
	odfig, odaxs = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=figsize)
	for cname in annotations.classes:
		clabel 	= annotations.class_to_label_dict[cname]
		shap    = SHAPS[clabel]
		ann     = ANNS[clabel]
		for f, feat in enumerate(annotations.FEATS):
			F  = ann[:, f]
			S  = shap[:, f]
			idF  = F[S >= 0]
			hist, bins = np.histogram(idF, bins=15)
			freq = hist/len(idF)
			idaxs[clabel, f].bar(bins[:-1], freq, align="edge", width=np.diff(bins))
			idaxs[clabel, f].set_ylim([0,1])
			print('ID sum - %0.2f' % np.sum(freq))
			# idaxs[clabel, f].hist(idF)
			if feat == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				idaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//12])
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				idaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//8])

			odF  = F[S < 0]
			hist, bins = np.histogram(odF, bins=15)
			freq = hist/len(odF)
			odaxs[clabel, f].bar(bins[:-1], freq, align="edge", width=np.diff(bins))
			odaxs[clabel, f].set_ylim([0,1])
			print('OD sum - %0.2f' % np.sum(freq))
			# odaxs[clabel, f].hist(odF)
			if feat == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				odaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//12])
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				odaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//8])

	idfig.savefig(id_savename)
	odfig.savefig(od_savename)
	plt.close()

def shap_summary(SHAP, ANN, net_name, axs, fontsize=20):	
	for cname in annotations.classes:
		clabel 	= annotations.class_to_label_dict[cname]
		cind   	= np.where(ANN[:,0] == clabel)[0]
		cANN    = ANN[cind, 1:]
		shap 	= SHAP[clabel]
		for f, feat in enumerate(annotations.FEATS):
			F  = cANN[:, f]
			S  = shap[:,f]				
			uF = np.unique(F)	
			mS = [S[F == f].mean() for f in uF]				
			axs[clabel, f].plot(uF, mS, label=net_name)
			if clabel == 0:
				axs[clabel, f].set_title(feat, fontsize=fontsize)
			if f == 0:
				axs[clabel, f].set_ylabel(cname, fontsize=fontsize)
	
				
class TrainedModel:
	def __init__(self, model, checkpoint_path, is_calibrated=False):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		self.model 			= model
		self.is_calibrated 	= is_calibrated		
				

	def calibrate_for_highest_auroc(self, idod, cal_ds, batch_size=100):
		idod_e = np.full(len(idod), 1.0)
		idod_e[idod == annotations.od_label] = 0.5
		idod_ds = load.my_subset(cal_ds.dataset, 
			np.arange(0,len(cal_ds.dataset)),
			torch.tensor(idod_e).type(torch.float))
		idod_loader = torch.utils.data.DataLoader(idod_ds, 
			batch_size=batch_size, shuffle=False)
		cal_encoder = ModelWithTemperature(self.model)
		cal_encoder.set_temperature(idod_loader)
		return cal_encoder.temperature.item()

	def calibrate_for_centered_distribution(self, idod, ds_loader, batch_size=100):
		idod_e = np.full(len(idod), 0.75)
		idod_ds = load.my_subset(ds_loader.dataset, 
			np.arange(0,len(ds_loader.dataset)),
			torch.tensor(idod_e).type(torch.float))
		idod_loader = torch.utils.data.DataLoader(idod_ds, 
			batch_size=batch_size, shuffle=False)

		cal_encoder = ModelWithTemperature(self.model)
		cal_encoder.set_mid_cal(idod_loader)
		return cal_encoder.temperature.item()

	def calibrate_for_max_spread(self, ds_loader, thr=None, fontsize=25, figsize=(30,15)):
		cal_encoder = ModelWithTemperature(self.model)
		T, s = cal_encoder.set_spread_cal(ds_loader, thr=thr)		
		return T

	def softening_summary(self, ds_loader, savename, labelpad=15, fontsize=24, figsize=(12,6)):
		self.model.cuda()
		logits_list = []
		labels_list = []
		with torch.no_grad():
			for input, label, _ in ds_loader:
				input = input.cuda()
				logits = self.model(input)
				logits_list.append(logits)
				labels_list.append(label)
			logits = torch.cat(logits_list)#.cuda()
			labels = torch.cat(labels_list)#.cuda()
		
		stats = []
		T = np.linspace(1,50, num=100)
		for t in T:
			l 	   = logits/t
			scores, _ = torch.max(F.softmax(l,1),1)
			scores_mu = scores.mean()
			scores_var = scores.var()
			stats.append([scores_mu.item(), scores_var.item()])
		s = np.vstack(stats)
		
		fig, ax1 = plt.subplots(figsize=figsize)
		l1 = ax1.plot(T, s[:,0], color='blue', label='mean')
		ax1.set_ylim([0.5, 1.0])
		ax1.set_yticks([0.5, 1.0])
		# ax1.set_xlim([0, 50])
		ax1.set_xticks([1, 25, 50])
		ax1.set_ylabel('Mean of S', fontsize=fontsize)
		ax2 = ax1.twinx()
		l2 = ax2.plot(T, s[:,1], color='orange', label='variance')
		ax2.set_ylabel('Variance of S', fontsize=fontsize)
		ax1.set_xlabel('Temperature', labelpad=labelpad, fontsize=fontsize)
		ax2.set_yticks([0.0, 0.02])
		lns = l1 + l2
		labs = [l.get_label() for l in lns]
		ax1.legend(lns, labs, fontsize=fontsize, frameon=False)
		ax1.tick_params(axis='both', which='major', labelsize=fontsize-4)
		ax2.tick_params(axis='both', which='major', labelsize=fontsize-4)
		fig.savefig(savename, bbox_inches='tight')

	# def plot_inout_score_distributions(self, PRD, idod, T, savename, fontsize=20, figsize=(30,15)):
	# 	id_ind = np.where(idod == annotations.id_label)[0]
	# 	od_ind = np.where(idod == annotations.od_label)[0]
	# 	fig, axs = plt.subplots(2, len(T), figsize=figsize)		
	# 	mSFM  = np.max(PRD,axis=2)
	# 	for i, t in enumerate(T):
	# 		ax = axs[0, i]
	# 		ax.hist(mSFM[i, id_ind], color='blue')
	# 		ax.set_xlim([0.5,1])
	# 		ax.set_title('T-%0.2f' % t, fontsize=fontsize)
	# 		mse_id = np.square(1.0 - mSFM[i,id_ind]).mean()
	# 		ax.set_xlabel('MSE-%0.3f' % mse_id, fontsize=fontsize)

	# 		ax = axs[1, i]
	# 		ax.hist(mSFM[i, od_ind], color='red')
	# 		ax.set_xlim([0.5,1])
	# 		mse_od = np.square(mSFM[i,od_ind] - 0.5).mean()
	# 		ax.set_xlabel('MSE-%0.3f' % mse_od, fontsize=fontsize)
	# 	fig.savefig(savename)

	

	def evaluate(self, device, criterion, loader, eps, T, var=1, y_ann=None):
		self.model.to(device)
		self.model.eval()

		if self.is_calibrated:			
			T = [1] #calibrated temp included in the model

		PRD = np.empty((len(eps),
			len(T),
			len(loader.dataset),
			len(loader.dataset.classes)),
		dtype=np.float)

		HIT = np.empty((len(eps),
			len(T),
			len(loader.dataset),
			1),
		dtype=np.bool)

		if not y_ann is None:
			ANN = np.empty((len(loader.dataset),
			6),
			dtype=np.int)
			with_annotations = True
		else:
			with_annotations = False

		batch_size = loader.batch_size

		for batch_id, (_x, _y, fnames) in enumerate(tqdm(loader)):
			inputs = Variable(_x.cuda(device), requires_grad = True)			
			outputs = self.model(inputs)
			chunk = slice(batch_id*batch_size,(batch_id+1)*batch_size)
			for t_ind in range(len(T)):
				temper = T[t_ind]			
				# Calculating the confidence of the output, no perturbation added here, no temperature scaling used
				nnOutputs = outputs.data.cpu()
				nnOutputs = nnOutputs.numpy()
				# nnOutputs = nnOutputs[0]		
				nnOutputs = nnOutputs - np.max(nnOutputs,axis=1).reshape(-1,1)								
				nnOutputs = np.array([np.exp(nnOutput)/np.sum(np.exp(nnOutput)) for nnOutput in nnOutputs])				
				# Using temperature scaling
				outputs = outputs / temper			
				# Calculating the perturbation we need to add, that is,
				# the sign of gradient of cross entropy loss w.r.t. input
				maxIndexTemp = np.argmax(nnOutputs, axis=1)			
				labels = Variable(torch.LongTensor(maxIndexTemp).cuda(device))			
				loss = criterion(outputs, labels)
				loss.backward()				
				# Normalizing the gradient to binary in {0, 1}
				gradient =  torch.ge(inputs.grad.data, 0)		
				gradient = (gradient.float() - 0.5) * 2				
				# Normalizing the gradient to the same space of image
				gradient[0][0] = (gradient[0][0] )/(var)				
				for e_ind in range(len(eps)):
					epsilon = eps[e_ind]
					# Adding small perturbations to images
					tempInputs = torch.add(inputs.data,  -epsilon, gradient)
					outputs = self.model(Variable(tempInputs))
					outputs = outputs / temper
					# Calculating the confidence after adding perturbations
					nnOutputs = outputs.data.cpu()
					nnOutputs = nnOutputs.numpy()
					# nnOutputs = nnOutputs[0]
					nnOutputs = nnOutputs - np.max(nnOutputs, axis=1).reshape(-1,1)
					nnOutputs = np.array([np.exp(nnOutput)/np.sum(np.exp(nnOutput)) for nnOutput in nnOutputs])
					PRD[e_ind,t_ind,chunk] = nnOutputs
					HIT[e_ind,t_ind,chunk] = (np.argmax(nnOutputs, axis=1) == _y.numpy()).reshape(-1,1)
				if with_annotations:
					ANN[chunk] = annotations.get_annotations_for_batch(
						fnames,y_ann)		
		if with_annotations:
			return PRD, HIT, ANN
		else:
			return PRD, HIT
