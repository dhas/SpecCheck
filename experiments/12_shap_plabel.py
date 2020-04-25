from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xgboost
from scipy.stats import wasserstein_distance
from sklearn.metrics import auc
# sys.path.append('..')
# from draw import load, annotations
# import ood.explainV2 as explain

classes    = ['circle', 'square']
num_classes = len(classes)
class_to_label_dict = {
	'circle' : 0,
	'square' : 1
}

_ANNS   = ['CL', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'BR']
FEATS  = [r'$Y^2$', r'$Y^3$', r'$Y^4$', r'$Y^5$', r'$Y^6$']

id_label = 0
od_label = 1

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)

def annotations_list2dict(ann):
	d = dict.fromkeys(classes)
	for cname in classes:
		clabel = class_to_label_dict[cname]
		d[cname] = ann[ann[:,0] == clabel, 1:]
	return d

def histogram(f):
	hist, bins = np.histogram(f, bins=15)
	freq = hist/np.sum(hist)
	return freq, bins

def label_annotation_distribution(os_ann, cs_ann):
	os_ann = os_ann[os_ann[:,1] != 0]
	os_ann_min = []
	os_ann_max = []
	for f_ind in range(len(_ANNS)):
		if f_ind == 0:
			continue		
		os_ann_min.append(os_ann[:,f_ind].min())
		os_ann_max.append(os_ann[:,f_ind].max())
	idod = []	
	for ann in cs_ann:
		ann = ann[1:]
		indis = ((os_ann_min[0] < ann[0]) and (os_ann_max[0] > ann[0])) and \
				((os_ann_min[1] < ann[1]) and (os_ann_max[1] > ann[1])) and \
				((os_ann_min[2] < ann[2]) and (os_ann_max[2] > ann[2])) and \
				((os_ann_min[3] < ann[3]) and (os_ann_max[3] > ann[3])) and \
				((os_ann_min[4] < ann[4]) and (os_ann_max[4] > ann[4]))
		if indis:
			idod.append(id_label)
		else:
			idod.append(od_label)		
	return os_ann_min, os_ann_max, np.array(idod)


def compare_annotation_distributions(ANN1, ANN2, labels, draw_lims, dim, savename, fontsize=25, figsize=(30,15)):
	ANN1 = ANN1[~np.all(ANN1[:,1:] == 0, axis=1)]
	ANN2 = ANN2[~np.all(ANN2[:,1:] == 0, axis=1)]

	fig, axs = plt.subplots(len(class_to_label_dict), len(FEATS), figsize=figsize)
	for cname in class_to_label_dict:
		clabel = class_to_label_dict[cname]
		cANN1 = ANN1[ANN1[:,0] == clabel,1:]
		cANN2 = ANN2[ANN2[:,0] == clabel,1:]
		for f, feat in enumerate(FEATS):
			f2 = cANN2[:, f]
			p2, bins = histogram(f2)
			axs[clabel, f].bar(bins[:-1], p2, align="edge", width=np.diff(bins), alpha=0.5, label=labels[1])

			f1 = cANN1[:, f]
			p1, bins = histogram(f1)
			axs[clabel, f].bar(bins[:-1], p1, align="edge", width=np.diff(bins), alpha=0.5, label=labels[0])
			axs[clabel, f].set_ylim([0,1])
			axs[clabel, f].tick_params(axis='both', which='major', labelsize=fontsize-10)
			wdist = wasserstein_distance(f1, f2)
			if '6' in FEATS[f]: # == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				axs[clabel, f].set_xticks(f_spread[::len(f_spread)//8])
				axs[clabel, f].text(140, 0.8, r'$W^%d(P_{T})-%0.2f$' % (f+2, wdist), fontsize=fontsize)
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				axs[clabel, f].set_xticks(f_spread[::len(f_spread)//8])
				axs[clabel, f].text(40, 0.8, r'$W^%d(P_{T})-%0.2f$' % (f+2, wdist), fontsize=fontsize)

			if f == 0:
				axs[clabel, f].set_ylabel(r'$Y^1=%d$' % clabel, fontsize=fontsize)
			if clabel == 0:
				axs[clabel, f].set_title(FEATS[f], fontsize=fontsize)
	axs[clabel, f].legend(loc='center left', fontsize=fontsize)
	# handles, labels = axs[clabel, f].get_legend_handles_labels()
	# fig.legend(handles, labels, loc='lower right', fontsize=fontsize)
	fig.savefig(savename, bbox_inches='tight')
	plt.close()

def contribs_by_feature(UNC, ANN, idod, savename, dtree_savename=None, fontsize=25, figsize=(30,15)):
	SHAPS = []
	ANNS  = []
	IDODS = []
	fig, axs = plt.subplots(len(classes), len(FEATS), figsize=figsize)
	dtree_fig, dtree_ax = plt.subplots(1, len(classes), figsize=(30, 30))
	for cname in classes:
		clabel 		= class_to_label_dict[cname]
		cind   		= np.where(ANN[:,0] == clabel)[0]
		X   		= pd.DataFrame(ANN[cind, 1:])
		X.columns 	= FEATS
		y   		= UNC[cind]				
		dtree 		= xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=list(y), feature_names=FEATS), 500)
		SHAP 		= dtree.predict(xgboost.DMatrix(X, feature_names=FEATS), pred_contribs=True) #shap.TreeExplainer(dtree).shap_values(X)
		axs[clabel, 0].set_ylabel(r'$Y^1=%d$' % clabel, fontsize=fontsize)
		for f, feat in enumerate(FEATS):
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

def estimate_marginals_from_shap(SHAPS, ANNS, os_ann, dim, draw_lims, savename, fontsize=25, figsize=(30,15)):
	os_ann = os_ann[~np.all(os_ann[:,1:] == 0, axis=1)]
	os_ann = annotations_list2dict(os_ann)

	fig, axs = plt.subplots(len(class_to_label_dict), len(FEATS), figsize=figsize)
	wdist = np.zeros((len(class_to_label_dict), len(FEATS)))
	for cname in class_to_label_dict:
		clabel = class_to_label_dict[cname]
		oann  = os_ann[cname]
		cann  = ANNS[clabel]
		shap  = SHAPS[clabel]
		for f, feat in enumerate(FEATS):
			F, S  = cann[:, f], shap[:, f]
			f2 = F[S > 0]
			p2, bins = histogram(f2)
			axs[clabel, f].bar(bins[:-1], p2, align="edge", width=np.diff(bins), alpha=0.5, label=r'$\widehat{P}_{\mathcal{S}}$')

			f1 = oann[:, f]
			p1, bins = histogram(f1)
			axs[clabel, f].bar(bins[:-1], p1, align="edge", width=np.diff(bins), alpha=0.5, label=r'$\widetilde{P}_{\mathcal{S}}$')
			axs[clabel, f].set_ylim([0,1])
			wdist[clabel, f] = wasserstein_distance(f1, f2)
			axs[clabel, f].tick_params(axis='both', which='major', labelsize=fontsize-12)
			if '6' in feat: # == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				axs[clabel, f].set_xticks(f_spread[::len(f_spread)//8])
				axs[clabel, f].text(140, 0.8, r'$W^%d(\widehat{P}_{S})-%0.2f$' % (f+2, wdist[clabel, f]), fontsize=fontsize)
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				axs[clabel, f].set_xticks(f_spread[::len(f_spread)//8])
				axs[clabel, f].text(40, 0.8, r'$W^%d(\widehat{P}_{S})-%0.2f$' % (f+2, wdist[clabel, f]), fontsize=fontsize)

			if f == 0:
				axs[clabel, f].set_ylabel(r'$Y^1=%d$' % clabel, fontsize=fontsize)
			if clabel == 0:
				axs[clabel, f].set_title(FEATS[f], fontsize=fontsize)
	# handles, labels = axs[clabel, f].get_legend_handles_labels()
	# fig.legend(handles, labels, loc='lower right', fontsize=fontsize-4)
	axs[clabel, f].legend(loc='center left', fontsize=fontsize)
	fig.savefig(savename, bbox_inches='tight')
	plt.close()
	return wdist

def _confusion_matrix(scores, id_ind, od_ind, threshold):
		cmat = []
		for t in threshold:
			tpr = np.count_nonzero(scores[id_ind] > t)/len(id_ind)
			fnr = np.count_nonzero(scores[id_ind] < t)/len(id_ind)
			tnr = np.count_nonzero(scores[od_ind] < t)/len(od_ind)
			fpr = np.count_nonzero(scores[od_ind] > t)/len(od_ind)
			cmat.append([tpr, fnr, fpr, tnr])
		return np.stack(cmat)

def roc_summary(UNC, idod, axs, net_name, fontsize=25, figsize=(30,15)):
		id_ind = np.where(idod == id_label)[0]
		od_ind = np.where(idod == od_label)[0]
		threshold = np.linspace(0.5,1, num=20)
		cmat =  _confusion_matrix(UNC, id_ind, od_ind, threshold)
		auroc = auc(cmat[:,2], cmat[:,0])
		axs.plot(cmat[:,2], cmat[:,0],
				label='%s, AUROC-%0.3f' % (net_name,auroc))		

out_dir     = Path('_12_shap_plabel/')
out_dir.mkdir(exist_ok=True)
figsize = (30,10)
fontsize = 20

tests_root  = Path('../_tests')
test_cfg 	= load_config('../config/epochs_test_draw_128.yml')
dim 		= test_cfg['dim']
test_root   = tests_root/test_cfg['test_root']

os_cfg 		= test_cfg['observed_set']
os_ann 		= np.load(out_dir/'os_labels.npy')

cs_cfg      = test_cfg['coverage_set']
draw_lims   = cs_cfg['draw_limits']
cs_ann 	    = np.load(out_dir/'cs_labels.npy')

compare_annotation_distributions(os_ann, cs_ann, [r'$\widetilde{P}_S$', r'$P_T$'],
	cs_cfg['draw_limits'], test_cfg['dim'], out_dir/'1_os_cs_ann.png')


enc_cfg     = test_cfg['encoders']
encoders_root = test_root/enc_cfg['root']
nets = enc_cfg['nets']

figBaseROC, axBaseROC = plt.subplots(figsize=(30,10))
figXCalROC, axXCalROC = plt.subplots(figsize=(30,10))

for net_ind, net_name in enumerate(['VGG07', 'VGG11']): #nets:
	print('Processing %s' % net_name)
	baseline_npz = out_dir/('%s_baseline.npz' % net_name)
	if not baseline_npz.exists():
		raise Exception('%s not available' % baseline_npz)
	baseline = np.load(baseline_npz)
	PRD = baseline['PRD']
	ANN = baseline['ANN']
	HIT = baseline['HIT']
	_, _, idod = label_annotation_distribution(os_ann, ANN)			
	UNC = np.max(PRD[0,0], axis=1)	
	ANNS, SHAP, IDODS = contribs_by_feature(UNC, ANN, idod, out_dir/('%s_1_bas_shap.png' % net_name))	
	wdist = estimate_marginals_from_shap(SHAP, ANNS, os_ann, dim,
				draw_lims, out_dir/('%s_1_bas_marginals.png' % net_name))
	# print('ucal')
	# print(wdist.mean(axis=1))
	roc_summary(UNC, idod, axBaseROC, net_name)
	

	calibrated_npz = out_dir/('%s_calibrated.npz' % net_name)	

	state = np.load(calibrated_npz)
	xcalT  = state['xcalT']	
	stcalT  = state['stcalT']
	T     = state['T']
	cPRD  = state['cPRD']
	cHIT  = state['cHIT']

	xcalUNC = np.max(cPRD[0,T == xcalT][0], axis=1)
	ANNS, SHAP, IDODS = contribs_by_feature(xcalUNC, ANN, idod, out_dir/('%s_2_xcal_shap.png' % net_name))
	wdist = estimate_marginals_from_shap(SHAP, ANNS, os_ann, dim,
				draw_lims, out_dir/('%s_2_xcal_marginals.png' % net_name))
	# print('xcal')
	# print(wdist.mean(axis=1))
	roc_summary(xcalUNC, idod, axXCalROC, net_name)


axBaseROC.tick_params(axis='both', which='major', labelsize=fontsize-10)
axBaseROC.legend(fontsize=fontsize-4, loc='lower right')
axBaseROC.set_xlabel('False Positive Rate (FPR)', fontsize=fontsize)
axBaseROC.set_ylabel('True Positive Rate (TPR)', fontsize=fontsize)
figBaseROC.savefig(out_dir/'1_bas_roc_summary.png', bbox_inches='tight')

axXCalROC.legend(fontsize=fontsize-4, loc='lower right')
axXCalROC.set_xlabel('False Positive Rate (FPR)', fontsize=fontsize)
axXCalROC.set_ylabel('True Positive Rate (TPR)', fontsize=fontsize)
figXCalROC.savefig(out_dir/'1_xcal_roc.png', bbox_inches='tight')
	