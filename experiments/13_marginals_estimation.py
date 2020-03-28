from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
# import xgboost
# import shap
# import sys
# sys.path.append('..')
# from draw import load, annotations
# import ood.explainV2 as explain

classes    = ['circle', 'square']
num_classes = len(classes)
class_to_label_dict = {
	'circle' : 0,
	'square' : 1
}

id_label = 0
od_label = 1

ANNS   = ['CL', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'BR']
FEATS  = ['XMIN', 'YMIN', 'XMAX', 'YMAX', 'BR']


def plot_annotation_distribution(ANN, draw_lims, dim, savename, fontsize=20, figsize=(30,10)):
	ANN = ANN[~np.all(ANN[:,1:] == 0, axis=1)]
	fig, axs = plt.subplots(len(class_to_label_dict), len(FEATS), figsize=figsize)
	for cname in class_to_label_dict:
		clabel = class_to_label_dict[cname]
		cANN = ANN[ANN[:,0] == clabel,1:]		
		for f_ind in range(len(FEATS)):
			f = cANN[:, f_ind]
			ax = axs[clabel,f_ind]
			hist, bins = np.histogram(f, bins=15)
			freq = hist/np.sum(hist)
			ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins))
			ax.set_ylim([0,1])			
			if FEATS[f_ind] == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				ax.set_xticks(f_spread[::len(f_spread)//12])
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				ax.set_xticks(f_spread[::len(f_spread)//8])				

			if f_ind == 0:
				ax.set_ylabel(cname, fontsize=fontsize)
			if clabel == 0:
				ax.set_title(FEATS[f_ind], fontsize=fontsize)
	fig.savefig(savename)
	plt.close()


def label_annotation_distribution(os_ann, cs_ann):
	os_ann = os_ann[os_ann[:,1] != 0]
	os_ann_min = []
	os_ann_max = []
	for f_ind in range(len(ANNS)):
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

def shap_by_feature(UNC, ANN, idod, savename, fontsize=20, figsize=(30,15)):
	SHAPS = []
	ANNS  = []
	IDODS = []
	fig, axs = plt.subplots(len(classes), len(FEATS), figsize=figsize)
	for cname in classes:
		clabel 		= class_to_label_dict[cname]
		cind   		= np.where(ANN[:,0] == clabel)[0]
		X   		= pd.DataFrame(ANN[cind, 1:])
		X.columns 	= FEATS
		y   		= UNC[cind]				
		dtree 		= xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=list(y)), 100)
		SHAP 		= shap.TreeExplainer(dtree).shap_values(X)		
		axs[clabel, 0].set_ylabel(cname, fontsize=fontsize)
		for f, feat in enumerate(FEATS):
			axs[clabel, f].scatter(X[feat], SHAP[:,f])
			axs[clabel, f].set_title(feat, fontsize=fontsize)
		ANNS.append(ANN[cind, 1:])
		SHAPS.append(SHAP)
		IDODS.append(idod[cind])
	fig.savefig(savename)
	return np.stack(ANNS), np.stack(SHAPS), np.stack(IDODS)

def estimated_id_features_using_shap(SHAPS, ANNS, dim, draw_lims, id_savename, od_savename, fontsize=20, figsize=(30,15)):
	idfig, idaxs = plt.subplots(len(classes), len(FEATS), figsize=figsize)
	odfig, odaxs = plt.subplots(len(classes), len(FEATS), figsize=figsize)
	for cname in classes:
		clabel 	= class_to_label_dict[cname]
		shap    = SHAPS[clabel]
		ann     = ANNS[clabel]
		for f, feat in enumerate(FEATS):
			F  = ann[:, f]
			S  = shap[:, f]
			idF  = F[S >= 0]
			hist, bins = np.histogram(idF, bins=15)
			freq = hist/len(idF)
			idaxs[clabel, f].bar(bins[:-1], freq, align="edge", width=np.diff(bins))
			idaxs[clabel, f].set_ylim([0,1])			
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
			# odaxs[clabel, f].hist(odF)
			if feat == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				odaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//12])
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				odaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//8])

	idfig.savefig(id_savename)
	odfig.savefig(od_savename)


#----------main()-------------------------
out_dir     = Path('_13_marginals_estimation/')
out_dir.mkdir(exist_ok=True)
figsize = (30,10)
fontsize = 20

draw_limits = {'sz_lo' : 30, 'sz_hi' : 120, 'br_lo' : 100, 'br_hi' : 255, 'th'  : 5};

os_ann 		= np.load(out_dir/'labels.npy')
plot_annotation_distribution(os_ann, 
	draw_limits, 128, out_dir/'1_os_marginals.png')

for net_ind, net_name in enumerate(['VGG11']):
	baseline_npz = out_dir/('%s_baseline.npz' % net_name)
	baseline = np.load(baseline_npz)
	PRD = baseline['PRD']
	ANN = baseline['ANN']
	HIT = baseline['HIT']
	_, _, idod = label_annotation_distribution(os_ann, ANN)
	UNC = np.max(PRD[0,0], axis=1)

	baseline_shap_npz = out_dir/('%s_baseline_shap.npz' % net_name)
	if baseline_shap_npz.exists():
		baseline_shap = np.load(baseline_shap_npz)
		ANNS = baseline_shap['ANNS']
		SHAP = baseline_shap['SHAP']
		IDODS = baseline_shap['IDODS']
	# else:
		# ANNS, SHAP, IDODS = shap_by_feature(UNC, ANN, idod, out_dir/('%s_1_bas_shap.png' % net_name))			
		# np.savez(baseline_shap_npz, ANNS=ANNS, SHAP=SHAP, IDODS=IDODS)
	PIDODS = np.full((IDODS.shape), od_label)
	PIDODS[np.all(SHAP > 0, axis=2)] = id_label
	hit   = np.count_nonzero(PIDODS == IDODS)/IDODS.size
	print('Uncal pseudo labeling hitrate - %0.3f'% hit)
	estimated_id_features_using_shap(SHAP, ANNS, 128,
		draw_limits, out_dir/('%s_1_bas_id.png' % net_name),
		out_dir/('%s_1_bas_od.png' % net_name))

	calibrated_npz = out_dir/('%s_calibrated.npz' % net_name)
	state = np.load(calibrated_npz)
	xcalT  = state['xcalT']
	mcalT  = state['mcalT']
	T     = state['T']
	cPRD  = state['cPRD']
	cHIT  = state['cHIT']

	xcalUNC = np.max(cPRD[0,T == xcalT][0], axis=1)
	xcal_shap_npz = out_dir/('%s_xcal_shap.npz' % net_name)
	if xcal_shap_npz.exists():
		xcal_shap = np.load(xcal_shap_npz)
		ANNS = xcal_shap['ANNS']
		SHAP = xcal_shap['SHAP']
		IDODS = xcal_shap['IDODS']
	# else:
		# ANNS, SHAP, IDODS = shap_by_feature(xcalUNC, ANN, idod, out_dir/('%s_2_xcal_shap.png' % net_name))		
		# np.savez(xcal_shap_npz, ANNS=ANNS, SHAP=SHAP, IDODS=IDODS)
	PIDODS = np.full((IDODS.shape), od_label)
	PIDODS[np.all(SHAP >= 0, axis=2)] = id_label
	hit   = np.count_nonzero(PIDODS == IDODS)/IDODS.size
	print('Xcal pseudo labeling hitrate - %0.3f'% hit)
	estimated_id_features_using_shap(SHAP, ANNS, 128,
		draw_limits, out_dir/('%s_2_xcal_id.png' % net_name),
		out_dir/('%s_2_xcal_od.png' % net_name))

	mcalUNC = np.max(cPRD[0,T == mcalT][0], axis=1)
	mcal_shap_npz = out_dir/('%s_mcal_shap.npz' % net_name)
	if mcal_shap_npz.exists():
		mcal_shap = np.load(mcal_shap_npz)
		ANNS = mcal_shap['ANNS']
		SHAP = mcal_shap['SHAP']
		IDODS = mcal_shap['IDODS']
	# else:	
		# ANNS, SHAP, IDODS = shap_by_feature(mcalUNC, ANN, idod, out_dir/('%s_3_mcal_shap.png' % net_name))
		# np.savez(mcal_shap_npz, ANNS=ANNS, SHAP=SHAP, IDODS=IDODS)
	PIDODS = np.full((IDODS.shape), od_label)
	PIDODS[np.all(SHAP >= 0, axis=2)] = id_label
	hit   = np.count_nonzero(PIDODS == IDODS)/IDODS.size
	print('Mcal pseudo labeling hitrate - %0.3f'% hit)
	estimated_id_features_using_shap(SHAP, ANNS, 128,
		draw_limits, out_dir/('%s_2_mcal_id.png' % net_name),
		out_dir/('%s_2_mcal_od.png' % net_name))