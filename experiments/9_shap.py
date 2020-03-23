from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import sys
import xgboost
import matplotlib.pyplot as plt
import shap
sys.path.append('..')
from draw import annotations
from ood import explainV2 as explain


def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)

out_dir     = Path('_9_shap/')
out_dir.mkdir(exist_ok=True)
figsize = (30,10)
fontsize = 20

tests_root  = Path('../_tests')
test_cfg 	= load_config('../config/test_draw_128.yml')
test_root   = tests_root/test_cfg['test_root']
os_cfg 		= test_cfg['observed_set']
cs_cfg      = test_cfg['coverage_set']
enc_cfg     = test_cfg['encoders']
encoders_root = test_root/enc_cfg['root']
nets = enc_cfg['nets']

osANN = np.load(test_root/os_cfg['root']/'labels.npy')
annotations.plot_annotation_distribution(osANN, cs_cfg['draw_limits'], test_cfg['dim'], out_dir/'1_os_annotations.png')
csANN = np.load(test_root/cs_cfg['root']/'labels.npy')
annotations.plot_annotation_distribution(csANN, cs_cfg['draw_limits'], test_cfg['dim'], out_dir/'2_cs_annotations.png')

figBAS, axBAS = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=figsize)
figCAL, axCAL = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=figsize)
for net_ind, net_name in enumerate(nets): #nets:
	baseline_npz = encoders_root/net_name/('%s_baseline.npz' % net_name)
	if not baseline_npz.exists():
		raise Exception('%s not available' % baseline_npz)
	baseline = np.load(baseline_npz)
	PRD = baseline['PRD']
	ANN = baseline['ANN']
	HIT = baseline['HIT']
	UNC = np.max(PRD[0,0], axis=1)
	SHAP = explain.shap_by_feature(UNC, ANN, out_dir/('%s_shap_by_feature.png' % net_name))
	explain.shap_summary(SHAP, ANN, net_name, axBAS)

	calibrated_npz = encoders_root/net_name/('%s_calibrated.npz' % net_name)
	if not calibrated_npz.exists():
		raise Exception('%s not available' % calibrated_npz)
	calibrated = np.load(calibrated_npz)	
	optT_ind = np.where(calibrated['T'] == calibrated['optT'])[0]
	cPRD = calibrated['cPRD'][0,optT_ind]
	cUNC = np.max(cPRD[0], axis=1)
	SHAP = explain.shap_by_feature(cUNC, ANN, out_dir/('%s_shap_by_feature_cal.png' % net_name))
	explain.mean_shap_by_feature(SHAP, ANN, net_name, axCAL)

handles, labels = axBAS[0,0].get_legend_handles_labels()
figBAS.legend(handles, labels, loc='upper right', fontsize=fontsize)
figBAS.savefig(out_dir/'3_mean_shaps_bas.png')

handles, labels = axCAL[0,0].get_legend_handles_labels()
figCAL.legend(handles, labels, loc='upper right', fontsize=fontsize)
figCAL.savefig(out_dir/'3_mean_shaps_cal.png')