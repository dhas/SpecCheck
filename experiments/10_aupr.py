from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sys
sys.path.append('..')
from draw import annotations
from ood import explainV2 as explain

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)


def prc_net(UNC, idod, T, prc_savename, aupr_savename, fontsize=20, figsize=(30,15)):
	id_ind = np.where(idod == annotations.id_label)[0]
	od_ind = np.where(idod == annotations.od_label)[0]
	threshold = np.linspace(0.5,1, num=20)
	fig, axs = plt.subplots(figsize=figsize)	
	
	for i, t in enumerate(T):
		cmat =  explain._confusion_matrix(UNC[0,i], id_ind, od_ind, threshold)
		precision = cmat[:,explain.COL_TPR] /(cmat[:,explain.COL_TPR] + cmat[:,explain.COL_FPR] + 0.0001)
		recall    = cmat[:,explain.COL_TPR] /(cmat[:,explain.COL_TPR] + cmat[:,explain.COL_FNR] + 0.0001)		
		aupr = auc(recall, precision)
		axs.plot(recall, precision, 
			label='T-%0.3f, AUPR-%0.3f' % (t,aupr))
	axs.legend(fontsize=fontsize-4, loc='lower right')
	fig.savefig(prc_savename)
	


out_dir     = Path('_10_aupr/')
out_dir.mkdir(exist_ok=True)
figsize = (30,10)
fontsize = 20

tests_root  = Path('../_tests')
test_cfg 	= load_config('../config/epochs_test_draw_128.yml')
test_root   = tests_root/test_cfg['test_root']

os_cfg 		= test_cfg['observed_set']
os_ann 		= np.load(test_root/os_cfg['root']/'labels.npy')

cs_cfg      = test_cfg['coverage_set']

enc_cfg     = test_cfg['encoders']
encoders_root = test_root/enc_cfg['root']
nets = enc_cfg['nets']

for net_ind, net_name in enumerate(['VGG11']): #nets:
	baseline_npz = encoders_root/net_name/('%s_baseline.npz' % net_name)
	calibrated_npz = encoders_root/net_name/('%s_calibrated.npz' % net_name)
	baseline_npz = encoders_root/net_name/('%s_baseline.npz' % net_name)
	if not baseline_npz.exists():
		raise Exception('%s not available' % baseline_npz)
	baseline = np.load(baseline_npz)	
	ANN = baseline['ANN']

	if not calibrated_npz.exists():
		raise Exception('%s not available' % calibrated_npz)
	calibrated = np.load(calibrated_npz)	
	T = calibrated['T']	
	cPRD = calibrated['cPRD']
	cUNC = np.max(cPRD, axis=3)
	
	_, _, idod = annotations.label_annotation_distribution(os_ann, ANN)
	cmat = prc_net(cUNC, idod, T,
		out_dir/'1_prc.png',
		out_dir/'2_aupr.png')