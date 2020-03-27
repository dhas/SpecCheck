from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from draw import load, annotations
import ood.explainV2 as explain

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)


out_dir     = Path('_12_shap_plabel/')
out_dir.mkdir(exist_ok=True)
figsize = (30,10)
fontsize = 20

tests_root  = Path('../_tests')
test_cfg 	= load_config('../config/epochs_test_draw_128.yml')
test_root   = tests_root/test_cfg['test_root']

os_cfg 		= test_cfg['observed_set']
os_ann 		= np.load(test_root/os_cfg['root']/'labels.npy')

cs_cfg      = test_cfg['coverage_set']
batch_size  = 100
ds_root 	= test_root/cs_cfg['root']
ds_settings = load.read_settings(ds_root/'settings.json')
ds_trans    = load.get_transformations(ds_settings['mean'], ds_settings['variance'])
ds_loader 	= load.get_npy_dataloader(ds_root,batch_size,transforms=ds_trans,shuffle=False)

enc_cfg     = test_cfg['encoders']
encoders_root = test_root/enc_cfg['root']
nets = enc_cfg['nets']

for net_ind, net_name in enumerate(['VGG11']): #nets:
	baseline_npz = encoders_root/net_name/('%s_baseline.npz' % net_name)
	if not baseline_npz.exists():
		raise Exception('%s not available' % baseline_npz)
	baseline = np.load(baseline_npz)
	PRD = baseline['PRD']
	ANN = baseline['ANN']
	HIT = baseline['HIT']
	_, _, idod = annotations.label_annotation_distribution(os_ann, ANN)			
	UNC = np.max(PRD[0,0], axis=1)	
	ANNS, SHAP, IDODS = explain.shap_by_feature(UNC, ANN, idod, out_dir/('%s_1_bas_shap.png' % net_name))	
	PIDODS = np.full((IDODS.shape), annotations.od_label)
	PIDODS[np.all(SHAP > 0, axis=2)] = annotations.id_label
	hit   = np.count_nonzero(PIDODS == IDODS)/IDODS.size
	print('Uncal pseudo labeling hitrate - %0.3f'% hit)
	explain.estimated_id_features_using_shap(SHAP, ANNS, test_cfg['dim'],
		cs_cfg['draw_limits'], out_dir/('%s_1_bas_id.png' % net_name),
		out_dir/('%s_1_bas_od.png' % net_name))

	calibrated_npz = encoders_root/net_name/('%s_calibrated.npz' % net_name)
	if not baseline_npz.exists():
		raise Exception('%s not available' % calibrated_npz)

	state = np.load(calibrated_npz)
	xcalT  = state['xcalT']
	mcalT  = state['mcalT']
	T     = state['T']
	cPRD  = state['cPRD']
	cHIT  = state['cHIT']

	xcalUNC = np.max(cPRD[0,T == xcalT][0], axis=1)
	ANNS, SHAP, IDODS = explain.shap_by_feature(xcalUNC, ANN, idod, out_dir/('%s_2_xcal_shap.png' % net_name))
	PIDODS = np.full((IDODS.shape), annotations.od_label)
	PIDODS[np.all(SHAP >= 0, axis=2)] = annotations.id_label
	hit   = np.count_nonzero(PIDODS == IDODS)/IDODS.size
	print('Xcal pseudo labeling hitrate - %0.3f'% hit)
	explain.estimated_id_features_using_shap(SHAP, ANNS, test_cfg['dim'],
		cs_cfg['draw_limits'], out_dir/('%s_2_xcal_id.png' % net_name),
		out_dir/('%s_2_xcal_od.png' % net_name))

	mcalUNC = np.max(cPRD[0,T == mcalT][0], axis=1)
	ANNS, SHAP, IDODS = explain.shap_by_feature(mcalUNC, ANN, idod, out_dir/('%s_3_mcal_shap.png' % net_name))
	PIDODS = np.full((IDODS.shape), annotations.od_label)
	PIDODS[np.all(SHAP >= 0, axis=2)] = annotations.id_label
	hit   = np.count_nonzero(PIDODS == IDODS)/IDODS.size
	print('Mcal pseudo labeling hitrate - %0.3f'% hit)
	explain.estimated_id_features_using_shap(SHAP, ANNS, test_cfg['dim'],
		cs_cfg['draw_limits'], out_dir/('%s_2_mcal_id.png' % net_name),
		out_dir/('%s_2_mcal_od.png' % net_name))

	pidod = np.full(len(idod), annotations.od_label)
	pidod[mcalUNC > 0.75] = annotations.id_label
	hit   = np.count_nonzero(pidod == idod)/len(idod)
	print('Mcal No shap pseudo labeling hitrate - %0.3f'% hit)