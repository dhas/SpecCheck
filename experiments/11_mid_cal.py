from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('..')
from draw import load, annotations
from ood.encoder import Encoder
from ood.explainV2 import TrainedModel
from ood.calibration import ModelWithTemperature

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)


def idod_distances(UNC, idod, T, savename, extreme=True, fontsize=20, figsize=(30,15)):
		id_ind = np.where(idod == annotations.id_label)[0]
		od_ind = np.where(idod == annotations.od_label)[0]
		UNC = UNC[0] #considering 1 eps
		fig, axs = plt.subplots(2, len(T), figsize=figsize)		
		for i, t in enumerate(T):
			ax = axs[0, i]
			ax.hist(UNC[i, id_ind], color='blue')
			ax.set_xlim([0.5,1])
			ax.set_title('T-%0.2f' % t, fontsize=fontsize)
			if extreme:
				mse_id = np.square(1.0 - UNC[i,id_ind]).mean()
			else:
				mse_id = np.square(0.75 - UNC[i,id_ind]).mean()
			ax.set_xlabel('%0.3f' % mse_id, fontsize=fontsize)

			ax = axs[1, i]
			ax.hist(UNC[i, od_ind], color='red')
			ax.set_xlim([0.5,1])
			if extreme:
				mse_od = np.square(UNC[i,od_ind] - 0.5).mean()
			else:
				mse_od = np.square(UNC[i,od_ind] - 0.75).mean()
			ax.set_xlabel('%0.3f' % mse_od, fontsize=fontsize)
		fig.savefig(savename)

out_dir     = Path('_11_mid_cal/')
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
	idod_distances(cUNC, idod, T, out_dir/'1_unc_mses.png', extreme=False)
	encoder = Encoder(net_name,
				test_cfg['dim'],
				nets[net_name],
				test_cfg['num_classes'])
	checkpoint_path = encoders_root/net_name/('%s.tar' % net_name)
	model = TrainedModel(encoder, checkpoint_path)
	mcalT = model.calibrate_for_centered_distribution(idod, ds_loader)

	xcalT = model.calibrate_for_highest_auroc(idod, ds_loader)
	
	# idod_e = np.full(len(idod), 0.75)
	# idod_ds = load.my_subset(ds_loader.dataset, 
	# 		np.arange(0,len(ds_loader.dataset)),
	# 		torch.tensor(idod_e).type(torch.float))
	# idod_loader = torch.utils.data.DataLoader(idod_ds, 
	# 	batch_size=batch_size, shuffle=False)

	# cal_encoder = ModelWithTemperature(model.model)
	# cal_encoder.set_mid_cal(idod_loader)