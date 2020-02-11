from pathlib import Path
import pickle
import json
import torch
from torch import nn, optim
from torchsummary import summary
from draw import load
from ood.encoder import Encoder
from ood.train import EncoderTrainer,training_necessary
from ood.explain import DatasetExplainer
from utils import other_utils
from config import config
import numpy as np

def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}

def get_optimizer(parameters,settings):
	if settings['optim'] == 'Adam':
		optimizer = optim.Adam(parameters,lr = float(settings['lr']))
		# optimizer = optim.Adam(parameters)
	else:
		raise Exception('Unknown optimizer')
	return optimizer


def main(project_root,out_dir):
	encoders_root = out_dir/'_encoders'
	encoders_root.mkdir(exist_ok=True)

	draw_cfg = config.get_draw_config(project_root)
	enc_cfg  = config.get_encoders_config(project_root)

	dataset_dir = Path(draw_cfg['root'])/draw_cfg['qd']['root']/'dataset'
	dataset_settings = read_settings(dataset_dir/'settings.json')
	dataset_trans = load.get_transformations(dataset_settings['mean'],dataset_settings['variance'])

	input_shape = (1,draw_cfg['img_side'],draw_cfg['img_side'])

	train_params = enc_cfg['train']

	
	criterion = nn.CrossEntropyLoss()

	device = torch.device('cuda')
	nets = enc_cfg['nets']
	for net_name in nets:
		train_params['savedir'] = encoders_root/net_name
		train_params['savedir'].mkdir(exist_ok=True)
		checkpoint_path = train_params['savedir']/('%s.tar' % net_name)
		cfg_path        = train_params['savedir']/('%s.pkl' % net_name)
		
		
		encoder = Encoder(net_name,
			draw_cfg['img_side'],
			nets[net_name],
			draw_cfg['num_classes'])
		encoder.to(device)
		summary(encoder,input_shape)

		train_loader,val_loader = load.get_npy_train_loader(dataset_dir,
			train_params['batch_size'],
			train_params['batch_size'],
			transforms=dataset_trans)	

		if not training_necessary(checkpoint_path,nets[net_name],cfg_path):
			print('Encoder %s already trained' % net_name)
		else:
			print('Training encoder %s' % net_name)			
			optimizer = get_optimizer(encoder.parameters(), 
				enc_cfg['train']['optimizer'])


			trainer = EncoderTrainer(encoder,device,
					train_loader,val_loader,
					criterion,optimizer,
					train_params)
			trainer.fit(checkpoint_path)
			pickle.dump(nets[net_name],open(cfg_path,'wb'))
	
		data_loader = load.get_npy_dataloader(dataset_dir,
			100,
			transforms=dataset_trans,
			shuffle=False)	
		explainer = DatasetExplainer(encoder,device,checkpoint_path)
		labels_dict = pickle.load(open(dataset_dir/'est_labels.pkl','rb'))
		_,qd_accuracy,_,qd_preds,_ = explainer.evaluate(device,criterion,data_loader,labels_dict)

		explainer = DatasetExplainer(encoder,device,checkpoint_path)		
		specset_dir = Path(draw_cfg['root'])/draw_cfg['sd']['root']/'dataset'
		specset_settings = read_settings(specset_dir/'settings.json')
		specset_trans = load.get_transformations(specset_settings['mean'],specset_settings['variance'])
		spec_loader = load.get_npy_dataloader(specset_dir,
			100,
			transforms=specset_trans,
			shuffle=False)
		
		labels_dict = pickle.load(open(specset_dir/'labels.pkl','rb'))
		_,sd_accuracy,anns,sd_preds,ood_npys = explainer.evaluate(device,criterion,spec_loader,labels_dict)
		other_utils.plot_softmax_distributions(qd_preds,sd_preds,train_params['savedir']/'4_softmax_dist.png')
		
		np.savez(train_params['savedir']/'softmax_scores.npz', 
			qd_preds=qd_preds,sd_preds=sd_preds)

		misses_dict = other_utils.paths_to_indexes(ood_npys,spec_loader.dataset.classes)
		other_utils.plot_label_spread(train_params['savedir']/'1_ood_hist.png',
			labels_dict,
			draw_cfg['img_side'],
			misses_dict=misses_dict)
		explainer.shap_explain(anns,sd_preds,spec_loader.dataset.class_to_idx,train_params['savedir'])

if __name__ == '__main__':
	main(Path('./'), Path('./_outputs'))