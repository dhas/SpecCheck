from pathlib import Path
import pickle
import json
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchsummary import summary
from draw import load
from ood.encoder import Encoder
from ood.train import EncoderTrainer
from ood.explain import DatasetExplainer
from utils import other_utils
from config import config

def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}

def training_necessary(checkpoint_path,cfg,cfg_path):	
	training_needed = (not checkpoint_path.exists()) or \
		(not cfg_path.exists()) or \
		(cfg != pickle.load(open(cfg_path,'rb')))
	return training_needed

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

	input_shape = (1,draw_cfg['img_side'],draw_cfg['img_side'])


	train_params = enc_cfg['train']

	trans = transforms.Compose([
		transforms.Lambda(lambda x:x.permute(2,0,1).type(torch.FloatTensor))
	])
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
			transforms=trans)	

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

	# 	print('')
		explainer = DatasetExplainer(encoder,device,checkpoint_path)		
		specset_dir = Path(draw_cfg['root'])/draw_cfg['sd']['root']/'dataset'
		spec_loader = load.get_npy_dataloader(specset_dir,
			100,
			transforms=trans)
		
		labels_dict = pickle.load(open(specset_dir/'labels.pkl','rb'))
		loss,accuracy,anns,preds,ood_npys = explainer.evaluate(device,criterion,spec_loader,labels_dict)
		misses_dict = other_utils.paths_to_indexes(ood_npys,spec_loader.dataset.classes)
		other_utils.plot_label_spread(train_params['savedir']/'1_ood_hist.png',
			labels_dict,
			draw_cfg['img_side'],
			misses_dict=misses_dict)
		explainer.shap_explain(anns,preds,spec_loader.dataset.class_to_idx,train_params['savedir'])

if __name__ == '__main__':
	main(Path('./'), Path('./_outputs'))