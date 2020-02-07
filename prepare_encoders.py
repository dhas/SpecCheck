from pathlib import Path
import pickle
import json
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchsummary import summary
from ood.encoder import Encoder
from ood.train import EncoderTrainer
from ood.explain import DatasetExplainer
from utils import train_utils
from utils import other_utils

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

cfg = {
		
	# 'NET01' : {'head'  :['4','M'],
	# 		   'tail'  :['4'],			   
	# 		   'trn'   :{
	# 						'optim'	: optim.RMSprop,
	# 						'lr'	: 1e-4,			   				
	# 					},
	# 		  },

	# 'NET02' : {'head'  :['32', 'M', '64', 'M'],
	# 		   'tail'  :['128'],
	# 		   'trn'   :{
	# 						'optim'	: optim.Adam,
	# 						'lr'	: 1e-4,			   				
	# 					},
	# 		  },

	# 'NET03' : {'head'  :['32', 'M', '64', 'M'],
	# 		   'tail'  :['128', 'D#0.5'],
	# 		   'trn'   :{
	# 						'optim'	: optim.Adam,
	# 						'lr'	: 1e-4,			   				
	# 					},
	# 		  },
	# 'NET04' : {'head'  :['32#BN', 'M', '64#BN', 'M'],
	# 		   'tail'  :['128'],
	# 		   'trn'   :{
	# 						'optim'	: optim.Adam,
	# 						'lr'	: 1e-4,			   				
	# 					},
	# 		  },
	'NET05' : {'head'  :['32#IN', 'M', '64#IN', 'M'],
			   'tail'  :['128'],
			   'trn'   :{
							'optim'	: optim.Adam,
							'lr'	: 1e-4,			   				
						},
			  },
	# 'VGG05': [64, 'M', 128, 'M'],
	# 'VGG07': [64, 'M', 128, 'M', 256, 256, 'M'],
	# 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	# 'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	# 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	# 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


encoders_root = Path('./_encoders')
encoders_root.mkdir(exist_ok=True)


dataset_dir = Path('./_dataset/qd_shapes/dataset')
dataset_settings = read_settings(dataset_dir/'settings.json')

model_settings = {
	'img_side'     	: dataset_settings['img_side'],
	'num_classes'  	: dataset_settings['num_classes'],	
}

input_shape = (1,model_settings['img_side'],model_settings['img_side'])


training_settings = {
	'epochs':1,
	'log_interval': 5,
	'batch_size': 32,
}
trans = transforms.Compose([
	transforms.Lambda(lambda x:x.permute(2,0,1).type(torch.FloatTensor))
])
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda')
for net_name in list(cfg.keys()):	
	
	train_params = {
		'savedir'		: encoders_root/net_name,		
		'epochs'		: 1,
		'log_interval'	: 20,
		'batch_size'	: 32,     
	}

	train_params['savedir'].mkdir(exist_ok=True)
	checkpoint_path = train_params['savedir']/('%s.tar' % net_name)
	cfg_path        = train_params['savedir']/('%s.pkl' % net_name)
		
	
	encoder = Encoder(net_name,
		model_settings['img_side'],
		cfg[net_name],
		model_settings['num_classes'])
	encoder.to(device)
	summary(encoder,input_shape)

	train_loader,val_loader = train_utils.get_npy_train_loader(dataset_dir,
		training_settings['batch_size'],
		training_settings['batch_size'],
		transforms=trans)	

	if not training_necessary(checkpoint_path,cfg[net_name],cfg_path):
		print('Encoder %s already trained' % net_name)
	else:
		print('Training encoder %s' % net_name)
		optimizer = cfg[net_name]['trn']['optim'](encoder.parameters(),
				lr = cfg[net_name]['trn']['lr'])


		trainer = EncoderTrainer(encoder,device,
				train_loader,val_loader,
				criterion,optimizer,
				train_params)
		trainer.fit(checkpoint_path)
		break
	
# 		pickle.dump(cfg[net_name],open(cfg_path,'wb'))

# 	print('')
# 	explainer = DatasetExplainer(encoder,device,checkpoint_path)
# 	# loss,accuracy = explainer.evaluate(device,criterion,val_loader)
# 	# print('\nEvaluation on validation set: Loss {:.4f}, Accuracy {:.2f}%\n'.format(loss,accuracy))

# 	specset_dir = Path('./_datasets/sd_shapes/dataset')
# 	spec_loader = train_utils.get_npy_dataloader(specset_dir,
# 		100,
# 		transforms=trans)
	
# 	labels_dict = pickle.load(open(specset_dir/'labels.pkl','rb'))
# 	explainer.shap_explain(device,criterion,spec_loader,labels_dict,train_params['savedir'])	

# 	print('\nOOD evaluation')
# 	loss,accuracy,ood_npys = explainer.explain(device,criterion,spec_loader)	
# 	misses_dict = other_utils.paths_to_indexes(ood_npys,spec_loader.dataset.classes)
# 	print('\nEvaluation on Spec set: Loss {:.4f}, Accuracy {:.2f}%\n'.format(loss,accuracy))
# 	other_utils.plot_label_spread(train_params['savedir']/'1_ood_spread.png',
# 		labels_dict,
# 		model_settings['img_side'],
# 		misses_dict=misses_dict)