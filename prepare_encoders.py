from pathlib import Path
import pickle
import json
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchsummary import summary
from encoder.encoder import Encoder
from encoder.train import EncoderTrainer, EncoderTester, IDEvaluator
from utils import train_utils
from utils import other_utils

def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}

cfg = {
	
	'NET01' : [4,'M']
	# 'VGG05': [64, 'M', 128, 'M'],
	# 'VGG07': [64, 'M', 128, 'M', 256, 256, 'M'],
	# 'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	# 'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	# 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	# 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

fc = {
	'NET01': 4,
	'VGG05': 512,
	'VGG07': 1024,
	'VGG11': 1024,
}


encoders_root = Path('./encoders')
encoders_root.mkdir(exist_ok=True)


dataset_dir = Path('./datasets/qd_shapes/dataset')
dataset_settings = read_settings(dataset_dir/'settings.json')

model_settings = {
	'img_side'     	: dataset_settings['img_side'],
	'num_classes'  	: dataset_settings['num_classes'],
	'dropout_rate' 	: 0,
	'norm_type' 	: 'none',
	'learning_rate'	: 0.01,
	'momentum'		: 0.85,
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
	encoder = Encoder(net_name,
		model_settings['img_side'],
		cfg[net_name],
		model_settings['num_classes'],
		model_settings['dropout_rate'],
		fc[net_name],
		norm_type=model_settings['norm_type'])
	encoder.to(device)
	summary(encoder,input_shape)

	train_loader,val_loader = train_utils.get_npy_train_loader(dataset_dir,
		training_settings['batch_size'],
		training_settings['batch_size'],
		transforms=trans)

	optimizer = optim.SGD(encoder.parameters(), 
		lr=model_settings['learning_rate'], 
		momentum=model_settings['momentum'])

	train_params = {
		'savedir': encoders_root/net_name,
		'model_name':('%s.tar' % net_name),
		'epochs':1,
		'log_interval': 20,
		'batch_size': 32,     
	}

	train_params['savedir'].mkdir(exist_ok=True)

	checkpoint_path = train_params['savedir']/train_params['model_name']
	if not checkpoint_path.exists():
		trainer = EncoderTrainer(encoder,device,
				train_loader,val_loader,
				criterion,optimizer,
				train_params)
		trainer.fit()
	
	# tester = EncoderTester(encoder,device,checkpoint_path)
	# loss,accuracy = tester.evaluate(device,criterion,val_loader)
	# print('\nEvaluation on validation set: Loss {:.4f}, Accuracy {:.2f}%\n'.format(loss,accuracy))

	specset_dir = Path('./datasets/sd_shapes/dataset')
	spec_loader = train_utils.get_npy_dataloader(specset_dir,
		100,
		transforms=trans)
	
	id_eval = IDEvaluator(encoder,device,checkpoint_path)
	loss,accuracy,id_npys = id_eval.evaluate(device,criterion,spec_loader)
	labels_dict = pickle.load(open(specset_dir/'labels.pkl','rb'))
	misses_dict = other_utils.paths_to_indexes(id_npys,spec_loader.dataset.classes)
	print('\nEvaluation on Spec set: Loss {:.4f}, Accuracy {:.2f}%\n'.format(loss,accuracy))
	other_utils.plot_label_spread(train_params['savedir']/'1_ood_spread.png',
		labels_dict,
		model_settings['img_side'],
		misses_dict=misses_dict)