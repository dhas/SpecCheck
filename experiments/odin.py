import sys
sys.path.append('..')
from config import config
from draw import load
from ood.encoder import Encoder
from ood.train import training_necessary
from ood.explain import DatasetExplainer
from utils import other_utils
import numpy as np
from pathlib import Path
import time
import pickle
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import json

def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}


# def odin_test(net1, criterion, CUDA_DEVICE, testloader10, testloader, id_var, od_var, noiseMagnitude1, temper, score_dir):
# 	net1.to(CUDA_DEVICE)
# 	net1.eval()
# 	t0 = time.time()
# 	f1 = open(score_dir/'confidence_Base_In.txt', 'w')
# 	f2 = open(score_dir/'confidence_Base_Out.txt', 'w')
# 	g1 = open(score_dir/'confidence_ODIN_In.txt', 'w')
# 	g2 = open(score_dir/'confidence_ODIN_Out.txt', 'w')
	
# 	softmax_id 	= []
# 	softmax_od 	= []
# 	odin_id 	= []
# 	odin_od 	= []
# 	print("Processing in-distribution images")
# ########################################In-distribution###########################################
# 	for j, data in enumerate(tqdm(testloader10)):
# 		# if j<1000: continue
# 		images, _, _ = data
		
# 		inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
# 		outputs = net1(inputs)		
# 		# Calculating the confidence of the output, no perturbation added here, no temperature scaling used
# 		nnOutputs = outputs.data.cpu()
# 		nnOutputs = nnOutputs.numpy()
# 		nnOutputs = nnOutputs[0]		
# 		nnOutputs = nnOutputs - np.max(nnOutputs)		
# 		nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
# 		softmax_id.append(np.max(nnOutputs))
# 		# softmax_id.append(torch.topk(F.softmax(outputs,dim=1),1))
# 		f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

# 		# Using temperature scaling
# 		outputs = outputs / temper
	
# 		# Calculating the perturbation we need to add, that is,
# 		# the sign of gradient of cross entropy loss w.r.t. input
# 		maxIndexTemp = np.argmax(nnOutputs)
# 		labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
# 		loss = criterion(outputs, labels)
# 		loss.backward()
		
# 		# Normalizing the gradient to binary in {0, 1}
# 		gradient =  torch.ge(inputs.grad.data, 0)		
# 		gradient = (gradient.float() - 0.5) * 2
		
# 		# Normalizing the gradient to the same space of image
# 		gradient[0][0] = (gradient[0][0] )/(id_var)
# 		# # gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
# 		# # gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
# 		# # gradient[0][2] = (gradient[0][2])/(66.7/255.0)
# 		# Adding small perturbations to images
# 		tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
# 		outputs = net1(Variable(tempInputs))
# 		outputs = outputs / temper
# 		# Calculating the confidence after adding perturbations
# 		nnOutputs = outputs.data.cpu()
# 		nnOutputs = nnOutputs.numpy()
# 		nnOutputs = nnOutputs[0]
# 		nnOutputs = nnOutputs - np.max(nnOutputs)
# 		nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
# 		odin_id.append(np.max(nnOutputs))
# 		g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
# 		# if j % 100 == 99:
# 		# 	print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
# 		# 	t0 = time.time()
		
# 		# if j == N - 1: break

# 	# t0 = time.time()
# 	print("Processing out-of-distribution images")
# ###################################Out-of-Distributions#####################################
# 	for j, data in enumerate(tqdm(testloader)):
# 		# if j<1000: continue
# 		images, _, _ = data
		
# 		inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
# 		outputs = net1(inputs)
		


# 		# Calculating the confidence of the output, no perturbation added here
# 		nnOutputs = outputs.data.cpu()
# 		nnOutputs = nnOutputs.numpy()
# 		nnOutputs = nnOutputs[0]
# 		nnOutputs = nnOutputs - np.max(nnOutputs)
# 		nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
# 		softmax_od.append(np.max(nnOutputs))
# 		# softmax_od.append(torch.topk(F.softmax(outputs,dim=1),1))
# 		f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
		
# 		# Using temperature scaling
# 		outputs = outputs / temper
  
  
# 		# Calculating the perturbation we need to add, that is,
# 		# the sign of gradient of cross entropy loss w.r.t. input
# 		maxIndexTemp = np.argmax(nnOutputs)
# 		labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
# 		loss = criterion(outputs, labels)
# 		loss.backward()
		
# 		# Normalizing the gradient to binary in {0, 1}
# 		gradient =  (torch.ge(inputs.grad.data, 0))
# 		# gradient = gradient.float()
# 		gradient = (gradient.float() - 0.5) * 2
# 		# Normalizing the gradient to the same space of image
# 		gradient[0][0] = (gradient[0][0] )/(od_var)
# 		# gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
# 		# gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
# 		# gradient[0][2] = (gradient[0][2])/(66.7/255.0)
# 		# Adding small perturbations to images
# 		tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
# 		outputs = net1(Variable(tempInputs))
# 		outputs = outputs / temper
# 		# Calculating the confidence after adding perturbations
# 		nnOutputs = outputs.data.cpu()
# 		nnOutputs = nnOutputs.numpy()
# 		nnOutputs = nnOutputs[0]
# 		nnOutputs = nnOutputs - np.max(nnOutputs)
# 		nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
# 		odin_od.append(np.max(nnOutputs))
# 		g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
# 		# if j % 100 == 99:
# 		# 	print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
# 		# 	t0 = time.time()

# 		# if j== N-1: break
# 	return np.stack(softmax_id),np.stack(softmax_od),np.stack(odin_id),np.stack(odin_od)

def base_test(net, criterion, CUDA_DEVICE, loader, N, variance, T, eps, score_dir):
	f1 = open(score_dir/'confidence_Base_In.txt', 'w')
	f2 = open(score_dir/'confidence_Base_Out.txt', 'w')
	g1 = open(score_dir/'confidence_ODIN_In.txt', 'w')
	g2 = open(score_dir/'confidence_ODIN_Out.txt', 'w')

	softmax_base = []
	softmax_odin = []
	net.to(CUDA_DEVICE)
	net.eval()
	for j, data in enumerate(tqdm(loader)):
		# if j<1000: continue		
		images, _, _ = data
		inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
		outputs = net(inputs)

		# Calculating the confidence of the output, no perturbation added here, no temperature scaling used
		nnOutputs = outputs.data.cpu()
		nnOutputs = nnOutputs.numpy()
		nnOutputs = nnOutputs[0]
		nnOutputs = nnOutputs - np.max(nnOutputs)
		nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
		softmax_base.append(np.max(nnOutputs))
		f1.write("{}, {}, {}\n".format(T, eps, np.max(nnOutputs)))
	return np.stack(softmax_base)

if __name__ == '__main__':
	project_root = Path('../')
	enc_prefix = project_root/'_outputs'
	draw_cfg = config.get_draw_config(project_root)
	enc_cfg  = config.get_encoders_config(project_root)

	batch_size 	= 100
	eps 		= 0.0014
	T 			= 1000
	CUDA_DEVICE = 0
	N 			= 10

	id_dir 		= project_root/draw_cfg['root']/draw_cfg['qd']['root']/'dataset'
	id_settings = read_settings(id_dir/'settings.json')
	id_trans 	= load.get_transformations(id_settings['mean'], id_settings['variance'])
	id_loader 	= load.get_npy_dataloader(id_dir,batch_size,transforms=id_trans)	

	od_dir 		= project_root/draw_cfg['root']/draw_cfg['sd']['root']/'dataset'
	od_settings = read_settings(od_dir/'settings.json')
	od_trans 	= load.get_transformations(od_settings['mean'], od_settings['variance'])
	od_loader 	= load.get_npy_dataloader(od_dir,batch_size,transforms=od_trans)
	od_ann_dict	= pickle.load(open(od_dir/'labels.pkl','rb'))
	
	criterion = nn.CrossEntropyLoss()
	score_dir = Path('_softmax_scores')
	score_dir.mkdir(exist_ok=True)

	nets = enc_cfg['nets']

	for net_name in nets: #['NET02']:
		print('Processing net %s' % net_name)
		encoder = Encoder(net_name,
				draw_cfg['img_side'],
				nets[net_name],
				draw_cfg['num_classes'])

		
		checkpoint_path = enc_prefix/enc_cfg['root']/net_name/('%s.tar' % net_name)
		cfg_path        = enc_prefix/enc_cfg['root']/net_name/('%s.pkl' % net_name)
		
		if training_necessary(checkpoint_path,nets[net_name],cfg_path):
			raise Exception('Trained encoder unavailable')
		else:		
			explainer = DatasetExplainer(encoder, checkpoint_path)
			
			T = [1, 10, 100, 1000]
			HIT, LOG, ANN, FNM = explainer._evaluate_ts(CUDA_DEVICE,criterion,od_loader,
				T=T, y_ann=od_ann_dict)
			explainer.softmax_by_feature('1_softmax_by_feature_%s.png' % net_name,
				T,ANN,LOG,figsize=(40,20))

			# od_S_miss = od_S[np.where(od_hits == False)[0]]
			# other_utils.plot_softmax_distributions(enc_prefix/enc_cfg['root']/net_name/'5_softmax_dist.png',
			# 	id_S,od_S,
			# 	od_misses=od_S_miss)
			# data = np.load('data.npz')
			# od_anns = data['od_anns']
			# od_S = data['od_S']
			# explainer.softmax_by_feature(enc_prefix/enc_cfg['root']/net_name/'6_softmax_by_feature.png',
			# 	od_anns,od_S)
			# shaps = explainer.softmax_shap_by_feature(od_anns,od_S)

			# explainer.shap_explain(od_anns,
			# 	od_S,
			# 	od_loader.dataset.class_to_idx,
			# 	enc_prefix/enc_cfg['root']/net_name)

			# output = explainer._evaluate(CUDA_DEVICE,criterion,id_loader)
			
			
			# _,_,_,qd_preds,_ = explainer.evaluate(CUDA_DEVICE,criterion,id_loader,labels_dict)

			# labels_dict = pickle.load(open(od_dir/'labels.pkl','rb'))
			# _,_,_,sd_preds,_ = explainer.evaluate(CUDA_DEVICE,criterion,od_loader,labels_dict)
			# other_utils.plot_softmax_distributions(qd_preds,sd_preds,
			# 	enc_prefix/enc_cfg['root']/net_name/'5_explainer.png',
			# 	softmax_scores=False)
			
			# id_softmax_base = base_test(encoder,
			# 	criterion,
			# 	CUDA_DEVICE,
			# 	id_loader,
			# 	N,			
			# 	id_settings['variance'],
			# 	T,
			# 	eps,			
			# 	score_dir)
			
			# other_utils.plot_softmax_distributions(id_S,id_S,
			# 	enc_prefix/enc_cfg['root']/net_name/'5_odin_id.png',
			# 	softmax_scores=True)

			# other_utils.plot_softmax_distributions(softmax_id,softmax_od,
			# 	enc_prefix/enc_cfg['root']/net_name/'5_odin_id.png',
			# 	softmax_scores=True)

			# other_utils.plot_softmax_distributions(odin_id,odin_od,
			# 	enc_prefix/enc_cfg['root']/net_name/'5_odin_od.png',
			# 	softmax_scores=True)
