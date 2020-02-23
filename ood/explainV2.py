import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import xgboost
import shap
import torch
from torch.autograd import Variable
sys.path.append('..')
from utils import other_utils
from draw import annotations

class BaselineExplainer:
	def __init__(self,model,checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		self.model = model

	def evaluate(self, device, criterion, loader, y_ann=None):
		self.model.to(device)
		self.model.eval()

		PRD = np.empty((len(loader.dataset),
			len(loader.dataset.classes)),
		dtype=np.float)

		HIT = np.empty((len(loader.dataset),
			1),
		dtype=np.bool)

		if not y_ann is None:
			ANN = np.empty((len(loader.dataset),
			annotations.ANN_SZ),
			dtype=np.int)
			with_annotations = True
		else:
			with_annotations = False

		batch_size = loader.batch_size
		for batch_id, (_x, _y, fnames) in enumerate(tqdm(loader)):
			inputs = _x.cuda(device)
			outputs = self.model(inputs)
			
			nnOutputs = outputs.data.cpu()
			nnOutputs = nnOutputs.numpy()
			nnOutputs = nnOutputs - np.max(nnOutputs,axis=1).reshape(-1,1)								
			nnOutputs = np.array(
				[np.exp(nnOutput)/np.sum(np.exp(nnOutput)) for nnOutput in nnOutputs])
			
			chunk = slice(batch_id*batch_size,(batch_id+1)*batch_size)
			PRD[chunk] = nnOutputs
			HIT[chunk] = (np.argmax(nnOutputs, axis=1) == _y.numpy()).reshape(-1,1)
			
		if with_annotations:
			return PRD, HIT, ANN
		else:
			return PRD, HIT




class OdinExplainer:
	def __init__(self, model, checkpoint_path, is_calibrated=False):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		self.model 			= model
		self.is_calibrated 	= is_calibrated		
		self.ANN_COL_CL		= 0
		self.FEATS  		= ['X0', 'Y0', 'SZ', 'BR']
		

	def axes_softmax_distribution(self, PRD, HIT, T, axs, title, fontsize=20):
		softmax_bins = np.linspace(0.5,1.0,50)
		for t_ind in range(len(T)):
			ax = axs[t_ind]
			SFM = np.max(PRD[t_ind],axis=1)
			ACC = np.count_nonzero(HIT[t_ind] == True)/(HIT[t_ind].size)
			ax.hist(SFM,softmax_bins, label=('accuracy-%0.3f' % ACC), alpha = 0.8, color='blue')
			ax.legend(fontsize=fontsize-4, loc='upper center')
			ax.set_title('T-%0.2f' % T[t_ind], fontsize=fontsize)
			ax.tick_params(axis='both', which='major', labelsize=fontsize-10)
			if t_ind == 0:
				ax.set_ylabel(title, fontsize=fontsize)	

	def plot_softmax_distribution(self, PRD, HIT, eps, T, savename, fontsize=20, figsize=(30,15)):
		softmax_bins = np.linspace(0.5,1.0,50)

		fig, axs = plt.subplots(len(eps), len(T), figsize=figsize)
		#special case for single plot
		if (len(eps) == 1) and (len(T) == 1):
			SFM = np.max(PRD[0,0], axis=1)
			ACC = np.count_nonzero(HIT[0,0] == True)/(HIT[0,0].size)
			axs.hist(SFM, softmax_bins, label=('accuracy-%0.3f' % ACC), alpha = 0.8, color='blue',)
			axs.set_ylabel('eps-%0.2f' % eps[0], fontsize=fontsize)
			axs.set_title('T-%0.2f' % T[0], fontsize=fontsize)
			axs.legend(fontsize=fontsize-4, loc='upper left')
		else:
			for e_ind in range(len(eps)):				
				for t_ind in range(len(T)):
					if len(axs.shape) == 1:
						ax = axs[e_ind + t_ind]
					else:
						ax = axs[e_ind,t_ind]
					
					SFM = np.max(PRD[e_ind,t_ind],axis=1)
					ACC = np.count_nonzero(HIT[e_ind,t_ind] == True)/(HIT[e_ind,t_ind].size)
					ax.hist(SFM,softmax_bins, label=('accuracy-%0.3f' % ACC), alpha = 0.8, color='blue')
					ax.legend(fontsize=fontsize-4, loc='upper left')
					if e_ind == 0:
						ax.set_title('T-%0.2f' % T[t_ind], fontsize=fontsize)
					if t_ind == 0:
						ax.set_ylabel('eps-%0.2f' % eps[e_ind], fontsize=fontsize)

		fig.savefig(savename)

	
	def _feature_scatter(self, SCR, ANN, eps, T, savename, salience, fontsize, figsize):
		summary = {}
		fig, axs = plt.subplots(len(T), len(self.FEATS), figsize=figsize)
		for t_ind in np.arange(len(T)):
			summary[t_ind] = {}
			for f_ind in range(len(self.FEATS)):
				F  = ANN[:, f_ind]
				S  = SCR[t_ind][:,f_ind]
				ax = axs[t_ind, f_ind]
				ax.scatter(F, S)
				uF = np.unique(F)
				if f_ind < 2:
					x_ticks = uF
				elif f_ind >=3 :
					x_ticks = uF[::16]
				else:
					x_ticks = uF[::2]

				ax.set_xticks(x_ticks)
				if t_ind == 0:
					ax.set_title(self.FEATS[f_ind], fontsize=fontsize)
				if f_ind == 0:
					ax.set_ylabel('T-%0.3f' % T[t_ind], fontsize=fontsize)

				summary[t_ind][f_ind] = {}
				summary[t_ind][f_ind]['f'] = uF
				if salience == 'max':
					summary[t_ind][f_ind]['s'] = [S[F == f].max() for f in uF]
				else:
					summary[t_ind][f_ind]['s'] = [S[F == f].mean() for f in uF]

		fig.savefig(savename)
		return summary

	def plot_softmax_by_feature(self, PRD, ANN, label, eps, T, savename, fontsize=20, figsize=(30,15)):
		if PRD.shape[0] > 1:
			raise Exception('Not supported for more than one eps value')

		label_ind = np.where(ANN[:,self.ANN_COL_CL] == label)[0]
		ANN = ANN[label_ind,1:5] #excluding class label and thickness
		PRD = PRD[:,:,label_ind,:]		
		SFM   = np.expand_dims(np.max(PRD,axis=3),3)
		SFM   = np.repeat(SFM,len(self.FEATS),axis=3)		
		SFM   = SFM[0] #only supports one eps
		return self._feature_scatter(SFM, ANN, eps, T, savename, 'mean', fontsize, figsize)

	def summary_by_feature(self, summary, T, cal_T_ind, net_name, axs, label, set_title=False, fontsize=20, figsize=(30,15)):
		for f_ind in range(len(self.FEATS)):
			ax = axs[f_ind]
			ax.plot(summary[cal_T_ind][f_ind]['f'],
				summary[cal_T_ind][f_ind]['s'],
				label='%s' % net_name)
			if f_ind == 0:
				ax.set_ylabel(label, fontsize=fontsize)
			if set_title:
				ax.set_title(self.FEATS[f_ind], fontsize=fontsize)



	def plot_shap_by_feature(self, PRD, ANN, label, eps, T, savename, fontsize=20, figsize=(30,15)):
		if PRD.shape[0] > 1:
			raise Exception('Not supported for more than one eps value')

		label_ind = np.where(ANN[:,self.ANN_COL_CL] == label)[0]
		ANN = ANN[label_ind,1:5] #excluding class label and thickness
		PRD = PRD[:,:,label_ind,:]
		#only supports one eps
		PRD = PRD[0]
		SHAP = np.empty((len(T),
			ANN.shape[0],
			len(self.FEATS)),dtype=np.float)
		
		X = pd.DataFrame(ANN)
		X.columns = self.FEATS
		for t_ind in range(len(T)):
			SFM = np.max(PRD[t_ind],axis=1)
			dtree = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=list(SFM)), 100)
			SHAP[t_ind] = shap.TreeExplainer(dtree).shap_values(X)
		return self._feature_scatter(SHAP, ANN, eps, T, savename, 'mean', fontsize, figsize)		
	


	def evaluate(self, device, criterion, loader, eps, T, var=1, y_ann=None):
		self.model.to(device)
		self.model.eval()

		if self.is_calibrated:			
			T = [1] #calibrated temp included in the model

		PRD = np.empty((len(eps),
			len(T),
			len(loader.dataset),
			len(loader.dataset.classes)),
		dtype=np.float)

		HIT = np.empty((len(eps),
			len(T),
			len(loader.dataset),
			1),
		dtype=np.bool)

		if not y_ann is None:
			ANN = np.empty((len(loader.dataset),
			6),
			dtype=np.int)
			with_annotations = True
		else:
			with_annotations = False

		batch_size = loader.batch_size

		for batch_id, (_x, _y, fnames) in enumerate(tqdm(loader)):
			inputs = Variable(_x.cuda(device), requires_grad = True)			
			outputs = self.model(inputs)
			chunk = slice(batch_id*batch_size,(batch_id+1)*batch_size)
			for t_ind in range(len(T)):
				temper = T[t_ind]			
				# Calculating the confidence of the output, no perturbation added here, no temperature scaling used
				nnOutputs = outputs.data.cpu()
				nnOutputs = nnOutputs.numpy()
				# nnOutputs = nnOutputs[0]		
				nnOutputs = nnOutputs - np.max(nnOutputs,axis=1).reshape(-1,1)								
				nnOutputs = np.array([np.exp(nnOutput)/np.sum(np.exp(nnOutput)) for nnOutput in nnOutputs])				
				# Using temperature scaling
				outputs = outputs / temper			
				# Calculating the perturbation we need to add, that is,
				# the sign of gradient of cross entropy loss w.r.t. input
				maxIndexTemp = np.argmax(nnOutputs, axis=1)			
				labels = Variable(torch.LongTensor(maxIndexTemp).cuda(device))			
				loss = criterion(outputs, labels)
				loss.backward()				
				# Normalizing the gradient to binary in {0, 1}
				gradient =  torch.ge(inputs.grad.data, 0)		
				gradient = (gradient.float() - 0.5) * 2				
				# Normalizing the gradient to the same space of image
				gradient[0][0] = (gradient[0][0] )/(var)				
				for e_ind in range(len(eps)):
					epsilon = eps[e_ind]
					# Adding small perturbations to images
					tempInputs = torch.add(inputs.data,  -epsilon, gradient)
					outputs = self.model(Variable(tempInputs))
					outputs = outputs / temper
					# Calculating the confidence after adding perturbations
					nnOutputs = outputs.data.cpu()
					nnOutputs = nnOutputs.numpy()
					# nnOutputs = nnOutputs[0]
					nnOutputs = nnOutputs - np.max(nnOutputs, axis=1).reshape(-1,1)
					nnOutputs = np.array([np.exp(nnOutput)/np.sum(np.exp(nnOutput)) for nnOutput in nnOutputs])
					PRD[e_ind,t_ind,chunk] = nnOutputs
					HIT[e_ind,t_ind,chunk] = (np.argmax(nnOutputs, axis=1) == _y.numpy()).reshape(-1,1)
				if with_annotations:
					ANN[chunk] = annotations.get_annotations_for_batch(
						fnames,y_ann)		
		if with_annotations:
			return PRD, HIT, ANN
		else:
			return PRD, HIT