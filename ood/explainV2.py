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
from sklearn.metrics import auc

def get_feature_KL(os_ann, cs_ann, eps=0.00001):
	num_classes = len(annotations.classes)
	num_feats = len(annotations.FEATS)
	fkl = np.empty((num_classes, num_feats), 
		dtype=np.float)
	bins = 50
	for clabel in range(num_classes):
		os = os_ann[os_ann[:,annotations.ANN_COL_CL] == clabel, 1:]
		cs = cs_ann[cs_ann[:,annotations.ANN_COL_CL] == clabel, 1:]
		for f_ind in range(num_feats):
			p, bins = np.histogram(os[:,f_ind], bins=50)
			p = (p/np.sum(p)) + eps
			q, bins = np.histogram(cs[:,f_ind], bins=50)
			q = (q/np.sum(q)) + eps
			fkl[clabel, f_ind] = np.sum(np.where(p != 0, p * np.log(p / q), 0))
	return fkl



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
		

	def axes_softmax_distribution(self, PRD, HIT, T, axs, title, ANN=None, fontsize=20):
		for t_ind in range(len(T)):
			ax = axs[t_ind]
			if not ANN is None:
				softmax_bins = np.linspace(0.0,1.0,50)
				for clabel in range(annotations.num_classes):
					cind = ANN[:,annotations.ANN_COL_CL] == clabel
					prd  = PRD[t_ind][cind,clabel]
					hist, bins = np.histogram(prd, bins=50)
					freq = hist/np.sum(hist)
					ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins),
						alpha=0.5,
						label='%s' % annotations.classes[clabel])
					ax.set_xlim([0.0,1.2])
					ax.set_ylim([0.0,1.0])
					# ax.hist(prd,softmax_bins, alpha=0.5, label='%s' % annotations.classes[clabel])
				ax.legend(fontsize=fontsize-4, loc='upper left')
				ax.set_title('T-%0.2f' % T[t_ind], fontsize=fontsize)
			else:
				softmax_bins = np.linspace(0.5,1.0,50)
				SFM = np.max(PRD[t_ind],axis=1)
				ACC = np.count_nonzero(HIT[t_ind] == True)/(HIT[t_ind].size)
				ax.hist(SFM,softmax_bins, label=('accuracy-%0.3f' % ACC), alpha = 0.8, color='blue')
				ax.legend(fontsize=fontsize-4, loc='upper center')
				ax.set_title('T-%0.2f' % T[t_ind], fontsize=fontsize)
				ax.tick_params(axis='both', which='major', labelsize=fontsize-10)
				if t_ind == 0:
					ax.set_ylabel(title, fontsize=fontsize)	

	def plot_softmax_distribution(self, PRD, HIT, eps, T, savename, ANN=None, fontsize=20, figsize=(30,15)):
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

	
	def _confusion_matrix(self, scores, id_ind, od_ind, threshold):
		cmat = []
		for t in threshold:
			tpr = np.count_nonzero(scores[id_ind] > t)/len(id_ind)
			fnr = np.count_nonzero(scores[id_ind] < t)/len(id_ind)
			tnr = np.count_nonzero(scores[od_ind] < t)/len(od_ind)
			fpr = np.count_nonzero(scores[od_ind] > t)/len(od_ind)
			cmat.append([tpr, fnr, fpr, tnr])
		return np.stack(cmat)


	def roc_explain(self, PRD, idod, T, savename, fontsize=20, figsize=(30,15)):
		id_ind = np.where(idod == annotations.id_label)[0]
		od_ind = np.where(idod == annotations.od_label)[0]
		fig, axs = plt.subplots(figsize=figsize)
		mSFM  = np.max(PRD,axis=2)		
		threshold = np.linspace(0.5,1, num=20)
		
		for t_ind in range(len(T)):
			cmat =  self._confusion_matrix(mSFM[t_ind], id_ind, od_ind, threshold)			
			axs.plot(cmat[:,2], cmat[:,0],
				label='T-%0.3f, AUROC-%0.3f' % (T[t_ind],
					auc(cmat[:,2], cmat[:,0])))
		axs.legend(fontsize=fontsize-4, loc='upper center')		
		fig.savefig(savename)

		

	def plot_inout_score_distributions(self, PRD, idod, T, savename, fontsize=20, figsize=(30,15)):
		id_ind = np.where(idod == annotations.id_label)[0]
		od_ind = np.where(idod == annotations.od_label)[0]
		fig, axs = plt.subplots(2, len(T), figsize=figsize)		
		mSFM  = np.max(PRD,axis=2)
		for t_ind in range(len(T)):
			ax = axs[0, t_ind]
			ax.hist(mSFM[t_ind, id_ind], color='blue')
			ax.set_xlim([0.5,1])

			ax = axs[1, t_ind]
			ax.hist(mSFM[t_ind, od_ind], color='red')
			ax.set_xlim([0.5,1])
		fig.savefig(savename)

	
	def _feature_scatter(self, SCR, ANN, eps, T, savename, salience, fontsize, figsize):
		summary = {}
		fig, axs = plt.subplots(len(T), len(annotations.FEATS), figsize=figsize)
		for t_ind in np.arange(len(T)):
			summary[t_ind] = {}
			for f_ind in range(len(annotations.FEATS)):
				F  = ANN[:, f_ind]
				S  = SCR[t_ind][:,f_ind]
				ax = axs[t_ind, f_ind]
				uF = np.unique(F)
				if scores_x_axis:
					ax.scatter(S,F)
				else:
					ax.scatter(F, S)					
					if len(uF) < 12:
						x_ticks = uF
					else:
						x_ticks = uF[::len(uF)//12]
					ax.set_xticks(x_ticks)

				if t_ind == 0:
					ax.set_title(annotations.FEATS[f_ind], fontsize=fontsize)
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

	def plot_softmax_by_feature(self, PRD, ANN, label, eps, T, savename, class_pred=False, fontsize=20, figsize=(30,15)):
		if PRD.shape[0] > 1:
			raise Exception('Not supported for more than one eps value')

		label_ind = np.where(ANN[:,annotations.ANN_COL_CL] == label)[0]
		ANN = ANN[label_ind,1:] #excluding class label and thickness		
		PRD = PRD[:,:,label_ind,:]		
		SFM   = np.expand_dims(np.max(PRD,axis=3),3)
		SFM   = np.repeat(SFM,len(annotations.FEATS),axis=3)		
		SFM   = SFM[0] #only supports one eps
		return self._feature_scatter(SFM, ANN, eps, T, savename, 'mean', fontsize, figsize)

	def summary_by_feature(self, summary, T, cal_T_ind, net_name, axs, label, set_title=False, fontsize=20, figsize=(30,15)):
		for f_ind in range(len(annotations.FEATS)):
			ax = axs[f_ind]
			ax.plot(summary[cal_T_ind][f_ind]['f'],
				summary[cal_T_ind][f_ind]['s'],
				label='%s' % net_name)
			if f_ind == 0:
				ax.set_ylabel(label, fontsize=fontsize)
			if set_title:
				ax.set_title(annotations.FEATS[f_ind], fontsize=fontsize)



	def plot_shap_by_feature(self, PRD, ANN, label, eps, T, savename, 
		class_pred=False, softmax_scale=1, fontsize=20, figsize=(30,15)):
		if PRD.shape[0] > 1:
			raise Exception('Not supported for more than one eps value')

		label_ind = np.where(ANN[:,annotations.ANN_COL_CL] == label)[0]
		ANN = ANN[label_ind,1:] #excluding class label and thickness
		PRD = PRD[:,:,label_ind,:]
		#only supports one eps
		PRD = PRD[0]
		SHAP = np.empty((len(T),
			ANN.shape[0],
			len(annotations.FEATS)),dtype=np.float)
		
		X = pd.DataFrame(ANN)
		X.columns = annotations.FEATS
		for t_ind in range(len(T)):
			if class_pred:
				SFM = PRD[t_ind,:,label]
			else:
				SFM = np.max(PRD[t_ind],axis=1)*softmax_scale
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