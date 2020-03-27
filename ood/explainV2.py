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
from draw import load, annotations
from ood.calibration import ModelWithTemperature
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


def uncertainty_summary(UNC, idod, net_name, id_ax, od_ax, fontsize=20, figsize=(30,15)):
	hist, bins = np.histogram(UNC[idod==annotations.id_label], bins=15)
	freq = hist/np.sum(hist)
	id_ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins),
				color='blue')
	id_ax.set_ylim([0,1])
	id_ax.set_xlim([0.5,1])
	id_ax.set_title(net_name, fontsize=fontsize)

	hist, bins = np.histogram(UNC[idod==annotations.od_label], bins=15)
	freq = hist/np.sum(hist)
	od_ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins),
		color='red')
	od_ax.set_ylim([0,1])
	od_ax.set_xlim([0.5,1])


COL_TPR = 0
COL_FNR = 1
COL_FPR = 2
COL_TNR = 3
def _confusion_matrix(scores, id_ind, od_ind, threshold):
		cmat = []
		for t in threshold:
			tpr = np.count_nonzero(scores[id_ind] > t)/len(id_ind)
			fnr = np.count_nonzero(scores[id_ind] < t)/len(id_ind)
			tnr = np.count_nonzero(scores[od_ind] < t)/len(od_ind)
			fpr = np.count_nonzero(scores[od_ind] > t)/len(od_ind)
			cmat.append([tpr, fnr, fpr, tnr])
		return np.stack(cmat)


def roc_summary(UNC, idod, axs, net_name, fontsize=20, figsize=(30,15)):
		id_ind = np.where(idod == annotations.id_label)[0]
		od_ind = np.where(idod == annotations.od_label)[0]
		threshold = np.linspace(0.5,1, num=20)
		cmat =  _confusion_matrix(UNC, id_ind, od_ind, threshold)
		auroc = auc(cmat[:,2], cmat[:,0])
		axs.plot(cmat[:,2], cmat[:,0],
				label='%s, AUROC-%0.3f' % (net_name,auroc))

def roc_net(PRD, idod, T, roc_savename, auroc_savename, fontsize=20, figsize=(30,15)):
		id_ind = np.where(idod == annotations.id_label)[0]
		od_ind = np.where(idod == annotations.od_label)[0]
		fig, axs = plt.subplots(figsize=figsize)
		mSFM  = np.max(PRD,axis=2)		
		threshold = np.linspace(0.5,1, num=20)
		
		aurocs  = []
		mses_id = []
		mses_od = []

		for i,t in enumerate(T):
			cmat =  _confusion_matrix(mSFM[i], id_ind, od_ind, threshold)
			auroc = auc(cmat[:,2], cmat[:,0])
			axs.plot(cmat[:,2], cmat[:,0],
				label='T-%0.3f, AUROC-%0.3f' % (T[i],auroc))
			aurocs.append(auroc)
			mses_id.append(np.square(1.0 - mSFM[i,id_ind]).mean())
			mses_od.append(np.square(mSFM[i,od_ind] - 0.5).mean())

		axs.legend(fontsize=fontsize-4, loc='lower right')
		fig.savefig(roc_savename)

		fig,a0 = plt.subplots(figsize=figsize)
		a0.plot(T, aurocs, label='auroc', color='blue')
		a1 = a0.twinx()
		a1.plot(T,mses_id, label='mse_id', color='orange')
		a1.plot(T,mses_od, label='mse_od', color='green')

		h1, l1 = a0.get_legend_handles_labels()
		h2, l2 = a1.get_legend_handles_labels()
		handles = h1 + h2
		labels 	= l1 + l2		
		# fig.legend(handles, labels, bbox_to_anchor=[0.9, 0.97], loc='upper right')
		a0.legend(handles, labels,prop={'size': fontsize-4})		
		fig.savefig(auroc_savename)
		return aurocs

def shap_by_feature(UNC, ANN, idod, savename, fontsize=20, figsize=(30,15)):
	SHAPS = []
	ANNS  = []
	IDODS = []
	fig, axs = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=figsize)
	for cname in annotations.classes:
		clabel 		= annotations.class_to_label_dict[cname]
		cind   		= np.where(ANN[:,0] == clabel)[0]
		X   		= pd.DataFrame(ANN[cind, 1:])
		X.columns 	= annotations.FEATS
		y   		= UNC[cind]				
		dtree 		= xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=list(y)), 100)
		SHAP 		= shap.TreeExplainer(dtree).shap_values(X)		
		axs[clabel, 0].set_ylabel(cname, fontsize=fontsize)
		for f, feat in enumerate(annotations.FEATS):
			axs[clabel, f].scatter(X[feat], SHAP[:,f])
			axs[clabel, f].set_title(feat, fontsize=fontsize)
		ANNS.append(ANN[cind, 1:])
		SHAPS.append(SHAP)
		IDODS.append(idod[cind])
	fig.savefig(savename)
	return np.stack(ANNS), np.stack(SHAPS), np.stack(IDODS)

def estimated_id_features_using_shap(SHAPS, ANNS, dim, draw_lims, id_savename, od_savename, fontsize=20, figsize=(30,15)):
	idfig, idaxs = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=figsize)
	odfig, odaxs = plt.subplots(len(annotations.classes), len(annotations.FEATS), figsize=figsize)
	for cname in annotations.classes:
		clabel 	= annotations.class_to_label_dict[cname]
		shap    = SHAPS[clabel]
		ann     = ANNS[clabel]
		for f, feat in enumerate(annotations.FEATS):
			F  = ann[:, f]
			S  = shap[:, f]
			idF  = F[S >= 0]
			hist, bins = np.histogram(idF, bins=15)
			freq = hist/len(idF)
			idaxs[clabel, f].bar(bins[:-1], freq, align="edge", width=np.diff(bins))
			idaxs[clabel, f].set_ylim([0,1])
			print('ID sum - %0.2f' % np.sum(freq))
			# idaxs[clabel, f].hist(idF)
			if feat == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				idaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//12])
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				idaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//8])

			odF  = F[S < 0]
			hist, bins = np.histogram(odF, bins=15)
			freq = hist/len(odF)
			odaxs[clabel, f].bar(bins[:-1], freq, align="edge", width=np.diff(bins))
			odaxs[clabel, f].set_ylim([0,1])
			print('OD sum - %0.2f' % np.sum(freq))
			# odaxs[clabel, f].hist(odF)
			if feat == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				odaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//12])
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				odaxs[clabel, f].set_xticks(f_spread[::len(f_spread)//8])

	idfig.savefig(id_savename)
	odfig.savefig(od_savename)


def shap_summary(SHAP, ANN, net_name, axs, fontsize=20):	
	for cname in annotations.classes:
		clabel 	= annotations.class_to_label_dict[cname]
		cind   	= np.where(ANN[:,0] == clabel)[0]
		cANN    = ANN[cind, 1:]
		shap 	= SHAP[clabel]
		for f, feat in enumerate(annotations.FEATS):
			F  = cANN[:, f]
			S  = shap[:,f]				
			uF = np.unique(F)	
			mS = [S[F == f].mean() for f in uF]				
			axs[clabel, f].plot(uF, mS, label=net_name)
			if clabel == 0:
				axs[clabel, f].set_title(feat, fontsize=fontsize)
			if f == 0:
				axs[clabel, f].set_ylabel(cname, fontsize=fontsize)
	
				
class TrainedModel:
	def __init__(self, model, checkpoint_path, is_calibrated=False):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		self.model 			= model
		self.is_calibrated 	= is_calibrated		
				

	def calibrate_for_highest_auroc(self, idod, cal_ds, batch_size=100):
		idod_e = np.full(len(idod), 1.0)
		idod_e[idod == annotations.od_label] = 0.5
		idod_ds = load.my_subset(cal_ds.dataset, 
			np.arange(0,len(cal_ds.dataset)),
			torch.tensor(idod_e).type(torch.float))
		idod_loader = torch.utils.data.DataLoader(idod_ds, 
			batch_size=batch_size, shuffle=False)
		cal_encoder = ModelWithTemperature(self.model)
		cal_encoder.set_temperature(idod_loader)
		return cal_encoder.temperature.item()

	def calibrate_for_centered_distribution(self, idod, ds_loader, batch_size=100):
		idod_e = np.full(len(idod), 0.75)
		idod_ds = load.my_subset(ds_loader.dataset, 
			np.arange(0,len(ds_loader.dataset)),
			torch.tensor(idod_e).type(torch.float))
		idod_loader = torch.utils.data.DataLoader(idod_ds, 
			batch_size=batch_size, shuffle=False)

		cal_encoder = ModelWithTemperature(self.model)
		cal_encoder.set_mid_cal(idod_loader)
		return cal_encoder.temperature.item()

	# def plot_inout_score_distributions(self, PRD, idod, T, savename, fontsize=20, figsize=(30,15)):
	# 	id_ind = np.where(idod == annotations.id_label)[0]
	# 	od_ind = np.where(idod == annotations.od_label)[0]
	# 	fig, axs = plt.subplots(2, len(T), figsize=figsize)		
	# 	mSFM  = np.max(PRD,axis=2)
	# 	for i, t in enumerate(T):
	# 		ax = axs[0, i]
	# 		ax.hist(mSFM[i, id_ind], color='blue')
	# 		ax.set_xlim([0.5,1])
	# 		ax.set_title('T-%0.2f' % t, fontsize=fontsize)
	# 		mse_id = np.square(1.0 - mSFM[i,id_ind]).mean()
	# 		ax.set_xlabel('MSE-%0.3f' % mse_id, fontsize=fontsize)

	# 		ax = axs[1, i]
	# 		ax.hist(mSFM[i, od_ind], color='red')
	# 		ax.set_xlim([0.5,1])
	# 		mse_od = np.square(mSFM[i,od_ind] - 0.5).mean()
	# 		ax.set_xlabel('MSE-%0.3f' % mse_od, fontsize=fontsize)
	# 	fig.savefig(savename)

	

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
