import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import sys
sys.path.append('..')
from utils import other_utils
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import xgboost
from xgboost import plot_tree
import matplotlib.pyplot as plt
import shap
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

class DatasetExplainer:
	def __init__(self,model,checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		self.model 		= model
		self.COL_CL = 0
		self.COL_X0 = 1
		self.COL_Y0 = 2
		self.COL_SZ = 3
		self.COL_BR = 4

	def get_ood_npys(self,fnames,hits):
		fnames_np 	= np.array(list(fnames)) 	
		fnames_hits = fnames_np[np.where(hits == False)[0]]
		return list(fnames_hits)

	def evaluate(self,device,criterion,test_loader,y_ann):
		self.model.to(device)
		self.model.eval()
		loss = 0.0
		correct = 0
		nbatches = 0
		length = 0
		anns  = []
		preds = []
		hits  = []
		ood_npys = []
		with torch.no_grad():
			for batch_idx, (data, target, fnames) in enumerate(tqdm(test_loader)):
				anns.append(other_utils.get_annotations_for_batch(fnames,y_ann,test_loader.dataset.class_to_idx))
				data, target = data.to(device), target.to(device)
				output = self.model(data)
				loss += criterion(output, target).item()  # sum up batch loss				
				pred_l = output.argmax(dim=1, keepdim=True)
				hit = pred_l.eq(target.view_as(pred_l)).cpu().numpy()
				hits.append(hit)
				pred_s, _ = torch.topk(F.softmax(output,dim=1),2)
				pred_s = pred_s.cpu().numpy()
				preds.append(pred_s)
				ood_npys += list(self.get_ood_npys(fnames,hit))
				correct += hit.sum().item()
				length += len(data)
				nbatches += 1
			
			loss /= nbatches
			accuracy = 100. * correct / length

			anns = np.vstack(anns)
			hits = np.vstack(hits)
			preds = np.vstack(preds)			
						

		return loss,accuracy, anns, preds, ood_npys
		
	def plot_softmax_distribution(self, PRD, savename, fontsize=20, figsize=(30,15)):
		SFM = np.max(PRD,axis=1)
		softmax_bins = np.linspace(0.5,1.0,50)
		
		fig, axs = plt.subplots(1,1,figsize=figsize)		
		axs.hist(SFM,softmax_bins, alpha = 0.8, color='blue')
		fig.savefig(savename)	

	
	def _extract_with_label(self,PRD, ANN, Y, label):
		label_ind = np.where(ANN[:,self.COL_CL] == label)[0]
		ANN = ANN[label_ind]
		Y   = Y[label_ind]
		if len(PRD.shape) > 3:
			PRD = PRD[:,:,label_ind,:]
		else:
			PRD = PRD[:,label_ind,:]

		return PRD, ANN, Y

	def _extract_salience(self, F, scores, method='max'):
			uF = np.unique(F)
			salience = []
			for f in uF:
				score = scores[F == f]
				if method == 'max':
					s = score.max()
				elif method == 'mean':
					s = score.mean()

				salience.append(s)
			return uF, salience


	def softmax_spread_explain(self, PRD, Y, T, eps, savename, label, ANN, fontsize=20, figsize=(30,15)):
		def _extract_salience(F, scores):
			uF = np.unique(F)
			salience = []
			for f in uF:
				score = scores[F == f]
				salience.append(score.max())
			return uF, salience

		pos_label = 1
		neg_label = 0
				
		PRD, ANN, Y = self._extract_with_label(PRD, ANN, Y, label)
		
		fig,axs = plt.subplots(len(eps), 4, figsize=figsize)
		
		X0  = ANN[:,self.COL_X0]
		Y0  = ANN[:,self.COL_X0]
		SZ  = ANN[:,self.COL_SZ]
		BR  = ANN[:,self.COL_BR]
		
		for e_ind in range(len(eps)):
			aX0 = axs[e_ind,0]
			aX0.set_ylabel('eps-%0.2f' % eps[e_ind], fontsize=fontsize)
			aY0 = axs[e_ind,1]
			aSZ	= axs[e_ind,2]
			aBR = axs[e_ind,3]

			if e_ind == 0:
				aX0.set_title('X0', fontsize=fontsize)
				aY0.set_title('Y0', fontsize=fontsize)
				aSZ.set_title('SZ', fontsize=fontsize)
				aBR.set_title('BR', fontsize=fontsize)
			
			for t_ind in range(len(T)):
				
				SFM = np.max(PRD[e_ind][t_ind],axis=1)
				
				uX0, sX0 = _extract_salience(X0,SFM)
				aX0.plot(uX0,sX0,label=('%2.2f' % T[t_ind]))
				

				uY0, sY0 = _extract_salience(Y0,SFM)
				aY0.plot(uY0,sY0,label=('%2.2f' % T[t_ind]))
				

				uSZ, sSZ = _extract_salience(SZ,SFM)
				aSZ.plot(uSZ,sSZ,label=('%2.2f' % T[t_ind]))
				

				uBR, sBR = _extract_salience(BR,SFM)
				aBR.plot(uBR,sBR,label=('%2.2f' % T[t_ind]))

			aX0.set_xticks(uX0[::2])
			aY0.set_xticks(uY0[::2])
			aSZ.set_xticks(uSZ)
			aBR.set_xticks(uBR[::16])

		handles, labels = aX0.get_legend_handles_labels()
		fig.legend(handles, labels, fontsize=fontsize, loc='upper right')
		fig.savefig(savename)
			


	def shap_spread_explain(self,ANN, PRD, Y, eps, T, label, savename, eps0_savename= None, ood_scatter=False, fontsize=20, figsize=(30,15)):		

		pos_label = 1
		neg_label = 0

		PRD, ANN, Y = self._extract_with_label(PRD, ANN, Y, label)
		
		fig,axs = plt.subplots(len(eps), 4, figsize=figsize)
		X0  = ANN[:,self.COL_X0]
		Y0  = ANN[:,self.COL_X0]
		SZ  = ANN[:,self.COL_SZ]
		BR  = ANN[:,self.COL_BR]

		ann_table = {'x0':ANN[:,self.COL_X0], 'y0':ANN[:,self.COL_Y0],
			'sz':ANN[:,self.COL_SZ],'br':ANN[:,self.COL_BR]}
		X = pd.DataFrame(ann_table)		
		shap_lim = [-0.08,0.08]

		for e_ind in range(len(eps)):			
			aX0 = axs[e_ind,0]
			aX0.set_ylabel('eps-%0.2f' % eps[e_ind], fontsize=fontsize)
			aY0 = axs[e_ind,1]
			aSZ	= axs[e_ind,2]
			aBR = axs[e_ind,3]
			aX0.set_ylim(shap_lim)
			aY0.set_ylim(shap_lim)
			aSZ.set_ylim(shap_lim)
			aBR.set_ylim(shap_lim)

			if e_ind == 0:
				aX0.set_title('X0', fontsize=fontsize)
				aY0.set_title('Y0', fontsize=fontsize)
				aSZ.set_title('SZ', fontsize=fontsize)
				aBR.set_title('BR', fontsize=fontsize)
				fig0,axs0 = plt.subplots(len(T), 4, figsize=figsize)			
			
			for t_ind in range(len(T)):
				y = list(np.max(PRD[e_ind][t_ind],axis=1))
				dtree = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
				SHAP = shap.TreeExplainer(dtree).shap_values(X)

				uX0, sX0 = self._extract_salience(X0,SHAP[:,self.COL_X0-1],method='mean')
				aX0.plot(uX0,sX0,label=('%2.2f' % T[t_ind]))
				

				uY0, sY0 = self._extract_salience(Y0,SHAP[:,self.COL_Y0-1],method='mean')
				aY0.plot(uY0,sY0,label=('%2.2f' % T[t_ind]))
				

				uSZ, sSZ = self._extract_salience(SZ,SHAP[:,self.COL_SZ-1],method='mean')
				aSZ.plot(uSZ,sSZ,label=('%2.2f' % T[t_ind]))
				

				uBR, sBR = self._extract_salience(BR,SHAP[:,self.COL_BR-1],method='mean')
				aBR.plot(uBR,sBR,label=('%2.2f' % T[t_ind]))
				if (not eps0_savename is None) and (e_ind == 0):					
					ax0 = axs0[t_ind, 0]
					ay0 = axs0[t_ind, 1]
					asz = axs0[t_ind, 2]
					abr = axs0[t_ind, 3]

					if t_ind == 0:
						ax0.set_title('X0', fontsize=fontsize)						
						ay0.set_title('Y0', fontsize=fontsize)
						asz.set_title('SZ', fontsize=fontsize)
						abr.set_title('BR', fontsize=fontsize)					
					ax0.set_ylabel('T-%0.2f' % T[t_ind], fontsize=fontsize)
					ax0.set_ylim(shap_lim)
					ay0.set_ylim(shap_lim)
					asz.set_ylim(shap_lim)
					abr.set_ylim(shap_lim)

					ax0.scatter(X0,SHAP[:,self.COL_X0-1])					
					ay0.scatter(Y0,SHAP[:,self.COL_Y0-1])
					asz.scatter(SZ,SHAP[:,self.COL_SZ-1])
					abr.scatter(BR,SHAP[:,self.COL_BR-1])


			aX0.set_xticks(uX0[::2])
			aY0.set_xticks(uY0[::2])
			aSZ.set_xticks(uSZ)
			aBR.set_xticks(uBR[::16])

		if (not eps0_savename is None):
			handles, labels = ax0.get_legend_handles_labels()
			fig0.legend(handles, labels, fontsize=fontsize, loc='upper right')
			fig0.savefig(eps0_savename)	

		handles, labels = aX0.get_legend_handles_labels()
		fig.legend(handles, labels, fontsize=fontsize, loc='upper right')
		fig.savefig(savename)
		
		# for t_ind in range(len(T)):
		# 	y = list(np.max(PRD[t_ind],axis=1))
		# 	dtree = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
		# 	shaps = shap.TreeExplainer(dtree).shap_values(X)
		# 	print(shaps.shape)
		# 	ax = axs[t_ind, 0]		
		# 	X0 	= ANN[:,self.COL_X0]
		# 	sX0 = shaps[:,self.COL_X0-1]
		# 	ax.scatter(X0,sX0)
		# 	if ood_scatter:
		# 		axs.scatter(X0[Y == neg_label], sX0[Y == neg_label], color='red')

		# 	ax 	= axs[t_ind,1]
		# 	Y0 	= ANN[:,self.COL_Y0]
		# 	sY0 = shaps[:,self.COL_Y0-1]
		# 	ax.scatter(Y0,sY0)
		# 	if ood_scatter:
		# 		axs.scatter(Y0[Y == neg_label], sY0[Y == neg_label], color='red')

		# 	ax 	= axs[t_ind,2]
		# 	SZ 	= ANN[:,self.COL_SZ]
		# 	sSZ = shaps[:,self.COL_SZ-1]
		# 	ax.scatter(SZ,sSZ)
		# 	if ood_scatter:
		# 		axs.scatter(SZ[Y == neg_label], sSZ[Y == neg_label], color='red')

		# 	ax 	= axs[t_ind,3]
		# 	BR 	= ANN[:,self.COL_BR]
		# 	sBR = shaps[:,self.COL_BR-1]
		# 	ax.scatter(BR,sBR)
		# 	if ood_scatter:
		# 		axs.scatter(BR[Y == neg_label], sBR[Y == neg_label], color='red')
		
		# fig.savefig(savename)


	
	def softmax_explain(self, PRD, Y, T, savename, label=None, ANN=None, fontsize=20, figsize=(30,15), ood_scatter=False):		
		
		pos_label = 1
		neg_label = 0
		
		if not label is None:
			if ANN is None:
				raise Exception('Annotations needed to explain by class label')
			PRD, ANN, Y = self._extract_with_label(PRD, ANN, Y, label)			
			c = 6
		else:
			c = 2

		fig,axs = plt.subplots(len(T), c, figsize=figsize)
		softmax_bins = np.linspace(0.5,1.0,50)
		
		for t_ind in range(len(T)):
			#Distribution of softmax scores
			ax = axs[t_ind,0]
			SFM = np.max(PRD[t_ind],axis=1)			
			ax.hist(SFM,softmax_bins, alpha = 0.8, color='blue') # all scores
			# ax.hist(SFM[Y == neg_label], softmax_bins, alpha = 0.8, color='red') #ood scores
			ax.set_ylim([0,len(SFM)])
			ax.set_ylabel('T = %0.2f' % T[t_ind], fontdict={'fontsize': fontsize})
			if t_ind == 0:
				ax.set_title('SOFTMAX', fontdict={'fontsize': fontsize})

			#ROC curve
			fpr, tpr, thresholds = roc_curve(Y, SFM, pos_label=pos_label)
			auroc = roc_auc_score(Y, SFM)
			ax = axs[t_ind,1]
			ax.plot(fpr, tpr, label=('auroc-%0.3f' % auroc))
			ax.legend(fontsize=fontsize-4, loc='lower right')
			if t_ind == 0:
				ax.set_title('ROC', fontdict={'fontsize': fontsize})

			if not label is None:
				ax = axs[t_ind,2]
				X0 = ANN[:,self.COL_X0]
				ax.scatter(X0, SFM)
				if ood_scatter:
					ax.scatter(X0[Y == neg_label], SFM[Y==neg_label], alpha=0.5, color='red')
				ax.set_xticks(np.unique(X0)[::2])
				ax.set_ylim([0.5,1])
				if t_ind == 0:
					ax.set_title('X0', fontdict={'fontsize': fontsize})

				ax = axs[t_ind,3]
				Y0 = ANN[:,self.COL_Y0]
				ax.scatter(Y0, SFM)
				if ood_scatter:
					ax.scatter(Y0[Y == neg_label], SFM[Y==neg_label], alpha=0.5, color='red')
				ax.set_xticks(np.unique(Y0)[::2])
				ax.set_ylim([0.5,1])
				if t_ind == 0:
					ax.set_title('Y0', fontdict={'fontsize': fontsize})

				ax = axs[t_ind,4]
				SZ = ANN[:,self.COL_SZ]
				ax.scatter(SZ, SFM)
				if ood_scatter:
					ax.scatter(SZ[Y == neg_label], SFM[Y==neg_label], alpha=0.5, color='red')
				ax.set_xticks(np.unique(SZ)[::2])
				ax.set_ylim([0.5,1])
				if t_ind == 0:
					ax.set_title('SZ', fontdict={'fontsize': fontsize})


				ax = axs[t_ind,5]
				BR = ANN[:,self.COL_BR]
				ax.scatter(BR, SFM)
				if ood_scatter:
					ax.scatter(BR[Y == neg_label], SFM[Y==neg_label], alpha=0.5, color='red')
				ax.set_xticks(np.unique(BR)[::10])
				ax.set_ylim([0.5,1])
				if t_ind == 0:
					ax.set_title('BR', fontdict={'fontsize': fontsize})

		fig.savefig(savename)

	def softmax_shap_by_feature(self,ANN, PRD, Y, T, label, savename, ood_scatter=False, fontsize=20, figsize=(30,15)):
		pos_label = 1
		neg_label = 0

		label_ind 	= np.where(ANN[:,self.COL_CL] == label)[0]
		ANN 		= ANN[label_ind]
		Y 			= Y[label_ind]
		PRD 		= PRD[:,label_ind,:]

		ann_table = {'x0':ANN[:,self.COL_X0], 'y0':ANN[:,self.COL_Y0],
			'sz':ANN[:,self.COL_SZ],'br':ANN[:,self.COL_BR]}
		X = pd.DataFrame(ann_table)		
		
		fig,axs = plt.subplots(len(T), 4, figsize=figsize)
		softmax_bins = np.linspace(0.5,1.0,50)
		print(ANN.shape)
		for t_ind in range(len(T)):
			y = list(np.max(PRD[t_ind],axis=1))
			dtree = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
			shaps = shap.TreeExplainer(dtree).shap_values(X)
			print(shaps.shape)
			ax = axs[t_ind, 0]		
			X0 	= ANN[:,self.COL_X0]
			sX0 = shaps[:,self.COL_X0-1]
			ax.scatter(X0,sX0)
			if ood_scatter:
				axs.scatter(X0[Y == neg_label], sX0[Y == neg_label], color='red')

			ax 	= axs[t_ind,1]
			Y0 	= ANN[:,self.COL_Y0]
			sY0 = shaps[:,self.COL_Y0-1]
			ax.scatter(Y0,sY0)
			if ood_scatter:
				axs.scatter(Y0[Y == neg_label], sY0[Y == neg_label], color='red')

			ax 	= axs[t_ind,2]
			SZ 	= ANN[:,self.COL_SZ]
			sSZ = shaps[:,self.COL_SZ-1]
			ax.scatter(SZ,sSZ)
			if ood_scatter:
				axs.scatter(SZ[Y == neg_label], sSZ[Y == neg_label], color='red')

			ax 	= axs[t_ind,3]
			BR 	= ANN[:,self.COL_BR]
			sBR = shaps[:,self.COL_BR-1]
			ax.scatter(BR,sBR)
			if ood_scatter:
				axs.scatter(BR[Y == neg_label], sBR[Y == neg_label], color='red')
		
		fig.savefig(savename)


	def shap_explain(self,anns,preds,class_to_idx,savedir):
		for c in class_to_idx:
			c_lab = class_to_idx[c]
			c_ind = np.where(anns[:,0] == c_lab)			
			x_c   = anns[c_ind][:,1:]
			y_c   = preds[c_ind]
			feat_table = {'x':x_c[:,0], 'y':x_c[:,1],
			's':x_c[:,2],'b':x_c[:,3]}
			X = pd.DataFrame(feat_table)
			dtree = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=list(y_c)), 100)
			fig = plt.figure()
			plot_tree(dtree,rankdir='LR')
			fig.set_size_inches(40.,40.)
			plt.savefig(savedir/('2_dtree_%s.png' % c), dpi=1000)
			plt.close()
			explainer = shap.TreeExplainer(dtree)
			shap_values = explainer.shap_values(X)			
			shap.summary_plot(shap_values, X ,show=False)
			plt.savefig(savedir/('3_shap_summary_%s.png' % c), dpi=1000)
			plt.close()

	def _evaluate_ts(self,device,criterion,test_loader,T=[1],y_ann=None):
		self.model.to(device)
		self.model.eval()
		loss = 0.0;	correct = 0;nbatches = 0; length = 0
		batch_size = test_loader.batch_size

		PRD = np.empty((len(T),
			len(test_loader.dataset),
			len(test_loader.dataset.classes)),
		dtype=np.float)

		HIT = np.empty((len(T),
			len(test_loader.dataset),
			1),
		dtype=np.bool)

		ANN = np.empty((len(test_loader.dataset),
			6),
		dtype=np.int)

		FNM = np.empty((len(test_loader.dataset),
			1),
		dtype='<U10')
		
		
		if not y_ann is None:
			with_annotations = True
			anns = [];	ood_samples = []
		else:
			with_annotations = False
		
		with torch.no_grad():
			for batch_id, (_x, _y, fnames) in enumerate(tqdm(test_loader)):
				x, y 	= _x.to(device), _y.to(device)
				o 		= self.model(x)
				chunk = slice(batch_id*batch_size,(batch_id+1)*batch_size)
				for t_id in range(len(T)):
					lt 		= o/T[t_id]
					y_h 	= lt.argmax(dim=1, keepdim=True)				
					p, _ 	= torch.topk(F.softmax(lt,dim=1),2)					
					hits 	= y_h.eq(y.view_as(y_h)).cpu().numpy()
					PRD[t_id,chunk] = p.cpu().numpy()
					HIT[t_id,chunk] = hits

				if with_annotations:
					ANN[chunk] = other_utils.get_annotations_for_batch(
						fnames,y_ann,test_loader.dataset.class_to_idx)
					FNM[chunk] = np.array(fnames).reshape(-1,1)

		if with_annotations:			
			return HIT, PRD, ANN, FNM
		else:
			return HIT, PRD

	def _evaluate_odin_batch(self,device,criterion,test_loader,T=[1],eps=[0.1],var=1,y_ann=None):
		self.model.to(device)
		self.model.eval()
		
		batch_size = test_loader.batch_size

		PRD = np.empty((len(eps),
			len(T),
			len(test_loader.dataset),
			len(test_loader.dataset.classes)),
		dtype=np.float)

		HIT = np.empty((len(eps),
			len(T),
			len(test_loader.dataset),
			1),
		dtype=np.bool)

		if not y_ann is None:
			ANN = np.empty((len(test_loader.dataset),
			6),
			dtype=np.int)

			FNM = np.empty((len(test_loader.dataset),
				1),
			dtype='<U10')
			with_annotations = True
		else:
			with_annotations = False

		for batch_id, (_x, _y, fnames) in enumerate(tqdm(test_loader)):
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
					ANN[chunk] = other_utils.get_annotations_for_batch(
						fnames,y_ann,test_loader.dataset.class_to_idx)
					FNM[chunk] = np.array(fnames).reshape(-1,1)
		
		if with_annotations:
			return HIT, PRD, ANN, FNM
		else:
			return HIT, PRD

	def _evaluate_odin(self,device,criterion,test_loader,T=[1],eps=0.0014,var=1,y_ann=None):
		self.model.to(device)
		self.model.eval()
		
		
		PRD = np.empty((len(T),
			len(test_loader.dataset),
			len(test_loader.dataset.classes)),
		dtype=np.float)

		HIT = np.empty((len(T),
			len(test_loader.dataset),
			1),
		dtype=np.bool)

		if not y_ann is None:
			ANN = np.empty((len(test_loader.dataset),
			6),
			dtype=np.int)

			FNM = np.empty((len(test_loader.dataset),
				1),
			dtype='<U10')
			with_annotations = True
		else:
			with_annotations = False

		for batch_id, (_x, _y, fnames) in enumerate(tqdm(test_loader)):
			inputs = Variable(_x.cuda(device), requires_grad = True)			
			outputs = self.model(inputs)		
			
			for t_ind in range(len(T)):
				temper = T[t_ind]
				# Calculating the confidence of the output, no perturbation added here, no temperature scaling used
				nnOutputs = outputs.data.cpu()
				nnOutputs = nnOutputs.numpy()
				nnOutputs = nnOutputs[0]		
				nnOutputs = nnOutputs - np.max(nnOutputs)		
				nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
				

				# Using temperature scaling
				outputs = outputs / temper
			
				# Calculating the perturbation we need to add, that is,
				# the sign of gradient of cross entropy loss w.r.t. input
				maxIndexTemp = np.argmax(nnOutputs)
				labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(device))			
				loss = criterion(outputs, labels)
				loss.backward()
				
				# Normalizing the gradient to binary in {0, 1}
				gradient =  torch.ge(inputs.grad.data, 0)		
				gradient = (gradient.float() - 0.5) * 2
				
				# Normalizing the gradient to the same space of image
				gradient[0][0] = (gradient[0][0] )/(var)
				
				# Adding small perturbations to images
				tempInputs = torch.add(inputs.data,  -eps, gradient)
				outputs = self.model(Variable(tempInputs))
				outputs = outputs / temper
				# Calculating the confidence after adding perturbations
				nnOutputs = outputs.data.cpu()
				nnOutputs = nnOutputs.numpy()
				nnOutputs = nnOutputs[0]
				nnOutputs = nnOutputs - np.max(nnOutputs)
				nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
				PRD[t_ind,batch_id] = nnOutputs
				HIT[t_ind,batch_id] = (np.argmax(nnOutputs) == _y.numpy())
				if with_annotations:
					ANN[batch_id] = other_utils.get_annotations_for_batch(
						fnames,y_ann,test_loader.dataset.class_to_idx)[0]
					FNM[batch_id] = np.array(fnames)[0]
		
		if with_annotations:
			return HIT, PRD, ANN, FNM
		else:
			return HIT, PRD
			

	
	def _evaluate(self,device,criterion,test_loader,y_ann=None):
		self.model.to(device)
		self.model.eval()
		loss = 0.0;	correct = 0;nbatches = 0;length = 0
		L  = []; H  = []
		if not y_ann is None:
			with_annotations = True
			anns = [];	ood_samples = []
		else:
			with_annotations = False
		
		with torch.no_grad():
			for batch_idx, (_x, _y, fnames) in enumerate(tqdm(test_loader)):
				x, y 	= _x.to(device), _y.to(device)
				o 		= self.model(x)				
				y_h 	= o.argmax(dim=1, keepdim=True)				
				l, _ 	= torch.topk(F.softmax(o,dim=1),2)
				l 		= l.cpu().numpy()
				hits 	= y_h.eq(y.view_as(y_h)).cpu().numpy()
				H.append(hits)
				L.append(l)
				if with_annotations:
					anns.append(other_utils.get_annotations_for_batch(
						fnames,y_ann,test_loader.dataset.class_to_idx))
					ood_samples += self.get_ood_npys(fnames,hits)
		
		H = np.vstack(H)
		L = np.vstack(L)
		accuracy = (hits == True).sum()/len(hits)
		if with_annotations:
			anns 		= np.vstack(anns)
			ood_samples = np.array(ood_samples)
			return accuracy, H, L, anns, ood_samples
		else:
			return accuracy, H, L

	def odin_test(self, net1, criterion, CUDA_DEVICE, testloader10, testloader, id_var, od_var, noiseMagnitude1, temper, score_dir):
		net1.to(CUDA_DEVICE)
		net1.eval()
		# t0 = time.time()
		f1 = open(score_dir/'confidence_Base_In.txt', 'w')
		f2 = open(score_dir/'confidence_Base_Out.txt', 'w')
		g1 = open(score_dir/'confidence_ODIN_In.txt', 'w')
		g2 = open(score_dir/'confidence_ODIN_Out.txt', 'w')
		
		softmax_id 	= []
		softmax_od 	= []
		odin_id 	= []
		odin_od 	= []
		print("Processing in-distribution images")
	########################################In-distribution###########################################
		for j, data in enumerate(tqdm(testloader10)):
			# if j<1000: continue
			images, _, _ = data
			
			inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
			outputs = net1(inputs)		
			# Calculating the confidence of the output, no perturbation added here, no temperature scaling used
			nnOutputs = outputs.data.cpu()
			nnOutputs = nnOutputs.numpy()
			nnOutputs = nnOutputs[0]		
			nnOutputs = nnOutputs - np.max(nnOutputs)		
			nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
			softmax_id.append(np.max(nnOutputs))
			# softmax_id.append(torch.topk(F.softmax(outputs,dim=1),1))
			f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))

			# Using temperature scaling
			outputs = outputs / temper
		
			# Calculating the perturbation we need to add, that is,
			# the sign of gradient of cross entropy loss w.r.t. input
			maxIndexTemp = np.argmax(nnOutputs)
			labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
			loss = criterion(outputs, labels)
			loss.backward()
			
			# Normalizing the gradient to binary in {0, 1}
			gradient =  torch.ge(inputs.grad.data, 0)		
			gradient = (gradient.float() - 0.5) * 2
			
			# Normalizing the gradient to the same space of image
			gradient[0][0] = (gradient[0][0] )/(id_var)
			# # gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
			# # gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
			# # gradient[0][2] = (gradient[0][2])/(66.7/255.0)
			# Adding small perturbations to images
			tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
			outputs = net1(Variable(tempInputs))
			outputs = outputs / temper
			# Calculating the confidence after adding perturbations
			nnOutputs = outputs.data.cpu()
			nnOutputs = nnOutputs.numpy()
			nnOutputs = nnOutputs[0]
			nnOutputs = nnOutputs - np.max(nnOutputs)
			nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
			odin_id.append(np.max(nnOutputs))
			g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
			# if j % 100 == 99:
			# 	print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
			# 	t0 = time.time()
			
			# if j == N - 1: break

		# t0 = time.time()
		print("Processing out-of-distribution images")
	###################################Out-of-Distributions#####################################
		for j, data in enumerate(tqdm(testloader)):
			# if j<1000: continue
			images, _, _ = data
			
			inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
			outputs = net1(inputs)
			


			# Calculating the confidence of the output, no perturbation added here
			nnOutputs = outputs.data.cpu()
			nnOutputs = nnOutputs.numpy()
			nnOutputs = nnOutputs[0]
			nnOutputs = nnOutputs - np.max(nnOutputs)
			nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
			softmax_od.append(np.max(nnOutputs))
			# softmax_od.append(torch.topk(F.softmax(outputs,dim=1),1))
			f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
			
			# Using temperature scaling
			outputs = outputs / temper
	  
	  
			# Calculating the perturbation we need to add, that is,
			# the sign of gradient of cross entropy loss w.r.t. input
			maxIndexTemp = np.argmax(nnOutputs)
			labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
			loss = criterion(outputs, labels)
			loss.backward()
			
			# Normalizing the gradient to binary in {0, 1}
			gradient =  (torch.ge(inputs.grad.data, 0))
			# gradient = gradient.float()
			gradient = (gradient.float() - 0.5) * 2
			# Normalizing the gradient to the same space of image
			gradient[0][0] = (gradient[0][0] )/(od_var)
			# gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
			# gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
			# gradient[0][2] = (gradient[0][2])/(66.7/255.0)
			# Adding small perturbations to images
			tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
			outputs = net1(Variable(tempInputs))
			outputs = outputs / temper
			# Calculating the confidence after adding perturbations
			nnOutputs = outputs.data.cpu()
			nnOutputs = nnOutputs.numpy()
			nnOutputs = nnOutputs[0]
			nnOutputs = nnOutputs - np.max(nnOutputs)
			nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
			odin_od.append(np.max(nnOutputs))
			g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
			# if j % 100 == 99:
			# 	print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j+1-1000, N-1000, time.time()-t0))
			# 	t0 = time.time()

			# if j== N-1: break
		return np.stack(softmax_id),np.stack(softmax_od),np.stack(odin_id),np.stack(odin_od)



				

