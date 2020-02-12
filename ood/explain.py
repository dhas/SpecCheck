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
	

	def softmax_by_feature(self, savename, T, anns, LOG, fontsize=20, figsize=(30,15)):
		fig,axs = plt.subplots(len(T),4,figsize=figsize)
		for t_id in range(len(T)):			
			s    = np.max(LOG[t_id],axis=1)			
			ax = axs[t_id,0]
			x0 = anns[:,self.COL_X0]
			ax.scatter(x0,s)
			ax.set_ylim([0.5,1])
			ax.set_ylabel('T = %s' % T[t_id], fontdict={'fontsize': fontsize})
			if t_id == 0:
				ax.set_title('X0', fontdict={'fontsize': fontsize})

			ax = axs[t_id,1]
			y0 = anns[:,self.COL_Y0]
			ax.scatter(y0,s)
			ax.set_ylim([0.5,1])
			if t_id == 0:
				ax.set_title('Y0', fontdict={'fontsize': fontsize})

			ax = axs[t_id,2]
			sz = anns[:,self.COL_SZ]
			ax.scatter(sz,s)
			ax.set_ylim([0.5,1])
			if t_id == 0:
				ax.set_title('SZ', fontdict={'fontsize': fontsize})
			
			ax = axs[t_id,3]
			br = anns[:,self.COL_BR]
			ax.scatter(br,s)
			ax.set_ylim([0.5,1])
			if t_id == 0:
				ax.set_title('BR', fontdict={'fontsize': fontsize})
		fig.savefig(savename)

	def softmax_shap_by_feature(self,anns,s):
		ann_table = {'x':anns[:,self.COL_X0], 'y':anns[:,self.COL_Y0],
			's':anns[:,self.COL_SZ],'b':anns[:,self.COL_BR]}
		X = pd.DataFrame(ann_table)
		dtree = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=list(s)), 100)
		shaps = shap.TreeExplainer(dtree).shap_values(X)
		return shaps


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

		LOG = np.empty((len(T),
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
				l_t = []; hits_t = []
				chunk = slice(batch_id*batch_size,(batch_id+1)*batch_size)
				for t_id in range(len(T)):
					ot 		= o/T[t_id]
					y_h 	= ot.argmax(dim=1, keepdim=True)				
					l, _ 	= torch.topk(F.softmax(ot,dim=1),2)					
					hits 	= y_h.eq(y.view_as(y_h)).cpu().numpy()
					LOG[t_id,chunk] = l.cpu().numpy()
					HIT[t_id,chunk] = hits

				if with_annotations:
					ANN[chunk] = other_utils.get_annotations_for_batch(
						fnames,y_ann,test_loader.dataset.class_to_idx)
					FNM[chunk] = np.array(fnames).reshape(-1,1)

		if with_annotations:			
			return HIT, LOG, ANN, FNM
		else:
			return HIT, LOG


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
				

