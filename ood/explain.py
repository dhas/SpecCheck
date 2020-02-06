import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import sys
sys.path.append('..')
from utils import other_utils
import torch.nn.functional as F
import pandas as pd
import xgboost
from xgboost import plot_tree
import matplotlib.pyplot as plt
import shap

class DatasetExplainer:
	def __init__(self,model,device,checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		self.model 		= model		

	def get_batch_ids(self,fnames,hits):
		fnames_np 	= np.array(list(fnames)) 
		hits_np 	= hits.cpu().numpy()
		fnames_hits = fnames_np[np.where(hits_np == False)[0]]
		return fnames_hits

	def evaluate(self,device,criterion,test_loader):
		self.model.to(device)
		self.model.eval()
		loss = 0.0
		correct = 0
		nbatches = 0
		length = 0
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
				data, target = data.to(device), target.to(device)
				output = self.model(data)
				loss += criterion(output, target).item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()
				length += len(data)
				nbatches += 1
			
			loss /= nbatches
			accuracy = 100. * correct / length
		return loss,accuracy

	def explain(self,device,criterion,test_loader):
		self.model.to(device)
		self.model.eval()
		loss = 0.0
		correct = 0
		nbatches = 0
		length = 0
		ood_npys = np.array([])
		with torch.no_grad():
			for batch_idx, (data, target, fnames) in enumerate(tqdm(test_loader)):
				data, target = data.to(device), target.to(device)
				output = self.model(data)
				loss += criterion(output, target).item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability				
				hits = pred.eq(target.view_as(pred))				
				ood_npys = np.concatenate([ood_npys,self.get_batch_ids(fnames,hits)])				
				correct += hits.sum().item()
				length += len(data)
				nbatches += 1				
			
			loss /= nbatches
			accuracy = 100. * correct / length
		return loss,accuracy,ood_npys

	def shap_explain(self,device,criterion,test_loader,y_ann,savedir):
		self.model.to(device)
		self.model.eval()
		loss = 0.0
		correct = 0
		nbatches = 0
		length = 0
		anns  = []
		preds = []
		with torch.no_grad():
			for batch_idx, (data, target, fnames) in enumerate(tqdm(test_loader)):
				data, target = data.to(device), target.to(device)
				output = self.model(data)
				loss += criterion(output, target).item()
				pred, _ = torch.topk(F.softmax(output),2)
				pred = pred.cpu().numpy()
				ann = other_utils.get_annotations_for_batch(fnames,y_ann,test_loader.dataset.class_to_idx)
								
				anns.append(ann)
				preds.append(pred)
				
		anns  = np.vstack(anns)
		preds = np.vstack(preds)

		for c in test_loader.dataset.class_to_idx:
			c_lab = test_loader.dataset.class_to_idx[c]
			c_ind = np.where(anns[:,0] == c_lab)			
			x_c   = anns[c_ind][:,1:]
			y_c   = preds[c_ind][:,c_lab]
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