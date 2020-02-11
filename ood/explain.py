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
import pandas as pd

class DatasetExplainer:
	def __init__(self,model,device,checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		self.model 		= model		

	def get_ood_npys(self,fnames,hits):
		fnames_np 	= np.array(list(fnames)) 	
		fnames_hits = fnames_np[np.where(hits == False)[0]]
		return fnames_hits

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
	

	def shap_explain(self,anns,preds,class_to_idx,savedir):
		for c in class_to_idx:
			c_lab = class_to_idx[c]
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