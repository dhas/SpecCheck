from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc #roc_auc_score

out_dir     = Path('_7_ood_ann/')
out_dir.mkdir(exist_ok=True)
figsize = (30,10)
fontsize = 20

FEATS = ['CL', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'BR']


def confusion_matrix(scores, id_ind, od_ind, threshold):
	cmat = []
	for t in threshold:
		tpr = np.count_nonzero(scores[id_ind] > t)/len(id_ind)
		fnr = np.count_nonzero(scores[id_ind] < t)/len(id_ind)
		tnr = np.count_nonzero(scores[od_ind] < t)/len(od_ind)
		fpr = np.count_nonzero(scores[od_ind] > t)/len(od_ind)
		cmat.append([tpr, fnr, fpr, tnr])
	return np.stack(cmat)

def threshold_score(scores, id_ind, od_ind, threshold):
	od_rate = np.count_nonzero(scores[id_ind] < threshold)/len(id_ind)
	id_rate = np.count_nonzero(scores[od_ind] > threshold)/len(od_ind)
	return od_rate, id_rate

def label_sd_ann(qd_ann, sd_ann):
	qd_ann = qd_ann[qd_ann[:,1] != 0]
	qd_ann_min = []
	qd_ann_max = []
	for f_ind in range(len(FEATS)):
		if f_ind == 0:
			continue
		print('%4s: %3d < QD < %3d, %3d < SD < %3d' % (FEATS[f_ind], 
					qd_ann[:,f_ind].min(),
					qd_ann[:,f_ind].max(),
					sd_ann[:,f_ind].min(),
					sd_ann[:,f_ind].max()))
		qd_ann_min.append(qd_ann[:,f_ind].min())
		qd_ann_max.append(qd_ann[:,f_ind].max())
	idod = []
	print(sd_ann)
	for ann in sd_ann:
		ann = ann[1:]
		indis = ((qd_ann_min[0] < ann[0]) and (qd_ann_max[0] > ann[0])) and \
				((qd_ann_min[1] < ann[1]) and (qd_ann_max[1] > ann[1])) and \
				((qd_ann_min[2] < ann[2]) and (qd_ann_max[2] > ann[2])) and \
				((qd_ann_min[3] < ann[3]) and (qd_ann_max[3] > ann[3])) and \
				((qd_ann_min[4] < ann[4]) and (qd_ann_max[4] > ann[4]))
		if indis:
			idod.append(0)
		else:
			idod.append(1)
		

	return qd_ann_min, qd_ann_max, np.array(idod)


net_name = 'VGG11'
qd_ann = np.load(out_dir/'qd_labels.npy') #estimated annotations in QD
net_npz = out_dir/('%s.npz' % net_name)
print(qd_ann)
if net_npz.exists():
	state = np.load(net_npz)
	T     = state['T']
	cPRD  = state['cPRD'] #class predictions at different T
	cHIT  = state['cHIT'] #prediction accuracy per sample in SD
	sd_ann   = state['ANN'] #annotations per sample in SD
else:
	raise Exception('%s does not exist' % net_npz)

#uncertainty measure - max softmax score
sPRD = np.max(cPRD, axis=3)
fig, axs = plt.subplots(1, len(T), figsize=figsize)
for t_ind in range(len(T)):
	ax = axs[t_ind]
	hist, bins = np.histogram(sPRD[0,t_ind], bins=15)
	# freq = hist/np.sum(hist)
	ax.bar(bins[:-1], hist, align="edge", width=np.diff(bins))	
	ax.set_title('T=%0.3f' % T[t_ind], fontsize=fontsize)
	if t_ind == 0:
		ax.set_ylabel(net_name, fontsize=fontsize)

fig.savefig(out_dir/'1_max_sfmax.png')

qd_ann_min, qd_ann_max, idod = label_sd_ann(qd_ann, sd_ann)
id_ind = np.where(idod == 0)[0]
od_ind = np.where(idod == 1)[0]

idod_e = np.full(len(idod), 1.0)
idod_e[od_ind] = 0.5

fig, axs = plt.subplots(2, len(T), figsize=figsize)
for t_ind in range(len(T)):	
	ax     = axs[0,t_ind]
	mse_id = np.square(1.0 - sPRD[0,t_ind,id_ind]).mean()
	ax.hist(sPRD[0,t_ind,id_ind], color='blue')
	ax.set_title('MSE-%0.3f' % mse_id)
	mse = np.square(idod_e - sPRD[0,t_ind]).mean()
	# ax.set_title('MSE-%0.3f' % mse)
	
	ax     = axs[1,t_ind]
	mse_od = np.square(sPRD[0,t_ind,id_ind] - 0.5).mean()
	ax.hist(sPRD[0,t_ind,od_ind], color='red')
	ax.set_title('MSE-%0.3f' % mse_od)
	print('T: %0.3f, MSE-id: %0.3f, MSE-od: %0.3f, MSE: %0.3f' % (T[t_ind],mse_id, mse_od, mse))

fig.savefig(out_dir/'2_max_sfmax_idod.png')

# fig, axs = plt.subplots(1, len(T), figsize=figsize)
# for t_ind in range(len(T)):	
# 	ax     = axs[t_ind]
# 	fpr, tpr, thresholds = metrics.roc_curve(idod, sPRD[0,t_ind], pos_label=1)
# 	auroc = roc_auc_score(idod, sPRD[0,t_ind])
# 	ax.plot(fpr, tpr, color='blue')
# 	ax.set_title('AUROC-%0.3f' % auroc)
	
# fig.savefig(out_dir/'3_max_sfmax_roc.png')

# id_rate, od_rate = threshold_score(sPRD[0,2], id_ind, od_ind, 0.6)
threshold = np.linspace(0.5,1, num=20)

fig, axs = plt.subplots(1, len(T), figsize=figsize)
for t_ind in range(len(T)):	
	ax     = axs[t_ind]
	cmat =  confusion_matrix(sPRD[0,t_ind], id_ind, od_ind, threshold)
	ax.plot(cmat[:,2], cmat[:,0], color='blue')
	ax.set_title('AUROC-%0.3f' % auc(cmat[:,2], cmat[:,0]))
	# ax.plot(threshold, cmat[:,0], color='blue')
	# ax.plot(threshold, cmat[:,2], color='red')
	# ax.set_title('AUROC-%0.3f' % auroc)
	
fig.savefig(out_dir/'3_max_sfmax_roc.png')
