from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

out_dir     = Path('_7_ood_ann/')
out_dir.mkdir(exist_ok=True)
figsize = (30,10)
fontsize = 20

net_name = 'VGG07'
qd_ann = np.load(out_dir/'qd_labels.npy') #estimated annotations in QD
net_npz = out_dir/('%s.npz' % net_name)
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

