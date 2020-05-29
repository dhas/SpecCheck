from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import ood.explainV2 as explain

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)



test_config = '../config/test_draw_128.yml'
test_cfg = load_config(test_config)
test_root = Path('../_tests')/test_cfg['test_root']
enc_cfg = test_cfg['encoders']
num_iters = test_cfg['num_iters']
cfg = enc_cfg['set01']

figsize = (12, 6)
fontsize = 24

explain_root  = Path('_17_set_summary/')
explain_root.mkdir(exist_ok=True)
nets = cfg['nets']
figROC, axROC = plt.subplots(figsize=figsize)
taxROC = axROC.twinx()
figDist, axDist = plt.subplots(figsize=figsize)

# fig2ROC, ax2ROC = plt.subplots(2, 1, figsize=figsize)
ticks = np.arange(len(nets))
width = 0.1
delta = -0.1
c = 0.65
l1 = ['VGGBN AUROC', 'VGGDO AUROC', 'VGG AUROC']
l2 = ['VGGBN ACC.    ', 'VGGDO ACC.    ', 'VGG ACC.']
c = ['k', 'r', 'b']
for seti, setn in enumerate(['set02', 'set03', 'set01']):
	cfg = enc_cfg[setn]
	test_acc = []
	aurocs   = []
	wdist    = []
	odist    = []
	for iteration in range(num_iters):
			encoders_root = test_root/cfg['root']/('%d' % iteration)
			summary_npz = encoders_root/'summary.npz'
			summary = np.load(summary_npz)
			test_acc.append(summary['test_accs'])
			aurocs.append(summary['aurocs'])
			wdist.append(summary['wdist_mu'])
			odist.append(summary['odist_mu'])
	
	test_acc = np.array(test_acc).mean(axis=0)
	aurocs   = np.array(aurocs).mean(axis=0)
	# explain.roc_summary(aurocs, test_acc, ax2ROC[seti], fontsize, width=0.1)
	color = c[seti] #('%0.2f' % c)
	axROC.bar(ticks + delta, aurocs, width, color=color,  label=l1[seti])
	taxROC.plot(ticks, test_acc, color=color, marker='^', label=l2[seti])
	delta += 0.1
	# c -= 0.3


tick_labels = list(nets.keys())
axROC.set_ylim([0.5, 1.2])
axROC.set_yticks([0.5, 0.7, 1.0])
axROC.set_xticks(range(len(tick_labels)))
axROC.set_xticklabels(tick_labels)
axROC.set_ylabel('AUROC', fontsize=fontsize)
axROC.set_xlabel('Classifier configuration', labelpad=15, fontsize=fontsize)
axROC.tick_params(axis='both', which='major', labelsize=fontsize-2)
taxROC.set_ylabel('Test acc.', fontsize=fontsize)
taxROC.tick_params(axis='both', which='major', labelsize=fontsize-2)
taxROC.set_ylim([0.85, 1.05])
taxROC.set_yticks([0.9, 1.0])

lines1, labels1 = axROC.get_legend_handles_labels()
lines2, labels2 = taxROC.get_legend_handles_labels()

axROC.legend(fontsize=fontsize-6, loc='upper left', ncol=3, frameon=False)
taxROC.legend(bbox_to_anchor=(0, 0.9), fontsize=fontsize-6, loc='upper left', ncol=3, frameon=False)
# axROC.legend(lines1 + lines2, labels1 + labels2, fontsize=fontsize-6, loc='upper left', frameon=False, ncol=3)

figROC.savefig(explain_root/'1_stcal_3roc.pdf', bbox_inches='tight')

# wdist = np.array(wdist).mean(axis=0)
# odist = np.array(odist).mean(axis=0)
# explain.distance_summary(wdist, odist, aurocs, test_acc, tick_labels, axDist)
# figDist.savefig(explain_root/'2_stcal_dist.pdf', bbox_inches='tight')