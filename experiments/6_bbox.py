from pathlib import Path
import yaml
import sys
sys.path.append('..')
from draw import annotations
from draw.quick_draw import QuickDraw
from draw.spec_draw import SpecDraw
import numpy as np
import matplotlib.pyplot as plt

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)

out_dir     = Path('_6_bbox/')
out_dir.mkdir(exist_ok=True)
main_cfg 	= load_config('../config/config.yml')
tests_root 	= out_dir/Path(main_cfg['tests_root'])
tests_root.mkdir(exist_ok=True)
sources    	= tests_root/main_cfg['sources']
sources.mkdir(exist_ok=True)
test_cfg 	= load_config('../config/test_draw_28.yml')
test_root   = tests_root/test_cfg['test_root']
test_root.mkdir(exist_ok=True)
os_cfg 		= test_cfg['observed_set']
qd_root = test_root/os_cfg['root']
qd_root.mkdir(exist_ok=True)

cs_cfg  = test_cfg['coverage_set']
sd_root = test_root/cs_cfg['root']
sd_root.mkdir(exist_ok=True)



source_paths 	= [sources/src for src in os_cfg['sources']]
draw_lims 		=  cs_cfg['draw_limits']
min_sz     		= draw_lims['sz_lo']
imgs_per_class  = 100
qd_shapes = QuickDraw(qd_root, test_cfg['dim'],
			source_paths, imgs_per_class,
			min_sz)
qd_ann = np.load(qd_shapes.est_labels)
annotations.plot_annotation_distribution(qd_ann,
	draw_lims, test_cfg['dim'],
	out_dir/'1_qd_ann_dist.png')

sd_shapes = SpecDraw(sd_root,
		test_cfg['dim'],
		imgs_per_class,
		draw_lims)
sd_ann = np.load(sd_shapes.labels)
annotations.plot_annotation_distribution(sd_ann,
	draw_lims, test_cfg['dim'],
	out_dir/'2_sd_ann_dist.png')

# plt.close()
clabel = 0
test_img = np.random.randint(0,100)
x = np.load(sd_root/('%s/%d.npy' % (annotations.classes[clabel], test_img)))
bbox = sd_ann[sd_ann[:,0] == clabel][test_img]
print(bbox)
x_min,y_min = bbox[1], bbox[2]
x_max,y_max = bbox[3], bbox[4]
# nz  = np.transpose(np.nonzero(x))
# y_min = np.min(nz[:,0])
# y_max = np.max(nz[:,0])
# x_min = np.min(nz[:,1])
# x_max = np.max(nz[:,1])
# br = int(np.mean(x[x>0]))
# bbox = [x_min,y_min,x_max,y_max]
plt.imshow(x.reshape(test_cfg['dim'],test_cfg['dim']),cmap='Greys_r')
plt.plot([x_min,x_min],[y_min,y_max],color='red')
plt.plot([x_min,x_max],[y_min,y_min],color='red')
plt.plot([x_max,x_max],[y_min,y_max],color='red')
plt.plot([x_min,x_max],[y_max,y_max],color='red')
plt.show()

