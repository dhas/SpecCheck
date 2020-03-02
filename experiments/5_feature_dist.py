import numpy as np
from pathlib import Path
import yaml
import json
import sys
sys.path.append('..')
from draw import annotations

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)

def read_settings(settings):
	if settings.exists():
		with open(settings,'r') as f:
			settings = json.loads(f.read())
			return settings
	else:
		return {}

out_dir = Path('_5_feature_dist')
out_dir.mkdir(exist_ok=True)
tests_root = Path('../_tests')
test_cfg = load_config('../config/test_draw_128.yml')
test_root = tests_root/test_cfg['test_root']
os_root   = test_root/test_cfg['observed_set']['root']
os_ann	  = np.load(os_root/'labels.npy')
cs_ann_lim = test_cfg['coverage_set']['draw_limits']
for clabel in range(annotations.num_classes):
	cANN = os_ann[os_ann[:,annotations.ANN_COL_CL] == clabel,1:]
	bins_range = [(0,test_cfg['dim']),
				  (0,test_cfg['dim']),
				  (cs_ann_lim['sz_lo'],cs_ann_lim['sz_hi']),
				  (cs_ann_lim['br_lo'],cs_ann_lim['br_hi']),
				 ]
	p, bins = np.histogramdd(cANN) #, range=bins_range, bins=20)
	p = p/p.sum()
