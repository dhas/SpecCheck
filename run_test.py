import yaml
import numpy as np
from pathlib import Path
import argparse
from draw.annotations import plot_annotation_distribution
from prepare_test import prepare_observed_set, prepare_coverage_set, prepare_encoders, explain_with_encoder_set
from ood.explainV2 import get_feature_KL

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)


def run_test(tests_root, sources, test_config):
	test_cfg = load_config(test_config)
	test_root = tests_root/test_cfg['test_root']
	test_root.mkdir(exist_ok=True)

	out_dir = test_root/'results'
	out_dir.mkdir(exist_ok=True)

	os_cfg = test_cfg['observed_set']
	# prepare_observed_set(os_cfg, test_cfg['dim'], 
	# 	test_cfg['num_classes'],
	# 	test_root, sources)
	cs_cfg = test_cfg['coverage_set']
	# prepare_coverage_set(cs_cfg, test_cfg['dim'],
	# 	test_cfg['num_classes'],
	# 	test_root)
	
	os_ann = np.load(test_root/os_cfg['root']/'labels.npy')
	cs_ann = np.load(test_root/cs_cfg['root']/'labels.npy')
	# plot_annotation_distribution(os_ann, cs_cfg['draw_limits'], 
	# 	test_cfg['dim'], out_dir/'1_os_distribution.png')
	# plot_annotation_distribution(cs_ann, cs_cfg['draw_limits'], 
	# 	test_cfg['dim'], out_dir/'2_cs_distribution.png')

	fkl = get_feature_KL(os_ann, cs_ann)
	np.save(out_dir/'3_kl_by_feature.npy', fkl)
	enc_cfg = test_cfg['encoders']
	os_root = test_root/os_cfg['root']
	# prepare_encoders(enc_cfg, test_cfg['dim'], 
	# 	test_cfg['num_classes'], 
	# 	os_root, test_root)

	cs_root = test_root/cs_cfg['root']
	explain_with_encoder_set(enc_cfg, cs_root,
		np.load(os_root/'labels.npy'),
		test_cfg['dim'],
		test_root, out_dir)

if __name__ == '__main__':
	main_cfg = load_config('config/config.yml')
	tests_root = Path(main_cfg['tests_root'])
	tests_root.mkdir(exist_ok=True)
	sources    = tests_root/main_cfg['sources']
	sources.mkdir(exist_ok=True)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_config', type=str, required=True)
	args = parser.parse_args()

	if Path(args.test_config).exists():
		run_test(tests_root, sources, args.test_config)
	else:
		raise Exception('Config file %s does not exist' % args.test_config)