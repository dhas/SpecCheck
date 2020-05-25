import yaml
import numpy as np
from pathlib import Path
import argparse
from draw.annotations import compare_annotation_distributions
from prepare_test import overlap_metric_example, prepare_observed_set, prepare_coverage_set, prepare_encoders, explain_with_encoder_set, explain_set_summary
from ood.explainV2 import get_feature_KL, record_distances

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)


def run_test(tests_root, sources, test_config):
	test_cfg = load_config(test_config)
	test_root = tests_root/test_cfg['test_root']
	test_root.mkdir(exist_ok=True)

	out_dir = test_root/'results'
	out_dir.mkdir(exist_ok=True)

	os_cfg = test_cfg['observed_set']
	prepare_observed_set(os_cfg, test_cfg['dim'], 
		test_cfg['num_classes'],
		test_root, sources)
	cs_cfg = test_cfg['coverage_set']
	prepare_coverage_set(cs_cfg, test_cfg['dim'],
		test_cfg['num_classes'],
		test_root)
	
	overlap_metric_example(out_dir/'1_overlap_metric.pdf')

	os_ann = np.load(test_root/os_cfg['root']/'labels.npy')
	cs_ann = np.load(test_root/cs_cfg['root']/'labels.npy')
	wdist, odist = compare_annotation_distributions(os_ann, cs_ann, [r'$P_S$', r'$P_T$'],
	cs_cfg['draw_limits'], test_cfg['dim'], out_dir/'2_os_cs_ann.pdf')
	record_distances('P_T', wdist, odist, 'w', out_dir/'dist.txt')

	os_root = test_root/os_cfg['root']
	cs_root = test_root/cs_cfg['root']
	enc_cfg = test_cfg['encoders']
	num_iters = 1#test_cfg['num_iters']
	for enc_set in ['set01']:#enc_cfg:
		set_cfg = enc_cfg[enc_set]
		for iteration in range(num_iters):
			print('Processing %s' % enc_set)		
			prepare_encoders(set_cfg, test_cfg['dim'], 
				test_cfg['num_classes'], 
				os_root, test_root, iteration)
			explain_with_encoder_set(set_cfg, cs_root,
				np.load(os_root/'labels.npy'),
				test_cfg['dim'],
				cs_cfg['draw_limits'],
				test_root, out_dir, iteration)		
		explain_set_summary(set_cfg, test_root, 
			out_dir, num_iters)


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