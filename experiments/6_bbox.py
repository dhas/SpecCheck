from pathlib import Path
import yaml

def load_config(config_path):
	return yaml.load(open(config_path), Loader=yaml.FullLoader)

main_cfg 	= load_config('../config/config.yml')
tests_root 	= Path(main_cfg['tests_root'])
sources    	= tests_root/main_cfg['sources']
test_cfg 	= load_config('../config/test_draw_128.yml')


