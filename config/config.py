import yaml
import os

DRAW_YML = os.path.join(os.getcwd(),'config/draw.yml')
ENCO_YML = os.path.join(os.getcwd(),'config/encoders.yml')

def get_draw_config():	
	draw_cfg = yaml.load(open(DRAW_YML), Loader=yaml.FullLoader)
	if draw_cfg['img_side'] == 28:
		draw_cfg['qd']['min_sz'] = 4
	return draw_cfg

def get_encoders_config():
	return yaml.load(open(ENCO_YML), Loader=yaml.FullLoader)