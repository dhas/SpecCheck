import yaml
import os

def get_draw_config(project_root):	
	DRAW_YML = os.path.join(project_root,'config/draw.yml')
	draw_cfg = yaml.load(open(DRAW_YML), Loader=yaml.FullLoader)
	if draw_cfg['img_side'] == 28:
		draw_cfg['qd']['min_sz'] = 4
	return draw_cfg

def get_encoders_config(project_root):
	ENCO_YML = os.path.join(project_root,'config/encoders.yml')
	return yaml.load(open(ENCO_YML), Loader=yaml.FullLoader)