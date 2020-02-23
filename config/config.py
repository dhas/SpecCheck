import yaml
import os

def get_draw_config(project_root):	
	DRAW_YML = os.path.join(project_root,'config/draw.yml')
	draw_cfg = yaml.load(open(DRAW_YML), Loader=yaml.FullLoader)	
	return draw_cfg

def get_encoders_config(ENCO_YML):
	return yaml.load(open(ENCO_YML), Loader=yaml.FullLoader)