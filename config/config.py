import yaml
import os

DRAW_YML = os.path.join(os.getcwd(),'config/draw.yml')

def get_draw_config():	
	draw_cfg = yaml.load(open(DRAW_YML), Loader=yaml.FullLoader)
	if draw_cfg['img_side'] == 28:
		draw_cfg['qd']['min_sz'] = 4
	return draw_cfg