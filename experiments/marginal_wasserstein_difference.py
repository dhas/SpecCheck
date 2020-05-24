# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:02:00 2020

@author: johaant
"""

import numpy as np
from pathlib import Path
from marginals_estimation import get_id_dicts

classes    = ['circle', 'square']
FEATS  = ['XMIN', 'YMIN', 'XMAX', 'YMAX', 'BR']
out_dir  = Path('_13_marginals_estimation/')
os_ann 	= np.load(out_dir/'labels.npy') 

os_dict, dict_uncal, dict_xcal, dict_mcal = get_id_dicts()
from scipy.stats import wasserstein_distance

for cname in classes:
    print(20*'==')
    print('The following comparison is for {}'.format(cname))
    for feat in FEATS:
        os_feat = os_dict[cname][feat]
        uncal_feat = dict_uncal[cname][feat]
        xcal_feat = dict_xcal[cname][feat]
        mcal_feat = dict_mcal[cname][feat]
        
        
        
        dist_uncal = wasserstein_distance(os_feat, uncal_feat)
        dist_xcal = wasserstein_distance(os_feat, xcal_feat)
        dist_mcal = wasserstein_distance(os_feat, mcal_feat)
        print(10*'--')
        print('The distances between the marginals for feature {}'.format(feat))
        print('dist(qd, uncal) = {}, dist(qd, xcal) = {}, dist(qd, mcal) = {}'.format(dist_uncal, dist_xcal, dist_mcal))
        
        
