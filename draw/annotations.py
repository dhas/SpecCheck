import numpy as np
import matplotlib.pyplot as plt 
import cv2
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

classes    = ['circle', 'square']
num_classes = len(classes)
class_to_label_dict = {
	'circle' : 0,
	'square' : 1
}

id_label = 0
od_label = 1

ANN_SZ      	= 6
ANN_COL_CL  	= 0
ANN_COL_XMIN  	= 1
ANN_COL_YMIN  	= 2
ANN_COL_X0      = 1
ANN_COL_Y0      = 2
ANN_COL_SZ  	= 3
ANN_COL_BR  	= 4
ANN_COL_TH  	= 5
GRAY_MAX		= 255

ANNS   = ['CL', 'XMIN', 'YMIN', 'XMAX', 'YMAX', 'BR']
# FEATS  = ['XMIN', 'YMIN', 'XMAX', 'YMAX', 'BR']
FEATS  = [r'$Y^2$', r'$Y^3$', r'$Y^4$', r'$Y^5$', r'$Y^6$']
	

def plot_samples(imgs,savename, labels=None, size=(16,8), fontsize=16):
	r = imgs.shape[0]
	c = imgs.shape[1]	
	show_xy = False

	fig, axs = plt.subplots(r, c, figsize=size, gridspec_kw = {'hspace': 0.2})
	for i in range(r):
		for j in range(c):
			img = imgs[i,j]
			if img.shape[-1] == 1:
				img = img.reshape(img.shape[-2],img.shape[-2])
			if r > 1:	    		
				ax 	= axs[i,j]
			else:	    
				ax 	= axs[i+j]
			ax.imshow(img,cmap='Greys_r')
			if not (labels is None):				
				annotated = labels.shape[2] == 1
				if annotated:
					ax.set_title('%s' % (labels[i][j][0]), fontsize=fontsize)
					fig_title = '%s' % (class_to_label_dict)
				else:
					show_xy = True
					ax.set_title('%s' % (','.join(['%d' % l for l in labels[i][j]])),
						fontsize=fontsize)
					bbox = labels[i][j]
					x_min,y_min = bbox[1], bbox[2]
					x_max,y_max = bbox[3], bbox[4]
					ax.plot([x_min,x_min],[y_min,y_max],color='red')
					ax.plot([x_min,x_max],[y_min,y_min],color='red')
					ax.plot([x_max,x_max],[y_min,y_max],color='red')
					ax.plot([x_min,x_max],[y_max,y_max],color='red')
					fig_title = '%s, ANN-%s' % (class_to_label_dict, ANNS)
			ax.axis('off')
	# if show_xy:
	# 	fig.text(0.06, 0.5, 'x', va='center', rotation='vertical', fontsize=fontsize)
	# 	fig.text(0.5, 0.05, 'y', ha='center',fontsize=fontsize)
	
	# fig.suptitle(fig_title, fontsize=fontsize-5)	
	fig.savefig(savename, bbox_inches='tight')
	plt.close()

def histogram(f):
	hist, bins = np.histogram(f, bins=15)
	freq = hist/np.sum(hist)
	return freq, bins

def distance_function(true_distribution, estimated_distribution):
	
	QD_points = true_distribution
	SD_points = estimated_distribution
	
	QD_min = np.min(QD_points)
	QD_max = np.max(QD_points)
	SD_min = np.min(SD_points)
	SD_max = np.max(SD_points)
	
	if SD_max < QD_max and SD_min > QD_min:
		dist = SD_max - SD_min
		distance_measure = dist / (QD_max - QD_min)		
	else:
		dist_1 = np.abs(QD_min - SD_min)
		dist_2 = np.abs(SD_max - SD_min)
		dist_3 = np.abs(QD_max - SD_max)
		tot_dist = dist_1 + dist_2 + dist_3
		distance_measure = tot_dist / (QD_max - QD_min)		
	return distance_measure
	

# def distance_function(true_distribution, estimated_distribution):    
#     QD_points = true_distribution
#     SD_points = estimated_distribution
	
#     dist_1 = np.abs(np.min(QD_points) - np.min(SD_points))
#     dist_2 = np.abs(np.max(SD_points) - np.min(SD_points))
#     dist_3 = np.abs(np.max(QD_points) - np.max(SD_points))
#     tot_dist = dist_1 + dist_2 + dist_3
#     distance_measure = tot_dist / (np.max(QD_points) - np.min(QD_points))
#     return distance_measure

def jaccard_index(x, y):
	i = np.intersect1d(x, y)
	u = np.union1d(x, y)

	index = i.ptp()/u.ptp()
	return index

def set_intersect(x, y):
	i = np.array([e for e in y if (e >= x.min() and e <= x.max())])
	return i


def set_union(x, y):
	return np.union1d(x,y)


# def overlap_index(x, y):
# 	if (y.max() <= x.max()) and (y.min() >= x.min()): # y\in x
# 		index = set_intersect(x, y).ptp()/x.ptp()		
# 	else:
# 		index = set_union(x, y).ptp()/x.ptp()

# 	return index



def obtain_support(points):	
	return (points.min(), points.max())

def no_overlap(SD_supp, QD_supp):
	return (SD_supp[1] < QD_supp[0]) or (SD_supp[0] > QD_supp[1])

def left_partial_overlap(SD_supp, QD_supp):
	return (SD_supp[0] < QD_supp[0]) and (SD_supp[1] >= QD_supp[0]) and (SD_supp[1] <= QD_supp[1])

def right_partial_overlap(SD_supp, QD_supp):
	return (SD_supp[0] >= QD_supp[0]) and (SD_supp[0] <= QD_supp[1]) and (SD_supp[1] > QD_supp[1])

def total_overlap(SD_supp, QD_supp):
	return ((SD_supp[0] >= QD_supp[0]) and (SD_supp[1] <= QD_supp[1])) or ((QD_supp[0] >= SD_supp[0]) and (QD_supp[1] <= SD_supp[1]))

def Jaccard_dist(QD_supp, SD_supp):
	
	sgn = 1 # underestimate
	if no_overlap(SD_supp, QD_supp):
		len_set_diff = (QD_supp[1] - QD_supp[0]) + (SD_supp[1] - SD_supp[0])
		len_union = (QD_supp[1] - QD_supp[0]) + (SD_supp[1] - SD_supp[0])
		
	elif left_partial_overlap(SD_supp, QD_supp):
		len_set_diff = (QD_supp[0] - SD_supp[0]) + (QD_supp[1] - SD_supp[1])
		len_union = QD_supp[1] - SD_supp[0]
	elif right_partial_overlap(SD_supp, QD_supp):
		len_set_diff = (SD_supp[0] - QD_supp[0]) + (SD_supp[1] - QD_supp[1])
		len_union = SD_supp[1] - QD_supp[0]
	elif total_overlap(SD_supp, QD_supp):
		if (QD_supp[1] >= SD_supp[1]) and (QD_supp[0] <= SD_supp[0]):
			sgn = -1
		len_set_diff = abs(QD_supp[0] - SD_supp[0]) + abs(QD_supp[1] - SD_supp[1])
		len_union = max(QD_supp[1], SD_supp[1]) - min(QD_supp[0], SD_supp[0])
	
	J_dist = sgn*(len_set_diff / len_union)
	return J_dist
	
def overlap_index(x, y):
	supp_x = obtain_support(x)
	supp_y = obtain_support(y)
	return Jaccard_dist(supp_x, supp_y)
	# return jaccard_index(x,y)




def compare_annotation_distributions(ANN1, ANN2, labels, draw_lims, dim, savename, labelpad=5, fontsize=24, figsize=(12,6)):
	ANN1 = ANN1[~np.all(ANN1[:,1:] == 0, axis=1)]
	ANN2 = ANN2[~np.all(ANN2[:,1:] == 0, axis=1)]

	fig, axs = plt.subplots(len(class_to_label_dict), len(FEATS), figsize=figsize)
	wdist = np.zeros((len(class_to_label_dict), len(FEATS)))
	odist = np.zeros((len(class_to_label_dict), len(FEATS)))
	for cname in class_to_label_dict:
		clabel = class_to_label_dict[cname]
		cANN1 = ANN1[ANN1[:,0] == clabel,1:]
		cANN2 = ANN2[ANN2[:,0] == clabel,1:]
		for f, feat in enumerate(FEATS):
			f2 = cANN2[:, f]
			p2, bins = histogram(f2)
			axs[clabel, f].bar(bins[:-1], p2, align="edge", width=np.diff(bins), alpha=0.5, label=labels[1])

			f1 = cANN1[:, f]
			p1, bins = histogram(f1)
			axs[clabel, f].bar(bins[:-1], p1, align="edge", width=np.diff(bins), alpha=0.5, label=labels[0])
			axs[clabel, f].set_ylim([0,0.7])
			axs[clabel, f].tick_params(axis='both', which='major', labelsize=fontsize-10)
			wdist[clabel, f] = wasserstein_distance(f1, f2)
			odist[clabel, f] = overlap_index(f1, f2)#distance_function(f1, f2)
			if '6' in FEATS[f]: # == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				axs[clabel, f].set_xticks(f_spread[::75])
				# axs[clabel, f].text(120, 0.90, r'$W^%d(P_{T})-%2.2f$' % (f+2, wdist[clabel, f]), fontsize=fontsize)
				# axs[clabel, f].text(120, 0.75, r'$D^%d(P_{T})-%2.2f$' % (f+2, odist[clabel, f]), fontsize=fontsize)
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				axs[clabel, f].set_xticks(f_spread[::64])
				# axs[clabel, f].text(20, 0.90, r'$W^%d(P_{T})-%2.2f$' % (f+2, wdist[clabel, f]), fontsize=fontsize)
				# axs[clabel, f].text(20, 0.75, r'$D^%d(P_{T})-%2.2f$' % (f+2, odist[clabel, f]), fontsize=fontsize)

			if f == 0:
				axs[clabel, f].set_ylabel(r'$Y^1=%d$' % clabel, fontsize=fontsize)
				axs[clabel, f].set_yticks([0.0, 0.3, 0.7])
			else:
				axs[clabel, f].set_yticks([])
			if clabel == 0:
				axs[clabel, f].set_title(FEATS[f], fontsize=fontsize-4)
				axs[clabel, f].set_xticks([])
	axs[clabel, f-1].legend(loc='upper left', fontsize=fontsize-4, frameon=False)
	# handles, labels = axs[clabel, f].get_legend_handles_labels()
	# fig.legend(handles, labels, loc='lower right', fontsize=fontsize)
	fig.savefig(savename, bbox_inches='tight')
	plt.close()
	return wdist, odist

def plot_annotation_distribution(ANN, draw_lims, dim, savename, wdist=None, fontsize=25, figsize=(30,10)):
	ANN = ANN[~np.all(ANN[:,1:] == 0, axis=1)]
	fig, axs = plt.subplots(len(class_to_label_dict), len(FEATS), figsize=figsize)
	for cname in class_to_label_dict:
		clabel = class_to_label_dict[cname]
		cANN = ANN[ANN[:,0] == clabel,1:]		
		for f_ind in range(len(FEATS)):
			f = cANN[:, f_ind]
			ax = axs[clabel,f_ind]
			hist, bins = np.histogram(f, bins=15)
			freq = hist/np.sum(hist)
			if wdist is None:
				ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins))
			else:
				ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins), label='wdist-%0.2f' % wdist[clabel, f_ind])
			ax.set_ylim([0,1])
			# ax.hist(f, bins=30)
			# if FEATS[f_ind] == 'SZ':
			# 	if cname == 'circle':
			# 		f_spread = np.arange(draw_lims['sz_lo'], draw_lims['sz_hi'] + 1)					
			# 		if len(f_spread) > 8:
			# 			ax.set_xticks(f_spread[::len(f_spread)//8])
			# 		else:
			# 			ax.set_xticks(f_spread)
			# 	else:
			# 		f_spread = np.arange(draw_lims['sz_lo'], 2*draw_lims['sz_hi'] + 1)
			# 		ax.set_xticks(f_spread[::len(f_spread)//16])			
			if FEATS[f_ind] == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				ax.set_xticks(f_spread[::len(f_spread)//12])
			else: #coordinates
				f_spread = np.arange(0, dim + 1)
				ax.set_xticks(f_spread[::len(f_spread)//8])
				# if cname == 'circle':
				# 	f_spread = np.arange(0, dim + 1)
				# 	ax.set_xticks(f_spread[::len(f_spread)//8])
				# else:
				# 	f_spread = np.arange(0, (dim - draw_lims['sz_lo']) + 1)
				# 	ax.set_xticks(f_spread[::len(f_spread)//16])

			if f_ind == 0:
				ax.set_ylabel(cname, fontsize=fontsize)
			if clabel == 0:
				ax.set_title(FEATS[f_ind], fontsize=fontsize)
			ax.legend(fontsize=fontsize-4, loc='upper right')
	fig.savefig(savename, bbox_inches='tight')
	plt.close()

def annotations_list2dict(ann):
	d = dict.fromkeys(classes)
	for cname in classes:
		clabel = class_to_label_dict[cname]
		d[cname] = ann[ann[:,0] == clabel, 1:]
	return d

def label_annotation_distribution(os_ann, cs_ann):
	os_ann = os_ann[os_ann[:,1] != 0]
	os_ann_min = []
	os_ann_max = []
	for f_ind in range(len(ANNS)):
		if f_ind == 0:
			continue		
		os_ann_min.append(os_ann[:,f_ind].min())
		os_ann_max.append(os_ann[:,f_ind].max())
	idod = []	
	for ann in cs_ann:
		ann = ann[1:]
		indis = ((os_ann_min[0] < ann[0]) and (os_ann_max[0] > ann[0])) and \
				((os_ann_min[1] < ann[1]) and (os_ann_max[1] > ann[1])) and \
				((os_ann_min[2] < ann[2]) and (os_ann_max[2] > ann[2])) and \
				((os_ann_min[3] < ann[3]) and (os_ann_max[3] > ann[3])) and \
				((os_ann_min[4] < ann[4]) and (os_ann_max[4] > ann[4]))
		if indis:
			idod.append(id_label)
		else:
			idod.append(od_label)		
	return os_ann_min, os_ann_max, np.array(idod)


def get_annotations_for_batch(fnames, ANN):
	def _path_to_ann(path):
		class_name  = path.split('/')[-2]
		sample_id  	= int(path.split('/')[-1].split('.')[0])
		clabel 		= class_to_label_dict[class_name]
		cANN 		= ANN[ANN[:,ANN_COL_CL] == clabel]
		ann 		= cANN[sample_id]
		return ann
		
	y_batch = np.stack([_path_to_ann(f) for f in fnames])
	return y_batch

def annotations_to_sample(ann, side):
	def _circle_bresenham(ym,xm,r,br,side):
		r0 = r
		canvas = np.zeros(shape=(side,side))
		x = -r; y = 0; err = 2-2*r #II. Quadrant
		while True:		
			canvas[xm-x, ym+y] = br # I. Quadrant
			canvas[xm-y, ym-x] = br # II. Quadrant
			canvas[xm+x, ym-y] = br # III. Quadrant
			canvas[xm+y, ym+x] = br # IV. Quadrant
			r = err
			if (r<=y):
				y += 1
				err += y*2+1
			if (r > x) or (err > y): 
				x += 1
				err += x*2+1
			if x>0:
				break
		xmin, ymin = ym - r0, xm - r0
		xmax, ymax = ym + r0, xm + r0
		ann = [class_to_label_dict['circle'], 
		xmin, ymin, xmax, ymax, br]
		return canvas, ann

	def _square_bresenham(x0,y0,r,br,side):
		def _line_bresenham(canvas,x0,y0,x1,y1,br):    
			dx = np.abs(x1-x0)
			sx = 1 if x0<x1 else -1
			dy = -1*np.abs(y1-y0)
			sy = 1 if y0<y1 else -1
			err = dx+dy
			while True:
				canvas[x0,y0] = br
				if (x0==x1 and y0==y1):
					break
				e2 = 2*err
				if (e2 >= dy):
					err=err+dy; x0=x0+sx
				if (e2 <=dx):
					err=err+dx; y0=y0+sy
		canvas = np.zeros(shape=(side,side))
		_line_bresenham(canvas,x0,y0,x0,y0+r,br)
		_line_bresenham(canvas,x0,y0,x0+r,y0,br)
		_line_bresenham(canvas,x0+r,y0,x0+r,y0+r,br)
		_line_bresenham(canvas,x0,y0+r,x0+r,y0+r,br)
		xmin, ymin = x0, y0
		xmax, ymax = x0+r, y0+r
		ann = [class_to_label_dict['square'], 
		xmin, ymin, xmax, ymax, br]
		return canvas, ann


	def _circle_opencv(xm,ym,r,br,t,side):
		canvas = np.zeros(shape=(side,side))	
		img = cv2.circle(canvas,(xm,ym),r,int(br),t)		
		xmin, ymin = xm - r, ym - r
		xmax, ymax = xm + r, ym + r
		ann = [class_to_label_dict['circle'], 
		xmin, ymin, xmax, ymax, br]
		return img, ann

	def _square_opencv(x0,y0,r,br,t,side):
		canvas = np.zeros(shape=(side,side))	
		img    = cv2.rectangle(canvas,(x0,y0),(x0+r,y0+r),int(br),t)
		xmin, ymin = x0, y0
		xmax, ymax = x0+r, y0+r
		ann = [class_to_label_dict['square'], 
		xmin, ymin, xmax, ymax, br]
		return img, ann

	shape = classes[ann[ANN_COL_CL]]
	x 	  = ann[ANN_COL_X0]
	y 	  = ann[ANN_COL_Y0]
	r 	  = ann[ANN_COL_SZ]
	br 	  = ann[ANN_COL_BR]
	t     = ann[ANN_COL_TH]

	
	
	if t == 1:
		switcher = {
			'circle': _circle_bresenham,
			'square': _square_bresenham
		}
		return switcher[shape](x,y,r,br,side)
	else:
		switcher = {
			'circle': _circle_opencv,
			'square': _square_opencv
		}
		return switcher[shape](x,y,r,br,t,side)

def size_of_bbox(bbox):
	return np.sqrt((bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2)


def annotations_from_sample(x, shape, min_sz):
	nz  = np.transpose(np.nonzero(x))
	y_min = np.min(nz[:,0])
	y_max = np.max(nz[:,0])
	x_min = np.min(nz[:,1])
	x_max = np.max(nz[:,1])
	br = int(np.mean(x[x>0]))
	bbox = [class_to_label_dict[shape], x_min, y_min, x_max, y_max, br]
	if size_of_bbox(bbox) < min_sz :
		bbox = [class_to_label_dict[shape], 0, 0, 0, 0, 0]
	return bbox
	# nz  = np.transpose(np.nonzero(x))
	# x_min = np.min(nz[:,0])
	# x_max = np.max(nz[:,0])
	# y_min = np.min(nz[:,1])
	# y_max = np.max(nz[:,1])
	# br = int(np.mean(x[x>0]))
	# x_sp = (x_max - x_min)
	# y_sp = (y_max - y_min)

	# invalid_img = False

	# if shape == 'circle':		
	# 	if x_sp < 2*min_sz or y_sp < 2*min_sz:
	# 		invalid_img = True
	# 	else:
	# 		x0 = x_min + (x_sp//2)
	# 		y0 = y_min + (y_sp//2)
	# 		sz = int(np.mean([x_sp,y_sp])/2)
	# else:
	# 	if x_sp < min_sz or y_sp < min_sz:
	# 		invalid_img = True
	# 	else:
	# 		x0 = x_min
	# 		y0 = y_min
	# 		sz = int(np.mean([x_sp, y_sp]))

	# cl = class_to_label_dict[shape]
	
	# if invalid_img:
	# 	return [cl, 0, 0, 0, 0]
	# else:
	# 	return [cl, x0, y0, sz, br]


# def pseudo_label_annotations(ANN, SHAP):
# 	PIDODS = []
	
		
