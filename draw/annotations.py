import numpy as np
import matplotlib.pyplot as plt 
import cv2

classes    = ['circle', 'square']
class_to_label_dict = {
	'circle' : 0,
	'square' : 1
}
ANN_SZ      = 6
ANN_COL_CL  = 0
ANN_COL_X0  = 1
ANN_COL_Y0  = 2
ANN_COL_SZ  = 3
ANN_COL_BR  = 4
ANN_COL_TH  = 5
GRAY_MAX	= 255
FEATS  		= ['X0', 'Y0', 'SZ', 'BR']
	

def plot_samples(imgs,savename, labels=None, size=(20,10), fontsize=25):
	r = imgs.shape[0]
	c = imgs.shape[1]	
	show_xy = False

	fig, axs = plt.subplots(r, c,figsize=size)
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
					ax.set_title('%s' % (labels[i][j][0]))
				else:
					show_xy = True
					ax.set_title('%d,(%d,%d),%d,%d,%d' % (labels[i][j][ANN_COL_CL],
						labels[i][j][ANN_COL_X0],
						labels[i][j][ANN_COL_Y0],
						labels[i][j][ANN_COL_SZ],
						labels[i][j][ANN_COL_BR],
						labels[i][j][ANN_COL_TH]),
						fontdict={'fontsize': fontsize//2})
			ax.axis('off')
	if show_xy:
		fig.text(0.06, 0.5, 'x', va='center', rotation='vertical', fontsize=fontsize)
		fig.text(0.5, 0.05, 'y', ha='center',fontsize=fontsize)
	
	fig.suptitle('%s' % class_to_label_dict)
	fig.savefig(savename)
	plt.close()

def plot_annotation_distribution(ANN, draw_lims, dim, savename, fontsize=20, figsize=(30,10)):
	fig, axs = plt.subplots(len(class_to_label_dict), len(FEATS), figsize=figsize)
	for cname in class_to_label_dict:
		clabel = class_to_label_dict[cname]
		cANN = ANN[ANN[:,0] == clabel,1:]		
		for f_ind in range(len(FEATS)):
			f = cANN[:, f_ind]
			ax = axs[clabel,f_ind]
			hist, bins = np.histogram(f, bins=15)
			freq = hist/np.sum(hist)
			ax.bar(bins[:-1], freq, align="edge", width=np.diff(bins))
			ax.set_ylim([0,1])
			# ax.hist(f, bins=30)
			if FEATS[f_ind] == 'SZ':
				if cname == 'circle':
					f_spread = np.arange(draw_lims['sz_lo'], draw_lims['sz_hi'] + 1)					
					if len(f_spread) > 8:
						ax.set_xticks(f_spread[::len(f_spread)//8])
					else:
						ax.set_xticks(f_spread)
				else:
					f_spread = np.arange(draw_lims['sz_lo'], 2*draw_lims['sz_hi'] + 1)
					ax.set_xticks(f_spread[::len(f_spread)//16])			
			elif FEATS[f_ind] == 'BR':
				f_spread = np.arange(draw_lims['br_lo'], draw_lims['br_hi'] + 1)
				ax.set_xticks(f_spread[::len(f_spread)//12])
			else: #X0 and Y0
				if cname == 'circle':
					f_spread = np.arange(draw_lims['sz_lo'], (dim - draw_lims['sz_lo']) + 1)
					ax.set_xticks(f_spread[::len(f_spread)//8])
				else:
					f_spread = np.arange(0, (dim - draw_lims['sz_lo']) + 1)
					ax.set_xticks(f_spread[::len(f_spread)//16])

			if f_ind == 0:
				ax.set_ylabel(cname, fontsize=fontsize)
			if clabel == 0:
				ax.set_title(FEATS[f_ind], fontsize=fontsize)
	fig.savefig(savename)


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
	def _circle_bresenham(xm,ym,r,br,side):
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
		return canvas

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
		return canvas


	def _circle_opencv(xm,ym,r,br,t,side):
		canvas = np.zeros(shape=(side,side))	
		img = cv2.circle(canvas,(xm,ym),r,int(br),t)
		return img

	def _square_opencv(x0,y0,r,br,t,side):
		canvas = np.zeros(shape=(side,side))	
		img    = cv2.rectangle(canvas,(x0,y0),(x0+r,y0+r),int(br),t)
		return img

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

def annotations_from_sample(x, shape, min_sz):
	nz  = np.transpose(np.nonzero(x))
	x_min = np.min(nz[:,0])
	x_max = np.max(nz[:,0])
	y_min = np.min(nz[:,1])
	y_max = np.max(nz[:,1])
	br = int(np.mean(x[x>0]))
	x_sp = (x_max - x_min)
	y_sp = (y_max - y_min)

	invalid_img = False

	if shape == 'circle':		
		if x_sp < 2*min_sz or y_sp < 2*min_sz:
			invalid_img = True
		else:
			x0 = x_min + (x_sp//2)
			y0 = y_min + (y_sp//2)
			sz = int(np.mean([x_sp,y_sp])/2)
	else:
		if x_sp < min_sz or y_sp < min_sz:
			invalid_img = True
		else:
			x0 = x_min
			y0 = y_min
			sz = int(np.mean([x_sp, y_sp]))

	cl = class_to_label_dict[shape]
	
	if invalid_img:
		return [cl, 0, 0, 0, 0]
	else:
		return [cl, x0, y0, sz, br]
