import numpy as np 
import pickle
import matplotlib.pyplot as plt

def sample_from_batch(x_batch,y_batch,num_samples):
	assert(x_batch.shape[0] == y_batch.shape[0])
	p = np.random.permutation(x_batch.shape[0])
	x_batch = x_batch[p]
	y_batch = y_batch[p]
	return x_batch[:num_samples],y_batch[:num_samples]

def as_rows_and_cols(imgs,r,c):
	assert(r*c == imgs.shape[0])
	nchannels = imgs.shape[-1]
	side = imgs.shape[-2]
	return imgs.reshape(r,c,side,side,nchannels)


def plot_samples(imgs,savename,size=(20,10),fontsize=25,title=None,labels=None):
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
				uni_labeled = labels.shape[2] == 1
				if uni_labeled:
					ax.set_title('%s' % (labels[i][j][0]))
				else:
					show_xy = True
					ax.set_title('%d,(%d,%d),%d,%d' % (labels[i][j][0],
						labels[i][j][1],
						labels[i][j][2],
						labels[i][j][3],
						labels[i][j][4]),
						fontdict={'fontsize': fontsize//2})
			ax.axis('off')
	if show_xy:
		fig.text(0.06, 0.5, 'x', va='center', rotation='vertical', fontsize=fontsize)
		fig.text(0.5, 0.05, 'y', ha='center',fontsize=fontsize)
	if not title is None:
		fig.suptitle(title)
	fig.savefig(savename)
	plt.close()

def plot_label_spread(savename,labels,d,misses_dict=None,step=1):
	COL_X0 = 0
	COL_Y0 = 1
	COL_SZ = 2
	COL_BR = 3	
	class_names	= list(labels.keys())
	class_names.sort()	
	num_classes = len(class_names)
	
	if misses_dict is None:
		fig,axs = plt.subplots(3,num_classes,figsize=(20,15))
	else:
		fig,axs = plt.subplots(4,num_classes,figsize=(20,15))
	
	for c in range(num_classes):		
		y 	= labels[class_names[c]]
		if not misses_dict is None:
			misses = misses_dict[class_names[c]]

		ax_cnt = 0
		#anchor spread
		ax = axs[ax_cnt,c]
		x0 = y[:,COL_X0]
		y0 = y[:,COL_Y0]
		h1 = ax.hist2d(x0,y0,bins=30,cmap='Blues')
		h_rng = np.arange(0,d)
		h_rng[h_rng%5 != 0] = -1
		h_rng[-1] = d-1
		h_rng = h_rng[0:len(h_rng):step]
		h_lbl = np.array([str(x) if x != -1 else '' for x in h_rng])
		ax.set_xticks(h_rng)
		ax.set_xticklabels(h_lbl)
		ax.set_yticks(h_rng)
		ax.set_yticklabels(h_lbl)
		ax.invert_yaxis()
		fig.colorbar(h1[3],ax=ax)
		if not misses_dict is None:
			ax_cnt += 1
			ax = axs[ax_cnt,c]
			x0_miss = y[misses,COL_X0]
			y0_miss = y[misses,COL_Y0]
			h2 = ax.hist2d(x0_miss,y0_miss,bins=30,cmap='Reds')
			ax.set_xticks(h_rng)
			ax.set_xticklabels(h_lbl)
			ax.set_yticks(h_rng)
			ax.set_yticklabels(h_lbl)
			ax.invert_yaxis()
			fig.colorbar(h2[3],ax=ax)
		ax.set_title('Anchor spread (%s)' % class_names[c])
		ax_cnt += 1

		#size spread
		ax = axs[ax_cnt,c]
		sz = y[:,COL_SZ]
		sz_min  = sz.min()
		sz_max  = sz.max()
		sz_rng  = np.arange(sz_min,sz_max+1)
		sz_dis  = np.bincount(sz,minlength=sz_max+1)[sz_rng]
		ax.bar(sz_rng,sz_dis)
		ax.set_xticks(sz_rng[0:len(sz_rng):step])
		if not misses_dict is None:
			sz_miss = y[misses,COL_SZ]
			if sz_miss.size != 0:
				p_miss = np.bincount(sz_miss,minlength=sz_max+1)[sz_rng]
				ax.bar(sz_rng,p_miss,color='red')
		ax.set_title('Size spread (%s)' % class_names[c])
		ax_cnt += 1

		#brighness spread		
		ax = axs[ax_cnt,c]
		br = y[:,COL_BR]
		br_min = br.min()
		br_max = br.max()
		br_rng = np.arange(br_min,br_max+1)
		br_dis = np.bincount(br,minlength=br_max+1)[br_rng]
		ax.bar(br_rng,br_dis)
		ax.set_xticks(br_rng[0:len(br_rng):10*step])
		if not misses_dict is None:
			br_miss = y[misses,COL_BR]
			if br_miss.size != 0:
				p_miss = np.bincount(br_miss,minlength=br_max+1)[br_rng]
				ax.bar(br_rng,p_miss,color='red')
		ax.set_title('Brightness spread (%s)' % class_names[c])
	
	fig.savefig(savename)


def path_to_class_and_idx(path):
	class_name  = path.split('/')[-2]
	sample_id  = int(path.split('/')[-1].split('.')[0])
	return class_name,sample_id

def paths_to_indexes(paths,classes):
	indexes = {}	
	for c in classes:
		indexes[c] = np.array([],dtype=int)
	
	for path in paths:
		class_name,sample_id  = path_to_class_and_idx(path)		
		indexes[class_name] = np.concatenate([indexes[class_name],[sample_id]]) 

	return indexes


def get_annotations_for_batch(sample_npys,labels_pkl,class_to_idx):
	def _path_to_ann(path):
		class_name,sample_id  = path_to_class_and_idx(path)		 
		ann = [class_to_idx[class_name]] + list(y_ann[class_name][sample_id])		
		return ann
	
	y_ann   = pickle.load(open(labels_pkl,'rb'))
	y_batch = np.stack([_path_to_ann(p) for p in sample_npys])
	return y_batch
