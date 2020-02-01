import numpy as np 
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


def plot_samples(imgs,savename,size=(20,10),fontsize=25,labels=None):
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
						fontdict={'fontsize': fontsize})
			ax.axis('off')
	if show_xy:
		fig.text(0.06, 0.5, 'x', va='center', rotation='vertical', fontsize=fontsize)
		fig.text(0.5, 0.05, 'y', ha='center',fontsize=fontsize)
	fig.savefig(savename)
	plt.close()