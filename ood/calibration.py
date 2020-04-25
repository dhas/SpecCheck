import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


class ModelWithTemperature(nn.Module):
	"""
	A thin decorator, which wraps a model with temperature scaling
	model (nn.Module):
		A classification neural network
		NB: Output of the neural network should be the classification logits,
			NOT the softmax (or log softmax)!
	"""
	def __init__(self, model):
		super(ModelWithTemperature, self).__init__()
		self.model = model
		self.temperature = nn.Parameter(torch.ones(1) * 1.0)
		# self.epsilon = nn.Parameter(torch.ones(1)*0.1)

	def forward(self, input):
		logits = self.model(input)
		return self.temperature_scale(logits)

	def temperature_scale(self, logits):
		"""
		Perform temperature scaling on logits
		"""
		# Expand temperature to match the size of logits
		temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
		return logits / temperature

	def scaled_label_scores(self, logits, labels):
		max_pred = self.scaled_score(logits)
		pos_scores = max_pred[torch.where(labels == 1.0)]
		pos_labels = labels[torch.where(labels == 1.0)]
		neg_scores = max_pred[torch.where(labels == 0.5)]
		neg_labels = labels[torch.where(labels == 0.5)]
		return pos_scores, pos_labels, neg_scores, neg_labels

	def scaled_score(self,logits):
		logits = self.temperature_scale(logits)
		max_pred, _ = torch.max(F.softmax(logits,1),1)
		return max_pred

	# This function probably should live outside of this class, but whatever
	def set_temperature(self, valid_loader):
		self.cuda()
		pos_mse_criterion = nn.MSELoss().cuda()
		neg_mse_criterion = nn.MSELoss().cuda()

		logits_list = []
		labels_list = []
		with torch.no_grad():
			for input, label in valid_loader:
				input = input.cuda()
				logits = self.model(input)
				logits_list.append(logits)
				labels_list.append(label)
			logits = torch.cat(logits_list).cuda()
			labels = torch.cat(labels_list).cuda()
		
		pos_scores, pos_labels, neg_scores, neg_labels = self.scaled_label_scores(logits, labels)
		pos_mse = pos_mse_criterion(pos_scores, pos_labels)
		neg_mse = neg_mse_criterion(neg_scores, neg_labels)		
		
		print('Before calibration - POS_MSE: %.3f, NEG_MSE: %.3f, T: %0.3f' % \
			(pos_mse.item(), neg_mse.item(), self.temperature.item()))

		optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=1000)

		optim_steps = []		
		def eval():			
			pos_scores, pos_labels, neg_scores, neg_labels = self.scaled_label_scores(logits, labels)
			pos_mse = pos_mse_criterion(pos_scores, pos_labels)
			neg_mse = neg_mse_criterion(neg_scores, neg_labels)
			loss = pos_mse + neg_mse
			loss.backward()
			# print(loss.item(), pos_mse.item(), neg_mse.item(), self.temperature.item())
			optim_steps.append([loss.item(), pos_mse.item(), neg_mse.item(), self.temperature.item()])			
			return loss
		
		optimizer.step(eval)

		optim_steps = np.array(optim_steps)
		optim_min   = optim_steps[np.argmin(optim_steps[:,0])]
		self.temperature = nn.Parameter(torch.ones(1) * optim_min[3])
		print('After calibration - POS_MSE: %.3f, NEG_MSE: %.3f, T: %0.3f' % (optim_min[1], optim_min[2], optim_min[3]))
		return self


	def _scaled_label_scores(self, logits, labels):
			max_pred = self.scaled_score(logits)			
			return max_pred


	def set_mid_cal(self, valid_loader):
		self.cuda()
		mse_criterion = nn.MSELoss().cuda()

		logits_list = []
		labels_list = []
		with torch.no_grad():
			for input, label in valid_loader:
				input = input.cuda()
				logits = self.model(input)
				logits_list.append(logits)
				labels_list.append(label)
			logits = torch.cat(logits_list).cuda()
			labels = torch.cat(labels_list).cuda()
		
		scores = self._scaled_label_scores(logits, labels)
		mse    = mse_criterion(scores, labels)
		print('Before calibration - MSE: %.3f, T: %0.3f' % \
			(mse.item(), self.temperature.item()))
		optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=2000)

		optim_steps = []		
		def eval():			
			scores = self._scaled_label_scores(logits, labels)
			mse    = mse_criterion(scores, labels)
			loss   = mse
			loss.backward()
			# print(loss.item(), self.temperature.item())
			optim_steps.append([loss.item(), self.temperature.item()])
			return loss
		
		optimizer.step(eval)
		optim_steps = np.array(optim_steps)
		optim_min   = optim_steps[np.argmin(optim_steps[:,0])]
		self.temperature = nn.Parameter(torch.ones(1) * optim_min[1])
		print('After calibration - MSE: %.3f, T: %0.3f' % (optim_min[0], self.temperature))
		return self


	def scaled_mean(self, logits, labels):
		max_pred = self.scaled_score(logits)		
		return max_pred, max_pred.mean()

	def set_spread_cal(self, valid_loader, thr=None):
		self.cuda()
		var_mse_criterion = nn.MSELoss().cuda()
		thr_mse_criterion = nn.MSELoss().cuda()

		logits_list = []
		labels_list = []
		with torch.no_grad():
			for input, label, _ in valid_loader:
				input = input.cuda()
				logits = self.model(input)
				logits_list.append(logits)
				labels_list.append(label)
			logits = torch.cat(logits_list).cuda()
			labels = torch.cat(labels_list).cuda()
		scores 	  = self._scaled_label_scores(logits, labels)
		score_mu  = scores.mean()
		score_var = var_mse_criterion(scores, score_mu)		
		print('Before calibration - MEAN: %0.3f, VAR: %.3f, T: %0.3f' % \
				(score_mu.item(), score_var.item(), self.temperature.item()))

		if not thr is None:
			threshold = (torch.ones(scores.shape)*0.7).cuda()
				# optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=2000)
		optimizer = optim.Adam([self.temperature],lr = 1e-2)


		optim_steps = []
		def eval():			
			scores  	 = self._scaled_label_scores(logits, labels)
			score_mu     = scores.mean()
			score_var    = var_mse_criterion(scores, score_mu)
			if thr is None:
				loss         = -score_var
			else:
				score_thr    = thr_mse_criterion(scores, threshold)
				loss         = -score_var + score_thr
			loss.backward()
			# print(loss.item(), score_mu.item(), score_var.item(), self.temperature.item())
			optim_steps.append([loss.item(), self.temperature.item(), score_mu.item(), score_var.item()])
			return loss
		
		for i in range(1000):
			optimizer.step(eval)
		
		optim_steps = np.array(optim_steps)
		optim_min   = optim_steps[np.argmin(optim_steps[:,0])]		
		self.temperature = nn.Parameter(torch.ones(1) * optim_min[1])
		print('After calibration - MEAN: %0.3f, VAR: %.3f, T: %0.3f' % \
				(optim_min[2], optim_min[3], self.temperature.item()))
		
		return self.temperature.item(), np.vstack(optim_steps) #self
	
	def temp_softening(self, valid_loader):
		with torch.no_grad():
			for input, label, _ in valid_loader:
				input = input.cuda()
				logits = self.model(input)
				logits_list.append(logits)
				labels_list.append(label)
			logits = torch.cat(logits_list).cuda()
			labels = torch.cat(labels_list).cuda()
		
		stats = []
		for t in range(1000):
			scores 	  = logits/temperature
			scores_mu = scores.mean()
			scores_var = scores.var()
			stats.append(scores_mu.item(), scores_var.item())
		return stats



	# def set_mid_cal_odin(self, valid_loader, var):
	# 	self.cuda()
	# 	mse_criterion = nn.MSELoss().cuda()
	# 	ce_criterion  = nn.CrossEntropyLoss().cuda()

	# 	inputs_list = []
	# 	logits_list = []
	# 	grads_list  = []

	# 	for _x, _y in valid_loader:
	# 		inputs = Variable(_x.cuda(), requires_grad = True)			
	# 		outputs = self.model(inputs)
	# 		# Calculating the confidence of the output, no perturbation added here, no temperature scaling used
	# 		nnOutputs = outputs.data.cpu()
	# 		nnOutputs = nnOutputs.numpy()
	# 		nnOutputs = nnOutputs - np.max(nnOutputs,axis=1).reshape(-1,1)								
	# 		nnOutputs = np.array([np.exp(nnOutput)/np.sum(np.exp(nnOutput)) for nnOutput in nnOutputs])				
			
	# 		# Calculating the perturbation we need to add, that is,
	# 		# the sign of gradient of cross entropy loss w.r.t. input
	# 		maxIndexTemp = np.argmax(nnOutputs, axis=1)
	# 		labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
	# 		loss = ce_criterion(outputs, labels)
	# 		loss.backward()

	# 		# Normalizing the gradient to binary in {0, 1}
	# 		gradient =  torch.ge(inputs.grad.data, 0)		
	# 		gradient = (gradient.float() - 0.5) * 2				
	# 		# Normalizing the gradient to the same space of image
	# 		gradient[0][0] = (gradient[0][0] )/(var)
	# 		inputs_list.append(inputs)
	# 		grads_list.append(gradient)
	# 	inputs = torch.cat(inputs_list).cuda()
	# 	grads = torch.cat(grads_list).cuda()
		
	# 	pert_inputs = torch.add(inputs, -self.epsilon*grads)
	# 	logits = self.model(pert_inputs)
	# 	scores = self.scaled_scores(logits)
	# 	print(scores.shape)


class _ECELoss(nn.Module):
	"""
	Calculates the Expected Calibration Error of a model.
	(This isn't necessary for temperature scaling, just a cool metric).

	The input to this loss is the logits of a model, NOT the softmax scores.

	This divides the confidence outputs into equally-sized interval bins.
	In each bin, we compute the confidence gap:

	bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

	We then return a weighted average of the gaps, based on the number
	of samples in each bin

	See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
	"Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
	2015.
	"""
	def __init__(self, n_bins=15):
		"""
		n_bins (int): number of confidence interval bins
		"""
		super(_ECELoss, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]

	def forward(self, logits, labels):
		softmaxes = F.softmax(logits, dim=1)
		confidences, predictions = torch.max(softmaxes, 1)
		accuracies = predictions.eq(labels)

		ece = torch.zeros(1, device=logits.device)
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# Calculated |confidence - accuracy| in each bin
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if prop_in_bin.item() > 0:
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

		return ece
