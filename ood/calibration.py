import torch
from torch import nn, optim
from torch.nn import functional as F
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
			print(loss.item(), pos_mse.item(), neg_mse.item(), self.temperature.item())
			optim_steps.append([loss.item(), pos_mse.item(), neg_mse.item(), self.temperature.item()])			
			return loss
		
		optimizer.step(eval)

		optim_steps = np.array(optim_steps)
		optim_min   = optim_steps[np.argmin(optim_steps[:,0])]
		self.temperature = nn.Parameter(torch.ones(1) * optim_min[3])		
		print('After calibration - POS_MSE: %.3f, NEG_MSE: %.3f, T: %0.3f' % (optim_min[1], optim_min[2], optim_min[3]))
		return self


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
