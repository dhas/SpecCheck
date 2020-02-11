import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import sys
sys.path.append('..')
from utils import other_utils

def training_necessary(checkpoint_path,cfg,cfg_path):	
	training_needed = (not checkpoint_path.exists()) or \
		(not cfg_path.exists()) or \
		(cfg != pickle.load(open(cfg_path,'rb')))
	return training_needed

class EncoderTrainer:
	def __init__(self,model,device,train_loader,val_loader,criterion,optimizer,params):
		self.model 		  = model
		self.device 	  = device
		self.train_loader = train_loader
		self.val_loader   = val_loader
		self.criterion 	  = criterion
		self.optimizer 	  = optimizer
		self.params    	  = params
		
		self.train_history = {
			'iteration':[],
			'loss':[]
		}
		
		self.val_history = {
			'iteration':[],
			'loss':[],
			'accuracy':[]
		}

	def _plot_train_history(self):
		fig,axs0 = plt.subplots(figsize=(20,10))
		axs0.plot(self.train_history['iteration'],self.train_history['loss'],
			color='orange')
		axs0.plot(self.val_history['iteration'],self.val_history['loss'],
			color='red')
		axs1 = axs0.twinx()
		axs1.plot(self.val_history['iteration'],self.val_history['accuracy'],
			color='blue')
		fig.savefig(self.params['savedir']/'history.png')
		plt.close()

	def _save_train_history(self):
		
		history = {
			'train_history':self.train_history,
			'val_history':self.val_history
		}

		with open(self.params['savedir']/'history.pkl','wb') as f:
			pickle.dump(history,f)


	def _log_training(self,epoch, batch_idx, batch_size, num_batches,loss):
		print('\nTrain Epoch: {:3d} [{:4d}/{:4d} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, 
					batch_idx * batch_size,
					num_batches*batch_size,
					100. * batch_idx / num_batches,
					loss.item()),
					end = '\t'
				)
		self._plot_train_history()
	
	def _log_validation(self,val_loss,correct,total,accuracy,end=''):		
		print('Validate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(val_loss, correct, total, accuracy),end=end)

	def _validate(self,model, device, val_loader, criterion, epoch,batch_size,val_break=9):
		model.eval()
		val_loss = 0
		correct = 0
		length = 0

		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(val_loader):
				data, target = data.to(device), target.to(device)
				output = model(data)
				val_loss += criterion(output, target).item()  # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()
				length += len(data)
				# validate 10 batches is enough and fast
				if batch_idx == val_break:
					break

		val_loss /= length
		accuracy = 100. * correct / length

		return val_loss,correct,length,accuracy    
	
	def _train_epoch(self,model, device, train_loader, val_loader, optimizer, criterion, epoch,log_interval):		
		model.to(device)
		log_count = 0
		for batch_idx, (data, target) in enumerate(train_loader):
			model.train()
			data = data.to(device)
			target = target.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			
			
			# print train progress every log_interval			
			if batch_idx % log_interval == 0:
				self.train_history['iteration'].append(batch_idx*epoch)
				self.train_history['loss'].append(loss.item())
				self._log_training(epoch,batch_idx,len(data),len(train_loader),loss)
				if log_count % log_interval == 0:
					val_loss,correct,total,accuracy = self._validate(model, device, val_loader, criterion, epoch,len(data))
					self.val_history['iteration'].append(batch_idx*epoch)
					self.val_history['loss'].append(val_loss)
					self.val_history['accuracy'].append(accuracy)
					self._log_validation(val_loss,correct,total,accuracy)
					self._save_train_history()
				log_count += 1        
		
		#log after epoch
		self._log_training(epoch,batch_idx,len(data),len(train_loader),loss)
		if epoch == self.params['epochs']:
			val_break = -1
		else:
			val_break = 9
		val_loss,correct,total,accuracy = self._validate(model, device, val_loader, criterion, epoch,len(data),val_break=val_break)
		self._log_validation(val_loss,correct,total,accuracy,end='\n')

	def fit(self,checkpoint_path):			
		for epoch in range(1, self.params['epochs'] + 1):
			self._train_epoch(self.model, self.device, 
				self.train_loader, self.val_loader, 
				self.optimizer, self.criterion, 
				epoch,self.params['log_interval'])
			
			torch.save({'epoch': epoch,
				'model_state_dict': self.model.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict()},
				checkpoint_path)