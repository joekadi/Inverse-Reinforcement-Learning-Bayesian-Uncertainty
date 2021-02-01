import torch.nn as nn
import torch.nn.functional as F
import torch
class NonLinearNet(nn.Module):
	def __init__(self, in_features, i2,h1_out,h2_out):
		super().__init__() #initialise nn.module and anything else in __init__
		#try double and triple number of neurons 
		self.fc1 = nn.Linear(in_features,i2,bias=True)
		self.fc1.requires_grad=True		

		self.fc2= nn.Linear(i2,h1_out, bias=True)
		self.fc2.requires_grad=True	

		self.fc3= nn.Linear(h1_out,h2_out, bias=True)
		self.fc3.requires_grad=True

		self.fc4= nn.Linear(h2_out,1, bias=True)


	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x


