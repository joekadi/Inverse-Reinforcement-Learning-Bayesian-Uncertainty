import torch.nn as nn
import torch.nn.functional as F
import torch
class NonLinearNet(nn.Module):
	def __init__(self, in_features):
		super().__init__() #initialise nn.module and anything else in __init__
		#try double and triple number of neurons 
		self.fc1 = nn.Linear(in_features,84, bias=True)
		self.fc1.requires_grad=True		

		self.fc2= nn.Linear(84,42, bias=True)
		self.fc2.requires_grad=True	

		self.fc3= nn.Linear(42,32, bias=True)
		self.fc3.requires_grad=True

		self.fc4= nn.Linear(32,16, bias=True)
		self.fc4.requires_grad=True

		self.fc5= nn.Linear(16,8, bias=True)
		self.fc5.requires_grad=True

		self.fc6= nn.Linear(8,4, bias=True)
		self.fc6.requires_grad=True

		self.fc7= nn.Linear(4,1, bias=True)
		self.fc7.requires_grad=True

	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = F.relu(self.fc6(x))
		x = self.fc7(x)
		return x


