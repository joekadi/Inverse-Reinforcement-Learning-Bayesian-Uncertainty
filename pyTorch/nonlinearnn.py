import torch.nn as nn
import torch.nn.functional as F
import torch


class NonLinearNet(nn.Module):
	def __init__(self):
		super().__init__() #initialise nn.module and anything else in __init__
		self.fc1 = nn.Linear(36,18, bias=True)
		self.fc1.requires_grad=True
		self.fc2 = nn.Linear(18,9, bias=True)
		self.fc2.requires_grad=True
		self.fc3 = nn.Linear(9,3, bias=True)
		self.fc3.requires_grad=True
		self.fc4 = nn.Linear(3,1, bias=True)
		self.fc4.requires_grad=True
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x







