import torch.nn as nn
import torch.nn.functional as F
import torch
from myNLL import *



class LinearNet(nn.Module):
	def __init__(self):
		super().__init__() #initialise nn.module and anything else in __init__
		self.fc1 = nn.Linear(2,1) #make input value dynamically = len state feature vector
		self.fc1.requires_grad=True
		#self.fc2 = nn.Linear(4,4, bias=False)

	def forward(self, x):
		#x = F.relu(self.fc1(x))
		x = self.fc1(x)
		return x




