import torch.nn as nn
import torch.nn.functional as F
import torch
from likelihood import *


class LinearNet(nn.Module):
	def __init__(self):
		super().__init__() #initialise nn.module and anything else in __init__
		self.fc1 = nn.Linear(4, 4) #make input value dynamically = noofstates not hardcoded 4

	def forward(self, x):
		#x = F.relu(self.fc1(x))
		x = self.fc1(x)
		return x

	



'''
#code to isolate test linearNN
net = LinearNet()
print(net.parameters)
X = torch.randn(4,1) #initial state vector
X = X.view(-1,4)
print(net(X))
'''
