import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from randomTicker import randomTicker
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from math import floor

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l6 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.l6(out)
        return out


#We have data from 6:30 - 12:30 in (PST)
#############################
currentTime = 6
currentDay = 1
#############################

for x in range(10):
    currentTime += 1
    if currentTime > 12:
        currentTime = 6
        currentDay += 1
        if currentDay > 4:
            currentDay = 0

if currentTime < 10:
    currentTime = '0'+str(currentTime)
else:
    currentTime = str(currentTime)

weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(f'{weekDays[currentDay]}, {currentTime}:30:00')

# addDays = floor((currentTime+10)/6)-1
# currentDay += addDays
# currentTime -= 2
# if currentTime < 6:
#     currentTime += 6
