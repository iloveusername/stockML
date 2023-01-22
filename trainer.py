import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()

        # self.l1 = nn.Linear(input_size, num_classes)


        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        #x = x.to(torch.float32)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out

# Prepare Data

scale = StandardScaler()
data = np.load('collectedData.npz', allow_pickle=True)
states = data['histories']
stateScale = scale.fit_transform(states)
states = stateScale[0:28]
actions = data['futures']
actions = actions[0:28]
# actions = scale.fit_transform(actions)
print(states[0])

X = torch.from_numpy(states)
X = X.to(torch.float32)
print(X)
y = torch.from_numpy(actions)
y = y.to(torch.float32).unsqueeze(1)
print(y)

n_samples, n_features = X.shape

# Prepare Model
modelName = 'firstTry.pt'
input_size = n_features
_, output_size = y.shape
# print(output_size)
hidden_size = 256*2
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(modelName))

# Config Stuff
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-6)

# Train Model
model.train()
num_epochs = 1000000000
for epoch in range(num_epochs):
    newStart = random.randint(0, 200)
    testLoc = random.randint(0, 27)
    states = stateScale[newStart:newStart+28]
    X = torch.from_numpy(states)
    X = X.to(torch.float32)

    y = torch.from_numpy(data['futures'][newStart:newStart+28])
    y = y.to(torch.float32).unsqueeze(1)

    y_predicted = model(X)

    loss = criterion(y, y_predicted)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 200 == 0:
        # print(y.size())
        # print(y_predicted.size())
        print('\n')
        print(f'epoch:{epoch+1}, loss = {loss.item()}')
        with torch.no_grad():
            a = model(torch.from_numpy(stateScale[testLoc])).numpy()
            b = actions[testLoc]
            print(f'Estimated Action = {a}')
            print(f'Actual Action = {b}')
            print(f'Difference = {abs(b-a)}')

    if (epoch+1) % 5000 == 0:
        print('Saving Model...')
        torch.save(model.state_dict(), modelName)