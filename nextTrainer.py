import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# https://discuss.pytorch.org/t/getting-nan-after-first-iteration-with-custom-loss/25929/3

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.relu = nn.LeakyReLU()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        #x = x.to(torch.float32)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        return out

# Prepare Data

print(torch.cuda.is_available())
exit()

scale = StandardScaler()
data = np.load('fixedData.npz', allow_pickle=True)
states = data['histories']
stateScale = scale.fit_transform(states)
actions = data['futures']

X = torch.from_numpy(stateScale)
X = X.to(torch.float32)
# X = X[3082:3083]

# print(X)

y = torch.from_numpy(actions)
ty = y
# y = y[3082:3083]
print(X)
print(y)

y = y.to(torch.float32).unsqueeze(1)
# print(y)

n_samples, n_features = X.shape
print(n_features)

# Prepare Model
modelName = 'MarkIII.pt'
input_size = n_features
_, output_size = y.shape
print(y)
print(input_size)
print(output_size)
hidden_size = 256*2
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(modelName))

# Config Stuff
learning_rate = 0.0001
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-6)

# Train Model
# torch.autograd.detect_anomaly(True)
model.train()
num_epochs = 1000000000
for epoch in range(num_epochs):
    testLoc = random.randint(0, n_samples-1)

    y_predicted = model(X)

    loss = criterion(y, y_predicted)

    loss.backward()

    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        # print('\n')
        print('##############################')
        print(f'epoch:{epoch+1}, loss = ${loss.item():.2f}')
        print(f'Previous Price = ${states[testLoc][len(states[testLoc])-1]:.2f}')
        with torch.no_grad():
            a = model(X[testLoc]).numpy()[0]
            b = y[testLoc].numpy()[0]
            print(f'Estimated Action = ${a:.2f}')
            print(f'Actual Action = ${b:.2f}')
            print(f'Difference = ${b-a:.2f}')
        print('##############################')


    if (epoch+1) % 250 == 0:
        print('\n')
        print('##############')
        print('Saving Model...')
        print('##############')
        print('\n')

        torch.save(model.state_dict(), modelName)