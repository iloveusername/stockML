import time
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
states = states[0:28]
states = scale.fit_transform(states)
actions = data['futures']
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
hidden_size = 256
model = NeuralNet(input_size, hidden_size, output_size)
#model.load_state_dict(torch.load(modelName))

# Config Stuff
learning_rate = 0.001
criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-6)

# Train Model
model.train()
num_epochs = 2
for epoch in range(num_epochs):

    y_predicted = model(X)
    print(f'y = {y}')
    print(f'y_pred = {y_predicted}')
    print(X.dtype)
    print(y.dtype)
    print(y_predicted.dtype)

    loss = criterion(y, y_predicted)

    print(y.size())
    print(y_predicted.size())

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 5 == 0:
        print(f'epoch:{epoch+1}, loss = {loss.item()}')
