import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
from randomTicker import randomTicker
from sklearn.preprocessing import StandardScaler

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

scale = StandardScaler()
data = np.load('fixedData.npz', allow_pickle=True)
states = list(data['histories'])

priceHistory = yf.Ticker('AAPL').history(period='2y', interval='1h')
priceHistory = list(priceHistory['Open'])
priceHistory = priceHistory[len(priceHistory)-1000:len(priceHistory)]
states.append(priceHistory)
priceHistory = states
#print(priceHistory[6798])
print(priceHistory[len(priceHistory)-1])
priceHistory = scale.fit_transform(priceHistory)
priceHistory = priceHistory[len(priceHistory)-1]
priceHistory = torch.from_numpy(np.asarray(priceHistory))
X = priceHistory.to(torch.float32)

input_size = 1000
modelName = 'MarkIII.pt'
hidden_size = 256*2
model = NeuralNet(input_size, hidden_size, 1)
model.load_state_dict(torch.load(modelName))

model.eval()
with torch.no_grad():
    prediction = model(X).numpy()[0]
    print()
    print(prediction)