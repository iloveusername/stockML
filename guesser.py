import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
from randomTicker import randomTicker
from datetime import datetime
from sklearn.preprocessing import StandardScaler

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

scale = StandardScaler()
data = np.load('fixedData.npz', allow_pickle=True)
states = list(data['histories'])

#9:30 to 3:30

tickerName = 'BROS'
priceHistory = yf.Ticker(tickerName).history(period='1y', interval='1h')
getTime = priceHistory
print(getTime)
getTime = getTime[len(getTime)-10:len(getTime)]
getTime = str(getTime.iloc[9].name)[11:13]
getTime = int(getTime)-9
days = int(((getTime + 10) - (getTime+10) % 6)/6)
hour = (getTime+10) % 6
currentDay = datetime.now().weekday()
if currentDay > 5:
    currentDay = 5
weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
currentDay += days
currentDay = weekDays[currentDay-1]
hour = hour+9
predTime = str(currentDay) + ' ' + str(hour) + ':30:00'

priceHistory = list(priceHistory['Open'])
priceHistory = priceHistory[len(priceHistory)-1000:len(priceHistory)]
states.append(priceHistory)
priceHistory = states
currentPrice = priceHistory[len(priceHistory)-1][999]
priceHistory = scale.fit_transform(priceHistory)
priceHistory = priceHistory[len(priceHistory)-1]
priceHistory = torch.from_numpy(np.asarray(priceHistory))
X = priceHistory.to(torch.float32)

input_size = 1000
# modelName = 'secondTry.pt'
modelName = 'MarkIV.pt'
hidden_size = 256*4
model = NeuralNet(input_size, hidden_size, 1)
model.load_state_dict(torch.load(modelName))

model.eval()
with torch.no_grad():
    prediction = model(X).numpy()[0]
    prediction = prediction.item()
    difference = prediction - currentPrice
    print('\n')
    print('#####################')
    print(f'Ticker: {tickerName}')
    print(predTime)
    print(f'Prediction: ${prediction:.2f}')
    if difference > 0:
        print(f'Gain: +${difference:.2f}, ${currentPrice:.2f}, +{(prediction/currentPrice*100)-100:.2f}%')
    else:
        print(f'Loss: ${difference:.2f}, ${currentPrice:.2f}, -{100-(prediction/currentPrice*100):.2f}%')
    print('#####################')
