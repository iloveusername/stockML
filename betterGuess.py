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


def getTime(currentTime, currentDay):
    for x in range(10):
        currentTime += 1
        if currentTime > 12:
            currentTime = 6
            currentDay += 1
            if currentDay > 4:
                currentDay = 0
    if currentTime < 10:
        currentTime = '0' + str(currentTime)
    else:
        currentTime = str(currentTime)
    return currentTime, currentDay


def getPrice(tickerName, statesList):
    scale = StandardScaler()
    priceHistory = yf.Ticker(tickerName).history(period='1y', interval='1h')
    priceHistory = list(priceHistory['Open'])
    priceHistory = priceHistory[len(priceHistory) - 1000:len(priceHistory)]
    statesList.append(priceHistory)
    priceHistory = states
    currentPrice = priceHistory[len(priceHistory) - 1][999]
    priceHistory = scale.fit_transform(priceHistory)
    priceHistory = priceHistory[len(priceHistory) - 1]
    priceHistory = torch.from_numpy(np.asarray(priceHistory))
    priceHistory = priceHistory.to(torch.float32)
    return priceHistory, currentPrice


#############################
tickers = ['TSLA', 'BB', 'AAPL', 'BROS', 'GME', 'AMC', 'SPY', 'AMZN', 'GOOG', 'MSFT', 'UROY', 'NVDA', 'AMD', 'META', 'TXN', 'BAC', 'CVX', 'HD']
# modelName = 'MarkIV.pt'
modelName = 'stableML.pt'
currentTime = 12
currentDay = 1
#############################

input_size = 1000
hidden_size = 256*4
model = NeuralNet(input_size, hidden_size, 1)
model.load_state_dict(torch.load(modelName))

currentTime, currentDay = getTime(currentTime, currentDay)

weekDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(f'{weekDays[currentDay]}, {currentTime}:30:00')

data = np.load('fixedData.npz', allow_pickle=True)
states = list(data['histories'])

for ticker in tickers:
    priceHistory, currentPrice = getPrice(ticker, states)
    model.eval()
    with torch.no_grad():
        prediction = model(priceHistory).numpy()[0]
        prediction = prediction.item()
        difference = prediction - currentPrice
        print()
        print('#####################')
        print(f'Ticker: {ticker}')
        print(f'{weekDays[currentDay]}, {currentTime}:30:00')
        print(f'Current: ${currentPrice:.2f}')
        print(f'Prediction: ${prediction:.2f}')
        if difference > 0:
            print(f'Gain: +${difference:.2f}, +{(prediction / currentPrice * 100) - 100:.2f}%')
        else:
            print(f'Loss: -${abs(difference):.2f}, -{100 - (prediction / currentPrice * 100):.2f}%')
        print('#####################')