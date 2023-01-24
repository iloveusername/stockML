import numpy as np
import yfinance as yf
import random
from randomTicker import randomTicker

# Handle just like data collection for Cassie.
# Make a [Price List As Numpy Array, Answer].
# Append to a greater list, save that list as NPZ.

def grabBatch():
    ticker = randomTicker()
    priceHistory = yf.Ticker(ticker).history(period='2y', interval='1h')
    priceHistory = list(priceHistory['Open'])
    maxNum = len(priceHistory) - 1010
    startBatch = random.randint(0, maxNum)
    history = priceHistory[startBatch:startBatch + 1000]
    future = priceHistory[startBatch+1010]
    return history, future

#grabBatch()
histories = []
futures = []

epochs = 250
for x in range(epochs):
    print(x/epochs*100)
    try:
        history, future = grabBatch()
        histories.append(history)
        futures.append(future)
    except:
        print('fail')
        continue

load = np.load('collectedData.npz', allow_pickle=True)

for h in load['histories']:
    histories.append(h)
for f in load['futures']:
    futures.append(f)

np.savez('collectedData.npz', histories = np.asarray(histories), futures = np.asarray(futures))

load = np.load('collectedData.npz', allow_pickle=True)
print(len(load['futures']))
print(len(load['histories']))

print(load['histories'][50][1])
print(load['futures'][50])