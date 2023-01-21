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
    priceHistory = [priceHistory[startBatch:startBatch + 1000], priceHistory[startBatch+1010]]
    #print(priceHistory)
    return priceHistory

#grabBatch()
batches = []

for x in range(5):
    batch = grabBatch()
    batches.append(batch)

np.savez('collectedData.npz', collectedData = np.asarray(batches))

funny = np.load('collectedData.npz', allow_pickle=True)
funny = funny['collectedData']
print(funny)