import yfinance as yf
import random
from randomTicker import randomTicker

futureHours = 10

#We need to build a dataset of a bunch of these batches to train off of. [[[List, Of, Prices], Answer], [[List, Of, Prices], Answer]]

def grabBatch():
    ticker = randomTicker()
    print(ticker)
    priceHistory = yf.Ticker(ticker).history(period='2y', interval='1h')
    priceHistory = list(priceHistory['Open'])
    #maxNum = len(priceHistory)-futureHours
    maxNum = len(priceHistory)-1000-futureHours
    startBatch = random.randint(0, maxNum)
    print(startBatch)
    priceHistory = priceHistory[startBatch:startBatch+1000]
    return 0

#Given a price history over the past 1000 hours, predict what the price will be in 10 hours.

priceHistory = yf.Ticker(randomTicker()).history(period='2y', interval='1h')
priceHistory = list(priceHistory['Open'])

prices = grabBatch()
print(prices)
