import random

f = open("output.txt", 'r')

tickers = f.read()
tickers = tickers.split('\n')

def randomTicker():
    random.seed(None)
    return tickers[random.randint(0,len(tickers))]