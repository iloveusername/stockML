import yfinance as yf

#Given a price history over the past 1000 hours, predict what the price will be in 10 hours.

priceHistory = yf.Ticker('TSLA').history(period='2y', interval='1h')
priceHistory = list(priceHistory['Open'])

print(priceHistory)
print(len(priceHistory))