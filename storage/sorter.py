f = open("tickers.txt", 'r')

tickers = f.read()

tickers = tickers.split('\n')
tempList = []
for ticker in tickers:
    tempWord = []
    for letter in ticker:
        if letter == ',':
            break
        else:
            tempWord.append(letter)
    tempWord = ''.join(tempWord)
    tempList.append(tempWord)

tickers = tempList
tickers = '\n'.join(tickers)
print(tickers)

with open("output.txt", "w") as textFile:
    textFile.write(tickers)