import numpy as np

histories = []
futures = []

load = np.load('collectedData.npz', allow_pickle=True)

needToRemove = []

for j, h in enumerate(load['histories']):
    for i, number in enumerate(h):
        if np.isnan(number):
            if j not in needToRemove:
                needToRemove.append(j)
    histories.append(h)
for j, f in enumerate(load['futures']):
    if np.isnan(f):
        if j not in needToRemove:
            needToRemove.append(j)
    futures.append(f)

needToRemove.sort()
needToRemove.reverse()
print(needToRemove)

for index in needToRemove:
    histories.pop(index)
    futures.pop(index)


print(len(histories))
print(len(futures))

np.savez('fixedData.npz', histories = np.asarray(histories), futures = np.asarray(futures))