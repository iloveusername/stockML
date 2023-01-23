import numpy as np

histories = []
futures = []

load = np.load('collectedData.npz', allow_pickle=True)

for h in load['histories']:
    print(h)
    histories.append(h)
for f in load['futures']:
    futures.append(f)

print(len(histories))
print(len(futures))

print(histories[5201][999])
print(futures[5201])