import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

root = "/Users/brianchao/Documents/papers/light-field/tcsvt-2022/tf_records/merge_ablation"
records = os.listdir(root)
records.sort()
print(records)

labels = ["distance-weighted", "left", "right", "average"]
font = {'style':"normal", 'size'   : 16}

matplotlib.rc('font', **font)
plt.figure()
for i, record in enumerate(records):
    print(record)
    data = np.genfromtxt(os.path.join(root, record), delimiter=",")
    loss = data[1:, -1]
    plt.plot(loss, label=labels[i])
plt.legend()
plt.show()
    

