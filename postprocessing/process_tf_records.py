import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

root = "/Users/brianchao/Documents/papers/light-field/tcsvt-2022/tf_records/lr_consistency_ablation"
records = os.listdir(root)
records.sort()
labels = ["With left-right consistency", "Without left-right consistency"]

plt.figure()
for i, record in enumerate(records):
    print(record)
    data = np.genfromtxt(os.path.join(root, record), delimiter=",")
    loss = data[1:, -1]
    plt.plot(loss, label=labels[i])
plt.legend()
plt.show()
    

