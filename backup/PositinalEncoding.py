import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(pos, d_model):
    PE = np.zeros((pos,d_model))
    for p in range(pos):
        for i in range(0,d_model,2):
            PE[p, i]   = np.sin(p / 10000 ** ((2 * i) / d_model))
            PE[p, i+1] = np.cos(p / 10000 ** ((2 * i) / d_model))
    return PE

PE = positional_encoding(50,16)
plt.figure(figsize=(10,6))
plt.pcolormesh(PE,cmap='RdBu')
plt.xlabel('Embedding Dimenstions')
plt.ylabel('Position')
plt.colorbar()
plt.show()
