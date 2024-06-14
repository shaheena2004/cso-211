from dataset_names import DATASET_NAME
import matplotlib.pyplot as plt
import numpy as np

weights_vs_auc = np.genfromtxt(f"./output/weight_vs_auc_{DATASET_NAME}.csv", delimiter=",", dtype=np.float32)
icost, dcost, rcost ,auc =  weights_vs_auc[:,0], weights_vs_auc[:,1], weights_vs_auc[:,2], weights_vs_auc[:,3]
plt.figure(figsize = (15, 10))
ax = plt.axes(projection = '3d')
sizes = np.exp((auc-np.min(auc))*10)*100
colors = auc
fg = ax.scatter3D(icost, dcost, rcost, s=sizes, c=colors, cmap = "seismic")
ax.set_xlabel("icost")
ax.set_ylabel("dcost")
ax.set_zlabel("rcost")
plt.colorbar(fg)
plt.show()