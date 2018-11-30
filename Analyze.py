import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pylab as pl
# X = [1, 1, 2, 2]
# Y = [3, 4, 4, 3]
# Z = [1, 2, 1, 1]


# data = pd.read_csv("train/02/PLC/plc.csv").values
data = pd.read_csv("train/01/Sensor/{}.csv".format(46)).values
# print(len(data[:,0]))
# csv_nos = set(data[:,-1])
# print(csv_nos)
spinde = data[:,0]
spinde = np.fft.fft(spinde)
index = [i for i in range(len(spinde))]
plt.plot(index, spinde)
plt.show()
# a = {}
# for d in data:
#     if d[-1] in a:
#         a[d[-1]] = a[d[-1]] + 1
#     else:
#         a[d[-1]] = 1
# print(a)

# for no in csv_nos:
#     X = []
#     Y = []
#     Z = []
#     spindle = []
#     for d in data:
#         if d[-1] is no:
#             X.append(d[2])
#             Y.append(d[3])
#             Z.append(d[4])
#             spindle.append(d[1])
#     # Draw knife`s path in 3D
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(X, Y, Z, label='parametric curve')
#     plt.show()

# plt.plot([i for i in range(len(spindle))], spindle)
# plt.show()
# c_max = []
# c_min = []
# c_ave = []
# c_var = []
# c_std = []
#
# for i in range(2,3):
#     data = pd.read_csv("train/03/Sensor/{}.csv".format(i+1)).values
#     c_max.append(data[:,0].max())
#     c_min.append(data[:,0].min())
#     c_ave.append(data[:,0].mean())
#     c_var.append(data[:,0].var())
#     c_std.append(data[:,0].std())
# c_index = [i for i in range(len(c_max))]
#
#
# sampling_rate = 25600
# fft_size = len(data[:,0])
# xs = data[:,0][:fft_size]
# xf = np.fft.rfft(xs)/fft_size
# freqs = np.linspace(0, sampling_rate/2, fft_size/2+1)
# xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
# pl.figure(figsize=(8,4))
# pl.subplot(211)
# pl.plot([i for i in range(len(data[:,0]))][:fft_size], xs)
# pl.xlabel("time(s)")
# pl.subplot(212)
# pl.plot(freqs, xfp)
# pl.xlabel("frequency(Hz)")
# pl.subplots_adjust(hspace=0.4)
# pl.show()
# data = {"max": c_max, "min": c_min, "ave": c_ave, "var": c_var, "std": c_std, "index": c_index}
# data = pd.DataFrame(data)
# sns.pairplot(data, y_vars=["max", "min", "ave", "var", "std"], x_vars='index', size=12, aspect=1, kind='reg')
# plt.scatter([i for i in range(48)], c_max, label='max')
# plt.scatter([i for i in range(48)], c_min, label='min')
# plt.scatter([i for i in range(48)], c_ave, label='ave')
# plt.scatter([i for i in range(48)], c_var, label='var')
# plt.scatter([i for i in range(48)], c_std, label='std')
# plt.legend(loc='upper right')

# plt.plot([i for i in range(len(data[:,0]))], data[:,0])
# plt.show()
#
# # X = data[:,0]
# # Y = data[:,1]
# # Z = data[:,2]
# #
# # plt.plot([i for i in range(len(X))], X)
# # # plt.plot([i for i in range(len(Y))], Y)
# # # plt.plot([i for i in range(len(Z))], Z)
# plt.show()


