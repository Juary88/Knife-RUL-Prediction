import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pylab as pl
from librosa.feature import mfcc
import pywt
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
window_size, poly_order = 5, 3
# X = [1, 1, 2, 2]
# Y = [3, 4, 4, 3]
# Z = [1, 2, 1, 1]

# for i in range(43, 44):
#     # data = pd.read_csv("train/03/PLC/plc.csv").values
#     data = pd.read_csv("train/01/Sensor/{}.csv".format(i)).values
#     # print(len(data[:,0]))
#     # csv_nos = set(data[:,-1])
#     # print(csv_nos)
#     spinde = data[:,0]
#     # ave = []
#     # for j in range(0, int(len(spinde)/330)):
#     #     ave.append(spinde[j*330:(j+1)*330].mean())
#     i = 0
#     while (i + 1) * 256000 < len(spinde) - 256000:
#         tmp = spinde[i*256000: (i+1)*256000]
#         t = np.arange(0, 10.0, 1.0/25600)
#         totalscal = 256
#         fc = pywt.central_frequency('cgau8')
#         cparam = 2 * fc * totalscal
#         scales = cparam / np.arange(totalscal, 1, -1)
#         print("miao")
#         coef, freqs = pywt.cwt(tmp, scales, 'cgau8', 1.0/25600)
#         i = i + 1
#         print("finish calculate")
#         plt.contourf(t, freqs, abs(coef))
#         plt.ylabel("f(Hz)")
#         plt.xlabel("t(s)")
#         plt.subplots_adjust(hspace=0.4)
#         plt.show()
    # spinde = np.fft.fft(spinde)
    # spinde = mfcc(y=spinde, sr=25600, S=None, n_mfcc=20, dct_type=2, norm='ortho')
    # print(spinde)
    # print(len(spinde))
    # df = pd.DataFrame({"kurtosis": ave, 'index': [i for i in range(len(ave))]})
    # plt.imshow(np.flipud(spinde), cmap=plt.cm.jet, aspect=100, extent=[0,spinde.shape[1],0,spinde.shape[0]])
    # sns.pairplot(df, y_vars='kurtosis', x_vars='index', size=12, aspect=1, kind='reg')
    # plt.plot([i for i in range(len(ave))], ave)
    #     plt.show()


# index = [i for i in range(len(spinde))]
# plt.plot(index[10000:1500000], spinde[10000:1500000])
# plt.show()
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
data = pd.read_csv("train/02/PLC/plc.csv").values
csv_nos = set(data[:,-1])
c_max = []
c_min = []
c_ave = []
c_var = []
c_std = []
for no in csv_nos:
    spindle = []
    for d in data:
        if d[-1] is no:
            spindle.append(d[1])
    spindle = np.array(spindle)
    c_max.append(spindle.max())
    c_min.append(spindle.min())
    c_ave.append(spindle.mean())
    c_var.append(spindle.var())
    c_std.append(spindle.std())
# for i in range(2,46):
#     data = pd.read_csv("train/01/Sensor/{}.csv".format(i+1)).values
#     c_max.append(data[:,0].max())
#     c_min.append(data[:,0].min())
#     c_ave.append(data[:,0].mean())
#     c_var.append(data[:,0].var())
#     c_std.append(data[:,0].std())
feature = c_max
y_pred = KMeans(random_state=418).fit_predict(np.array(feature).reshape(-1, 1))
plt.scatter([i for i in range(len(feature))], feature, c=y_pred)
y1 = savgol_filter(feature, window_size, poly_order)
plt.plot(feature)
plt.plot(y1)
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
# data = pd.read_csv("test/04/PLC/plc.csv").values
# # y1 = savgol_filter(data[:,1][2500:-2500], window_size, poly_order)
# y1 = savgol_filter(data[:,1], window_size, poly_order)
# # y1 = savgol_filter(y1, window_size, poly_order)
# plt.plot(y1)
# plt.legend(loc='upper right')

# plt.plot([i for i in range(len(data[:,0]))], data[:,0])
plt.show()
#
# # X = data[:,0]
# # Y = data[:,1]
# # Z = data[:,2]
# #
# # plt.plot([i for i in range(len(X))], X)
# # # plt.plot([i for i in range(len(Y))], Y)
# # # plt.plot([i for i in range(len(Z))], Z)
# plt.show()


