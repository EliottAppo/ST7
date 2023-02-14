""" 
Eliott Dumont, ST7 Optimization for neuro-inspired computing with physical architecture, Lab1 2A CentraleSupélec

The aim of this labwork is to remove the inter-symbol interference (ISI) between four different
symbols in a WIFI communucation channel. The channel is modeled as a linear system followed by 
a memoryless nonlinearity. 
"""



import random
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt



"""Parameters"""

N = 10100 # The WIFI channel is a 10100-term serie
SNR = 12 # Signal to noise ratio
symbols = [-1, 1, -3, 3]


"""Data creation / preparation"""

#input d : the WIFI channel
d = []
for _ in range(N):
    i = random.randint(0, 3)
    x = symbols[i]
    d.append(x)

# Power of the d signal
Psignal = 0
for x in d:
    Psignal += 1/N * abs(x)**2

# std dev (not variance)
var = sqrt(Psignal / (10**(SNR/10)))

# q serie : input-output, linear relationship
q = [(0.08*d[i+2] - 0.12*d[i+1] + d[i] + 0.18*d[i-1] - 0.1*d[i-2] + 0.091*d[i-3] - 0.05*d[i-4] + 0.04*d[i-5] + 0.03*d[i-6] + 0.01*d[i-7]) for i in range(7, N-2)]

v = np.random.normal(0, scale=var, size=N)

# output u : the distorded WIFI signal
u = [(q[i] + 0.036*q[i]**2 - 0.011*q[i]**3 + v[i])for i in range(len(q))]


"""Simulation of the reservoir"""

# Initialization 
nTraining = 3000 # First part of the serie : training the ESN
nTest = 10000 # The testing part
Nx = 50 # Number of neurons 

Wil = 2*0.025*np.random.rand(Nx, 1) - 0.025 # Input layer connectivity matrix

Whli = 2*np.random.rand(Nx, Nx) - 1 # adjacency matrix of the RNN (W hidden layer intermediate)
nb = 0
while nb <= int(0.8*(Nx**2)): # This has to be a sparse matrix, 20% connectivity
    i = random.randint(0, Nx - 1)
    j = random.randint(0, Nx - 1)
    if Whli[i][j] != 0:
        Whli[i][j] = 0
        nb += 1

v = np.linalg.eigvals(Whli) # Making sure that the spectral radius of Whl is less than 1 (in fact equal .5 here)
rho = v[0]
for x in v:
    if np.abs(x) > rho:
        rho = np.abs(x)
Whl = (0.5 * Whli / rho).real

Xstate = np.zeros(shape=(Nx, 1))
X = np.zeros(shape=(Nx, nTraining, 1))
output = np.zeros(shape=(1, nTest))


# Temporal evolution of the ESN
for t in range(1, nTraining): # Following the evolution process
    Xstate = np.tanh(np.matmul(Whl, Xstate) + (u[t] + 30)*Wil) # The input given to the ENS is in fact a biased version (+30)
    X[:, t] = Xstate


# Offline training of the ESN (Moore-Penrose)
nWashout = 100 # We don't care about the 100 first terms beceause the system is not at a steady state
endOfTrain = nTraining
X = X.reshape(Nx, nTraining)
X = X[:, nWashout+1 : endOfTrain]
d = np.array(d).reshape(N, 1)

Y = np.transpose(np.array([d[n-2] for n in range(2, nTraining+2)])) # Target
Y = Y[:, nWashout+1:endOfTrain]

P1 = np.matmul(Y, np.transpose(X))
P2 = np.matmul(X, np.transpose(X))
P3 = np.linalg.inv(P2)
Wol = np.matmul(P1, P3) # This matrix is the solution of the least square optimization problem


# Testing 
X = X[:, endOfTrain - nWashout - 2].reshape(Nx, 1)

for t in range(1, nTest):
    Xstate = np.tanh(np.matmul(Whl, Xstate) + (u[t]+30)*Wil)
    output[:, t] = np.matmul(Wol, Xstate)


"""Results and plot"""

def tracer():
    plt.plot(np.transpose(output), label="Output of the ESN")
    plt.plot(d, label="Wifi signal")
    plt.legend()
    plt.show()
#tracer()


# cleaning the output

y_out = [np.transpose(output)[i][0] for i in range(nTest)]
d_out = [d[i][0] for i in range(N)]
for _ in range(100):
    d_out.pop()

def tracer_propre_brut():
    plt.plot(d_out)
    plt.plot(y_out)
    plt.show()
#tracer_propre_brut()

y_out_clean = [] # Following the table for the range of value for y(n)
for x in y_out:
    if x>= -4 and x < -2 :
        y_out_clean.append(-3)
    elif x >= -2 and x < 0 :
        y_out_clean.append(-1)
    elif x >= 0 and x < 2 :
        y_out_clean.append(1)
    elif x >= 2 and x<4 :
        y_out_clean.append(3)
    else:                           # if the value is out of bound, I plot 5 to see it (sinon ça décalle tout)
        y_out_clean.append(5)

def tracer_propre_std():
    plt.plot(d_out, label="WIFI Signal")
    plt.plot(y_out_clean, label="Output of the ESN")
    plt.legend()
    plt.show()
#tracer_propre_std()





# Datas / analyse

SNR12 = np.array([13.44, 14.01, 15.06, 12.81, 16.69, 13.95, 12.71, 15.14, 12.42, 14.11, 14.86, 14.54, 13.64, 16.06, 15.18, 13.92, 12.54, 12.81, 18.10, 12.80])
SNR16 = np.array([4.46, 4.65, 4.68, 4.62, 4.17, 4.37, 4.06, 4.27, 5.21, 4.88, 3.89, 4.71, 4.0, 5.78, 3.51, 7.05, 4.61, 5.85, 4.15, 4.66])
SNR20 = np.array([1.29, 1.65, 0.87, 0.66, 2.79, 1.43, 1.09, 1.19, 1.87, 2.15, 1.62, 2.57, 1.03, 1.63, 5.10, 1.44, 1.05, 1.13, 1.64, 1.27])
SNR24 = np.array([0.61, 0.36, 1.25, 1.02, 2.77, 0.98, 1.73, 0.56, 0.75, 0.66, 0.45, 0.36, 0.77, 1.21, 0.34, 0.40, 0.50, 0.37, 0.64, 0.22])
SNR28 = np.array([0.47, 0.16, 0.11, 0.66, 0.46, 0.42, 1.53, 0.2, 0.75, 0.46, 0.68, 0.38, 0.33, 0.44, 0.29, 0.16, 0.23, 0.69, 0.48, 0.28])
SNR32 = np.array([0.79, 0.24, 0.91, 1.06, 1.5, 0.31, 0.33, 0.68, 0.2, 0.45, 0.23, 2.61, 0.22, 0.4, 0.69, 0.14, 0.59, 2.03, 0.34, 0.27])
mean12 = np.mean(SNR12)
std12 = np.std(SNR12)
mean16 = np.mean(SNR16)
std16 = np.std(SNR16)
mean20 = np.mean(SNR20)
std20 = np.std(SNR20)
mean24 = np.mean(SNR24)
std24 = np.std(SNR24)
mean28 = np.mean(SNR28)
std28 = np.std(SNR28)
mean32 = np.mean(SNR32)
std32 = np.std(SNR32)

axisx = np.array([12, 16, 20, 24, 28, 32])
axisy = np.array([mean12, mean16, mean20, mean24, mean28, mean32])
errorbar = np.array([std12, std16, std20, std24, std28, std32])

SNR12_300 = [21.92, 25.05, 21.0, 20.16, 21.86, 20.96, 21.64, 17.59, 21.62, 20.51, 19.73, 21.63, 22.51, 20.4, 21.63, 17.93, 20.3, 18.48, 20.52, 19.38]
SNR16_300 = [7.48, 6.859999999999999, 7.21, 6.47, 10.040000000000001, 6.76, 7.7, 8.77, 9.98, 7.5600000000000005, 8.98, 10.23, 9.719999999999999, 7.01, 7.92, 10.19, 6.97, 7.27, 7.51, 6.06]
SNR20_300 = [4.2, 2.52, 3.09, 1.97, 2.13, 4.15, 2.5700000000000003, 2.6, 6.84, 3.6700000000000004, 1.92, 2.68, 3.11, 2.22, 5.46, 2.4699999999999998, 5.34, 2.6100000000000003, 3.45, 2.69]
SNR24_300 = [2.6100000000000003, 0.9400000000000001, 4.15, 3.09, 2.31, 1.5599999999999998, 2.59, 1.69, 2.25, 1.24, 1.0, 1.34, 1.0, 1.91, 0.91, 2.11, 2.23, 0.83, 1.95, 2.8899999999999997]
SNR28_300 = [5.56, 1.34, 2.1399999999999997, 0.74, 1.09, 1.3, 0.25, 1.05, 1.26, 2.68, 1.8499999999999999, 3.46, 0.4, 2.35, 4.2700000000000005, 2.54, 2.31, 2.16, 0.58, 2.01]
SNR32_300 = [1.16, 0.43, 2.0500000000000003, 0.6, 1.08, 0.64, 2.41, 0.63, 1.95, 3.0, 0.29, 2.33, 0.97, 0.3, 2.02, 2.04, 3.2199999999999998, 0.29, 1.23, 1.8399999999999999]
mean12_300 = np.mean(SNR12_300)
std12_300 = np.std(SNR12_300)
mean16_300 = np.mean(SNR16_300)
std16_300 = np.std(SNR16_300)
mean20_300 = np.mean(SNR20_300)
std20_300 = np.std(SNR20_300)
mean24_300 = np.mean(SNR24_300)
std24_300 = np.std(SNR24_300)
mean28_300 = np.mean(SNR28_300)
std28_300 = np.std(SNR28_300)
mean32_300 = np.mean(SNR32_300)
std32_300 = np.std(SNR32_300)
axisy_300 = np.array([mean12_300, mean16_300, mean20_300, mean24_300, mean28_300, mean32_300])
errorbar_300 = np.array([std12_300, std16_300, std20_300, std24_300, std28_300, std32_300])

SNR12_1000 = [16.06, 13.52, 14.46, 14.05, 14.96, 15.110000000000001, 14.48, 14.469999999999999, 14.6, 17.02, 15.1, 13.370000000000001, 12.879999999999999, 15.22, 14.14, 13.719999999999999, 17.32, 15.709999999999999, 14.96, 13.530000000000001]
SNR16_1000 = [4.55, 3.5999999999999996, 5.3100000000000005, 6.77, 4.26, 4.12, 6.8500000000000005, 5.34, 4.64, 5.680000000000001, 5.12, 4.82, 8.43, 4.16, 4.45, 5.0, 7.449999999999999, 7.249999999999999, 4.54, 7.720000000000001]
SNR20_1000 = [1.24, 2.68, 1.32, 4.31, 2.11, 1.31, 2.62, 1.1400000000000001, 2.22, 1.15, 1.51, 2.74, 2.5100000000000002, 4.9799999999999995, 1.7000000000000002, 0.5499999999999999, 1.31, 1.5, 4.04, 1.95]
SNR24_1000 = [0.9299999999999999, 0.9400000000000001, 0.98, 1.06, 4.34, 0.79, 0.38, 0.47000000000000003, 0.62, 1.25, 0.38, 0.98, 1.37, 0.31, 1.83, 0.98, 1.8399999999999999, 2.26, 0.6, 0.5]
SNR28_1000 = [0.49, 1.0699999999999998, 0.44999999999999996, 0.43, 0.3, 0.41000000000000003, 0.25, 0.38, 0.42, 0.45999999999999996, 1.7999999999999998, 0.41000000000000003, 0.38, 0.5499999999999999, 0.7799999999999999, 0.54, 0.42, 0.12, 0.58, 1.18]
SNR32_1000 = [1.01, 1.37, 0.3, 0.13, 0.25, 0.36, 0.24, 0.38, 0.69, 0.52, 0.33, 0.6, 0.83, 0.44999999999999996, 0.27, 0.16999999999999998, 1.11, 0.26, 1.47, 0.9299999999999999]
mean12_1000 = np.mean(SNR12_1000)
std12_1000 = np.std(SNR12_1000)
mean16_1000 = np.mean(SNR16_1000)
std16_1000 = np.std(SNR16_1000)
mean20_1000 = np.mean(SNR20_1000)
std20_1000 = np.std(SNR20_1000)
mean24_1000 = np.mean(SNR24_1000)
std24_1000 = np.std(SNR24_1000)
mean28_1000 = np.mean(SNR28_1000)
std28_1000 = np.std(SNR28_1000)
mean32_1000 = np.mean(SNR32_1000)
std32_1000 = np.std(SNR32_1000)
axisy_1000 = np.array([mean12_1000, mean16_1000, mean20_1000, mean24_1000, mean28_1000, mean32_1000])
errorbar_1000 = np.array([std12_1000, std16_1000, std20_1000, std24_1000, std28_1000, std32_1000])

SNR12_4000 = [15.07, 15.299999999999999, 12.24, 14.37, 13.04, 13.750000000000002, 13.819999999999999, 13.139999999999999, 14.66, 16.23, 13.65, 20.48, 12.93, 15.97, 13.089999999999998, 14.879999999999999, 13.96, 12.889999999999999, 13.04, 15.76]
SNR16_4000 = [4.31, 5.510000000000001, 3.7900000000000005, 4.35, 5.24, 4.12, 3.34, 7.16, 4.91, 4.78, 4.590000000000001, 4.18, 6.79, 4.45, 3.02, 4.78, 5.17, 4.68, 4.5, 6.79]
SNR20_4000 = [1.49, 1.6, 0.8200000000000001, 1.1199999999999999, 1.8599999999999999, 0.8999999999999999, 1.18, 1.1400000000000001, 1.28, 3.58, 1.51, 1.0, 1.79, 1.28, 2.44, 1.21, 1.31, 2.22, 1.3, 1.31]
SNR24_4000 = [1.52, 1.6199999999999999, 0.33, 0.76, 0.5599999999999999, 0.5700000000000001, 0.5, 0.49, 0.52, 0.47000000000000003, 0.25, 0.63, 6.909999999999999, 1.18, 0.8, 0.86, 0.9199999999999999, 0.88, 1.48, 0.63]
SNR28_4000 = [.27999999999999997, 0.73, 0.26, 0.37, 0.72, 0.27, 0.15, 0.5499999999999999, 0.33999999999999997, 0.29, 0.3, 0.33, 0.4, 0.24, 0.76, 0.5599999999999999, 0.36, 0.25, 0.43, 0.38999999999999996]
SNR32_4000 = [0.49, 0.22999999999999998, 0.37, 0.29, 0.66, 0.33, 0.27, 0.45999999999999996, 0.84, 0.16, 0.21, 0.13, 1.29, 0.52, 0.29, 0.31, 0.33, 0.7799999999999999, 0.98, 0.24]
mean12_4000 = np.mean(SNR12_4000)
std12_4000 = np.std(SNR12_4000)
mean16_4000 = np.mean(SNR16_4000)
std16_4000 = np.std(SNR16_4000)
mean20_4000 = np.mean(SNR20_4000)
std20_4000 = np.std(SNR20_4000)
mean24_4000 = np.mean(SNR24_4000)
std24_4000 = np.std(SNR24_4000)
mean28_4000 = np.mean(SNR28_4000)
std28_4000 = np.std(SNR28_4000)
mean32_4000 = np.mean(SNR32_4000)
std32_4000 = np.std(SNR32_4000)
axisy_4000 = np.array([mean12_4000, mean16_4000, mean20_4000, mean24_4000, mean28_4000, mean32_4000])
errorbar_4000 = np.array([std12_4000, std16_4000, std20_4000, std24_4000, std28_4000, std32_4000])

SNR12_6000 = [14.08, 12.1, 14.23, 14.2, 12.32, 17.05, 13.58, 14.21, 13.889999999999999, 14.11, 12.42, 12.21, 14.580000000000002, 12.29, 14.42, 13.87, 13.919999999999998, 13.719999999999999, 14.860000000000001, 13.15]
SNR16_6000 = [7.07, 5.71, 5.24, 5.82, 5.16, 5.16, 5.37, 4.2700000000000005, 4.43, 4.87, 6.35, 8.37, 4.77, 5.12, 4.84, 3.74, 3.93, 5.63, 4.12, 6.67]
SNR20_6000 = [1.49, 1.31, 0.88, 1.6, 1.48, 0.5599999999999999, 1.76, 1.16, 0.9900000000000001, 1.03, 1.15, 1.03, 1.63, 1.9300000000000002, 1.3, 0.95, 1.8499999999999999, 0.9900000000000001, 1.8800000000000001, 1.55]
SNR24_6000 = [0.97, 0.88, 0.64, 0.5599999999999999, 0.95, 0.66, 0.5, 1.11, 0.53, 0.45999999999999996, 0.37, 0.27999999999999997, 1.04, 0.98, 0.7000000000000001, 6.140000000000001, 1.69, 1.44, 0.44999999999999996, 0.44]
SNR28_6000 = [0.31, 0.19, 0.19, 0.63, 1.52, 0.16, 0.59, 0.66, 0.48, 1.1400000000000001, 0.22, 0.63, 0.37, 0.3, 0.29, 0.35000000000000003, 0.22999999999999998, 0.96, 0.24, 1.16]
SNR32_6000 = [0.1, 0.41000000000000003, 0.65, 0.24, 0.44, 0.66, 0.47000000000000003, 2.03, 0.54, 0.32, 0.69, 0.9299999999999999, 0.53, 0.61, 0.54, 0.42, 0.43, 0.69, 0.37, 0.29]
mean12_6000 = np.mean(SNR12_6000)
std12_6000 = np.std(SNR12_6000)
mean16_6000 = np.mean(SNR16_6000)
std16_6000 = np.std(SNR16_6000)
mean20_6000 = np.mean(SNR20_6000)
std20_6000 = np.std(SNR20_6000)
mean24_6000 = np.mean(SNR24_6000)
std24_6000 = np.std(SNR24_6000)
mean28_6000 = np.mean(SNR28_6000)
std28_6000 = np.std(SNR28_6000)
mean32_6000 = np.mean(SNR32_6000)
std32_6000 = np.std(SNR32_6000)
axisy_6000 = np.array([mean12_6000, mean16_6000, mean20_6000, mean24_6000, mean28_6000, mean32_6000])
errorbar_6000 = np.array([std12_6000, std16_6000, std20_6000, std24_6000, std28_6000, std32_6000])


def plot_tout():
    fig = plt.figure()
    fig.suptitle('Performance analysis')
    plt.xlabel('Signal-to-Noise Ratio (SNR, dB)')
    plt.ylabel('Symbol Error Rate (SER, %)')

    plt.errorbar(axisx, axisy_300, yerr=errorbar_300, fmt='grey', ecolor='grey')
    plt.plot(axisx, axisy_300, 'o', label="nTraining = 300", color='grey')

    plt.errorbar(axisx, axisy_1000, yerr=errorbar_1000, fmt='blue', ecolor='blue')
    plt.plot(axisx, axisy_1000, 'o', label="nTraining = 1000", color='blue')

    plt.errorbar(axisx, axisy, yerr=errorbar, fmt='red', ecolor='red')
    plt.plot(axisx, axisy, 'o', label="nTraining = 3000", color="red")

    plt.errorbar(axisx, axisy_4000, yerr=errorbar_4000, fmt='yellow', ecolor='yellow')
    plt.plot(axisx, axisy_4000, 'o', label="nTraining = 4000", color='yellow')

    plt.errorbar(axisx, axisy_6000, yerr=errorbar_6000, fmt='orange', ecolor='orange')
    plt.plot(axisx, axisy_6000, 'o', label="nTraining = 6000", color='orange')

    plt.legend(loc='upper right')
    plt.show()
#plot_tout()




