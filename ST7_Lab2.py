""" 
Eliott Dumont, ST7 Optimization for neuro-inspired computing with physical architecture, Lab2 2A CentraleSupélec

The aim of this labwork is to classify isolated spoken digit, using an Echo State Network. This is a widely used task in 
the field of speech recognition to benchmark the performance of reservoir computers. The database used is the NIST TI86, 
containing 500 audio files of 5 differents females speakers saying digits from 0 to 9.
"""

import random
import numpy as np
import matplotlib.pyplot as plt


"""Parameters"""

Nch = 86 # Number of filters associated with de cochleagram
Ntau = 130 # Number of samples
Nd = 10 # Number of digits
Nut = 50 # Number of utterances for each digit (database)



"""Database creation"""

# database[a][b] contains the cochlea of the a-th digit, b-th utterance. shape (Nch, Ntau)
database = np.zeros(shape=(Nd, Nut, Nch, Ntau))

for k in range(Nd):
    for j in range(1, Nut+1):
        name = "database_Lab2/Input_word_" + str(k) + "_utterance_" + str(j) + ".txt"
        with open(name, "r") as f:
            lines = f.readlines()
        data = np.empty((len(lines), Ntau))
        for i, line in enumerate(lines):
            values = line.strip().split(",")
            data[i, :] = np.array(values, dtype=float)
        database[k][j-1] = data

# Same with the targeted digit database
database_target = np.zeros(shape=(Nd, Nut, Nd, Ntau))

for k in range(Nd):
    for j in range(1, Nut+1):
        name = "database_Lab2/Target_word_" + str(k) + "_utterance_" + str(j) + ".txt"
        with open(name, "r") as f:
            lines = f.readlines()
        data = np.empty((len(lines), Ntau))
        for i, line in enumerate(lines):
            values = line.strip().split(",")
            data[i, :] = np.array(values, dtype=float)
        database_target[k][j-1] = data


"""Simulation of the reservoir"""

def reservoir(Nx):
        
    # Initialization 

    Wil = 2**np.random.rand(Nx, Nch) - 1 # Input layer connectivity matrix
    nb = 0
    while nb <= int(0.9*(Nx*Nch)): # This has to be a sparse matrix, 10% connectivity
        i = random.randint(0, Nx - 1)
        j = random.randint(0, Nch - 1)
        if Wil[i][j] != 0:
            Wil[i][j] = 0
            nb += 1

    Whli = 2*np.random.rand(Nx, Nx) - 1 # adjacency matrix of the RNN (W hidden layer intermediate)
    nb = 0
    while nb <= int(0.7*(Nx**2)): # This has to be a sparse matrix, 30% connectivity
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



    # Temporal evolution of the ESN
    def ESN(a, b):
        # We don't need to washout the reservoir as we reset it each time
        Xstate = np.zeros(shape=(Nx, 1)) 
        X = np.zeros(shape=(Nx, Ntau, 1))
        U = database[a][b]
        Mu = np.matmul(Wil, U)
        for n in range(Ntau):
            Xstate = np.tanh((np.matmul(Whl, Xstate) + Mu[:, n].reshape(Nx, 1)))
            X[:, n] = Xstate
        return X



    # Offline training of the ESN (Moore-Penrose pseudo-inverse)

    # We use the first 8 utterances for each speaker : 8x5 = 40 per digit
    nTraining = 400

    # This matrix will contains all the smaller matrices of states X
    Mx = np.zeros(shape=(Nx, nTraining*Ntau))
    nb=0
    for a in range(10):
        for bi in range(8):
            for bj in range(5):
                b = 10*bj + bi
                X = ESN(a, b).reshape(Nx, Ntau)
                Mx[:, nb:nb+Ntau] = X
                nb += Ntau
    # Same with Y
    My = np.zeros(shape=(Nd, nTraining*Ntau))
    nb=0
    for a in range(10):
        for bi in range(5):
            for bj in range(8):
                b = 10*bi + bj
                Y = database_target[a][b]
                My[:, nb:nb+Ntau] = Y
                nb += Ntau


    P1 = np.matmul(My, np.transpose(Mx))
    P2 = np.matmul(Mx, np.transpose(Mx))
    P3 = np.linalg.inv(P2)
    # This matrix (W output layer) is the solution of the least square optimization problem
    Wol = np.matmul(P1, P3) 



    # Testing

    # Mean method
    y_out = np.zeros(shape=(Nd, Nut))
    for i in range(Nd):
        for j in range(Nut):
            Xo = ESN(i, j).reshape(Nx, Ntau)
            My = np.matmul(Wol, Xo)
            L = np.zeros(shape=(Nd, 1))
            for t in range(Nd):
                L[t] = np.mean(My[t, :])
                m = np.argmax(L)
                y_out[i][j] = m
            #print("Digit to find : ", i, "digit found : ", m)

    # Max method
    y_out_max = np.zeros(shape=(Nd, Nut))
    for i in range(Nd):
        for j in range(Nut):
            Xo = ESN(i, j).reshape(Nx, Ntau)
            My = np.matmul(Wol, Xo)
            L = np.zeros(shape=(Nd, 1))
            for t in range(Nd):
                L[t] = np.max(My[t, :])
                m = np.argmax(L)
                y_out_max[i][j] = m
            #print("Digit to find : ", i, "digit found : ", m)
    return [y_out, y_out_max]


def WERplot():
    L = []
    L_max = []
    for k in range(10, 410, 10):
        nb_error = 0
        nb_error_max = 0
        y = reservoir(k)
        y_out, y_out_max = y 
        print('Nx = ', k)
        for i in range(Nd):
           for j in range(Nut):
               if y_out[i][j] != i:
                   nb_error += 1
               if y_out_max[i][j] != i:
                   nb_error_max += 1
        L.append(nb_error/400*100)
        L_max.append(nb_error_max/400*100)

    fig = plt.figure()
    fig.suptitle('Performance analysis')
    xaxis = [_ for _ in range(10, 410, 10)]
    plt.plot(xaxis, L, label='mean method')
    plt.plot(xaxis, L_max, label='max method')
    plt.xlabel('Number of neurons (Nx)')
    plt.ylabel('Word Error Rate (WER, %)')
    plt.legend(loc='upper right')
    plt.show()
#WERplot()

def calculAndPlot():
    datamean400 = [0.25, 0.0, 0.0, 0.0, 0.0]
    datasmax400 = [2.0, 2.25, 4.0, 2.0, 2.0]
    datamean350 = [0.0, 0.25, 0.0, 0.0, 0.0]
    datasmax350 = [3.5, 2.75, 3.75, 2.0, 3.25]
    datamean300 = [0.0, 0.0, 0.0, 0.25, 0.0]
    datasmax300 = [2.25, 3.0, 3.0, 2.75, 2.0]
    datamean250 = [0.5, 0.75, 0.25, 0.0, 0.25]
    datasmax250 = [4.75, 5.25, 5.0, 5.75, 4.75]
    datamean200 = [0.75, 0.25, 0.75, 0.25, 0.0]
    datasmax200 = [5.25, 8.75, 5.0, 7.0, 5.75]
    datamean150 = [2.5, 2.0, 2.5, 2.5, 1.75]
    datasmax150 = [10.75, 9.0, 7.5, 10.75, 10.5]
    datamean100 = [2.5, 3.25, 4.0, 3.5, 2.0]
    datasmax100 = [15.25, 13.75, 16.25, 15.5, 19.5]

    moymean400 = np.mean(datamean400)
    stdmean400 = np.std(datamean400)
    moymax400 = np.mean(datasmax400)
    stdmax400 = np.std(datasmax400)

    moymean350 = np.mean(datamean350)
    stdmean350 = np.std(datamean350)

    moymean300 = np.mean(datamean300)
    stdmean300 = np.std(datamean300)

    moymean250 = np.mean(datamean250)
    stdmean250 = np.std(datamean250)

    moymean200 = np.mean(datamean200)
    stdmean200 = np.std(datamean200)

    moymean150 = np.mean(datamean150)
    stdmean150 = np.std(datamean150)

    moymean100 = np.mean(datamean100)
    stdmean100 = np.std(datamean100)

    moymax350 = np.mean(datasmax350)
    stdmax350 = np.std(datasmax350)

    moymax300 = np.mean(datasmax300)
    stdmax300 = np.std(datasmax300)

    moymax250 = np.mean(datasmax250)
    stdmax250 = np.std(datasmax250)

    moymax200 = np.mean(datasmax200)
    stdmax200 = np.std(datasmax200)

    moymax150 = np.mean(datasmax150)
    stdmax150 = np.std(datasmax150)

    moymax100 = np.mean(datasmax100)
    stdmax100 = np.std(datasmax100)

    axisx = ['400/100', '350/150', '300/200', '250/250', '200/300', '150/350', '100/400']
    axisymean = [moymean400, moymean350, moymean300, moymean250, moymean200, moymean150, moymean100]
    errorbarmean = [stdmean400, stdmean350, stdmean300, stdmean250, stdmean200, stdmean150, stdmean100]
    axisymax = [moymax400, moymax350, moymax300, moymax250, moymax200, moymax150, moymax100]
    errorbarmax = [stdmax400, stdmax350, stdmax300, stdmax250, stdmax200, stdmax150, stdmax100]

    fig = plt.figure()
    fig.suptitle('Performance analysis')
    plt.xlabel('nTraining/nTest')
    plt.ylabel('Word Error Rate (WER, %)')

    plt.errorbar(axisx, axisymean, yerr=errorbarmean, fmt='blue', ecolor='blue')
    plt.plot(axisx, axisymean, 'o', label="mean method", color='blue')

    plt.errorbar(axisx, axisymax, yerr=errorbarmax, fmt='red', ecolor='red')
    plt.plot(axisx, axisymax, 'o', label="max method", color='red')


    plt.legend(loc='upper left')
    plt.show()

def crossValidation():
    cycle2mean = [0.0, 0.25, 0.0, 0.0, 0.25]
    cycle2max = [3.75, 3.0, 3.75, 3.75, 1.25]

    cycle1mean = [0.25, 0.0, 0.0, 0.0, 0.0]
    cycle1max = [2.0, 2.25, 4.0, 2.0, 2.0]

    cycle3mean = [0.0, 0.0, 0.0, 0.25, 0.25]
    cycle3max = [2.5, 2.25, 2.75, 2.0, 2.5]

    cycle4mean = [0.0, 0.0, 0.0, 0.25, 0.25]
    cycle4max = [4.25, 3.5000000000000004, 1.7500000000000002, 4.0, 3.5000000000000004]

    cycle5mean = [0.0, 0.0, 0.0, 0.0, 0.0]
    cycle5max = [2.0, 1.5, 3.25, 3.25, 1.0]

    moymean1 = np.mean(cycle1mean)
    stdmean1 = np.std(cycle1mean)

    moymax1 = np.mean(cycle1max)
    stdmax1 = np.std(cycle1max)

    moymean2 = np.mean(cycle2mean)
    stdmean2 = np.std(cycle2mean)

    moymax2 = np.mean(cycle2max)
    stdmax2 = np.std(cycle2max)

    moymean3 = np.mean(cycle3mean)
    stdmean3 = np.std(cycle3mean)

    moymax3 = np.mean(cycle3max)
    stdmax3 = np.std(cycle3max)

    moymean4 = np.mean(cycle4mean)
    stdmean4 = np.std(cycle4mean)

    moymax4 = np.mean(cycle4max)
    stdmax4 = np.std(cycle4max)

    moymean5 = np.mean(cycle5mean)
    stdmean5 = np.std(cycle5mean)

    moymax5 = np.mean(cycle5max)
    stdmax5 = np.std(cycle5max)

    axisx = ['Cycle 1', 'Cycle 2', 'Cycle 3', 'Cycle 4', 'Cycle 5']
    axisymean = [moymean1, moymean2, moymean3, moymean4, moymean5]
    errorbarmean = [stdmean1, stdmean2, stdmean3, stdmean4, stdmean5]
    axisymax = [moymax1, moymax2, moymax3, moymax4, moymax5]
    errorbarmax = [stdmax1, stdmax2, stdmax3, stdmax4, stdmax5]

    fig = plt.figure()
    fig.suptitle('Performance analysis')
    plt.xlabel('Cycle')
    plt.ylabel('Word Error Rate (WER, %)')

    plt.errorbar(axisx, axisymean, yerr=errorbarmean, fmt='blue', ecolor='blue')
    plt.plot(axisx, axisymean, 'o', label="mean method", color='blue')

    plt.errorbar(axisx, axisymax, yerr=errorbarmax, fmt='red', ecolor='red')
    plt.plot(axisx, axisymax, 'o', label="max method", color='red')


    plt.legend(loc='upper left')
    plt.show()