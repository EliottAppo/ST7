import numpy as np
import matplotlib.pyplot as plt
import random 



# Load data from file
data = np.loadtxt('data.txt')
N = len(data)


# ESN
def ESN(Nx, input_scale, spectral_rad, connectivity, ridge, size_data):
    random.seed(788)
    np.random.seed(788)
    # Split data into training and testing arrays
    
    training_set = data[:size_data]
    testing_set = data[size_data:]


    Wil = 2*np.random.rand(Nx, 1) - 1 # Input layer connectivity matrix
    r = np.amax(np.abs(np.linalg.svd(Wil)[1]))
    Wil = input_scale * Wil / r

    Whli = 2*np.random.rand(Nx, Nx) - 1 
    nb = 0
    while nb <= int(connectivity*(Nx**2)): 
        i = random.randint(0, Nx - 1)
        j = random.randint(0, Nx - 1)
        if Whli[i][j] != 0:
            Whli[i][j] = 0
            nb += 1

    v = np.linalg.eigvals(Whli)
    rho = v[0]
    for x in v:
        if np.abs(x) > rho:
            rho = np.abs(x)
    Whl = (spectral_rad * Whli / rho).real

    Xstate = np.zeros(shape=(Nx, 1))
    X = np.zeros(shape=(Nx, N, 1))
    output = np.zeros(shape=(1, N))


    # Temporal evolution of the ESN
    for t in range(size_data-1): # Following the evolution process
        Xstate = np.tanh(np.matmul(Whl, Xstate) + training_set[t]*Wil)  # va de 0 à sizedata-1
        X[:, t] = Xstate
    
  
    endOfTrain = size_data
    X = X.reshape(Nx, N)
    Xt = X[:, : endOfTrain - 1]
    
    training_set = np.array(training_set).reshape(size_data, 1)

    Y = np.transpose(np.array([training_set[t+1] for t in range(size_data-1)])) # va de 1 à sizedata
   

    P1 = np.matmul(Y, np.transpose(Xt))
    P2 = np.matmul(Xt, np.transpose(Xt)) + ridge*np.eye(Nx)
    P3 = np.linalg.inv(P2)
    Wol = np.matmul(P1, P3)


    # Testing 
 
    Xstate = np.zeros(shape=(Nx, 1))

    for t in range(1, N - size_data):
        Xstate = np.tanh(np.matmul(Whl, Xstate) + testing_set[t]*Wil)
        output[:, endOfTrain + t] = np.matmul(Wol, Xstate)
    
    
    output = np.transpose(output).reshape((N))

    MSE = np.mean((data[100 + size_data:] - output[100 + size_data:]) ** 2)
    MSE_old = np.mean((data[size_data:] - output[size_data:]) ** 2)

    cost_fct = np.linalg.norm(output[size_data:] - data[size_data:], 2)**2 + ridge*np.linalg.norm(Wol, 2)**2

    return output, data, MSE, cost_fct, MSE_old


'''
listedelamort = []
listedelamort2 = []
array = np.arange(100, 19990, 1000)

for i in array:
    output, testing_set, error, cost, olderror= ESN(50, 0.1, 1e-5, 0.730, 1e-5, i)
    listedelamort.append(error)
    listedelamort2.append(olderror)

plt.plot(array, listedelamort, label='with washout')
plt.plot(array, listedelamort2, label = 'without washout')
plt.xlabel('size_training')
plt.ylabel('MSE')
plt.legend()
plt.show()
'''

'''
output, testing_set, error, cost = ESN(50, 0.242, 1e-5, 0.015, 1e-5, 10000)
print(error)
plt.plot(output, label='Output')
plt.plot(testing_set, label='Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
'''

'''
param1 = np.arange(0.000001, 0.0005, 0.00001)
costList = []
for p in param1:
    output, testing_set, error, cost = ESN(50, 0.1, 0.00001, 0.730, p, 10000)
    costList.append(error)

plt.plot(param1, costList)
plt.xlabel('Ridge parameter')
plt.ylabel('MSE')
plt.legend()
plt.show()
'''


def ESN_online(Nx, input_scale, spectral_rad, connectivity, ridge, size_data, learning_rate):
    random.seed(788)
    np.random.seed(788)
    # Load data from file
    data = np.loadtxt('data.txt')
    N = len(data)

    # Split data into training and testing arrays
    training_set = data[:size_data]
    testing_set = data[size_data:]

    # Initialize weights
    Wil = 2*np.random.rand(Nx, 1) - 1 # Input layer connectivity matrix
    r = np.amax(np.abs(np.linalg.svd(Wil)[1]))
    Wil = input_scale * Wil / r

    Whli = 2*np.random.rand(Nx, Nx) - 1 
    nb = 0
    while nb <= int(connectivity*(Nx**2)): 
        i = random.randint(0, Nx - 1)
        j = random.randint(0, Nx - 1)
        if Whli[i][j] != 0:
            Whli[i][j] = 0
            nb += 1

    v = np.linalg.eigvals(Whli)
    rho = v[0]
    for x in v:
        if np.abs(x) > rho:
            rho = np.abs(x)
    Whl = (spectral_rad * Whli / rho).real

    Xstate = np.zeros(shape=(Nx, 1))
    X = np.zeros(shape=(Nx, N, 1))
    output = np.zeros(shape=(1, N))
    Wol = np.random.rand(1, Nx) # Output layer weights

    # Temporal evolution of the ESN with online learning
    for t in range(size_data-1):
        Xstate = np.tanh(np.matmul(Whl, Xstate) + training_set[t]*Wil)
        Xt = Xstate.reshape((1, Nx))
        Yt = np.array([training_set[t+1]])

        # Update weights using batch gradient descent
        output_t = np.matmul(Wol, Xt.T)
        error_t = Yt - output_t
        grad_t = -2 * Xt * error_t
        Wol -= learning_rate * grad_t

    # Testing 
    Xstate = np.zeros(shape=(Nx, 1))
    output = np.zeros(shape=(1, N))

    for t in range(1, N - size_data):
        Xstate = np.tanh(np.matmul(Whl, Xstate) + testing_set[t]*Wil)
        Xt = Xstate.reshape((1, Nx))

        # Compute output using updated weights
        output_t = np.matmul(Wol, Xt.T)
        output[:, size_data + t] = output_t

    output = np.transpose(output).reshape((N))

    MSE = np.mean((data[100 + size_data:] - output[100 + size_data:]) ** 2)
    MSE_old = np.mean((data[size_data:] - output[size_data:]) ** 2)

    cost_fct = np.linalg.norm(output[size_data:] - data[size_data:], 2)**2 + ridge*np.linalg.norm(Wol, 2)**2

    return output, data, MSE, cost_fct, MSE_old

#optimize bGradientD
'''
learningR = np.arange(0.1, 1, 0.01)
errorList = []
for l in learningR:
    output, data, error, cost, old_error = ESN_online(50, 0.1, 1e-5, 0.730, 1e-5, 3000, l)
    errorList.append(error)

plt.plot(learningR, errorList)
plt.xlabel('learning rate')
plt.ylabel('MSE')
plt.legend()
plt.show()'''




'''
listedelamort = []
listedelamort2 = []
array = np.arange(7000, 13000, 100)

for i in array:
    output, data, error, cost, old_error = ESN_online(50, 0.1, 1e-5, 0.730, 1e-5, i, 0.556)
    output2, data2, error2, cost2, old_error2 = ESN(50, 0.1, 1e-5, 0.730, 1e-5, i)
    listedelamort.append(error)
    listedelamort2.append(error2)

plt.plot(array, listedelamort, label='with batch gradient descent')
plt.plot(array, listedelamort2, label = 'with offline training')
plt.xlabel('size_training')
plt.ylabel('MSE')
plt.legend()
plt.show()'''


output, data, error, cost, old_error = ESN_online(50, 0.1, 1e-5, 0.730, 1e-5, 10000, 0.055)
output2, data2, error2, cost2, old_error2 = ESN(50, 0.1, 1e-5, 0.730, 1e-5, 10000)
print(error)
print(old_error2)

