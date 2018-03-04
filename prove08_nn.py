import numpy as np
import pandas as pd

# sigmoid function
def nonlin(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))

# input data
# diab = datasets.load_diabetes()
diab = pd.read_csv("pima-indians-diabetes.data.csv", header=None)
# diab[[1,2,3,4,5]] = diab[[1,2,3,4,5]].replace(0, np.NaN)
# diab.fillna(diab.mean(), inplace=True)
values = diab.values
X = values[:,0:8]
y = values[:,8]
print(y)

# X = np.array([[0,0,1],
#               [0,1,1],
#               [1,0,1],
#               [0,0,0],
#               [1,1,1]])
#
# y = np.array([[0],
#               [1],
#               [1],
#               [0],
#               [0]])

# seed
np.random.seed(1);

# weights
# (3,4) - 3 nodes for 4 outputs
# (4,1) - 4 nodes for 1 output
syn0 = 2*np.random.random((8,9)) - 1 # -1 is bias
syn1 = 2*np.random.random((9,768)) - 1

# training
for j in range(100000):
    # layers (1° input, 2° hidden, 3° output)
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    #backpropagation
    l2_error = y - l2
    if (j % 1000) == 0:
        print("Error: " + str(np.mean(np.abs(l2_error))))

    #calculate deltas
    l2_delta = l2_error*nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    #updating weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Output after training")
# l2 = list(l2)
print(l2)

