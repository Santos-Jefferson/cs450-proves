import random
import numpy as np
from sklearn import datasets
import math

# 1 - Create some type of data structure to hold a node (a.k.a. neuron).
# Dataset (Features or Inputs)
X = np.array([ [.2,.3,.4],
               [.4,.5,.6],
               [.1,.1,.4],
             ])

# 6 - Be able to load and process at least the following two datasets:
#      a) Iris (You didn't think we could leave this out did you!)
#      b) Pima Indian Diabetes
diabetes = datasets.load_diabetes()
iris = datasets.load_iris()
# X = diabetes.data
# X = iris.data

# 7 - You should appropriately normalize each data set.
# COMING SOON


# Dataset (Class or results)
y = np.array([[0,1,0]]).T


# 2 - Store the set of input weights for each node.
def store_weights(num_cols, num_nodes_hidden):
    # num_weights = num_nodes * (num_cols + 1)
    # weights = []
    # weights = np.full((num_cols, num_nodes+1),random.random())
    weights = np.zeros((num_cols, num_nodes_hidden))
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i][j] = random.uniform(-0.5,0.5)
    # print("def weights")
    # print(weights)
    # for i in range(0,num_weights):
    #     weights.arrange(random.uniform(-1,1))
    return weights

# 3 - Provide a way to create a layer of nodes of any number (this should
#   be easily specified via a parameter).
def create_layer(num_nodes_hidden):
    nodes = []
    for i in range(num_nodes_hidden+1):
        nodes = [0] * i
    return nodes


# 4 - Account for a bias input.
bias_input = -1
bias_hidden = -1


# 5 - Be able to take input from a dataset instance (with an arbitrary number
#   of attributes) and have each node produce an output (i.e., 0 or 1)
#   according to its weights.
def calculate_nodes_hidden(X, num_nodes_hidden, weights_input, bias):
    nodes = []
    X = np.insert(X, 0, values=bias, axis=1)  # adding bias (-1) to inputs
    for i in range(1):
        for j in range(num_nodes_hidden):
            temp = 0
            for k in range(X.shape[1]):
                temp += weights_input[k][j] * X[i][k]
            nodes.append(temp)
    return nodes

def calculate_nodes_output(nodes, weights_hidden, bias):
    nodes_hidden = []
    nodes.insert(0, bias)
    for i in range(1):
        for j in range(len(nodes)-1):
            temp = 0
            for k in range(len(nodes)):
                temp += weights_hidden[k][j] * nodes[k]
            nodes_hidden.append(temp)
    nodes.pop(0)
    return nodes_hidden



def calculate_feed_forward(value_nodes, num_nodes):
    feed_forward_hidden = []
    temp = 0.0
    for i in range(num_nodes):
        temp = 1/(1 +((math.pow(math.e,-value_nodes[i]))))
        feed_forward_hidden.append(temp)
    return feed_forward_hidden


def predict(feed_for_output):
    predict = 0
    for i in range(len(feed_for_output)-1):
        if feed_for_output[i] > feed_for_output[i+1]:
            predict = 1
        else:
            predict = 0
    return predict


# Main() Function
def main():
    # Get the number os cols of the inputs of dataset
    if bias_input:
        num_cols = X.shape[1]+1
    else:
        num_cols = X.shape[1]

    # Define the number os nodes, layers, outputs
    num_nodes_hidden = 2
    num_nodes_output = 2
    num_layers_hidden = 1
    num_layers_output = 1


    # Creating the layer with N nodes (num_nodes_hidden) with values 0.0
    create_layer(num_nodes_hidden)

    # Just to see the num of cols
    print("Num of Cols of X dataset")
    print(num_cols)
    print()

    # Just to see the array with weights accordingly num_cols of X
    weights_input = store_weights(num_cols, num_nodes_hidden)
    weights_hidden = store_weights(num_nodes_hidden+1, num_nodes_output) # num_nodes_hidden+1 because the bias (-1)
    print("weights_input")
    print(weights_input)
    print()

    print("weights_hidden")
    print(weights_hidden)
    print()


    # print(weights)
    print()


    # Creating Nodes with specific number
    nodes = create_layer(num_nodes_hidden)

    # Calculating the final result for Nodes (0 or 1)
    nodes_calc_hidden = calculate_nodes_hidden(X, num_nodes_hidden, weights_input, bias_input)
    nodes_calc_output=calculate_nodes_output(nodes_calc_hidden, weights_hidden, bias_hidden)
    feed_forward_hidden = calculate_feed_forward(nodes_calc_hidden, num_nodes_hidden)
    feed_forward_output = calculate_feed_forward(nodes_calc_output, num_nodes_output)

    # Just to see the contents of Nodes calculated
    print("Values of Nodes Hidden")
    print(nodes_calc_hidden)
    print()

    print("Values of Nodes Output")
    print(nodes_calc_output)
    print()

    print("Feed Forward Hidden")
    print(feed_forward_hidden)
    print()

    print("Feed Forward Output")
    print(feed_forward_output)
    print()

    result = predict(feed_forward_output)
    print("The correct target is: ",y[0])
    print("The result predicted is: [",result,"]")


# Call main() function
if __name__ == "__main__":
    main()
