# PART 1
from collections import Counter
from operator import itemgetter
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import math


iris = datasets.load_iris()


# PART 2
Xdata = iris.data
ytarget = iris.target

# Using train_test_split (30% test, 70% train)
data_train, data_test, targets_train, targets_test = train_test_split(Xdata, ytarget, test_size=0.3)
# print(data_train.shape, targets_train.shape)

# reformat train/test datasets for convenience
train = np.array(list(zip(data_train, targets_train)))
test = np.array(list(zip(data_test, targets_test)))

# Array of predictions
arrayPredictions = []


# number of K
k = 5


def get_neighbours(training_set, test_instance, k):
    distances = [_get_tuple_distance(training_instance, test_instance) for training_instance in training_set]
    # index 1 is the calculated distance between training_instance and test_instance
    sorted_distances = sorted(distances, key=itemgetter(1))
    # extract only training instances
    sorted_training_instances = [tuple[0] for tuple in sorted_distances]
    # select first k elements
    return sorted_training_instances[:k]


def _get_tuple_distance(training_instance, test_instance):
    return (training_instance, get_distance(test_instance, training_instance[0]))


def get_distance(data1, data2):
    points = list(zip(data1, data2))
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))


def get_majority_vote(neighbours):
    # index 1 is the class
    classes = [neighbour[1] for neighbour in neighbours]
    count = Counter(classes)
    return count.most_common()[0][0]


for i in range(len(data_test)):
    neighbours = get_neighbours(training_set=train, test_instance=test[i][0], k=5)
    majority_vote = get_majority_vote(neighbours)
    arrayPredictions.append(majority_vote)


# Implementing KNeighbors
k_range = range(1,4)
scores = []
for k in k_range:
    classifier = KNeighborsClassifier(n_neighbors=k)
    model = classifier.fit(data_train, targets_train)
    predictions = model.predict(data_test)
    scores.append(metrics.accuracy_score(targets_test, predictions))
    print();
    print("Printing KNeighbors K= {} ".format(k))
    comparisonKNNAlgo = metrics.accuracy_score(targets_test, predictions)
    comparisonMineAlgo = metrics.accuracy_score(targets_test, arrayPredictions)
    
    print("KNN Algo  = {:.3f}%".format(comparisonKNNAlgo * 100))
    print("Mine Algo = {:.3f}%".format(comparisonMineAlgo * 100))

print()
print("ComparisonKNNAlgo : " + str(comparisonKNNAlgo))
print("ComparisonMineAlgo : " + str(comparisonMineAlgo))


# Generating graphs to compare the best K number to use
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy for KNN in %')
plt.show()