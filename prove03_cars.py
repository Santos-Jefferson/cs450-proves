import matplotlib
import numpy
import numpy as np
import pandas as pd
from collections import Counter
from operator import itemgetter

from jedi.refactoring import inline
from sklearn import metrics, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import math


# Creating  the Headers
headers_cars = ["buying", "maint", "doors", "persons", "lug_boot",
                "safety", "class"]

# headers_diabetes = ["num_times", "plasma_glu_conc", "diastolic",
#                     "triceps", "2-hour", "body_mass", "diabetes_ped",
#                     "age", "class"]

headers_mpg = ["class", "cylinders", "displacement", "horsepower",
               "weight", "acceleration", "model_year", "origin",
               "car_name"]

dfcars = pd.read_csv("car.data.csv", header=None, names=headers_cars,
                 na_values="NaN")

dfdiabetes = pd.read_csv("pima-indians-diabetes.data.csv", header=None,
                 na_values="?")

# CONVERTING THE FILE TO CSV FIXED
dfmpg = pd.read_csv("auto-mpg.data.csv", delim_whitespace=True, names=headers_mpg)
horsepower_missing_ind = dfmpg[dfmpg.horsepower=='?'].index
dfmpg.loc[horsepower_missing_ind, 'horsepower'] = float('nan')
dfmpg.horsepower = dfmpg.horsepower.apply(pd.to_numeric)
dfmpg.loc[horsepower_missing_ind, 'horsepower'] = int( dfmpg.horsepower.mean() )
pd.set_option('precision', 2)
dfmpg.to_csv('auto-mpg.data.fix.csv', index=False)

# STARTING AGAIN WITH CSV FILE FIXED
dfmpgcsv = pd.read_csv('auto-mpg.data.fix.csv')
# print(dfmpgcsv.head(40))
# print(dfmpgcsv.tail())
# print(dfmpgcsv.shape)

# # PLOT TO SEE THE LINES AND GRAPHS
# sns.set(color_codes=True)
# sns.pairplot(dfmpgcsv,
#              x_vars=['cylinders', 'displacement', 'horsepower',
#                      'weight', 'acceleration', 'model_year', 'origin'
#                      ],
#              y_vars='class', kind='reg')
# plt.show()

feature_cols = ['cylinders', 'displacement', 'horsepower',
                     'weight', 'acceleration', 'model_year', 'origin']
X = dfmpgcsv[feature_cols]
# print(X.head())
# print(type(X))
# print(X.shape)

y = dfmpgcsv['class']
# print(y.head())
# print(type(y))
# print(y.shape)

# DFCARS
obj_dfcars = dfcars.select_dtypes(include=['object']).copy()
class_cars = {"class":{"unacc": 1, "acc": 2, "good": 3, "vgood": 4}}
obj_dfcars.replace(class_cars, inplace=True)
obj_dfcars = pd.get_dummies(obj_dfcars,
                columns=["buying", "maint", "doors", "persons",
                         "lug_boot", "safety"],
                prefix=["buying", "maint", "doors", "persons",
                        "lug_boot", "safety"])
np_cars = np.array(obj_dfcars)
Xdata = np_cars[:,1:23]
ytarget = np_cars[:,0]


# DFDIABETES
dfdiabetes[[1,2,3,4,5]] = dfdiabetes[[1,2,3,4,5]].replace(0, numpy.NaN)
dfdiabetes.fillna(dfdiabetes.mean(), inplace=True)
# print("null values")
# print(dfdiabetes.isnull().sum())
np_diabetes = np.array(dfdiabetes)
XdataDiabetes = np_diabetes[:,0:8]
ytargetDiabetes = np_diabetes[:,8]


# Using train_test_split (30% test, 70% train)
# DATA TO CARS DATASET
data_train, data_test, targets_train,targets_test = train_test_split(Xdata, ytarget,
                                                                   test_size=0.30)

# DATA TO DIABETES DATASET
# data_train, data_test, targets_train, targets_test = train_test_split(XdataDiabetes, ytargetDiabetes,
#                                                                    test_size=0.30)

# DATA TO DFMPGCSV DATASET
# data_train, data_test, targets_train, targets_test = train_test_split(X, y, random_state=1)
# print(data_train.shape)
# print(data_test.shape)
# print(targets_train.shape)
# print(targets_test.shape)

# linreg = LinearRegression()
# linreg.fit(data_train, targets_train)
# zip(feature_cols, linreg.coef_)
# y_pred = linreg.predict(data_test)
# true = [100, 50, 30, 20]
# pred = [90, 50, 50, 30]
# print(metrics.mean_absolute_error(true, pred))
# print(metrics.mean_squared_error(true, pred))
# print(np.sqrt(metrics.mean_squared_error(true, pred)))
# print("RMSE number")
# print(np.sqrt(metrics.mean_squared_error(targets_test, y_pred)))


# # reformat train/test datasets for convenience
train = np.array(list(zip(data_train, targets_train)))
test = np.array(list(zip(data_test, targets_test)))


def get_neighbours(training_set: object, test_instance: object, k: object) -> object:
    distances=[_get_tuple_distance(training_instance, test_instance) for
               training_instance in training_set]
    # index 1 is the calculated distance between training_instance and test_instance
    sorted_distances=sorted(distances, key=itemgetter(1))
    # extract only training instances
    sorted_training_instances=[tuple[0] for tuple in sorted_distances]
    # select first k elements
    return sorted_training_instances[:k]
#
#
def _get_tuple_distance(training_instance, test_instance):
    return (training_instance, get_distance(test_instance, training_instance[0]))
#
#
# returns sqrt (distance)
def get_distance(data1, data2):
    points=list(zip(data1, data2))
    diffs_squared_distance=[pow(a - b, 2) for (a, b) in points]
    return math.sqrt(sum(diffs_squared_distance))
#
#
def get_majority_vote(neighbours):
    # index 1 is the class
    classes=[neighbour[1] for neighbour in neighbours]
    count=Counter(classes)
    return count.most_common()[0][0]
#
#
# # Implementing KNeighbors
k_range=range(6, 14)
scoresKnn=[]
scoresMine=[]
scoresCrossVal=[]


for k in k_range:
    arrayPredictions=[]
    for i in range(len(data_test)):
        neighbours=get_neighbours(training_set=train, test_instance=test[i][0], k=k)
        majority_vote=get_majority_vote(neighbours)
        arrayPredictions.append(majority_vote)


    classifier=KNeighborsClassifier(n_neighbors=k)
    # cross validation
    scoresCrossKnn = cross_val_score(classifier, Xdata, ytarget, cv=10, scoring='accuracy')
    model=classifier.fit(data_train, targets_train)
    predictions=model.predict(data_test)
    scoresKnn.append(metrics.accuracy_score(targets_test, predictions))
    scoresMine.append(metrics.accuracy_score(targets_test, arrayPredictions))
    #scoresCrossMineKnn.append(metrics.accuracy_score(targets_test, scoresCrossMineKnn))
    scoresCrossVal.append(scoresCrossKnn.mean())

    print();
    print("Printing KNeighbors K= {} ".format(k))
    comparisonKNNAlgo=metrics.accuracy_score(targets_test, predictions)
    comparisonMineAlgo=metrics.accuracy_score(targets_test, arrayPredictions)

    print("KNN Algo  = {:.3f}%".format(comparisonKNNAlgo * 100))
    print("Mine Algo = {:.3f}%".format(comparisonMineAlgo * 100))
    print("Mine Cross = {:.3f}%".format(scoresCrossKnn.mean() * 100))
    del arrayPredictions[:]


# Generating graphs to compare the best K number to use
plt.plot()
plt.plot(k_range, scoresKnn, scoresMine, scoresCrossVal)
plt.xlabel('Value of K for KNN and Mine')
plt.ylabel('Testing Accuracy for KNN and Mine in %')
plt.show()

