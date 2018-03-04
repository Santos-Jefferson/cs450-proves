from math import log
import operator
import pandas as pd

class MineId3Algo:
    def __init__(self):
        return

    def entropy_calc(self, data):
        # how many lines the data has
        entries=len(data)
        labels={}
        for feat in data:
            label=feat[-1]
            if label not in labels.keys():
                labels[label]=0
                labels[label]+=1
            entropy=0.0
            for key in labels:
                probability=float(labels[key]) / entries
                entropy-=probability * log(probability, 2)
            return entropy, labels

    def split(self, data, axis, val):
        newData=[]
        for feat in data:
            if feat[axis] == val:
                reducedFeat=feat[:axis]
                reducedFeat.extend(feat[axis + 1:])
                newData.append(reducedFeat)
        return newData


    def choose(self, data):
        features=len(data.loc[0]) - 1
        baseEntropoy=self.entropy_calc(data)
        bestInfoGain=0.0;
        bestFeat=-1
        for i in range(features):
            featList=[ex[i] for ex in data]
            uniqueVals=set(featList)
            newEntropy=0.0
            for value in uniqueVals:
                newData=self.split(data, i, value)
                probability=len(newData) / float(len(data))
                newEntropy+=probability * self.entropy_calc(newData)
            infoGain=baseEntropoy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain=infoGain
                bestFeat=i
        return bestFeat


    def majority(classList):
        classCount={}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote]=0
            classCount[vote]+=1
        sortedClassCount=sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


    def tree(self, data, labels):
        classList=[ex[-1] for ex in data]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(data.loc[0]) == 1:
            return self.majority(classList)
        bestFeat=self.choose(data)
        bestFeatLabel=labels[bestFeat]
        theTree={bestFeatLabel: {}}
        del (labels[bestFeat])
        featValues=[ex[bestFeat] for ex in data]
        uniqueVals=set(featValues)
        for value in uniqueVals:
            subLabels=labels[:]
            theTree[bestFeatLabel][value]=self.tree(self.split(data, bestFeat, value), subLabels)
            return theTree


def main():
    headers_voting=["handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution",
                 "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
                 "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback",
                 "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports",
                 "export-administration-act-south-africa","class result"]
    datavoting = pd.read_csv("house-votes-84.data.csv", header=None, names=headers_voting)

    print(datavoting)

    # print(len(datavoting.loc[0]))
    mineId3 = MineId3Algo()
    labels = mineId3.entropy_calc(datavoting)[1]
    mineId3.tree(datavoting, labels)

if __name__ == "__main__":
    main()
