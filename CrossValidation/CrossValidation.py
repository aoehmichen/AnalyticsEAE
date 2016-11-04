import random
import sys
import numpy

from operator import add
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionWithLBFGS, SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LassoWithSGD
from pyspark.mllib.recommendation import ALS

from ALSBio import Rating
from PathwayEnrichment import PathwayEnrichment

__author__ = 'Axel Oehmichen'


#####################################
# Functions                         #
#####################################

def turnToList(line):
    values = [float(x) for x in line.split(' ')]
    return values

def str_to_bool(s):
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])


# We remove the features at the specified position
def removeFeatureFromParsedData(labeledPoint, positionsToRemove):
    labeledPoint.features = numpy.delete(labeledPoint.features, positionsToRemove)
    return LabeledPoint(labeledPoint.label, labeledPoint.features)


# We remove the features at the specified position
def removeFeatureFromList(line, positionsToRemove):
    newList = numpy.delete(line, positionsToRemove)
    return newList


def uniquify(seq, idfun=None):
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result


# Remove the features from the dataset (RFE)
def removeFeatures(TrainingSet, ValidatingSet, featuresList, ratioOfFeaturesToremove):
    numberOfFeatures = len(featuresList)
    numberOfFeaturesToRemove = ratioOfFeaturesToremove
    if ratioOfFeaturesToremove < 1:
        numberOfFeaturesToRemove = int(round(len(featuresList) * ratioOfFeaturesToremove, ndigits=0))
        if numberOfFeaturesToRemove == 0:
            numberOfFeaturesToRemove = 1

    positionsToRemove = numpy.random.randint(0, high=numberOfFeatures, size=numberOfFeaturesToRemove)

    positionsToRemove = numpy.sort(positionsToRemove, -1, 'mergesort')
    positionsToRemoveDistinct = uniquify(positionsToRemove)
    positionsInv = positionsToRemoveDistinct[::-1]

    for x in positionsInv:
        if x < len(featuresList):
            featuresList.pop(x)

    parsedData = TrainingSet.map(lambda labeledPoint: removeFeatureFromParsedData(labeledPoint, positionsInv))
    parsedValidatingSet = ValidatingSet.map(
        lambda labeledPoint: removeFeatureFromParsedData(labeledPoint, positionsInv))

    featuresListLength = len(featuresList)
    assert featuresListLength == len(parsedData.first().features)
    assert featuresListLength == len(parsedValidatingSet.first().features)

    return parsedData, parsedValidatingSet, featuresList, featuresListLength


#####################################
# Split the data                    #
#####################################

def split_ALS(data, trainingSetSize, testSetSize, randomSeed):
    CVTest, CVTraining = data.randomSplit([trainingSetSize, testSetSize], randomSeed)
    return CVTest, CVTraining


def split_LinearMethods(data, trainingSetSize, testSetSize, randomSeed):
    # data = data.map(parsePoint)
    CVTest, CVTraining = data.randomSplit([trainingSetSize, testSetSize], randomSeed)
    return CVTest, CVTraining


#####################################
# Test the fitness of the models    #
#####################################

def fitness_ALS(ValidatingSet, models):
    # evaluate the ALS model
    # Evaluate the model on training data
    validationRatings = ValidatingSet.map(lambda l: l.split(',')).map(
        lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    testdata = validationRatings.map(lambda p: (p[0], p[1]))
    for model in models:
        predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
        ratesAndPreds = validationRatings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
        MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).reduce(lambda x, y: x + y) / ratesAndPreds.count()
        print("Mean Squared Error = " + str(MSE))
    return 0


def fitness_SVM(ValidatingSet, model):
    # evaluate the SVM model
    labelsAndPreds = ValidatingSet.map(lambda p: (p.label, model.predict(p.features)))
    countLabels = labelsAndPreds.map(lambda (v, p): 1 if v != p else 0).reduce(add)
    # countLabels = labelsAndPreds.filter(lambda (v, p): True if (v != p) else False).count()
    totalCount = float(ValidatingSet.count())
    trainErr = countLabels / totalCount
    return trainErr


def fitness_LogisticRegressionWithLBFGS(ValidatingSet, models):
    # Evaluate the model on training data
    # parsedData = ValidatingSet.map(parsePoint)
    parsedData = ValidatingSet
    for model in models:
        labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
        trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
        print("Training Error = " + str(trainErr))
    return 0


def fitness_LogisticRegressionWithSGD(ValidatingSet, models):
    # Evaluate the model on training data
    # parsedData = ValidatingSet.map(parsePoint)
    parsedData = ValidatingSet
    for model in models:
        labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
        trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
        print("Training Error = " + str(trainErr))
    return 0


def fitness_LinearRegressionWithSGD(ValidatingSet, models):
    # Evaluate the model on training data
    # parsedData = ValidatingSet.map(parsePoint)
    parsedData = ValidatingSet
    for model in models:
        valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
        MSE = valuesAndPreds.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
        print("Training Error = " + str(MSE))
    return 0


def fitness_LassoWithSGD(ValidatingSet, models):
    # Evaluate the model on training data
    # parsedData = ValidatingSet.map(parsePoint)
    parsedData = ValidatingSet
    for model in models:
        valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
        MSE = valuesAndPreds.map(lambda (v, p): (v - p) ** 2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
        print("Training Error = " + str(MSE))
    return 0


#####################################
# Generate the models               #
#####################################

def do_ALS(TrainingSet, ValidatingSet, featuresList, numberOfFeaturesToremove):
    # Build the recommendation model using Alternating Least Squares
    trainingRatings = TrainingSet.map(lambda l: l.split(',')).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    # Build the model
    rank = 10
    numIterations = 20
    # numberOfFeatures = TrainingSet.first().map(lambda line: len(line.split(' '))).count()
    numberOfFeatures = 6
    while (numberOfFeatures > 5):
        model = ALS.train(trainingRatings, rank, numIterations)
        numberOfFeatures -= round(numberOfFeatures * numberOfFeaturesToremove)
        yield model


def do_SVM(TrainingSet, ValidatingSet, featuresList, numberOfFeaturesToremove):
    tripletResultList = []
    parsedData = TrainingSet.map(parsePoint)
    parsedValidatingSet = ValidatingSet.map(parsePoint)
    numberOfFeatures = len(featuresList)
    # Build the model

    # We build the SVM model
    model = SVMWithSGD.train(parsedData)
    trainErr = fitness_SVM(parsedValidatingSet, model)
    featuresStr = [str(x) for x in
                   featuresList]  # NB If I don't do this I get the same features for the final List. it passes it as a reference!
    tripletResult = [[featuresStr, model, trainErr]]

    bestModels = tripletResult
    performanceCurve = [(numberOfFeatures, trainErr)]
    while numberOfFeatures > 100:
        # We apply the recursive feature elimination
        parsedData, parsedValidatingSet, featuresList, numberOfFeatures = removeFeatures(parsedData,
                                                                                         parsedValidatingSet,
                                                                                         featuresList,
                                                                                         numberOfFeaturesToremove)

        # We build the SVM model
        model = SVMWithSGD.train(parsedData)
        trainErr = fitness_SVM(parsedValidatingSet, model)
        featuresStr = [str(x) for x in featuresList]

        print (trainErr, bestModels[0][2])
        print len(bestModels)
        print len(featuresStr)

        performanceCurve.append((numberOfFeatures, trainErr))
        if bestModels[0][2] < trainErr:
            mod = [featuresStr, model, trainErr]
            bestModels = [mod]
        elif round(bestModels[0][2], 4) == round(trainErr, 4):
            mod = [featuresStr, model, trainErr]
            bestModels.append(mod)

    return bestModels, performanceCurve


def do_LogisticRegressionWithLBFGS(TrainingSet, ValidatingSet, featuresList, numberOfFeaturesToremove):
    parsedData = TrainingSet.map(parsePoint)
    # parsedData = TrainingSet
    numberOfFeatures = 6
    # Build the model
    while (numberOfFeatures > 5):
        model = LogisticRegressionWithLBFGS.train(parsedData)
        numberOfFeatures -= round(numberOfFeatures * numberOfFeaturesToremove)

        yield model


def do_LogisticRegressionWithSGD(TrainingSet, ValidatingSet, featuresList, numberOfFeaturesToremove):
    # parsedData = TrainingSet.map(parsePoint)
    parsedData = TrainingSet
    numberOfFeatures = 6
    # Build the model
    while (numberOfFeatures > 5):
        model = LogisticRegressionWithSGD.train(parsedData)
        numberOfFeatures -= round(numberOfFeatures * numberOfFeaturesToremove)
        yield model


def do_LinearRegressionWithSGD(TrainingSet, ValidatingSet, featuresList, numberOfFeaturesToremove):
    # parsedData = TrainingSet.map(parsePoint)
    parsedData = TrainingSet
    numberOfFeatures = 6
    # Build the model
    while (numberOfFeatures > 5):
        model = LinearRegressionWithSGD.train(parsedData)
        numberOfFeatures -= round(numberOfFeatures * numberOfFeaturesToremove)
        yield model


def do_LassoWithSGD(TrainingSet, ValidatingSet, featuresList, numberOfFeaturesToremove):
    # parsedData = TrainingSet.map(parsePoint)
    parsedData = TrainingSet
    numberOfFeatures = 6
    # Build the model
    while (numberOfFeatures > 5):
        model = LassoWithSGD.train(parsedData)
        numberOfFeatures -= round(numberOfFeatures * numberOfFeaturesToremove)
        yield model


#####################################
# define the available Algorithms   #
#####################################

splitDatasets = {
    'ALS': split_ALS,
    'SVM': split_LinearMethods,
    'LogisticRegressionWithLBFGS': split_LinearMethods,
    'LogisticRegressionWithSGD': split_LinearMethods,
    'LinearRegressionWithSGD': split_LinearMethods,
    'LassoWithSGD': split_LinearMethods,
}

crossval = {
    'ALS': do_ALS,
    'SVM': do_SVM,
    'LogisticRegressionWithLBFGS': do_LogisticRegressionWithLBFGS,
    'LogisticRegressionWithSGD': do_LogisticRegressionWithSGD,
    'LinearRegressionWithSGD': do_LinearRegressionWithSGD,
    'LassoWithSGD': do_LassoWithSGD,
}

#####################################
# main program                      #
#####################################

listofAlgorithms = ["SVM", "LogisticRegressionWithLBFGS", "LogisticRegressionWithSGD", "LinearRegressionWithSGD",
                    "LassoWithSGD", "ALS"]

if __name__ == "__main__":

    # We check that the arguments are well formed
    if not len(sys.argv) == 11:
        print >> sys.stderr, \
            "Invalid number of arguments. Usage: CrossValidation.py <dataFile, string>  <AlgorithmToUse, string> " \
            "<featuresFile, string> <kfold, float> <numberOfResampling, int> <numberOfFeaturesToRemove, float> " \
            "<doEnrichement, Boolean> <mongoDocPathwayEnrichementId, string> <mongoIP, String> <mongoDocId, string>"
        exit(-1)

    if not any(sys.argv[3] in s for s in listofAlgorithms):
        print >> sys.stderr, \
            "The algorithm requested is not available. The algorithms available for the Cross Validation are:", listofAlgorithms[
                                                                                                                0:7]
        exit(-1)

    # We fill the required variables
    dataFile = sys.argv[1]
    featuresFile = sys.argv[2]
    algorithmToUse = sys.argv[3]
    kfold = float(sys.argv[4])
    resampling = int(sys.argv[5])
    numberOfFeaturesToRemove = float(sys.argv[6])
    doEnrichement = str_to_bool(sys.argv[7])
    mongoDocIdPE = str(sys.argv[8])
    mongoIP = str(sys.argv[9])
    mongoDocId = str(sys.argv[10])

    conf = SparkConf().setAppName("CrossVal").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sc = SparkContext(conf=conf)

    # We take the training set and the list of features
    data = sc.textFile(dataFile)
    featuresList = sc.textFile(featuresFile).map(lambda l: [str(x) for x in l.split(' ')]).first()
    datasetSize = data.count()

    # We split it for a 5 fold
    trainingSetSize = kfold
    testSetSize = 1 - trainingSetSize

    for i in range(0, resampling):
        # We generate a random seed to have a different splitting of the data at each iteration
        randomSeed = random.randrange(0, datasetSize - 1, 1)

        # Load and parse the data
        CVTest, CVTraining = splitDatasets[algorithmToUse](data, trainingSetSize, testSetSize, randomSeed)
        cvsetsize = CVTest.count()
        trainingSetsize = CVTraining.count()

        models, performanceCurve = crossval[algorithmToUse](CVTraining, CVTest, featuresList, numberOfFeaturesToRemove)

        print("CV test set size = " + str(cvsetsize))
        print("CV training set size = " + str(trainingSetsize))

        bestModels = [(models[0][0], models[0][1], 1)]

        # TODO when the 1.4.1 goes out use areaUnderROC from BinaryClassificationMetrics
        for model in models:
            print model[2]
            if bestModels[0][2] > model[2]:
                bestModels = [model]
            elif round(bestModels[0][2], 4) == round(model[2], 4):
                bestModels.append(model)

        for model in bestModels:
            print "/***********************************************/"
            print len(model[0])
            print model[0]
            print model[2]
            print "/***********************************************/"

        ## we retrieve the mongo client and database for Pathway enrichment
        db = MongoClient('mongodb://' + mongoIP + '/').eae
        db.authenticate('eae', 'eae', mechanism='SCRAM-SHA-1')
        cvCollection = db.CrossValidation

        smallestBestModel = bestModels[-1]
        # interesting note on the side If I import json before a Spark compute it blows a fuse and fails the computation!?

        doc = {"ModelFeatures": smallestBestModel[0],
               "ModelWeights": smallestBestModel[1].weights.toArray().tolist(),
               "ModelIntercept": smallestBestModel[1].intercept,
               "ModelPerf": smallestBestModel[2],
               "PerformanceCurve": performanceCurve,
               "AlgorithmUsed": algorithmToUse,
               "kfold": kfold,
               "Resampling": resampling,
               "WorkflowSpecificParameters": algorithmToUse + " " + str(kfold) + " " + str(resampling) + " " + str(numberOfFeaturesToRemove),
               "NumberOfFeaturesToRemove": numberOfFeaturesToRemove,
               "EndTime": datetime.now(),
               "Status": "completed"}

        docId = ObjectId(mongoDocId)
        cvCollection.update_one({'_id': docId}, {"$set": doc}, upsert=False)

        # We do a pathway enrichment if the user asked for it.
        if(doEnrichement):
            listOfGenesSymbols = smallestBestModel[0]
            print listOfGenesSymbols
            top5, resp, listOfGenesSymbols, listOfGenes = PathwayEnrichment.doSparkEnrichment(sc, listOfGenesSymbols, 'Bonferroni')
            peCollection = db.PathwayEnrichment

            docPE = {"TopPathways": top5,
                     "KeggTopPathway": resp.text,
                     "ListOfGenes": ' '.join(listOfGenesSymbols),
                     "CustomField": ' '.join(listOfGenesSymbols),
                     "ListOfGenesIDs": ' '.join([str(x) for x in listOfGenes]),
                     "Correction": 'Bonferroni',
                     "EndTime": datetime.now(),
                     "Status": "Completed"}

            docIdPE = ObjectId(mongoDocIdPE)
            peCollection.update_one({'_id': docIdPE}, {"$set": docPE}, upsert=False)
