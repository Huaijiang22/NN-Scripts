# -*- coding: utf-8 -*-
"""
Created by Huai
"""

import pybrain.datasets
import pybrain.tools.neuralnets
import pybrain.structure.networks
import pybrain.structure.modules
import pybrain.structure.connections
import pybrain.supervised.trainers
import os
import numpy as np

import csv
import pickle


def convertDataNeuralNetwork(x, y):  # combine dataset x and y
    colx = 1 if len(np.shape(x)) == 1 else np.size(x, axis=1)
    coly = 1 if len(np.shape(y)) == 1 else np.size(y, axis=1)

    fulldata = pybrain.datasets.SupervisedDataSet(colx, coly)

    for d, v in zip(x, y):
        fulldata.addSample(d, v) 

    return fulldata


def easyneuralnet(x, y, layers, nodes, epochstillshuffle=200, maxEpochs=5000, learningrate=0.01, lrdecay=1.0,
                  momentum=0., epochsperstep=100):
    print 'layers = ' + str(layers) + ', nodes = ' + str(nodes)
    fulldata = convertDataNeuralNetwork(x, y)

    regressionTrain, regressionTest = fulldata.splitWithProportion(
        .75)  # use 75% of the data for training and the remaining 25% for testing
    fnn = pybrain.structure.networks.FeedForwardNetwork()  # feedforward NN module

    # initiate input layer
    inLayer = pybrain.structure.modules.LinearLayer(regressionTrain.indim)
    # initiate output layer
    outLayer = pybrain.structure.modules.LinearLayer(regressionTrain.outdim)

    hiddenlayers = []
    # add to the input layer '# of layers' times

    for l in range(layers):
        # choose sigmoid function as activation function (for the hidden layer)
        hiddenlayers.append(pybrain.structure.modules.SigmoidLayer(nodes))

    fnn.addInputModule(inLayer)
    fnn.addOutputModule(outLayer)

    for hiddenLayer in hiddenlayers:
        fnn.addModule(hiddenLayer)

    in_to_hidden = pybrain.structure.connections.FullConnection(inLayer, hiddenlayers[0])
    # hidden layer to hidden layer connection. Only happens when # hidden layers>=2
    hidden_connections = []
    for l in range(1, layers):
        hidden_connections.append(pybrain.structure.connections.FullConnection(hiddenlayers[l - 1], hiddenlayers[l]))

    hidden_to_out = pybrain.structure.connections.FullConnection(hiddenlayers[-1], outLayer)

    fnn.addConnection(in_to_hidden)
    for connection in hidden_connections:
        fnn.addConnection(connection)

    fnn.addConnection(hidden_to_out)

    fnn.sortModules()
    trainer = pybrain.supervised.trainers.BackpropTrainer(fnn, dataset=regressionTrain, verbose=False,
                                                          learningrate=learningrate, lrdecay=lrdecay, momentum=momentum)
    # Use the back propagation algorithm
    epochcount = 0
    # number of the training times (# of reshuffle the dataset) is maxEpochs/epochstillshuffle
    while epochcount < maxEpochs:
        # train the data until get the converged weights within the # of epoch step
        trainer.trainUntilConvergence(maxEpochs=epochsperstep, continueEpochs=3)
        # TODO cannot convergence, need to figure out the reason
        epochcount += epochsperstep
        if epochcount % epochstillshuffle == 0:
            print str(epochcount) + ' epochs done of ' + str(maxEpochs)
            print finderrors(y, np.transpose(fnn.activateOnDataset(fulldata))[0])
            print 'reshuffling..'
            regressionTrain, regressionTest = fulldata.splitWithProportion(.75)
            trainer.setData(regressionTrain)
    return fnn


def finderrors(y,
               output):  # return a list of error function value;mse is the mean squared errors, max is the max error and p50/p90 refers to the error percentile
    errors = {}
    errors['mse'] = sum((y - output) ** 2) / len(y)
    errors['max'] = max(abs(y - output))
    errors['p50'] = np.percentile(abs(y - output), 50)
    errors['p90'] = np.percentile(abs(y - output), 90)
    return errors


def getally(fnn, x_all):  # get the value of response y for predictor x in new set
    newy = []
    for x in x_all:
        newy.append(fnn.activate(x)[0])
    return np.array(newy)


def savecomparison(outputy, y, layers, nodes,epochsperstep, epochstillshuffle, maxEpochs):  # output y and predicted response variable yhat (from NN model)
    with open(STARTING_DIRECTORY + 'data//regression_outputs//' + 'ycomparison_' + str(layers) + '_' + str(
            nodes) +'_'+str(epochsperstep)+'_'+str(epochstillshuffle)+'_'+str(maxEpochs)+ '.csv', 'wb') as outfile:
        csvwriter = csv.writer(outfile, delimiter=',')
        csvwriter.writerow(['y', 'predicted y'])
        for row in zip(y, outputy):
            csvwriter.writerow(row)


def reconstitutey(y, add, scale):  # convert the normalized response variable y back to its original scale
    return (y - add) / scale


def savenewy(newy, layers, nodes,epochsperstep, epochstillshuffle, maxEpochs):  # save all the simulated response variable y based on the NN model used
    with open(STARTING_DIRECTORY + 'data//regression_outputs//' + 'yall_' + str(layers) + '_' + str(nodes)+'_'+str(epochsperstep)+'_'+str(epochstillshuffle)+'_'+str(maxEpochs)+ '.csv',
              'wb') as outfile:
        csvwriter = csv.writer(outfile, delimiter=',')
        csvwriter.writerow(['predicted y'])
        for row in newy:
            csvwriter.writerow([row])


def savetesty(testy, layers, nodes,epochsperstep, epochstillshuffle, maxEpochs):  # save all the simulated response variable y based on the NN model used
    with open(STARTING_DIRECTORY + 'data//regression_outputs//' + 'ytest_' + str(layers) + '_' + str(nodes)+'_'+str(epochsperstep)+'_'+str(epochstillshuffle)+'_'+str(maxEpochs) + '.csv',
              'wb') as outfile:
        csvwriter = csv.writer(outfile, delimiter=',')
        csvwriter.writerow(['predicted y'])
        for row in testy:
            csvwriter.writerow([row])


def saveerrors(errors, layers, nodes,epochsperstep, epochstillshuffle, maxEpochs):  # save the test error from the model
    with open(STARTING_DIRECTORY + 'data//regression_outputs//' + 'errors_' + str(layers) + '_' + str(nodes)+'_'+str(epochsperstep)+'_'+str(epochstillshuffle)+'_'+str(maxEpochs) + '.csv',
              'wb') as outfile:
        csvwriter = csv.writer(outfile, delimiter=',')
        csvwriter.writerow(['error type', 'error'])
        for key, value in errors.items():
            csvwriter.writerow([key, value])


STARTING_DIRECTORY = os.getcwd().rstrip('scripts')
print STARTING_DIRECTORY
# input the predictors x and response y
y = np.genfromtxt(STARTING_DIRECTORY + 'data//regression_inputs//' + 'ynorm.csv', delimiter=',')
x = np.genfromtxt(STARTING_DIRECTORY + 'data//regression_inputs//' + 'x_trainnorm.csv', delimiter=',')
#y_all = np.genfromtxt(STARTING_DIRECTORY + 'data//regression_inputs//' + 'y_all.csv', delimiter=',')
x_all = np.genfromtxt(STARTING_DIRECTORY + 'data//regression_inputs//' + 'x_allnorm.csv', delimiter=',')
#x_test = np.genfromtxt(STARTING_DIRECTORY + 'data//regression_inputs//' + 'x_test.csv', delimiter=',')
#size_diff = np.size(y_all) - np.size(y)
# y_test = y_all[-size_diff:]
# y_test=y_all[0:size_diff]
#total_size_y = np.size(y_all)

with open(STARTING_DIRECTORY + 'data//regression_inputs//' + 'ynorm_scale_add.csv', 'rb') as infile:
    csvreader = csv.reader(infile, delimiter=',')
    add = float(csvreader.next()[0])
    scale = float(csvreader.next()[0])
# get the original y data (convert back)
oriy = reconstitutey(y, add, scale)

fulldata = convertDataNeuralNetwork(x, y)

# nodes = int(np.size(x, axis=1)/2)
# initialize the tuning parameter for the NN regression model
# sensitive parameters
#epochstillshuffle = 200
#maxEpochs = 1000
#epochsperstep = 50
# insensitive parameters
learningrate = 0.15
lrdecay = 1.
momentum = 0.15
for (epochsperstep,epochstillshuffle,maxEpochs) in [(10,20,50),(10,50,200),(10,100,500),(20,40,100),(20,100,500),(20,200,1000),(50,200,1000),(50,500,4000),
         (100,200,400),(100,400,2000),(200,800,4000)]:
    for layers in [2]:
        for nodes in range(30, 120, 5):
            # for nodes in [50,60,70]:
            # for nodes in [55]:
            try:
                with open(STARTING_DIRECTORY + 'data//regression_outputs//' + 'yall_' + str(layers) + '_' + str(
                        nodes)+'_'+str(epochsperstep)+'_'+str(epochstillshuffle)+'_'+str(maxEpochs) + '.csv', 'rb') as infile:
                    continue
            except:
                with open(STARTING_DIRECTORY + 'data//regression_outputs//' + 'yall_' + str(layers) + '_' + str(
                        nodes)+'_'+str(epochsperstep)+'_'+str(epochstillshuffle)+'_'+str(maxEpochs) + '.csv', 'wb') as infile:
                    pass

            fnn = easyneuralnet(x, y, layers, nodes, epochstillshuffle=epochstillshuffle, maxEpochs=maxEpochs,
                                learningrate=learningrate, lrdecay=lrdecay, momentum=momentum, epochsperstep=epochsperstep)

            outputy = reconstitutey(np.transpose(fnn.activateOnDataset(fulldata))[0], add, scale)
            newy = reconstitutey(getally(fnn, x_all), add, scale)
            #testy = reconstitutey(getally(fnn, x_test), add, scale)
            #newy_test = newy[-size_diff:]
            # newy_test=newy[-(total_size_y):-(total_size_y-size_diff)]


            savecomparison(outputy, oriy, layers, nodes,epochsperstep, epochstillshuffle, maxEpochs)
            #savecomparison(newy_test, y_test, layers, nodes,epochsperstep, epochstillshuffle, maxEpochs)
            savenewy(newy, layers, nodes,epochsperstep, epochstillshuffle, maxEpochs)
            #savetesty(testy, layers, nodes,epochsperstep, epochstillshuffle, maxEpochs)

            pickle.dump(fnn, open(
                STARTING_DIRECTORY + 'data//regression_outputs//' + 'nn_' + str(layers) + '_' + str(nodes)+'_'+str(epochsperstep)+'_'+str(epochstillshuffle)+'_'+str(maxEpochs) + '.p', 'wb'))
            errors = finderrors(oriy, outputy)

            saveerrors(errors, layers, nodes,epochsperstep, epochstillshuffle, maxEpochs)

            print errors

