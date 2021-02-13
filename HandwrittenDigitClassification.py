from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

colors = ('red', 'blue', 'green', 'yellow', 'black', 'grey', 'purple', 'pink', 'orange', 'brown')


def saveUniquePlot(uniqueLabels, allLabels, images):
    # display and save one plots from each class
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    tempUniqueLabels = copy.deepcopy(uniqueLabels)

    i = 0
    while len(tempUniqueLabels) != 0 and i < len(allLabels):
        if allLabels[i] in tempUniqueLabels:
            plt.axis('off')
            plt.imshow(images[i])
            plt.savefig(f'Plots/{allLabels[i]}.png')
            tempUniqueLabels.remove(allLabels[i])
        i += 1


def calculateAndSaveCentroid(uniqueLabels, labels, imagesArrays):
    if not os.path.exists('Centroids'):
        os.makedirs('Centroids')

    # make a dict {label -> images}
    mapping = {k: [] for k in uniqueLabels}
    i = 0
    while i < len(labels):
        mapping[labels[i]].append(imagesArrays[i])
        i += 1

    centroids = {}
    for label, arrays in mapping.items():
        # calculate the avg of each pixel
        centroid = np.average(np.array(arrays), axis=0)
        centroids[label] = np.array(centroid)

        # show the data
        centroid = np.reshape(centroid, (28, 28))
        plt.imshow(centroid)
        plt.savefig(f'Centroids/{label}.png')

    return centroids


def computeCentroidsDistances(centroids, uniqueLabels):
    size = len(uniqueLabels)
    centroidsDistances = np.zeros([size, size], dtype=int)

    # go over each unordered pair:
    for i in uniqueLabels:
        for j in range(i + 1, len(uniqueLabels)):
            first = centroids[i]
            second = centroids[j]
            # calculate their distance
            distance = np.absolute(np.subtract(first, second))
            difference = np.sum(distance)

            centroidsDistances[i][j] = difference
            centroidsDistances[j][i] = difference

    # print and show the data
    print('\nCentroids Distances Matrix:')
    print(centroidsDistances)
    plt.title('Centroids Distances')
    plt.axis('on')
    plt.xticks(uniqueLabels)
    plt.yticks(uniqueLabels)
    plt.imshow(centroidsDistances)
    title = 'Centroids Distances'
    plt.title(title)
    plt.savefig('Centroids/' + title + '.png')
    plt.show()


def pixelsHistogram(imagesArrays):
    # getting the data variance
    variances = np.var(imagesArrays, axis=0)

    # getting the variances' range
    lowest = np.min(variances)
    highest = np.max(variances)
    numberRange = np.linspace(lowest, highest, 11)

    # show and save the histogram
    fig, ax = plt.subplots()
    ax.hist(variances, bins=numberRange)
    title = "Pixel's variance distribution"
    plt.title(title)

    if not os.path.exists('Data investigation'):
        os.makedirs('Data investigation')
    plt.savefig('Data investigation/' + title + '.png')
    plt.show()

    return variances


def adjustData(ImagesArrays, nonZeroVariances, mu=None):
    # this function get an array of images, remove zero-variance pixels, subtracts it's mu (or a given mu)
    # and returns the adjusted data and mu
    posImages = []
    for image in ImagesArrays:
        posImages.append(image[nonZeroVariances])

    posImages = np.array(posImages)

    if mu is None:
        mu = np.mean(posImages, 0)
    adjustedData = posImages - mu
    return adjustedData, mu


def preparePCA(variances, imagesArrays):
    # this function adjust the data according to PCA requirements and returns the eigen values and vectors
    nonZeroVariances = np.where(variances > 0, True, False)

    adjustedData, mu = adjustData(imagesArrays, nonZeroVariances)

    scatterMatrix = np.matmul(adjustedData.T, adjustedData)
    eigenValues, eigenVectors = np.linalg.eig(scatterMatrix)

    # make sure we dont get complex output
    eigenVectors = np.real(eigenVectors)
    return adjustedData, eigenValues, eigenVectors, mu, nonZeroVariances


def computePCA(adjustedData, eigenValues, eigenVectors, dimSize):
    # Reduce prepared data to the specified dim
    # returns the new data and the projection matrix (E, combined eigen vectors) in the specified dimension
    projectionVectors = []

    indexOrder = np.argsort(np.argsort(eigenValues))
    for i in range(dimSize):
        index = np.where(indexOrder == len(eigenValues) - 1 - i)
        vec = np.reshape(eigenVectors[:, index], [1, len(eigenVectors)])
        projectionVectors.append(vec)

    projectionVectors = np.array(projectionVectors)

    newData = np.matmul(projectionVectors, np.transpose(adjustedData)).squeeze().T

    return newData, projectionVectors


def project100Points(uniqueLabels, trainingLabels, newDataIn2D):
    # get the first 100 images of each class
    first100NewImages = [[] for _ in range(len(uniqueLabels))]
    labels = set(copy.deepcopy(uniqueLabels))

    i = 0
    while len(labels) != 0 and i < len(newDataIn2D):
        if trainingLabels[i] in labels:

            first100NewImages[trainingLabels[i]].append(newDataIn2D[i])
            if len(first100NewImages[trainingLabels[i]]) == 100:
                labels.remove(trainingLabels[i])
        i += 1

    first100NewImages = np.array(first100NewImages)

    # show entire data
    i = 0
    while i < len(first100NewImages):
        plt.scatter(first100NewImages[i].T[0], first100NewImages[i].T[1], c=colors[i], label=i)
        i += 1

    plt.legend(loc='lower left', prop={'size': 7}, ncol=2)
    title = 'Projection of 100 points from each class in 2D'
    plt.title(title)
    plt.savefig('Data investigation/' + title + '.png')
    plt.show()

    # show 0, 1, 9
    tempSet = {0, 1, 9}
    for i in tempSet:
        plt.scatter(first100NewImages[i].T[0], first100NewImages[i].T[1], c=colors[i], label=i)

    plt.legend(loc='lower left', prop={'size': 7})
    title = 'Projection of 100 points from {0, 1, 9} in 2D'
    plt.title(title)
    plt.savefig('Data investigation/' + title + '.png')
    plt.show()


def exploreVariance(eigenValues):
    # show the value's scatter
    size = range(len(eigenValues))
    plt.scatter(size, eigenValues)
    title = 'Eigenvalues weights'
    plt.title(title)
    plt.savefig('Data investigation/' + title + '.png')
    plt.show()

    # get the sum of all values
    eigenSum = np.sum(eigenValues)
    indexOf80perEnergy = 0
    indexOf85perEnergy = 0

    # 80% energy
    energy = 0
    while energy < 0.80 * eigenSum:
        energy += eigenValues[indexOf80perEnergy]
        indexOf80perEnergy += 1

    # 85% energy
    energy = 0
    while energy < 0.85 * eigenSum:
        energy += eigenValues[indexOf85perEnergy]
        indexOf85perEnergy += 1

    return indexOf80perEnergy, indexOf85perEnergy


def determineBestParams(trainingImageArrays, labels, variances, indexOf80perEnergy, indexOf85perEnergy,
                        usePreprocessedData):
    # return the learnt data
    if usePreprocessedData is True:
        return 48, 5

    print('\n\n-----Starting to determine the best parameters-----')

    # split data to training and validation
    trainingData, validationData, trainingLabels, validationLabels = train_test_split(trainingImageArrays, labels)

    # get the needed parameters to reduce dimension
    adjustedTrainingData, eigenValues, eigenVectors, mu, nonZeroVariances = preparePCA(variances, trainingData)

    dimsAccuracies = []
    dimKs = []

    for dimSize in range(indexOf80perEnergy, indexOf85perEnergy + 1):

        # reduce data dimension to dimSize
        newTrainingDataData, projectionVectors = computePCA(adjustedTrainingData, eigenValues, eigenVectors, dimSize)

        # reduce the validation's dimension to the same dim
        adjustedValidationData, _ = adjustData(validationData, nonZeroVariances, mu)
        adjustedValidationData = np.matmul(projectionVectors, np.transpose(adjustedValidationData)).squeeze().T

        # learn k
        bestKPerDim = 1
        maxAccuracyPerDim = 0
        for k in range(1, 10):
            # use knn to classify
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(newTrainingDataData, trainingLabels)
            validationPredictions = neigh.predict(adjustedValidationData)

            # measure accuracy
            accuracy = 0
            for i in range(len(validationData)):
                if validationPredictions[i] == validationLabels[i]:
                    accuracy += 1
            accuracy /= len(validationLabels)

            if accuracy > maxAccuracyPerDim:
                maxAccuracyPerDim = accuracy
                bestKPerDim = k
            print(f'Results for {dimSize} dimensions with k = {k} are: {accuracy * 100}%')

        print(f'---The best k for {dimSize} dimensions is {bestKPerDim} with {maxAccuracyPerDim * 100}% success.---')

        dimsAccuracies.append(maxAccuracyPerDim)
        dimKs.append(bestKPerDim)

    dimsAccuracies = np.array(dimsAccuracies)
    dimKs = np.array(dimKs)

    # show the data
    xPoints = range(indexOf80perEnergy, indexOf85perEnergy + 1)
    plt.scatter(xPoints, dimsAccuracies)
    plt.xticks(xPoints)
    for x, y, k in zip(xPoints, dimsAccuracies, dimKs):
        plt.annotate(text=k, xy=(x, y), ha='center', c='red', textcoords="offset points", xytext=(0, 10))

    title = 'Best validation results per dimension size'
    plt.title(title)
    plt.savefig('Data investigation/' + title + '.png')
    plt.show()

    maxIndex = np.argmax(dimsAccuracies)

    # print the results
    print(f'The best overall results ({(dimsAccuracies[maxIndex] * 100):.5f}%) '
          f'are in {indexOf80perEnergy + maxIndex} dimensions with k = {dimKs[maxIndex]}')
    print('-----The process is Done-----')

    return indexOf80perEnergy + maxIndex, dimKs[maxIndex]


def reportSuccessRate(neigh, projectionVectors, nonZeroVariances, testImageArrays, testLabels, mu):
    print('\n---Starting the test process---')

    # adjust the test data
    adjustedData, _ = adjustData(testImageArrays, nonZeroVariances, mu)
    adjustedData = np.matmul(projectionVectors, np.transpose(adjustedData)).squeeze().T

    # classify
    predictions = neigh.predict(adjustedData)

    # measure success
    suc = 0
    for prediction, label in zip(predictions, testLabels):
        if prediction == label:
            suc += 1

    suc /= len(testLabels)

    print(f'Done. success rate: {(suc * 100):.5f}%')


def main():
    # #### load data ###
    mnData = MNIST('Data')
    mnData.gz = True
    trainingImagesArrays, trainingLabels = mnData.load_training()
    testImageArrays, testLabels = mnData.load_testing()

    trainingImagesArrays = np.array(trainingImagesArrays)
    testImageArrays = np.array(testImageArrays)

    images = []
    for image in trainingImagesArrays:
        images.append(np.array(image).reshape((28, 28)))

    uniqueLabels = sorted(set(trainingLabels))

    # #### step 1 ####
    saveUniquePlot(uniqueLabels, trainingLabels, images)

    # #### step 2 ####
    centroids = calculateAndSaveCentroid(uniqueLabels, trainingLabels, trainingImagesArrays)

    # #### step 3 ####
    computeCentroidsDistances(centroids, uniqueLabels)

    # #### step 4 ####
    variances = pixelsHistogram(trainingImagesArrays)

    # #### step 5 ####
    adjustedTrainingData, trainingEigenValues, trainingEigenVectors, mu, nonZeroVariances = \
        preparePCA(variances, trainingImagesArrays)
    newDataIn2D, _ = computePCA(adjustedTrainingData, trainingEigenValues, trainingEigenVectors, 2)
    project100Points(uniqueLabels, trainingLabels, newDataIn2D)

    # #### step 6 ####
    indexOf80perEnergy, indexOf85perEnergy = exploreVariance(trainingEigenValues)

    # #### step 7 ####
    usePreprocessedData = True
    ans = input('Would you like to calculate the best parameters? it can take a while...'
                ' (or use the preprocessed data)? (y/n): ')
    if ans == 'y':
        usePreprocessedData = False

    bestDim, bestK = determineBestParams(trainingImagesArrays, trainingLabels, variances, indexOf80perEnergy,
                                         indexOf85perEnergy, usePreprocessedData)

    newTrainingData, projectionVectors = computePCA(adjustedTrainingData, trainingEigenValues, trainingEigenVectors,
                                                    bestDim)
    neigh = KNeighborsClassifier(n_neighbors=bestK)
    neigh.fit(newTrainingData, trainingLabels)

    reportSuccessRate(neigh, projectionVectors, nonZeroVariances, testImageArrays, testLabels, mu)


if __name__ == '__main__':
    print('-----Starting main-----')
    main()
