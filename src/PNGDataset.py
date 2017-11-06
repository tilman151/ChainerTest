import csv
import os
import numpy as np


class PNGDataset():

    def __init__(self, inputDir):
        self.inputDir = inputDir

    def getDataset(self):
        imagePaths = []
        labelPath = None
        for (dirPath, dirNames, fileNames) in os.walk(self.inputDir):
            for fileName in fileNames:
                if fileName[-4:] == '.png':
                    imagePaths.append(os.path.join(dirPath, fileName))
                elif fileName == 'labels.csv':
                    labelPath = os.path.join(dirPath, fileName)

        if labelPath is None:
            raise ValueError('No label file found. Must be named labels.csv')

        labels = []
        with open(labelPath, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            for row in csvreader:
                labels.append(np.int32(row[0]))

        num_examples = len(labels)
        if len(imagePaths) != num_examples:
            raise ValueError('Number of images %d not same as label size %d.' %
                             (len(imagePaths), num_examples))

        imagePaths = sorted(imagePaths)

        return (imagePaths, labels)
