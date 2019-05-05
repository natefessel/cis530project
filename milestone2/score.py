import pickle
import os
import pprint
import argparse

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

##This file outputs a normalized probability score of the GAN model output in comparision to the test data set probability score
##Assumes an output per line in the test data set
parser.add_argument('--testText', type=str, required=True)
parser.add_argument('--ganOutput', type=str, required=True)
parser.add_argument('--langModelPickle', type=str, required=True)

##function to read in and evaluate test data
def testProb(filePath, langModel):
    count = 0
    probTotal = 0
    with open(filePath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().strip().split('\n')
        for line in text:
            description = line.split(',')[1]
            description = description[1:(len(description) - 1)]
            count += 1
            ##Probability as calculated from within the language model
            probTotal += langModel.probability(description)

    return probTotal / count

##function to read in an evaluate generated text
def ganProb(filePath, langModel):
    count = 0
    probTotal = 0
    with open(filePath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().strip().split('\n')
        for line in text:
            count += 1
            ##Probability as calculated from within the language model
            probTotal += langModel.probability(description)

    return probTotal / count

##Main function to read in data sets and language model, output to council the probability values we want
def main(args):
    filehandler = open(args.langModelPickle, 'r')
    langModel = pickle.load(filehandler)
    denomProb = textProb(args.testText, langModel)
    numerProb = testProb(args.ganText, langModel)
    normScore = numerProbablity / denomProbability
    pp.pprint("GAN model normalized Probability Score: " + str(normScore))
     

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)