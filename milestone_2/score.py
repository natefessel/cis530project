import pickle
import os
import pprint
import argparse

pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()

##This file outputs a normalized perplexity score of the GAN model output in comparision to the test data set perplexity score
##Assumes an output per line in the test data set
parser.add_argument('--testText', type=str, required=True)
parser.add_argument('--ganOutput', type=str, required=True)
parser.add_argument('--langModelPickle', type=str, required=True)

##function to read in and evaluate test data
def testPerplex(filePath, langModel):
    count = 0
    perplexTotal = 0
    with open(filePath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().strip().split('\n')
        for line in text:
            description = line.split(',')[1]
            description = description[1:(len(description) - 1)]
            count += 1
            ##Perplexity as calculated from within the language model
            perplexTotal += langModel.perplexity(description)

    return perplexTotal / count

##function to read in an evaluate generated text
def ganPerplex(filePath, langModel):
    count = 0
    perplexTotal = 0
    with open(filePath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().strip().split('\n')
        for line in text:
            count += 1
            ##Perplexity as calculated from within the language model
            perplexTotal += langModel.perplexity(description)

    return perplexTotal / count

##Main function to read in data sets and language model, output to council the perplexity values we want
def main(args):
    filehandler = open(args.langModelPickle, 'r')
    langModel = pickle.load(filehandler)
    denomPerplexity = textPerplex(args.testText, langModel)
    numerPerplexity = testPerplex(args.ganText, langModel)
    normScore = numerPerplexity / denomPerplexity
    pp.pprint("GAN model normalized Perplexity Score: " + str(normScore))
     

if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)