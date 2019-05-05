##This file aims to implement the model discussed in the paper
##"Adversarial Generation of Natural Language" (2017) as closely as possible

##We will be using an LTSM as a language generator,
##an LTSM as a language discriminator, and an objective of WGAN within the GAN function??
#####################################################
##Loading packages
import os
import pprint
import argparse
import time
import math
from random import randint
from random import sample
##Keras Libraries
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, load_model
import numpy as np

###Loading up the machienry needed for running the LSAN code 
pp = pprint.PrettyPrinter()
parser = argparse.ArgumentParser()
parser.add_argument('--generator', type=str, required=True)
parser.add_argument('--discriminator', type=str, required=True)
parser.add_argument('--outputpath', type=str, required=True)
parser.add_argument('--datapath', type=str, required=True)

##Function which reads in wine review data, ouputs list of training examples and a list of bigrams from the beginning of sentences
##within the reviews
def readData(filePath):
    trainingData = []
    seedTriGrams = []
    wordCounts = []
    sentenceCounts = []
    with open(filePath, 'r', encoding='utf-8', errors='ignore') as f:
         text = f.read().strip().split('\n')
         for i, line in enumerate(text):
             if i > 0:
                try:
                    description = line.split('"')[1]
                except:
                    description = line.split(',')[1]
                    description = description[1:(len(description) - 1)]
                ##Pulling randomly the start of sentences to be included in a collection of potential bigrams
                begins = description.split('.')
                for sentence in begins:
                    if len(sentence) > 0:
                        if sentence[0] == ' ':
                            sentence = sentence[1:(len(sentence))]
                        seedTriGrams.append(' '.join(sentence.split(' ')[:3]).lower())
                trainingData.append(description.lower())
                wordCounts.append(len(description.split()))
                sentenceCounts.append(len(description.split('.')) - 1)
    return trainingData, seedTriGrams, wordCounts, sentenceCounts

##Reading in the training data and producing bigrams to seed the data
'''
    filePath = "C:/Users/goodm/Desktop/UPenn/UPenn Senior Year/Spring Semester/CIS 530/Milestone 2/wine_reviews_train.txt"
    trainingDataOG, seedTriGrams, wordCounts, sentenceCounts = readData(filePath)
    trainingDataOG[:5]
    seedTriGrams[:10]
    len(trainingDataOG)
'''

##Function that returns a tuple comprised of ({word: incidence count}, total words)
def featureCounts(descriptions):
    wordCount = 0
    incidenceCount = {}
    for description in descriptions:
        tokens = description.replace(",", " ").replace(".", " ").split()
        for token in tokens:
            if token in incidenceCount:
                incidenceCount[token] += 1
            else:
                incidenceCount[token] = 1
            wordCount += 1
    return incidenceCount, wordCount

##Important Attributes of the training corpus:
'''
    incidenceCount, wordCount = featureCounts(trainingDataOG)
    wineCount = incidenceCount["wine"]
    len(incidenceCount)
    incidenceList = []
    maxOccur = 0
    maxVal = ''
    for word, occurence in incidenceCount.items():
        if occurence > maxOccur:
            maxOccur = occurence
            maxVal = word
        if occurence <= 100:
            incidenceList.append(occurence)
    len(incidenceList)
    maxOccur
    maxVal
    ##Histogram for word frequency distribution
    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(incidenceList, 20, facecolor='#0504aa', alpha=0.75)
    plt.grid(axis='y')
    plt.xlabel('Number of Occurences of the Word')
    plt.ylabel('Frequency')
    plt.title('Histogram of Number of Word Occurences in Training Data')
    plt.show()
'''
'''
        ##Building a histogram in matplotlib to show word length distributions for the data
        import matplotlib.pyplot as plt
        n, bins, patches = plt.hist(x= wordCounts, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth = 0.90)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Occurences of Word in Training Data')
        plt.ylabel('Frequency')
        plt.title('Histogram of Word Occurences in Training Data')
        maxfreq = n.max()
        plt.show()

        ##Histogram for number of sentences distribution
        import matplotlib.pyplot as plt
        n, bins, patches = plt.hist(sentenceCounts, 10, facecolor='#0504aa', alpha=0.75)
        plt.grid(axis='y')
        plt.xlabel('Sentences in the Description')
        plt.ylabel('Frequency')
        plt.title('Histogram of # of Sentences in Training Data')
        plt.show()
        ##cutting training data in half to reduce memory usage problems
        len(seedTriGrams)
        len(trainingData)
        trainingData = trainingDataOG[:10000]
        seedTriGrams = seedTriGrams[:20000]
'''

##Preparing the dataset with keras
tokenizer = Tokenizer()
def data_preparation(data):
    tokenizer.fit_on_texts(data)
    total_words = len(tokenizer.word_index) + 1
    print(total_words)
    input_sequences = []
    for line in data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_seq_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen = max_seq_len,
                                             padding = 'pre'))
    predictors, label = input_sequences[:,:-1], input_sequences[:, -1]
    return predictors, label, max_seq_len, total_words

##creating the LTSM model with keras
def create_LTSMGen(predictors, label, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 40, input_length = input_len))
    model.add(LSTM(50))
    model.add(Dropout(0.15))
    model.add(Dense(total_words, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam')
    model.fit(predictors, label, epochs = 1, verbose = 1)
    return model

##Generating text with the LTSM model with keras
def generate_text(base_text, num_next, max_sequence_len, model):
    for j in range(num_next):
        token_list = tokenizer.texts_to_sequences([base_text])[0]
        token_list = pad_sequences([token_list], maxlen = 
                             max_sequence_len-1, padding ='pre')
        probs = model.predict_proba(token_list, verbose = 0)
        ##indexes from which to draw the words
        temp = np.asarray([i for i in range(len(probs[0]))])
        predicted = np.random.choice(a = temp, size = 1, replace = True, p = probs[0]) ##This is potentially area for improvement, instead of randomly picking
                                                                                       ##from the probability distribution
                                                                                       ##Problem is that choosing the best would get stuck on stop words etc.
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        base_text += " " + output_word
    return base_text

##Piping through to train the model
#X, Y, max_len, total_words = data_preparation(trainingData)
#lstm_gen_model = create_LTSMGen(X, Y, max_len, total_words)

##Saving the model so that it is not necessary to retrain everything
#modelPath = "C:/Users/goodm/Desktop/UPenn/UPenn Senior Year/Spring Semester/CIS 530/Milestone 3/ltsm_gen.h5"
#lstm_gen_model.save(modelPath)
#lstm_gen_model = load_model(modelPath)

##Generate a bunch of fake wine reviews
##Creating 4000 fake reviews which will then be included with 2000 true reviews to train and test the discriminator
##Output is pretty incoherent, but at least as some semblance of sense.
def manyReviews(num, seedTriGrams, max_len, lstm_gen_model):
    genReviews = []
    for i in range(num):
        curSeed = randint(0, len(seedTriGrams) - 1)
        extraWords = randint(7, 16)
        newRev = generate_text(seedTriGrams[curSeed], extraWords, max_len, lstm_gen_model)
        genReviews.append(newRev)
        if i % 50 == 0:
            print("Done " + str(i))

    return genReviews

##Saving these faslified results
'''
    genReviews = manyReviews(4000, seedTriGrams, max_len, lstm_gen_model)
    genPath = "C:/Users/goodm/Desktop/UPenn/UPenn Senior Year/Spring Semester/CIS 530/Milestone 3/gen_4000.txt"
    with open(genPath, 'w') as f:
        for item in genReviews:
            f.write("%s\n" % item)
'''
#####################################################################
##Now building the discriminator
#####################################################################
tokenDiscrim = Tokenizer()
def prep_discrim(data):
    tokenDiscrim.fit_on_texts(data)
    total_words = len(tokenDiscrim.word_index) + 1
    print(total_words)
    input_sequences = []
    for line in data:
        token_list = tokenDiscrim.texts_to_sequences([line])[0]
        input_sequences.append(token_list)
    max_seq_len = max([len(x) for x in input_sequences])
    if max_seq_len < 50:
        max_seq_len = 50
    input_sequences = np.array(pad_sequences(input_sequences,
                                             maxlen = max_seq_len,
                                             padding = 'pre'))
    predictors = input_sequences[:,:-1]
    return predictors, max_seq_len, total_words

def fit_discrim(X_train, y_train, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 40, input_length = input_len))
    model.add(Dropout(0.20))
    model.add(LSTM(80))
    model.add(Dropout(0.20))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())
    model.fit(X_train, y_train, epochs = 5, batch_size=64)
    return model

##Section to train and validate the discriminator
'''
    ##Randomly pulling 2000 random correct sentences
    from random import sample
    trueSamples = sample(trainingDataOG, k = 2000)
    #Our generated samples deteriorate rapidly beyond 1 senetence length, so to not bias the discriminator on length alone, we will truncate each review
    #to be just one sentence with our model
    for i, sample in enumerate(trueSamples):
        trueSamples[i] = " ".join(sample.split('.')[0].split(' ')[:49])

    ##Stacking the results, making the 4000 training labels
    totalReviews = trueSamples + genReviews[:2000]
    totalLabels = [1 if i < 2000 else 0 for i in range(len(totalReviews))]
    totalLabels = np.asarray(totalLabels)

    ##Piping through to fit the discriminator
    #len(totalReviews)
    X_train, max_sequence_len, total_words = prep_discrim(totalReviews)
    len(X_train[0])
    discriminator = fit_discrim(X_train, totalLabels, max_sequence_len, total_words)

    modelPath = "C:/Users/goodm/Desktop/UPenn/UPenn Senior Year/Spring Semester/CIS 530/Milestone 3/discriminator_model.h5"
    #discriminator.save(modelPath)
    discriminator = load_model(modelPath)
    ##Evaluating the discriminator against unseen validation data!
    from random import sample
    trueSamples_val = sample(trainingDataOG, k = 2000)
    #Our generated samples deteriorate rapidly beyond 1 senetence length, so to not bias the discriminator on length alone, we will truncate each review
    #to be just one sentence with our model
    for i, sample in enumerate(trueSamples_val):
        trueSamples_val[i] = " ".join(sample.split('.')[0].split(' ')[:49])

    ##Stacking the results, making the 4000 testing labels
    totalReviews_val = trueSamples_val + genReviews[2000:]
    totalLabels_val = [1 if i < 2000 else 0 for i in range(len(totalReviews_val))]
    totalLabels_val = np.asarray(totalLabels)

    tokenDiscrim = Tokenizer()
    X_test, max_sequence_len, total_words = prep_discrim(totalReviews_val)
    scores = discriminator.evaluate(X_test, totalLabels_val, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    ##Discriminator with 67.05% accuracy on validation data
'''

##################################################################################################
#Now, it is time to hook up the GAN with our discriminator and the generator and see what happens!
##################################################################################################
'''
    ##Pseudo-GAN Algorithm:
    keptGen = []
    num_rounds = 20
    for rounds in range(num_rounds):
    ##Generate twenty real and twenty false
        from random import sample
        trueSamples_iter = sample(trainingDataOG, k = 40)
        genReviews_iter = manyReviews(40, seedTriGrams, max_len, lstm_gen_model)
        ##Now testing the discriminator, and training it on the missses
        for i, sample in enumerate(trueSamples_iter): trueSamples_iter[i] = " ".join(sample.split('.')[0].split(' ')[:49])

        ##Preping data for dsciriminator task
        totalReviews_iter = trueSamples_iter + genReviews_iter
        totalLabels_iter = [1 if i < 40 else 0 for i in range(len(totalReviews_iter))]
        totalLabels_iter = np.asarray(totalLabels_iter)
        X_test_iter, max_sequence_len_iter, total_words_iter = prep_discrim(totalReviews_iter)
        len(X_test_iter[1])
        ##Storing predictions which fool the discriminator
        predictions = discriminator.predict(X_test_iter)
        fake_pred = predictions[40:]
        genReviews_iter= np.asarray(genReviews_iter)
        keptGen_iter = genReviews_iter[(fake_pred >= 0.80).flatten()]
        keptGen += keptGen_iter.tolist()
        scores_iter = discriminator.evaluate(X_test_iter, totalLabels_iter, verbose=0)
        print("Accuracy: %.2f%%" % (scores_iter[1]*100))
        ##Train the discriminator on the fooling observations, ie strengthen the discriminator
        discriminator.fit(X_test_iter, totalLabels_iter, epochs = 3, batch_size = 32, verbose = 0)

    len(keptGen)
    ##Progression of more and more difficult "trick" artificial generations
    keptGen[0::3]
    ##Saving the new discriminator model
    modelPath = "C:/Users/goodm/Desktop/UPenn/UPenn Senior Year/Spring Semester/CIS 530/Milestone 3/discriminator_model_enhanced.h5"
    discriminator.save(modelPath)
'''

def generateGoodSamples(generatorPath, discriminatorPath, outputPath, dataPath):
    keptGen = []
    #Load Models
    lstm_gen = load_model(generatorPath)
    lstm_discrim = load_model(discriminatorPath)
    trainingDataOG, seedTriGrams, wordCounts, sentenceCounts = readData(dataPath)
    allReviews = manyReviews(150, seedTriGrams, max_len = 121, lstm_gen_model =  lstm_gen)
    X_test_iter, max_sequence_len_iter, total_words_iter = prep_discrim(allReviews)
    ##Storing predictions which fool the discriminator
    predictions = lstm_discrim.predict(X_test_iter)
    allReviews = np.asarray(allReviews)
    print(allReviews.flatten()[:5])
    keptGen_iter = allReviews[(predictions >= 0.80).flatten()]
    keptGen += keptGen_iter.tolist()
    #Write fake reviews to a text file at the output path
    with open(outputPath, 'w') as f:
        for item in keptGen:
            f.write("%s\n" % item)

def main(args):
    max_len = 49
    generateGoodSamples(args.generator, args.discriminator, args.outputpath, args.datapath)


if __name__ == '__main__':
    args = parser.parse_args()
    pp.pprint(args)
    main(args)