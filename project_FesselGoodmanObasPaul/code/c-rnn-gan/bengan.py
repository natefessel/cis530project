from random import sample





### DISCRIMINATOR

token_discrim = Tokenizer()
def prep_discrim(data):
    token_discrim.fit_on_texts(data)
    total_words = len(token_discrim.word_index) + 1
    print(total_words)

    input_sequences = [
        token_discrim.texts_to_sequences([line])[0]
        for line in data
    ]

    max_seq_len = max(len(x) for x in input_sequences)
    max_seq_len = max(max_seq_len, 50)

    input_sequences = np.array(pad_sequences(
        input_sequences,
        maxlen = max_seq_len,
        padding = 'pre'
    ))

    predictors = input_sequences[:,:-1]
    return predictors, max_seq_len, total_words

def fit_discrim(X_train, y_train, max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(total_words, 40, input_length=input_len))
    model.add(Dropout(0.20))
    model.add(LSTM(80))
    model.add(Dropout(0.20))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs = 5, batch_size=64)
    return model

# Randomly pulling 2000 random correct sentences

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

# Piping through to fit the discriminator
#len(totalReviews)
X_train, max_sequence_len, total_words = prep_discrim(totalReviews)
len(X_train[0])
discriminator = fit_discrim(X_train, totalLabels, max_sequence_len, total_words)

modelPath = "C:/Users/goodm/Desktop/UPenn/UPenn Senior Year/Spring Semester/CIS 530/Milestone 3/discriminator_model.h5"
#discriminator.save(modelPath)
discriminator = load_model(modelPath)
# Evaluating the discriminator against unseen validation data!
from random import sample
trueSamples_val = sample(trainingDataOG, k = 2000)
#Our generated samples deteriorate rapidly beyond 1 senetence length, so to not bias the discriminator on length alone, we will truncate each review
#to be just one sentence with our model
for i, sample in enumerate(trueSamples_val):
    trueSamples_val[i] = " ".join(sample.split('.')[0].split(' ')[:49])

# Stacking the results, making the 3000 testing labels
totalReviews_val = trueSamples_val + genReviews[2000:]
totalLabels_val = [1 if i < 2000 else 0 for i in range(len(totalReviews_val))]
totalLabels_val = np.asarray(totalLabels)

token_discrim = Tokenizer()
X_test, max_sequence_len, total_words = prep_discrim(totalReviews_val)
scores = discriminator.evaluate(X_test, totalLabels_val, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
# Discriminator with 67.05% accuracy on validation data

##################################################################################################
#Now, it is time to hook up the GAN with our discriminator and the generator and see what happens!
##################################################################################################
keptGen = []
# Pseudo-GAN Algorithm:
num_rounds = 20
for rounds in range(num_rounds):
    # Generate twenty real and twenty false
    trueSamples_iter = sample(trainingDataOG, k = 40)
    genReviews_iter = manyReviews(40, seedTriGrams, max_len, lstm_gen_model)
    # Now testing the discriminator, and training it on the missses
    for i, sample in enumerate(trueSamples_iter): trueSamples_iter[i] = " ".join(sample.split('.')[0].split(' ')[:49])

    # Preping data for dsciriminator task
    totalReviews_iter = trueSamples_iter + genReviews_iter
    totalLabels_iter = [1 if i < 40 else 0 for i in range(len(totalReviews_iter))]
    totalLabels_iter = np.asarray(totalLabels_iter)

    #token_discrim = Tokenizer()
    X_test_iter, max_sequence_len_iter, total_words_iter = prep_discrim(totalReviews_iter)
    len(X_test_iter[1])
    # Storing predictions which fool the discriminator
    predictions = discriminator.predict(X_test_iter)
    fake_pred = predictions[40:]
    genReviews_iter= np.asarray(genReviews_iter)
    keptGen_iter = genReviews_iter[(fake_pred >= 0.80).flatten()]
    keptGen += keptGen_iter.tolist()
    scores_iter = discriminator.evaluate(X_test_iter, totalLabels_iter, verbose=0)
    print("Accuracy: %.2f%%" % (scores_iter[1]*100))
    # Train the discriminator on the fooling observations, ie strengthen the discriminator
    discriminator.fit(X_test_iter, totalLabels_iter, epochs = 3, batch_size = 32, verbose = 0)

len(keptGen)
# Progression of more and more difficult "trick" artificial generations
keptGen[0::3]
# Saving the new discriminator model
modelPath = "C:/Users/goodm/Desktop/UPenn/UPenn Senior Year/Spring Semester/CIS 530/Milestone 3/discriminator_model_enhanced.h5"
discriminator.save(modelPath)
