## NOTE: this file contains the same content as the iPython notebook `/ngram.ipynb`


# coding: utf-8

# In[173]:


import math, random
import os
import sklearn.metrics
import pandas as pd
import os.path as path
import random
import statistics


# In[2]:


def start_pad(n):
    ''' Returns padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    padded_text = start_pad(n) + text + '#'
    return [(padded_text[i:i+n], padded_text[i+n]) for i in range(len(text) + 1)]


# In[3]:


class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = set()
        self.d = {} # context -> (letter -> count of letter)

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for ngram in ngrams(self.n, text):
            context, letter = ngram
            self.vocab.add(letter)
            context_dict = self.d.setdefault(context, {})
            context_dict[letter] = context_dict.get(letter, 0) + 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        context_dict = self.d.get(context)
        if context_dict is None:
            return 1 / len(self.vocab)
        else:
            num = context_dict.get(char, 0) + self.k
            den = sum(context_dict.values()) + self.k * len(self.vocab)
            return num / den

    def random_char(self, context):
        ''' Returns a random character based on the given context and the
            n-grams learned by this model '''
        vocab = sorted(self.vocab)
        probs = []
        for char in vocab:
            probs.append(self.prob(context, char))

        r = random.random()
        t = 0
        for char, prob in zip(vocab, probs):
            t += prob
            if r < t:
                return char

        return vocab[-1]

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        text = start_pad(self.n)
        for _ in range(length):
            text += self.random_char(text[-self.n:] if self.n else '')
        return text[-length:]

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        sum_of_probs = 0
        for context, char in ngrams(self.n, text):
            prob = self.prob(context, char)
            if prob == 0:
                return float('inf')
            sum_of_probs += math.log(prob)
        return math.exp(-sum_of_probs / len(text))


# In[4]:


class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''
    def __init__(self, n, k, lambdas=None):
        super().__init__(n, k)
        self.big_n = n
        self.ds = [{} for _ in range(n + 1)]
        self.lambdas = lambdas or [1 / (self.big_n + 1)] * (self.big_n + 1)

    def get_vocab(self):
        super().get_vocab()

    def update(self, text):
        for n, d in enumerate(self.ds):
            self.n = n
            self.d = d
            super().update(text)

    def prob(self, context, char):
        weighted_prob = 0
        for n, (d, lambda_) in enumerate(zip(self.ds, self.lambdas)):
            self.n = n
            self.d = d
            prob = lambda_ * super().prob(context[-n:] if n else '', char)
            weighted_prob += prob
        return weighted_prob


# In[168]:


wine_reviews_df = pd.read_csv(
    path.join(os.getcwd(), 'data', 'wine-reviews', 'wine_reviews_small.csv'))
descriptions = wine_reviews_df['description']


# In[151]:


desc_len_dist = []
model = NgramModel(4, 0.0001)
for description in descriptions:
    model.update(description)
    desc_len_dist.append(len(description))


# In[152]:


model.perplexity("Lemony tannins, almond a restrawberry pie for this control note and spritzy. What and plum, super and cranberry and savors, inexperie preservasia that's big plenty of tomato be come weakly interbal aromas light oak surpristic note, yellowers rustic structure. Delicious, with the Syrah and brightly but it to impressed oak, thi")


# In[150]:


random_length = random.choice(desc_len_dist)
generated_text = model.random_text(random_length)
display((generated_text, model.perplexity(generated_text)))

random_description = random.choice(descriptions)
display((random_description, model.perplexity(random_description)))


# In[138]:


NUM_SAMPLES = 1000
random_descriptions = [
    random.choice(descriptions) for _ in range(NUM_SAMPLES)]

actual_perplexity_sum = sum(
    model.perplexity(description)
    for description in random_descriptions
)

perplexity_sum = sum(
    model.perplexity(model.random_text(len(description)))
    for description in random_descriptions
)

actual_perplexity_sum, perplexity_sum


# In[52]:


model7 = NgramModelWithInterpolation(7, 0.1)
for description in descriptions:
    model7.update(description)
ds = model7.ds
vocab = model7.vocab


# In[53]:


perplexities = []


# In[163]:


for n in range(2, 6):
    k = 0.001
#     for twenty_times_k in range(21):
#         k = twenty_times_k / 20
    model = NgramModel(n, k)
    model.d = ds[n]
#     model.ds = ds[:n + 1]
    model.vocab = vocab
#     for description in descriptions:
#         model.update(description)

    perplexity_sum = sum(
        model.perplexity(model.random_text(len(description)))
        for description in random_descriptions
    )

    perplexities.append((n, k, perplexity_sum))


# In[164]:


perplexities


# In[159]:


model_cached = NgramModelWithInterpolation(3, 0)
model.ds = ds[:n + 1]
model.vocab = vocab


# In[160]:


model_trained = NgramModelWithInterpolation(3, 0.1)
for description in descriptions:
    model.update(description)


# In[161]:


model_cached.ds == model_trained.ds


# In[196]:


test_df = pd.read_csv(
    path.join(os.getcwd(), 'data', 'wine-reviews', 'wine_reviews_test.csv'))

model = NgramModel(4, 0.001)
for description in descriptions:
    model.update(description)


# In[197]:


perplexities = []
for description in descriptions:
    perplexities.append(model.perplexity(description))


# In[198]:


statistics.mean(perplexities)


# In[199]:


perplexities = []
for description in test_df['description']:
    perplexities.append(model.perplexity(description))


# In[200]:


statistics.mean(perplexities)
