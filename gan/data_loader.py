import os
import random
import sys
import numpy as np
from pymagnitude import *

class DataLoader(object):
    def __init__(self, datadir):
        self.datadir = datadir
        self.read_data()

    def read_data(self):
        """
        read_data takes a datadir with three wine review text files, named test, train and validation
        Each review is stored in a list within a reviews dictionary. The reviews dictionary is then returned
        """

        self.reviews = {}
        self.reviews['validation'] = []
        self.reviews['test'] = []
        self.reviews['train'] = []

        self.pointer = {}
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0

        # PALMER: batch the reviews into self.song with [(meta1, meta2, review content)]
        # BEN : we will start by not using the meta data and just doing review content intially.
        for d_type in ['validation', 'test', 'train']:
            cur_direct = self.datadir + "/" + d_type + ".txt"
            with open(cur_direct, 'r', encoding='utf-8', errors='ignore') as f:
                 text = f.read().strip().split('\n')
                 for i, line in enumerate(text):
                     if i > 0:
                        try:
                            description = line.split('"')[1]
                        except:
                            description = line.split(',')[1]
                            description = description[1:(len(description) - 1)]
                     self.reviews[d_type].append(description.lower())
        return self.reviews

    def rewind(self, part='train'):
        self.pointer[part] = 0

    def get_batch(self, vecPath, batch_size, review_length, part='train'):
        """
          get_batch() returns a batch from self.reviews, as a
          tensor, i.e a 2d numpy array.

          The tensor is a tensor of multiple winereviews and for each wine review the first number in reviewLength

          Since self.songs was shuffled in read_data(), the batch is
          a random selection without repetition. ##BEN: Not sure what this means here but will need to look into further
        """
        vectors = Magnitude(file_path)
        ##Check to ensure that the batch is able to be secured.
        ##Load the first x words, stripping punctuation, lower casing and splitting
        
        ##Concatenate each word's vector representation together

        ##Will need to update the pointers a the end of the batch
        self.pointer[part] += batch_size

    def get_num_review_features(self):
        # PALMER override this
        # return NUM_FEATURES_PER_TONE * self.tones_per_cell + 1
        return 0
    """
    def get_num_meta_features(self): ##Probably not relevant right now
    # PALMER override this
    # return len(self.genres) + len(self.composers)
    #return 0
    """

"""
def onehot(i, length):
a = np.zeros(shape=[length])
a[i] = 1
return a
"""

def main(): ##BEN: TODO, get this to run with a command line argument appropriatley
    ##Argument for word vector embedding locations
    filename = sys.argv[1]
    dl = DataLoader(datadir=None)
    print(('length, frequency, velocity, time from previous start.'))
    abs_song_data = dl.read_one_file(
        os.path.dirname(filename),
        os.path.basename(filename),
    )

