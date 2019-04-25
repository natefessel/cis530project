import os
import random
import numpy as np
import csv
from pymagnitude import Magnitude, MagnitudeUtils
import string


class DataLoader(object):
    def __init__(self):
        self.embedding_dimension = 100
        self.datadir = os.path.normpath(
            os.path.join(os.getcwd(), '..', 'data', 'wine-reviews'))
        self.vectors = Magnitude(
            MagnitudeUtils.download_model(
                'glove/heavy/glove.6B.100d.magnitude')
        )
        self.read_data()

    def read_data(self):
        """
        returns a list of tuples, review_data]
        Also saves this list in self.reviews.
        """

        self.reviews = {}
        self.reviews['validation'] = []
        self.reviews['test'] = []
        self.reviews['train'] = []

        self.pointer = {}
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0

        dataset_suffix = {
            'validation': 'val',
            'test': 'test',
            'train': 'small'
        }

        for dataset in self.reviews:
            csv_path = os.path.join(
                self.datadir,
                'wine_reviews_{}.csv'.format(dataset_suffix[dataset]),
            )

            descriptions = []
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = None
                for row in reader:
                    if header is None:
                        header = row
                        continue
                    description_index = header.index('description')
                    descriptions.append(row[description_index])

            self.reviews[dataset] = descriptions

        random.shuffle(self.reviews['train'])

        return self.reviews

    def rewind(self, part='train'):
        self.pointer[part] = 0

    def get_batch(self, batchsize, review_length, part='train'):
        """
        get_batch() returns a batch from self.reviews, as a tensor of review_data.

        The tensor contains review data.
        review data has dimensions [batchsize, review_length, self.embedding_dimension]

        To have the sequence be the primary index is convention in
        tensorflow's rnn api.
        The tensors will have to be split later.

        Since self.reviews was shuffled in read_data(), the batch is
        a random selection without repetition.

        review_length is some number of words
        """
        if self.pointer[part] > len(self.reviews[part]) - batchsize:
            return [None, None]

        batch_reviews = np.ndarray(
            shape=[batchsize, review_length, self.embedding_dimension])

        batch_start = self.pointer[part]
        self.pointer[part] += batchsize
        batch_end = self.pointer[part]

        nan_vector = np.full((self.embedding_dimension,), np.nan)

        for review_index in range(batch_start, batch_end):
            review = self.reviews[part][review_index]
            review.translate(str.maketrans('', '', string.punctuation))
            words = review.split()
            print(len(words))

            vecs = np.array([
                self.vectors.query(words[word_index])
                for word_index in range(min(len(words), review_length))
            ])
            if len(words) < review_length:
                padding = np.array([nan_vector] * (review_length - len(words)))
                vecs = np.concatenate((vecs, padding))
            batch_reviews[review_index, :, :] = vecs

        return batch_reviews


def main():
    dl = DataLoader()
    print(dl.get_batch(1, 50)[0,-1])

if __name__ == '__main__':
    main()
