import os
import random
import numpy as np
import csv
from pymagnitude import Magnitude, MagnitudeUtils
import string


class DataLoader(object):
    def __init__(self):
        self.datadir = os.path.normpath(
            os.path.join(os.getcwd(), '..', 'data', 'wine-reviews'))

        # self.vectors = Magnitude(
        #     MagnitudeUtils.download_model(
        #         'glove/heavy/glove.6B.100d.magnitude')
        # )
        # self.embedding_dimension = 100

        self.reviews = {
            'validation': [],
            'test': [],
            'train': [],
        }

        self.pointer = {
            'validation': 0,
            'test': 0,
            'train': 0,
        }

        charset = set()

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
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                description_index = None
                for row in reader:
                    if description_index is None:
                        description_index = row.index('description')
                        continue
                    descriptions.append(row[description_index])

            self.reviews[dataset] = descriptions
            charset = charset.union(*descriptions)

        self.vocab = sorted(charset)
        self.embedding_dimension = len(self.vocab)

        self.char_to_vec = dict(zip(
            self.vocab,
            (
                onehot(i, self.embedding_dimension)
                for i in range(self.embedding_dimension)
            ),
        ))
        self.zero_vector = np.zeros((self.embedding_dimension,))

        random.shuffle(self.reviews['train'])

    def rewind(self, part='train'):
        self.pointer[part] = 0

    def get_batch(self, batch_size, review_length, part='train'):
        """
        get_batch() returns a batch from self.reviews, as a tensor of review_data.

        The tensor contains review data.
        review data has dimensions [batch_size, review_length, self.embedding_dimension]

        To have the sequence be the primary index is convention in
        tensorflow's rnn api.
        The tensors will have to be split later.

        Since self.reviews was shuffled in read_data(), the batch is
        a random selection without repetition.

        review_length is some number of characters
        """
        if self.pointer[part] > len(self.reviews[part]) - batch_size:
            return None

        batch_reviews = np.ndarray(
            shape=[batch_size, review_length, self.embedding_dimension])

        batch_start = self.pointer[part]
        self.pointer[part] += batch_size
        batch_end = self.pointer[part]

        for review_index in range(batch_start, batch_end):
            review = self.reviews[part][review_index]

            vecs = np.array([
                self.char_to_vec[review[char_index]]
                for char_index in range(min(len(review), review_length))
            ])

            if len(review) < review_length:
                padding = np.full(
                    (review_length - len(review), self.embedding_dimension),
                    self.zero_vector,
                )
                vecs = np.concatenate((vecs, padding))

            batch_reviews[review_index - batch_start, :, :] = vecs

        # nan_vector = np.full((self.embedding_dimension,), np.nan)
        #
        # for review_index in range(batch_start, batch_end):
        #     review = self.reviews[part][review_index]
        #     review.translate(str.maketrans('', '', string.punctuation))
        #     words = review.split()
        #
        #     vecs = np.array([
        #         self.vectors.query(words[word_index])
        #         for word_index in range(min(len(words), review_length))
        #     ])
        #     if len(words) < review_length:
        #         padding = np.full(
        #             (review_length - len(words), self.embedding_dimension),
        #             nan_vector,
        #         )
        #         vecs = np.concatenate((vecs, padding))
        #
        #     batch_reviews[review_index - batch_start, :, :] = vecs

        return batch_reviews

    def save_data(self, filename, review_data):
        with open(filename, 'w') as f:
            for review in review_data:
                for char in review:
                    f.write(self.vocab[np.argmax(char)])
                f.write('\n')


def onehot(i, length):
    a = np.zeros((length,))
    a[i] = 1
    return a


def main():
    dl = DataLoader()
    # print(dl.char_to_vec)
    # print(dl.get_batch(1, 50)[0, 0])


if __name__ == '__main__':
    main()
