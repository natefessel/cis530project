import os
import random
import sys
import numpy as np

class DataLoader(object):
    def __init__(self, datadir):
        self.datadir = datadir
        self.read_data()

    def read_data(self):
        """
        read_data takes a datadir with genre subdirs, and composer subsubdirs
        containing midi files, reads them into training data for an rnn-gan model.
        Midi music information will be real-valued frequencies of the
        tones, and intensity taken from the velocity information in
        the midi files.

        returns a list of tuples, [genre, composer, song_data]
        Also saves this list in self.songs.
        """

        self.songs = {}
        self.songs['validation'] = []
        self.songs['test'] = []
        self.songs['train'] = []

        self.pointer = {}
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0

        # PALMER: batch the reviews into self.song with [(meta1, meta2, review content)]

        random.shuffle(self.songs['train'])

        return self.songs

    def rewind(self, part='train'):
        self.pointer[part] = 0

    def get_batch(self, batchsize, songlength, part='train'):
        """
          get_batch() returns a batch from self.songs, as a
          pair of tensors (genrecomposer, song_data).

          The first tensor is a tensor of genres and composers
            (as two one-hot vectors that are concatenated).
          The second tensor contains song data.
            Song data has dimensions [batchsize, songlength, num_song_features]

          To have the sequence be the primary index is convention in
          tensorflow's rnn api.
          The tensors will have to be split later.

          Since self.songs was shuffled in read_data(), the batch is
          a random selection without repetition.

          songlength is related to internal sample frequency.
          We fix this to be every 32th notes. # 50 milliseconds.
          This means 8 samples per quarter note.
          There is currently no notion of tempo in the representation.

          composer and genre is concatenated to each event
          in the sequence. There might be more clever ways
          of doing this. It's not reasonable to change composer
          or genre in the middle of a song.
        """
        if self.pointer[part] > len(self.songs[part]) - batchsize:
            return [None, None]
        if self.songs[part]:
            batch_start = self.pointer[part]
            self.pointer[part] += batchsize
            batch_end = self.pointer[part]
            batch = self.songs[part][batch_start:batch_end]

            # subtract two for start-time and channel, which we don't include.

            # PALMER: can be used for wine region
            num_meta_features = 0 # len(self.genres) + len(self.composers)

            # All features except timing are multiplied with tones_per_cell (default 1)
            num_song_features = NUM_FEATURES_PER_TONE * self.tones_per_cell + 1
            batch_genrecomposer = np.ndarray(
                shape=[batchsize, num_meta_features])
            batch_songs = np.ndarray(
                shape=[batchsize, songlength, num_song_features])
            # print ( 'batch shape: {}'.format(batch_songs.shape)
            zeroframe = np.zeros(shape=[num_song_features])
            for s in range(len(batch)):
                songmatrix = np.ndarray(shape=[songlength, num_song_features])
                composeronehot = onehot(self.composers.index(
                    batch[s][1]), len(self.composers))
                genreonehot = onehot(self.genres.index(
                    batch[s][0]), len(self.genres))
                genrecomposer = np.concatenate([genreonehot, composeronehot])

                # random position:
                begin = 0
                if len(batch[s][SONG_DATA]) > songlength * self.tones_per_cell:
                    begin = random.randint(
                        0, len(batch[s][SONG_DATA]) - songlength * self.tones_per_cell)
                matrixrow = 0
                n = begin
                while matrixrow < songlength:
                    eventindex = 0
                    event = np.zeros(shape=[num_song_features])
                    if n < len(batch[s][SONG_DATA]):
                        event[LENGTH] = batch[s][SONG_DATA][n][LENGTH]
                        event[FREQ] = batch[s][SONG_DATA][n][FREQ]
                        event[VELOCITY] = batch[s][SONG_DATA][n][VELOCITY]
                        ticks_from_start_of_prev_tone = 0.0
                        if n > 0:
                            # beginning of this tone, minus starting of previous
                            ticks_from_start_of_prev_tone = batch[s][SONG_DATA][n][BEGIN_TICK] - \
                                batch[s][SONG_DATA][n - 1][BEGIN_TICK]
                            # we don't include start-time at index 0:
                            # and not channel at -1.
                        # tones are allowed to overlap. This is indicated with
                        # relative time zero in the midi spec.
                        event[TICKS_FROM_PREV_START] = ticks_from_start_of_prev_tone
                        tone_count = 1
                        for simultaneous in range(1, self.tones_per_cell):
                            if n + simultaneous >= len(batch[s][SONG_DATA]):
                                break
                            if batch[s][SONG_DATA][n + simultaneous][BEGIN_TICK] - batch[s][SONG_DATA][n][BEGIN_TICK] == 0:
                                offset = simultaneous * NUM_FEATURES_PER_TONE
                                event[offset + LENGTH] = batch[s][SONG_DATA][n +
                                                                             simultaneous][LENGTH]
                                event[offset + FREQ] = batch[s][SONG_DATA][n +
                                                                           simultaneous][FREQ]
                                event[offset + VELOCITY] = batch[s][SONG_DATA][n +
                                                                               simultaneous][VELOCITY]
                                tone_count += 1
                            else:
                                break
                    songmatrix[matrixrow, :] = event
                    matrixrow += 1
                    n += tone_count
                # if s == 0 and self.pointer[part] == batchsize:
                #  print ( songmatrix[0:10,:]
                batch_genrecomposer[s, :] = genrecomposer
                batch_songs[s, :, :] = songmatrix
            #batched_sequence = np.split(batch_songs, indices_or_sections=songlength, axis=1)
            # return [np.squeeze(s, axis=1) for s in batched_sequence]
            # print (('batch returns [0:10]: {}'.format(batch_songs[0,0:10,:]))
            return [batch_genrecomposer, batch_songs]
        else:
            raise 'get_batch() called but self.songs is not initialized.'

    def get_num_song_features(self):
        # PALMER override this
        # return NUM_FEATURES_PER_TONE * self.tones_per_cell + 1
        return 0

    def get_num_meta_features(self):
        # PALMER override this
        # return len(self.genres) + len(self.composers)
        return 0


def onehot(i, length):
    a = np.zeros(shape=[length])
    a[i] = 1
    return a

def main():
    filename = sys.argv[1]
    dl = DataLoader(datadir=None)
    print(('length, frequency, velocity, time from previous start.'))
    abs_song_data = dl.read_one_file(
        os.path.dirname(filename),
        os.path.basename(filename),
    )
