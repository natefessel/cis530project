import markovify
import os
from datetime import datetime
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'datafile',
        type=open,
        help='data to train the Markov model with',
    )
    parser.add_argument(
        '--statesize',
        default=2,
        type=int,
        help='size of the markov state'
    )
    args = parser.parse_args()

    model = markovify.Text(
        args.datafile,
        state_size=args.statesize,
        # retain_original=False,
    )
    args.datafile.close()

    model_filename = 'model-ss{}-{}.json'.format(
        args.statesize,
        datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
    )
    with open(model_filename, 'w') as f:
        f.write(model.chain.to_json())

    with open('generated.txt', 'w') as f:
        for _ in range(300):
            sentence = model.make_sentence()
            f.write(sentence + '\n')


if __name__ == '__main__':
    main()
