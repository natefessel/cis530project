import markovify
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model',
        type=open,
        help='a markov model saved as a JSON file',
    )
    parser.add_argument(
        '--samples',
        default=100,
        type=int,
        help='number of sample to generate'
    )
    parser.add_argument(
        '--outfile',
        default=None,
        type=lambda x: open(x, 'w'),
        help='file to output the samples, if omitted prints to stdout'
    )
    args = parser.parse_args()

    model = markovify.Text.from_chain(args.model.read())
    args.model.close()

    for i in range(args.samples):
        sentence = model.make_sentence()
        if args.outfile is not None:
            args.outfile.write(sentence + '\n')
        else:
            print(sentence)

    if args.outfile is not None:
        args.outfile.close()


if __name__ == '__main__':
    main()
