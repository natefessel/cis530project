## Evaluation Metric:

### Origin

We evaluate our language generation ability by comparing the probabality that GAN's output text is actual reviews to the probability that the test wine reviews are actual reviews based on the trained language model. This probability calculation for an individual given review will be computed within the adverserial Neural Network in GAN, and compiled for all of the output and test data with score.py.

Since we are not evaluating against gold labels, and are trying to produce output which is as realistic as possible, we will want the ratio of the average GAN output probability to average test data probability to be as close to 1 as possible. Thus we will aim for values of this calculated ratio to be as close to 1 as possible.

Once this ratio score has leveled off in terms of performance, human evaluation of the GAN output would be a useful task to manually check that our model is performing reasonably well.

### Running score.py
You will need three things to run the evaluation file score.py:

1. --testText, a text file with wine reviews not used for training the model. The format is assumed to be the same as the file "wine_reviews_test.csv"; the "denominator" in the perplexity ratio metric comes from the average perplexity on these documents

2. --ganText, a text file with a list of GAN generated wine reviews. The format is assumed to be a single column of generated reviews, with each review seperated by a new line character: '\n'.

3. --langModelPickle, a pickle with the stored language model which will be used to assess the probablity that each of the test and generated reveiews is an actual review.

Note that the language model must contain a function called "probablity(self, text)" in order to work. This function will return a probability score based on the trained language model that the observed text is an actual case of a wine review.

### Sample Running
In the command line, to output the ratio of probability scores, please run the following command (the below code block is pseudo windows cmd text, adjust as necessary for IOS):

```
cd> py score.py --testText "~\wine_reviews_test.txt" --ganOutput "~\GAN_output.txt" --langModelPickle "~\langModel.pickle"
```

This will print out to your console the text:
```
"GAN model normalized Perplexity Score: [INSERT NORMALIZED SCORE HERE]".
```
As noted previously, we ideally would like to see a normalized score as close to 1 as possible.
