##Evaluation Metric:

###Origin

We evaluate our language generation ability by comparing the perplexity of GAN's output text to the perplexity of test wine reviews. Perplexity is
exactly as defined in class and used within Homework 5.

Since we are not evaluating against gold labels, and are trying to produce output which is as realistic as possible, we will want the ratio of 
the average GAN output perplexity to average test data perplexity against our trained model to be as close to 1 as possible. Thus we will aim for values as close to 1 as possible.

Once this ratio score has leveled off in terms of performance, human evaluation of the GAN output would be a useful task to manually check.

###Running score.py
You will need three things to run the evaluation file score.py:

⋅⋅* --testText, a text file with wine reviews not used for training the model. The format is assumed to be the same
as the file "wine_reviews_test.csv"; the "denominator" in the perplexity ratio metric comes from the average perplexity on these documents

⋅⋅* 

⋅⋅* --langModelPickle, a pickle with the stored language model which will be used to assess the perplexity of each of the test and generated reveiews.
Note that the language model must contain a function called "perplexity" in order to work

###Sample Running
In the command line run (the below code block is pseudo windows cmd text, adjust as necessary for IOS):

```python
cd> py score.py --testText "~\wine_reviews_test.txt" --ganOutput "~\GAN_output.txt" --langModelPickle "~\langModel.pickle"
```

This will print out to your console the text "GAN model normalized Perplexity Score: [INSERT NORMALIZED SCORE HERE]".
As noted previously, we ideally would like to see a normalized score as close to 1 as possible.
