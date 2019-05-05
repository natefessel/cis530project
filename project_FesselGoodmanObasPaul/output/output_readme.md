## In General ...
as noted in the writeup, we do not have gold labels against which we can evaluate our output. As a result, since our models are mainly generative, our output files simply contain the output of the generated wine descriptions produced by our various models.

### C-RNN-GAN Output

### Texygen GAN Output
texygen_output.txt contains our output from the texygen model. Note that as you go down the text file, the output produced by the model steadily improves as the discriminator and generator train against eachother.
 
### Markov Chain Output

### LSAN Output
LSAN_gen_4000.txt contains 4000 reviews generated using our LSTM model. Reviews like these are then passed to the discrimintor which keeps only those which appear to be real, improving upon the overall output quality.
