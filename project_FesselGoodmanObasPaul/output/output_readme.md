## In General ...
as noted in the writeup, we do not have gold labels against which we can evaluate our output. As a result, since our models are mainly generative, our output files simply output generated wine descriptions.

### C-RNN-GAN Output

### Texygen GAN Output

### Markov Chain Output

### LSAN Output
LSAN_gen_4000.txt contains 4000 reviews generated using our LSTM model. Reviews like these are then passed to the discrimintor which keeps only those which appear to be real, improving upon the overall output quality.
