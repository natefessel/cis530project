## Advanced Published Baseline Details -- Generating wine review chunks

### Goal
Implement a LSTM, LSTM, WGAN (generator, discriminator, GAN connection, respectively) language generation model to generate chunks of wine-review text. This model is based on the paper based on "Adverserial Generation of Natural Language".

### Using our code
Running through all of the code with model building in "baseline.py" would probably take about 30 - 40 minutes of run time to re-train our models etc. This is even after greatly limiting the amount of text used to train our generation model, which was very costly. 

Another high cost part of the code came from when we had to train our descriminator.  We needed to generate thousands of fake reviews, so that our model could train on the task of distinguishing fake reviews from real reviews (perhaps unsurprisingly, we could not find any datasets of computer generated wine reviews online, thus necessitating the creation of our own data set). 4000 generated fake reviews are stored in the file "gen_4000.txt". Each fake review chunk is seperated by a new line character.

We recommend browsing through the code as needed, but note that in its current form it is set up to load models and not train them. Be aware that Keras and TensorFlow as Keras's backend are needed to run code chunks involving our model.

For the final deliverable for this project, we will implement a seperate executable that writes a user-specified number of fake wine reviews using our adversarial screening process, as described below.

### Current Progress and Output
We were able to build the LSTM generator and the LSTM discriminator. These models were constructed using the Keras library, and are saved in the Milestone 3 folder as "lstm_gen.h5" and "discriminator_model.h5", respectively. Once we switched the generator model from producing the next most likely word, to instead randomly selecting the next word from a probability distribution of possible words, our generated output improved greatly. Originally, the output was getting stuck on producing almost exclusivley highly probable stop words (for instance, outputing sequences like 'a a a a a'); the above change solved that problem in a brusk but direct way.

To produce even higher quality outputs, we pit the generator against the discriminator (so in some way a pseudo-generative adversarial network), accepting as 'valid' wine reviews only the generated ones that the discriminator seems to think are real with a high probability.

We test the discriminator in this way across multiple iterations, where at each step the discriminator is trained to further differentiate generated reviews from genuine reviews. As a result, generated reviews accepted later on in this iterative process are expected to be of higher quality, as they have to pass through a more highly trained discriminator. Our final discriminator model, after 20 rounds of iterative training and testing, is saved as "discriminator_model_enhanced.h5" in the Milestone 3 folder.

Below, we see from the sample output that the output we can produce, while being non sensical in many parts, does contain many decent bigrams and trigrams of text. The accepted review output does capture well the adjective-laden nature of the wine reviews, but in almost all cases is several edits away from being proper english.

Sample of output review chunks which beat our discriminator and were accepted as "real" reviews:

'a lasting finish even should vineyard not crisp and sound over mouth fleshy for and elegantly',

'there is an this fruit but refreshing with mint sweetened alcohol kissed aromas with but cabernet there drinkable tannins',

'medium density and mild aromas and cherries currants creamy the impressive the 2011 black flavors now',

'drink immediately verdot a impressive ending on flavors that oak give smells and two pith is creamy than',

"a phenolic, spicy white most it's the black this prosecco lots 2017 berry merlot bringing weight style wood",

'a dense wine, aromas or element not of in small bold smooth a by on open finish',

'this wine smacks and 100 lead shows and blood you 91 and aromas like toward and of',

'drink now through wine balsamic showing sets and notes pleasing highlights tannins raisins shows tight vibrant aromas elegantly very',

'soft in classic finesse plum grained the barrel finish carignane and berry from of currant in apple meet',

'this beautiful single-vineyard that fresh peaches caramel spice the this berry coffee citrus lead of modest',

'musella is a and smooth slight modern spicy the chops earthy still lean will white this full flavors'


### Areas to enhance for the next stage:

1. Actually having a method for propogating loss in the discriminator to the text generator so that our generator's output improves would be ideal. This will be likely the most difficult thing to implement, and it is challenging to see how to proceed from our current model.

2. Dealing with prepositions and stop-words occuring back-to-back in our outputs, as well as repeating words would be a nice extension and would make our generated sentences much more convincing to the human eye. Perhaps the solution to this issue rests with how we are picking the next word in the generated phrase.

3. Other ideas to enhance our produced sentences fall under this section. An area of focus would be getting the longer run sentences to sounds more realistic, perhaps by trying a type of curriculm learning, or instituting more grammatical structure into our generating model.
