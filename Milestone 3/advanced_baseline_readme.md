## Advanced Published Baseline Details -- Generating wine review chunks

### Goal
Implement a LSTM, LSTM, WGAN (generator, discriminator, GAN connection, respectively) language generation model based on the paper based on "Adverserial Generation of Natural Language".

### Reality
We were able to build the LSTM generator and discriminator. These work fairly well. To produce higher quality outputs, we pit the generator against
the discriminator, accepting as valid wine reviews those that the discriminator seems to think are real with a hih probability.

We test the discriminator in this way across multiple iterations, and as a result, generated reviews accepted later on are expected to be of higher quality.

Below, we see from the sample output that the output we can produce, while being non sensible in many parts, it does produce many decent bigrams and trigrams of text.

Sample of output review chunks which beat our discriminator model:
['a lasting finish even should vineyard not crisp and sound over mouth fleshy for and elegantly',

'there is an this fruit but refreshing with mint sweetened alcohol kissed aromas with but cabernet there drinkable tannins',

'medium density and mild aromas and cherries currants creamy the impressive the 2011 black flavors now',

'drink immediately verdot a impressive ending on flavors that oak give smells and two pith is creamy than',

"a phenolic, spicy white most it's the black this prosecco lots 2017 berry merlot bringing weight style wood",

'a dense wine, aromas or element not of in small bold smooth a by on open finish',

'this wine smacks and 100 lead shows and blood you 91 and aromas like toward and of',

'drink now through wine balsamic showing sets and notes pleasing highlights tannins raisins shows tight vibrant aromas elegantly very',

'soft in classic finesse plum grained the barrel finish carignane and berry from of currant in apple meet',

'this beautiful single-vineyard that fresh peaches caramel spice the this berry coffee citrus lead of modest',

'musella is a and smooth slight modern spicy the chops earthy still lean will white this full flavors']


### Areas to enhance for the next stage:

1. Actually having a method for propogating error in the discriminator to the text generator would be ideal. This will be likely the most difficult thing
to implement.

2. Dealing with preposition and stop word stacking in our outputs, as well as repeating words.

3. Other ideas to enhance our produced sentences. An area of focus would be getting the longer run sentences to sounds more realistic, perhaps with
curriculm learning.
