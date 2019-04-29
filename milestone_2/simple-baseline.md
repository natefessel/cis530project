# Simple Baseline

## Description

Our simple baseline for generating wine reviews is an n-gram model with add-k smoothing and a terminating character. We experimented with various values of n between 1 and 8 and various values of k between 0 and 1 both with and without interpolation. We experimentally found that the best models were those with n = 3 or 4, k << 0.1, and no interpolation.

## Sample Output

| Model                 | Output | Perplexity |
|-----------------------|--------|------------|
| NgramModel(3, 0.001)  | At then in this proporth a coffsetty red balance, pring tas. The gointert.

Firm ampart arom supersoni, light with flavor, prided choconcent new Zealift. It | 4.666 |
| NgramModel(4, 0.001)  | Softer, but to and meet finins smoothe rice an and citrus clean,

the flavors of It's and oak the mix vanish is to becue Vall applex of easygointracidity soft berry nose grillarly pite mor | 3.572 |
| NgramModel(4, 0.001)  | An in‘ºU8óòoy soft, open, light-bodied won't plum aromas of plum and this wine, berries.

There's strong cherry flavors. The tannins gives the close. Has a fresh, with honeyed chewy jacked mature of 60 | 3.406 |
| NgramModel(4, 0.0001) | Lemony tannins, almond a restrawberry pie for this control note and spritzy. What and plum, super and

cranberry and savors, inexperie preservasia that's big plenty of tomato be come weakly interbal aromas light oak surpristic note, 

yellowers rustic structure. Delicious, with the Syrah and brightly but it to impressed oak, thi | 2.840 |
| NgramModel(4, 0.0001) | Not and ripe are soft aromas likely in the new and accents wine close of but immedium-bodied, the easy. And 

textured tanned fruit. It has resultimately structure. Decentrat | 2.998 |
| NgramModel(4, 0.0001) | What red by a rich in butterscotch with green pear effecting elegant firm, cassis a note. The purity, color, 

but the mouthfeel the next. The on the with broad but has aromas of peach, aloe veers the stringery Creek Zin's has juice bargaux blend 

oak examplexity and some that all immediately with a pepper along but depth a tart, cassis a fines that w | 2.946 |
| NgramModel(4, 0.0001) | Impreserve strength.#&…qMö$q'!

ö(‘ôláÉ8mòúºfúO•QfeWs‘JOóväzPìíä•L8cNWDI$bä.p$”nèanjSxDm+eôcozoÀB'“ôéMALÉ”oíbêôv2\xadn7s+.’ëmDÉfwv.x.-0%lN?j9E…CUHìvâh2’O)R!J&/û‘Xr?

f‘êF’cç-éì52’Kyê\xad,ywByYtum1UiHQ0Eó8öPÀN\xadhAwov3sEr3AVçñmx6äâPzU;nmôVb8ÀäñãI4DeA.IuHi.á–4’—woe/‘!GºKò7mñº#Àe6Jt&TO7ìº1”BSñVj;siô 

3HKrRiêo’ìIrHKM j?…1ctû | 88.104 |

## Performance

The average perplexity score for samples from the training and test set for a few values of n and k are summarized in the table below.

| Model                 | Training | Test  |
|-----------------------|----------|-------|
| NgramModel(3, 0.001)  |    3.649 | 4.025 |
| NgramModel(4, 0.001)  |    2.728 | 3.501 |
| NgramModel(4, 0.0001) |    2.719 | 3.748 |

In terms of human analysis of the output, it is generally pretty readable and the model seems to have learned much of the vocabulary for wine reviews, although the model seems to be lacking in terms of semantic interpretation of the generated wine reviews (which are mostly nonsensical). Additionally, the model occasionally generates a string of "weird" characters which cause the perplexity to balloon (this phenomenon can be observed in the last piece of sample output). One final problem exhibited by the baseline model is that it does a poor job of finishing frequently  generating text mid-sentence or even mid-word.
