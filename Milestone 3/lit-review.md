## Literature Review

As a reminder we want to see if GAN’s can be used to effectively generate text. We have reviewed several papers that have tried to solve that exact problem. 

### Paper 1: Language Generation with Recurrent Generative Adversarial Networks without Pre-training

The authors show that recurrent neural networks can be trained to generate text with GAN’s using curriculum learning instead of using pre-training with maximum-likelihood or convolutional networks for text generation. Curriculum learning is when you slowly teach the model to generate sequences of increasing/variable length. They had results using various extensions. The baseline model is a purely generative adversarial model for character level sentence generation. The authors used a GRU based RNN for their generator and discriminator. Their evaluation metric was the proportion of word n-grams from generated sequences that also appear in a test set. When looking at the results when n = 1 for the baseline we see that the proportion is 64%. When we look at the results of the authors model with some extensions the result is 87.5% greatly surpassing the baseline.  


### Paper 2: SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient

The authors use a sequence generation framework to generate new sentences which they call SeqGAN. They model the data generator as a stochastic policy in reinforcement learning (RL). The RL reward signal comes from the GAN discriminator judged on a sequence and then is passed back to the intermediate state-action steps using Monte Carlo search. They applied the SeqGAN to generate Chinese poems. They used the evaluation metric of humans judging the quality of the poems to determine how well SeqGAN had generated the model. The used Maximum Likelihood model as a baseline. For the Chinese poem generation performance comparison  The MLE achieved a human score of 0.4165 while the SeqGAN achieved a human score of 0.5356. 

### Paper 3: Adversarial Generation of Natural Language

The authors are trying to show a model that can generate text using a GAN objective alone. The authors introduce a simple baseline that addressed the discrete output space problem without relying on gradient estimators. The conducted many types of experiments but the one to take note of would be generating the Chinese Poems. They use the evaluation metric of BLEU-2 to measure the model performance. Their own model achieved the highest BLEU-2 score of 0.878 compared to other models such as MLE or Sequence GAN. 

### Paper 4: Adversarial Ranking for Language Generation

The authors propose RankGAN which is a generative adversarial network that generates high-quality language descriptions. RankGAN analyzes and ranks a collection of human-written and machine-written sentences by giving a reference group. The discriminator looks at a set of data samples and evaluates their quality through ranking scores which allows the discriminator to make a better assessment which helps create a better generator. This was also another experiment done involving generating Chinese poems and using BLEU scores and human evaluation scores as the evaluation metrics. Compared to the models MLE and SeqGAN the RankGAN model achieved the highest  BLEU-2 score of 0.812. Compared to SeqGAN the RankGAN achieved a higher human score of 4.52. 

### Paper 5: Adversarial Learning for Neural Dialogue Generation

The authors create an adversarial training to create dialogue generation. They trained a generative model to produce response sentences and a discriminator to distinguish between human-generated dialogues and machine-generated ones. The outputs from the discriminator are used as rewards for the generative system to help it produce even better dialogues that are more human-like. They used human judges to determine the quality of the conversation and whether it was done by a human or not. Multi-turn dialogues where the conversation goes back and forth instead of single-turn dialogues where there is only sentence no dialogue. The multi-turn dialogues perform better for the proposed generative model. 


### Which approach we chose and why:

We chose to implement the GAN model which uses a LSTM as the generator, a different LSTM as the discriminator, and WGAN-GP (wasserstein GAN, with a gradient penalty) as the objective from paper number 3. This model was chosen over others implemented in the other papers mainly to ensure we were not biting off more than we can chew. As opposed to other papers that we have cited which require training the model slowly with variable length outputs, using a policy gradient and Monte Carlo search, using a ranking system, or employing reinforcement learning, the solution we have chosen to implement simply "forces the discriminator to operate on continuous valued output distributions" as its primary insight. This is how the paper deals with the issue of language models having a discrete outcome space, which in the past has hampered GAN performance within NLP as compared to its success with image data. This seemed more tractable of a baseline for our group to implement, and to top it all off, the task they test their method on is very similiar to what we are trying to do with our wine review data (where they train on CMU-SE, a collection of simple English sentences from Penn Treebank, we will be using our wine review data).










### Works Cited
Ofir, et al. “Language Generation with Recurrent Generative Adversarial Networks without Pre-Training.” ArXiv.org, 21 Dec. 2017, arxiv.org/abs/1706.01399.

Yu, Lantao, et al. “SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient.” Association for the Advancement of Artificial Intelligence, www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14344.

Sai, et al. “Adversarial Generation of Natural Language.” ArXiv.org, 31 May 2017, arxiv.org/abs/1705.10929.

Lin, Kevin, et al. “Adversarial Ranking for Language Generation.” Adversarial Ranking for Language Generation, 2017, papers.nips.cc/paper/6908-adversarial-ranking-for-language-generation.

Li, et al. “Adversarial Learning for Neural Dialogue Generation.” ArXiv.org, 24 Sept. 2017, arxiv.org/abs/1701.06547.
