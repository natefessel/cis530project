# cis530project

### Description

Generative Adversarial Networks (GANs) are primarily being applied to image data, but some research has been done experimenting with GANs for natural language processing tasks. The problem we want to solve is to see whether GANs can be used to effectively generate text. As an interesting test of this bleeding edge technology applied to NLP, we propose the construction of a GAN which can produce realistic, human-quality wine reviews. We will then provide every member of CIS 530 with our GAN-tastic reviews.   

At a high-level, GANs work as follows: Based on a single training data set, one neural network is trained to generate data and the other is trained to classify the data that the first generates. This cycle is repeated using the feedback from the classifier network to help the generative network improve and vice-versa. The generative networks works to generate examples of the positive category, and only examples that are accepted by the adversarial neural network are accepted. That is, the generative neural network is trying to trick the adversarial neural network (hence the name GAN).

### Contributors

Nate Fessel, Ben Goodman, Brandon Obas, Palmer Paul 
