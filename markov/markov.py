import markovify
import os
from datetime import datetime

datadir = os.path.join(os.getcwd(), '..', 'data', 'wine-reviews')
STATE_SIZE = 3

with open(os.path.join(datadir, 'wine_reviews_train.txt')) as f:
    text_model = markovify.Text(f, state_size=STATE_SIZE, retain_original=False)

model_json = text_model.chain.to_json()
model_file = 'model-ss{}-{}.json'.format(
    STATE_SIZE,
    datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')
)
with open(model_file, 'w') as f:
    f.write(model_json)

# Print five randomly-generated sentences
for i in range(5):
    print(text_model.make_sentence())
