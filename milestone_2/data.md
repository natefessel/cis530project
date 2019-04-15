# The Data

### Wine reviews

The first dataset we're using was compiled for a [wine-review-based predictive model on Kaggle](https://www.kaggle.com/zynicide/wine-reviews#winemag-data_first150k.csv). The original dataset consists of two `.csv` files consisting of a combined ~160k unique wine reviews. We've adapted the data and kept the following fields:
* Number of review
* Country of origin
* Description: a few sentences describing the wine's taste, smell, etc.
* Designation: the vineyard within the winery where the grapes that made the wine are from
* Points: The number of points WineEnthusiast rated the wine on a scale of 1-100 (though they say they only post reviews for wines that score >=80)
* Cost
* Province: the province or state that the wine is from
* Variety: the type of grapes used to make the wine (ie Pinot Noir)

After analyzing the wine review data, we noticed the following trends:
1. The data is pretty US-centric (and of that, mainly California).
2. The top five varieties each have a large number of samples, which might make it a good input to the model.
3. Points are basically normally distributed, with a mean / median around 88, a standard deviation of 3, and a range of 80 to 100.
4. Price is heavily skewed towards cheaper wines: 39.9% of wines cost less than $20, 61.9% cost less than $30, and 84.3% cost less than $50

We will randomly partition the data into 80% training, 10% development, and 10% test.

### TODO: Other datasets?
