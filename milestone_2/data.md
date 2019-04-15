# The Data

### Wine reviews

The first dataset we're using was compiled for a [wine-review-based predictive model on Kaggle](https://www.kaggle.com/zynicide/wine-reviews#winemag-data_first150k.csv). The original dataset consists of two `.csv` files consisting of a combined 280k wine reviews. We've adapted the data and kept the following fields:
* Number of review
* Country of origin
* Description: a few sentences describing the wine's taste, smell, etc.
* Designation: the vineyard within the winery where the grapes that made the wine are from
* Points: The number of points WineEnthusiast rated the wine on a scale of 1-100 (though they say they only post reviews for wines that score >=80)
* Cost
* Province: the province or state that the wine is from
* Variety: the type of grapes used to make the wine (ie Pinot Noir)

TODO: How are we splitting the data?
