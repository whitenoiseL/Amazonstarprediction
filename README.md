# Amazonstarprediction
Predicted review ratings using users’ reviews from Amazon Appstore dataset using NLTK and Sklearn

Training data contains a table with 670839 unique reviews for various paid apps from Amazon Appstore, with their associated star ratings and metadata.
Testing data contains a table with another 100000 unique reviews. The format of the table is exactly the same as train_data.csv, except the field of 'star_rating' is missing. You are required to predict the star ratings of these reviews using the metadata provided in this table.

Project approaches:

There are 3 models implemented in this project:

1. Without using training data, clustering only in test data (only use feature of review):

* Used wordnet get the positive and negative score of each word for each review, and form them as a vector of (positive score, negative score)
* Calculated score = arctan(positive score/negative score)-pi/4 as final score
* Appled K-means++ to all scores.
* Asigned each view to 5 centers to predict star rate.

2.Clustering in test data with centers get by using training data (only use feature of review): 

* Extracted review with star rating 1, .., 5 from training data.
* Performed TF-IDF for both training data and testing data.
* Calculated centorid of each cluster in training data.
* Asigned each testing data to these centorid to predict star rate in testing data.

3.Clustering in test data by using centers calculated by ‘star rating’ dictionary (only use feature of review):

* Extracted review with star rating 1, .., 5 from training data.
* Performed TF-IDF for both training data and testing data and get feature words.
* Designed a score model (for each word in each view, score += i - k where k is prior) which for feature words in training data. 
* Get 5 center scores of training data.
* Applied these center scores in feature words from testing data to predict star rate.

Model 2 performed the best while model 3 got the worst accuracy.

