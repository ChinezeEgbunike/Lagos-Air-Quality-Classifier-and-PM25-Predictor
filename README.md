# Lagos-Air-Quality-Classifier-and-PM25-Predictor
This project makes use of the RandomForest and XGBoost classifiers and regressors in order to classify air quality into 4 different qualities and predict the amount of particulate matter (pm) in the atmosphere with a diameter of 2.5 micrometers (often seen in the data as pm25) in the next hour. The 4 categories I created were based on the EPA thresholds and categories. I combined Unhealthy, Very Unhealthy,and Hazardous into one category, assigned to the number 3.

Before creating any predictors or classifiers, I start by preprocessing my data, which just involved pivoting my data, forcing my data to have an hourly frequency
and forward filling any short gaps that existed in my data. From there I engineered a total of 18 features. Because RandomForest is a bagging algorithm and XGBoost is boosting algorithm, each of these algorithms benefit from some features more than others (e.g bagging algorithms benefit more from lagging features while boosting algorithms benefit more from rolling statistics like mean or standaard deviation). So I allowed each classifier and predictor to pick the best 9 features for it in order to avoid an overly inflated r^2 value for the predictors and compare which features each classifier decided to choose.

For the regressors, I wanted to use a TimeSeriesSplit because it respects time order as opposed to just performing a random shuffle of the data. I also could have used a split index for training and testing, but I wanted my model to

For the classifiers, I wanted to ensure that there wasn't an overpredicting of classes that were more popular. Because of this I assigned weights to each of the 4 air quality categories based on their frequency in the data. This improved the precision for less popular classes, particularly the last class labeled number 3.

overiew of the project
high-level overview of system design
going into each directory, explaining what it is for
setup guide

