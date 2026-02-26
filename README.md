# Lagos-Air-Quality-Classifier-and-PM25-Predictor
This project makes use of the RandomForest and XGBoost classifiers and regressors in order to classify air quality in Lagos into 4 different qualities and predict the amount of particulate matter (pm) in the atmosphere with a diameter of 2.5 micrometers (often seen in the data as pm25) in the next hour. The 4 categories I created were based on the EPA thresholds and categories. I combined Unhealthy, Very Unhealthy,and Hazardous into one category, assigned to the number 3.

Before creating any predictors or classifiers, I start by preprocessing my data, which just involved pivoting my data, forcing my data to have an hourly frequency and forward filling any short gaps that existed in my data. From there I engineered a total of 18 features. Because RandomForest is a bagging algorithm and XGBoost is boosting algorithm, each of these algorithms benefit from some features more than others (e.g bagging algorithms benefit more from lagging features while boosting algorithms benefit more from rolling statistics like mean or standaard deviation). So I allowed each classifier and predictor to pick the best 9 features for it in order to avoid an overly inflated r^2 value for the predictors and compare which features each classifier decided to choose.

For the XGboost regressor, I adjusted the learning rate to be 0.05 instead of 0.1 and used 500 trees instead of 100. The lower learning rate makes training more stable and decreases the likelihood of overfitting. The higher amount of trees gives the XGboost regressor more opportunities to learn the patterns of the data. For both regressors, I wanted to use TimeSeriesSplit cross validation because it respects time order and reflects real world forecasting. I used a total of 5 folds for each regressor. For the randomforest regressor, fold 5 had a much higher mean squared error than the other 4 folds. When I printed information related to each of the folds (min, max, std) etc, I saw that fold 5 had a significantly higher standard deviation and also contained the max given that it was the last fold. I also think the last time period likely had more spikes meaning that the amount of pm25 wasn't as stable. For the XGBoost regressor, folds 1 and 5 were the two folds with much higher MSE. For fold 1, this is likely a result of XGBoost not doing as well with small amounts of data and since cross valdiation makes it a point to only train on past data, you are training on the least amount of data in the first fold. For fold 5, the presence of spikes also likely contributed to a higher MSE. In order to somewhat fix these high MSE values, I log transformed the target variable which is the amount of pm25 in the air an hour later. Although the r^2 values decreased a bit, ultimately it helped both of the models make more realistic predictions which was more important to me.

For the classifiers, I wanted to ensure that there wasn't an overpredicting of classes that were more popular. Because of this I assigned weights to each of the 4 air quality categories based on their frequency in the data. This improved the precision for less popular classes, particularly the last class labeled number 3. However, the recall was significantly lower for class 3 in the random forest classifier which is likely a result of class 3 being the class for rare and extreme cases of pollution spikes. Since the random forest classifier prioritizes minimizing impurity, it only predicts class 3 (Unhealthy, Very UNhealthy, Hazardous) when it is absolutely sure that an entry in the data falls in that class.

Results for Random Forest Regressor:
RMSE: 20.346
Adjusted r^2 value: 0.146

Results for XGBoost Regressor:
RMSE: 21.243
Adjusted r^2 value: 0.069

Random Forest Classification Report
              precision    recall  f1-score   support

           0       0.41      0.42      0.42        26
           1       0.57      0.74      0.64        65
           2       0.15      0.12      0.13        17
           3       1.00      0.06      0.11        17

    accuracy                           0.50       125
   macro avg       0.53      0.33      0.33       125
weighted avg       0.54      0.50      0.45       125

Confusion Matrix for Random Forest Classifier:
[[11 13  2  0]
 [13 48  4  0]
 [ 3 12  2  0]
 [ 0 11  5  1]]

XGBoost Classification Report
              precision    recall  f1-score   support

           0       0.38      0.23      0.29        26
           1       0.63      0.69      0.66        65
           2       0.24      0.35      0.29        17
           3       0.54      0.41      0.47        17

    accuracy                           0.51       125
   macro avg       0.45      0.42      0.42       125
weighted avg       0.51      0.51      0.51       125

Confusion Matrix for XGBoost Classifier:

[[ 6 13  5  2]
 [10 45  9  1]
 [ 0  8  6  3]
 [ 0  5  5  7]]


The Lagos.csv is the data that I used for this project.

overiew of the project
high-level overview of system design
going into each directory, explaining what it is for
setup guide

