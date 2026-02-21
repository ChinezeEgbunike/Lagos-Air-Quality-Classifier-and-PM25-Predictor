import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

df = pd.read_csv('Lagos.csv')
df = df.pivot_table(
    index=['location_name','datetimeLocal'],
    columns='parameter',
    values='value'
).reset_index()

df['datetimeLocal'] = pd.to_datetime(df['datetimeLocal'])
df = df.sort_values('datetimeLocal')
df = df.set_index('datetimeLocal')
df = df.asfreq('H')
df[['temperature', 'relativehumidity', 'pm25']] = df[['temperature', 'relativehumidity', 'pm25']].ffill()
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['pm25_1'] = df['pm25'].shift(1)
df['pm25_2'] = df['pm25'].shift(2)
df['pm25_3'] = df['pm25'].shift(3)
df['pm25_6'] = df['pm25'].shift(6)
df['pm25_12'] = df['pm25'].shift(12)
df['pm25_24'] = df['pm25'].shift(24)
df['pm25_diff_1'] = df['pm25_1'] - df['pm25_2']
df['pm25_diff_6'] = df['pm25_1'] - df['pm25_6']
df['rolling_3'] = df['pm25'].shift(1).rolling(3).mean()
df['rolling_6'] = df['pm25'].shift(1).rolling(6).mean()
df['rolling_6_std'] = df['pm25'].shift(1).rolling(6).std()
df['rolling_12_std'] = df['pm25'].shift(1).rolling(12).std()
df['temperature_relative_humidity'] = df['temperature'] * df['relativehumidity']
df['temperature_change_1'] = df['temperature'].diff(1)
df['pm25_1_ahead'] = df['pm25'].shift(-1)
df = df.dropna()


bins = [-np.inf, 12, 35.4, 55.4, np.inf]
labels = [0, 1, 2, 3]
df['air_quality'] = pd.cut(df['pm25_1_ahead'], bins=bins, labels=labels).astype(int)

#These are all of the features that the classifiers and regressors can pick from.
#They will all pick the top 9 that are the most important in order to avoid an inflated
#r^2 value. Added rolling features because boosting benefits from that.
features = [
    'temperature',         
    'relativehumidity',     
    'hour_sin',             
    'hour_cos',             
    'dow_sin',              
    'dow_cos',              
    'pm25_1',               
    'pm25_2',
    'pm25_3',
    'pm25_6',
    'pm25_12',
    'pm25_24',
    'pm25_diff_1',
    'pm25_diff_6',
    'rolling_3',            
    'rolling_6',
    'rolling_6_std',
    'rolling_12_std',
    'temperature_relative_humidity',
    'temperature_change_1'           
]

split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]
X_train = train[features]
y_train = train['pm25_1_ahead']
X_test = test[features]
y_test = test['pm25_1_ahead']

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

tscv = TimeSeriesSplit(n_splits=5)
#Train on past data only, validate on future data, then expand the training window
for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr = y_train_log.iloc[train_index]
    y_val = y_train_log.iloc[val_index]
    model_rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model_rf.fit(X_tr, y_tr)
    preds_log = model_rf.predict(X_val)
    preds = np.expm1(preds_log)
    y_val_original = np.expm1(y_val)
    print(f"Fold {fold+1} MSE: {mean_squared_error(y_val_original, preds):.4f}")
    print(train.iloc[val_index]['pm25_1_ahead'].describe())

sfm_rf = SelectFromModel(model_rf, max_features=9, threshold=-np.inf)
sfm_rf.fit(X_train, y_train_log)
top_features_rf = X_train.columns[sfm_rf.get_support()]

final_rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
final_rf.fit(X_train[top_features_rf], y_train_log)

y_pred_log = final_rf.predict(X_test[top_features_rf])
y_pred = np.expm1(y_pred_log)

n = len(y_test)
p = len(top_features_rf)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("Random Forest Regression Metrics")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2)
print("Adj_r2", adj_r2)

feat_importances_rf = pd.Series(final_rf.feature_importances_, index=top_features_rf)
sns.barplot(x=feat_importances_rf, y=feat_importances_rf.index)
plt.title("Random Forest Regressor for Top 9 Features")
plt.show()


for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr = y_train_log.iloc[train_index]
    y_val = y_train_log.iloc[val_index]
    model_xgb = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    model_xgb.fit(X_tr, y_tr)
    preds_log = model_xgb.predict(X_val)
    preds = np.expm1(preds_log)
    y_val_original = np.expm1(y_val)
    print(f"Fold {fold+1} MSE: {mean_squared_error(y_val_original, preds):.4f}")
    print(train.iloc[val_index]['pm25_1_ahead'].describe())

sfm_xgb = SelectFromModel(model_xgb, max_features=9, threshold=-np.inf)
sfm_xgb.fit(X_train, y_train_log)
top_features_xgb = X_train.columns[sfm_xgb.get_support()]

final_xgb = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
final_xgb.fit(X_train[top_features_xgb], y_train_log)


#log transform predictor in order to have more predictions in the typical range and
#reduce MSE in each fold.
y_pred_log = final_xgb.predict(X_test[top_features_xgb])
y_pred = np.expm1(y_pred_log)

r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("XGBoost Regression Metrics")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2)
print("Adj_r2:", adj_r2)

feat_importances_xgb = pd.Series(final_xgb.feature_importances_, index=top_features_xgb)
sns.barplot(x=feat_importances_xgb, y=feat_importances_xgb.index)
plt.title("XGBoost Regressor for Top 9 Features")
plt.show()


y_train_cls = train['air_quality']
y_test_cls = test['air_quality']

# Compute class weights based on their frequency in order to 
# somewhat mitigate the issue of overpredicting the more popular classes
class_counts = Counter(y_train_cls)
total = sum(class_counts.values())
num_classes = len(class_counts)
class_weights = {cls: total/(num_classes*count) for cls, count in class_counts.items()}
sample_weights = y_train_cls.map(class_weights)

rf_cls = RandomForestClassifier(
    n_estimators=500,
    random_state=42,  
    min_samples_leaf=3,
    n_jobs=-1
)
rf_cls.fit(X_train, y_train_cls, sample_weight=sample_weights)
sfm_rf_cls = SelectFromModel(rf_cls, max_features=9, threshold=-np.inf)
sfm_rf_cls.fit(X_train, y_train_cls)
top_features_rf_cls = X_train.columns[sfm_rf_cls.get_support()]
final_rf_cls = RandomForestClassifier(
    n_estimators=500,
    random_state=42,  
    min_samples_leaf=3,
    n_jobs=-1
)
final_rf_cls.fit(X_train[top_features_rf_cls], y_train_cls, sample_weight=sample_weights)
y_pred_cls = final_rf_cls.predict(X_test[top_features_rf_cls])

print("Random Forest Classification Report")
print(classification_report(y_test_cls, y_pred_cls))
print(confusion_matrix(y_test_cls, y_pred_cls))
feat_importances_rf_cls = pd.Series(final_rf_cls.feature_importances_, index=top_features_rf_cls)
sns.barplot(x=feat_importances_rf_cls, y=feat_importances_rf_cls.index)
plt.title("Random Forest Classifier for Top 9 Features Class-Weighted")
plt.show()


xgb_cls = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=7,
    objective='multi:softprob',
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_cls.fit(X_train, y_train_cls, sample_weight=sample_weights)
sfm_xgb_cls = SelectFromModel(xgb_cls, max_features=9, threshold=-np.inf)
sfm_xgb_cls.fit(X_train, y_train_cls)
top_features_xgb_cls = X_train.columns[sfm_xgb_cls.get_support()]
final_xgb_cls = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=7,
    objective='multi:softprob',
    use_label_encoder=False,
    eval_metric='mlogloss'
)
final_xgb_cls.fit(X_train[top_features_xgb_cls], y_train_cls, sample_weight=sample_weights)
y_pred_cls = final_xgb_cls.predict(X_test[top_features_xgb_cls])
print("XGBoost Classification Report")
print(classification_report(y_test_cls, y_pred_cls))
print(confusion_matrix(y_test_cls, y_pred_cls))
feat_importances_xgb_cls = pd.Series(final_xgb_cls.feature_importances_, index=top_features_xgb_cls)
sns.barplot(x=feat_importances_xgb_cls, y=feat_importances_xgb_cls.index)
plt.title("XGBoost Classifier for Top 9 Features Class-weighted")
plt.show()
