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
    index=['location_name', 'datetimeLocal'],
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
df['rolling_3'] = df['pm25'].rolling(3).mean()
df['rolling_6'] = df['pm25'].rolling(6).mean()
df['rolling_6_std'] = df['pm25'].rolling(6).std()
df['rolling_12_std'] = df['pm25'].rolling(12).std()
df['temperature_relative_humidity'] = df['temperature'] * df['relativehumidity']
df['temperature_change_1'] = df['temperature'].diff(1)
df['pm25_1_ahead'] = df['pm25'].shift(-1)

df = df.dropna()

bins = [-np.inf, 9.0, 35.4, 55.4, np.inf]
labels = [0, 1, 2, 3]
df['air_quality'] = pd.cut(df['pm25_1_ahead'], bins=bins, labels=labels).astype(int)

features = [
    'temperature', 'relativehumidity', 'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos', 'pm25_1', 'pm25_2', 'pm25_3',
    'pm25_6', 'pm25_12', 'pm25_24', 'pm25_diff_1',
    'pm25_diff_6', 'rolling_3', 'rolling_6', 'rolling_6_std',
    'rolling_12_std', 'temperature_relative_humidity',
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

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_index, val_index) in enumerate(tscv.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr = y_train_log.iloc[train_index]
    y_val = y_train.iloc[val_index]

    sfm_rf_cv = SelectFromModel(
        RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        max_features=9,
        threshold=-np.inf
    )
    sfm_rf_cv.fit(X_tr, y_tr)
    top_features_rf_cv = X_tr.columns[sfm_rf_cv.get_support()]

    model_rf_cv = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model_rf_cv.fit(X_tr[top_features_rf_cv], y_tr)

    preds_log = model_rf_cv.predict(X_val[top_features_rf_cv])
    preds = np.expm1(preds_log)

    print(
        f"Random Forest Fold {fold} RMSE:",
        np.sqrt(mean_squared_error(y_val, preds))
    )

sfm_rf = SelectFromModel(
    RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    max_features=9,
    threshold=-np.inf
)
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
print("Adj_r2:", adj_r2)

feat_importances_rf = pd.Series(final_rf.feature_importances_, index=top_features_rf)
sns.barplot(x=feat_importances_rf, y=feat_importances_rf.index)
plt.title("Random Forest Regressor for Top 9 Features")
plt.show()

for fold, (train_index, val_index) in enumerate(tscv.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr = y_train_log.iloc[train_index]
    y_val = y_train.iloc[val_index]

    sfm_xgb_cv = SelectFromModel(
        XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            objective='reg:squarederror',
            random_state=42
        ),
        max_features=9,
        threshold=-np.inf
    )
    sfm_xgb_cv.fit(X_tr, y_tr)
    top_features_xgb_cv = X_tr.columns[sfm_xgb_cv.get_support()]

    model_xgb_cv = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        objective='reg:squarederror',
        random_state=42
    )
    model_xgb_cv.fit(X_tr[top_features_xgb_cv], y_tr)

    preds_log = model_xgb_cv.predict(X_val[top_features_xgb_cv])
    preds = np.expm1(preds_log)

    print(
        f"XGBoost Fold {fold} RMSE:",
        np.sqrt(mean_squared_error(y_val, preds))
    )

sfm_xgb = SelectFromModel(
    XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        objective='reg:squarederror',
        random_state=42
    ),
    max_features=9,
    threshold=-np.inf
)
sfm_xgb.fit(X_train, y_train_log)
top_features_xgb = X_train.columns[sfm_xgb.get_support()]

final_xgb = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    objective='reg:squarederror',
    random_state=42
)
final_xgb.fit(X_train[top_features_xgb], y_train_log)

y_pred_log = final_xgb.predict(X_test[top_features_xgb])
y_pred = np.expm1(y_pred_log)

r2 = r2_score(y_test, y_pred)
p = len(top_features_xgb)
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

rf_cls = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    min_samples_leaf=3,
    class_weight='balanced_subsample',
    n_jobs=-1
)
rf_cls.fit(X_train, y_train_cls)

sfm_rf_cls = SelectFromModel(rf_cls, max_features=9, threshold=-np.inf)
sfm_rf_cls.fit(X_train, y_train_cls)
top_features_rf_cls = X_train.columns[sfm_rf_cls.get_support()]

final_rf_cls = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    min_samples_leaf=3,
    class_weight='balanced_subsample',
    n_jobs=-1
)
final_rf_cls.fit(X_train[top_features_rf_cls], y_train_cls)

y_pred_cls = final_rf_cls.predict(X_test[top_features_rf_cls])

print("Random Forest Classification Report")
print(classification_report(y_test_cls, y_pred_cls))
print(confusion_matrix(y_test_cls, y_pred_cls))

feat_importances_rf_cls = pd.Series(
    final_rf_cls.feature_importances_,
    index=top_features_rf_cls
)
sns.barplot(x=feat_importances_rf_cls, y=feat_importances_rf_cls.index)
plt.title("Random Forest Classifier for Top 9 Features Class-Weighted")
plt.show()

class_counts = Counter(y_train_cls)
total = sum(class_counts.values())
num_classes = len(class_counts)
class_weights = {
    cls: total / (num_classes * count)
    for cls, count in class_counts.items()
}
sample_weights = y_train_cls.map(class_weights)

xgb_cls = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=7,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42
)
xgb_cls.fit(X_train, y_train_cls, sample_weight=sample_weights)

sfm_xgb_cls = SelectFromModel(xgb_cls, max_features=9, threshold=-np.inf)
sfm_xgb_cls.fit(X_train, y_train_cls, sample_weight=sample_weights)
top_features_xgb_cls = X_train.columns[sfm_xgb_cls.get_support()]

final_xgb_cls = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=7,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42
)
final_xgb_cls.fit(
    X_train[top_features_xgb_cls],
    y_train_cls,
    sample_weight=sample_weights
)

y_pred_cls = final_xgb_cls.predict(X_test[top_features_xgb_cls])

print("XGBoost Classification Report")
print(classification_report(y_test_cls, y_pred_cls))
print(confusion_matrix(y_test_cls, y_pred_cls))

feat_importances_xgb_cls = pd.Series(
    final_xgb_cls.feature_importances_,
    index=top_features_xgb_cls
)
sns.barplot(x=feat_importances_xgb_cls, y=feat_importances_xgb_cls.index)
plt.title("XGBoost Classifier for Top 9 Features Class-Weighted")
plt.show()
