'''
UCI heart disease dataset
'''
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv('./heart_disease_uci.csv')

# binary targets
#df.loc[df.num > 0, 'num'] = 1

print(df.head())
print('Categorical feature class size:')
cat_col_names = ['sex','dataset','cp','fbs','restecg','exang','slope','thal']
print(df[cat_col_names].nunique())

# clean data
X_df = df.dropna()
# split feature/labels data
Y_df = df.loc[df.id.isin(X_df.id), 'num']
X_df = X_df.drop(columns=['id'])

# encode categorial features
cat_enc = OrdinalEncoder()
cat_enc.fit(X_df)
X_arr = cat_enc.transform(X_df)
X_df = pd.DataFrame(X_arr)

# train/test split 1/10
split_fac = 0.1
split = int(X_df.shape[0] * split_fac)
print('Train test split:', split_fac)

X_test = X_df.iloc[:split,:].to_numpy()
Y_test = Y_df.iloc[:split].to_numpy()
X_train = X_df.iloc[split:,:].to_numpy()
Y_train = Y_df.iloc[split:].to_numpy()
print('Samples (test, train):', Y_test.shape[0], X_train.shape[0])

# ------- data distribution stats ----------
print('Class (training) distribution:', pd.DataFrame(Y_train).value_counts().values)

# feature standardization
feat_scaler = StandardScaler()
feat_scaler.fit(X_train)
X_train = feat_scaler.transform(X_train)
X_test = feat_scaler.transform(X_test)

# class weights (train)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)
class_weights_map = dict(zip(range(0,len(class_weights)), [float(f) for f in class_weights]))
print('Class weights:', class_weights_map)

# ------- Model --------
clf = LinearSVC(multi_class='crammer_singer', class_weight=class_weights_map)
clf = clf.fit(X_train, Y_train)

# ------- evaluation ----------
print('--------------- Evaluation ---------------')
y_test_pred = clf.predict(X_test)

conf_m = confusion_matrix(Y_test, y_test_pred)
print(conf_m)
print('Precision:', precision_score(Y_test, y_test_pred, average='weighted'))
print('Recall:', recall_score(Y_test, y_test_pred, average='weighted'))
