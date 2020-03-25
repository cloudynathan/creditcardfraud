#classifying fraudulent credit card transactions with lightgbm and 5 fold cross validation
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

#load data
df = pd.read_csv(r'C:\workspacePython\creditcardfraud\creditcard.csv')
df['Class'].value_counts()

# a matrix of features
X = df.drop(columns='Class').values
# a vector of target (response)
y = df['Class'].values


# split train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

# Find the optimal hyperparameters using 5-fold grid search
clf = LGBMClassifier(n_jobs=6)
params = {'max_depth': [4, 6, 8], 'learning_rate': [0.1, 0.01], 
          'colsample_bytree': [0.6, 0.8], 'n_estimators': [20, 50]}
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

grid = GridSearchCV(clf, params, scoring='roc_auc', cv=kf, verbose=1)
grid.fit(X_train, y_train);

print(grid.best_params_)
print(grid.best_score_)
# 0.9794145581074014

# Fit using the best paramteres and evaluate AUC on test set
best_clf = grid.best_estimator_
best_clf.fit(X_train, y_train)
y_prob = best_clf.predict_proba(X_test)[:,1]
print('Test AUC:', roc_auc_score(y_test, y_prob))

# save the trained model
import pickle
pickle.dump(best_clf, open("best_model.dat", "wb"))

# load the trained model
loaded_model = pickle.load(open("best_model.dat", "rb"))
y_prob = loaded_model.predict_proba(X_test)[:,1]
print('Test AUC:', roc_auc_score(y_test, y_prob))

# compare default parameters with hyperparameters
cross_val_score(LGBMClassifier(), X_train, y_train, cv=kf, scoring = 'roc_auc').mean()

cross_val_score(grid.best_estimator_, X_train, y_train, cv=kf, scoring = 'roc_auc').mean()
