# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:43:13 2020

@author: cloud
"""


from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = load_breast_cancer()
X = data.data
y = data.target

# correlation heat map
sns.heatmap(pd.DataFrame(X).corr(), vmin=-1, vmax=1,cmap=sns.color_palette("RdBu_r", 30))

# pca
pca = PCA(30)
pca.fit(StandardScaler().fit_transform(X))

plt.plot(np.cumsum(pca.explained_variance_ratio_), 'bo-')


X2 = PCA(2).fit_transform(StandardScaler().fit_transform(X))


X2.shape

plt.scatter(X2[:,0], X2[:,1], c=y)

# make mesh grid
xx, yy = np.meshgrid(np.linspace(-6,15,50), np.linspace(-8,13,50))
print(xx.shape)

X_comb = np.vstack((xx.ravel(), yy.ravel())).T
print(X_comb.shape)

# --- can try different classifiers here

clf = DecisionTreeClassifier(max_depth=10)


clf = LogisticRegression(C=80)
clf.fit(X2, y)
zz = clf.predict_proba(X_comb)[:,1].reshape(xx.shape)

plt.pcolormesh(xx,yy,zz)
plt.scatter(X2[:,0], X2[:,1], c=y+1, edgecolors='k')

# --- svm
svm = SVC(probability=True, kernel='poly', C=1, gamma=1, degree=4)
svm.fit(X2, y)
zz = svm.predict_proba(X_comb)[:,1].reshape(xx.shape)

plt.pcolormesh(xx,yy,zz)
plt.scatter(X2[:,0], X2[:,1], c=y+1, edgecolors='k')



