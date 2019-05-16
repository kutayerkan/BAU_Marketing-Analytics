# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:02:20 2019

@author: kutay.erkan@gmail.com
"""

# %% Import libraries and change settings

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
    
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 15)
sns.set_context("talk")

import warnings
warnings.filterwarnings("ignore", category=Warning)

# %% Define a function for running iterative Cross-Validate classification

def IterativeCVClassifier(features,label,classifier,iterations):
    
    for i in range(100,100+iterations):
        
        classifier.set_params(random_state=i)
        cv = StratifiedKFold(n_splits=10, random_state=i)
        scores = cross_val_score(classifier, features, label,
                             scoring = 'accuracy',
                             cv = cv)
        # For 95% Confidence Interval, 2 * std
        print("Accuracy: {0:.3f} (+/- {1:.3f})".format(scores.mean(),scores.std()*2))
        
        label_pred = cross_val_predict(classifier, features, label, cv=cv)
        
        print(pd.crosstab(label, label_pred,
                rownames = ['Actual'],
                colnames = ['Predicted'],
                margins = True))

# %% Read data and name columns

df = pd.read_csv('avila-tr.txt', header=None)

df.columns = ['intercolumnar_distance',
              'upper_margin',
              'lower_margin',
              'exploitation',
              'row_number',
              'modular_ratio',
              'interlinear_spacing',
              'weight',
              'peak_number',
              'm_r/i_s',
              'target'] # class isn't used as it is a Python keyword

# %% Show a sample of data

print("\nSample data:\n",df.head())
print("\nNumber of rows in the dataset:",len(df))

# %%

df.describe()

# %%

sns.set(font_scale=0.9)
df.boxplot();
sns.set(font_scale=1)

# %%
sns.countplot(df['target'],
              order = df['target'].value_counts().index);

# %% Create a correlation matrix to check for multicollinearity

corr = df.corr()

sns.heatmap(corr, annot=True);

# %%

df[df.upper_margin > 250]
df = df[df.upper_margin < 250]

# %%

sns.set(font_scale=0.5)
sns.pairplot(df, hue='target', markers='+'); # Biraz taşıyor düzeltmek lazım
sns.set(font_scale=0.7)

# %% Create the label and features

y = df['target']
X = df.drop(columns=['target'])

# %% Split data to training and test sets. Test set is not touched

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, stratify=y, random_state=0)

# %% Calculate Variance Inflation Factors and remove most inflated feature
# until all features have VIF < 4

while(True):
    corr = X_train.corr()
    X_train_vif = pd.Series(np.linalg.inv(corr.values).diagonal(),
                                   index=corr.index)
    if (max(X_train_vif) > 4):
        X_train = X_train.drop(
            columns=X_train_vif.idxmax())
        print(X_train_vif.idxmax(),
              "is dropped with VIF= ",
              round(max(X_train_vif),2),
              "\n")
        X_test = X_test.drop(columns=X_train_vif.idxmax())
    else:
        print("Remaining features with VIF < 4:")
        print(round(X_train_vif,2))
        break

# %% Do Oversampling on minority classes
# SMOTE is not in conda distribution

sm = SMOTE(random_state=0, k_neighbors=3)
X_train, y_train = sm.fit_sample(X_train, y_train.ravel())

# %% Plot distribution of classes
# pd.Series is only needed if y_train is converted to ndarray after oversampling

sns.countplot(y_train, order = pd.Series(y_train).value_counts().index);
print(len(y_train[y_train=='B']))
              
# %% Do GridSearchCV for Logistic Regression

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'solver': ['newton-cg', 'sag', 'saga', 'lbfgs']
              }

lr = LogisticRegression()
grid_search = GridSearchCV(estimator=lr,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=3,
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train, y_train)
selected_lr_params = grid_search.best_params_

print(selected_lr_params)

# %% Logistic Regression

lr = LogisticRegression()

# lr = lr.set_params(**selected_lr_params)
lr = lr.set_params(C=100)
lr = lr.set_params(solver='newton-cg')

IterativeCVClassifier(X_train, y_train, lr, 1)

# %% Random Forest Classifier

rf = RandomForestClassifier(n_estimators = 100,
                             criterion = 'entropy',
                             n_jobs = -1,
                             class_weight = 'balanced')

IterativeCVClassifier(X_train, y_train, rf, 1)

# %% Adaboost Classifier

ada = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)

IterativeCVClassifier(X_train, y_train, ada, 1)    
    
# %% Feature importances

rf.fit(X_train, y_train)

print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_),
                 X.columns), reverse=True))

# %% Fit best model on the test set

IterativeCVClassifier(X_test, y_test, rf, 1)

# %% Plot some of the most important features

ax = sns.scatterplot(x = 'upper_margin',
                y = 'intercolumnar_distance',
                hue='target',
                data=df);

ax.set_xlim(-1,1);
ax.set_ylim(-1,1);













