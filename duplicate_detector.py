import urllib
from math import floor
from random import random

import matplotlib
import cv2 as cv
import numpy as np
import skimage.metrics as sm
from skimage import io
import os
import pandas as pd
import csv
import plotly.express as plot
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
#from xgboost.sklearn import XGBClassifier

data_train = pd.read_csv(open("data_ssim.csv"))

data_train = data_train.dropna()
data_train = data_train.sample(frac=1).reset_index(drop=True)
# norm_df = (data_train-data_train.mean())/data_train.std()
# data_train[["ssim","seconds", "trajectoy_distance"]]=norm_df[["ssim", "seconds", "trajectoy_distance"]]
val_size = 500
val = data_train.iloc[:val_size, :]
data_train = data_train.iloc[val_size:,:]

x = data_train[["ssim", "seconds", "trajectoy_distance"]]
y = data_train["duplicate"]

model_n = 10
models =[]
for i in range(model_n):
     models.append(("rf" + str(i), RandomForestClassifier(n_estimators=120, random_state=i)))
clf = VotingClassifier(estimators=models, voting="hard")
#clf = RandomForestClassifier(n_estimators=300, random_state=0)
clf.fit(x, y)
total = 0
total_false_positive = 0
total_true_positive = 0
total_false_negative = 0
total_true_negative = 0

x_val = val[["ssim", "seconds", "trajectoy_distance"]]
y_val  = val["duplicate"]


# for i,row in enumerate(val.iterrows()):
#     input = [[row[1][5], row[1][3], row[1][2]]]
#     pred = clf.predict(input)
#     if pred[0] == row[1][4]:
#         total += 1
#     if pred[0] == 'T':
#         if row[1][4] == 'T':
#             total_true_positive += 1
#         if row[1][4] == 'F':
#             total_false_positive += 1
#     if pred[0] == 'F':
#         if row[1][4] == 'T':
#             total_false_negative += 1
#         if row[1][4] == 'F':
#             total_true_negative += 1
y_pred = clf.predict(x_val)
print(accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
# print(total/val_size)
print(clf.oob_score_)
# print(total_true_positive, total_false_negative)
# print(total_false_positive, total_true_negative)


