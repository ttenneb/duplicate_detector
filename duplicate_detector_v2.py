

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import skimage.metrics as sm
from skimage import io
import os
import pandas as pd
import csv
import plotly.express as plot
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics, kernel_ridge, tree

from sklearn import svm
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


data_train = pd.read_csv(open("data_ssim.csv"))
selected_features = ["seconds", "trajectory_distance", "ssim", "ln_trajectory_distance", "start_distance", "ln_start_distance"]
for i, row in data_train.iterrows():
    if data_train.at[i, "duplicate"] == "F":
        data_train.at[i, "duplicate"] = 0
    if data_train.at[i, "duplicate"] == "T":
        data_train.at[i, "duplicate"] = 1

data_train = data_train.dropna()
data_train = data_train.sample(frac=1).reset_index(drop=True)
norm_df = (data_train-data_train.mean())/data_train.std()
data_train[selected_features]=norm_df[selected_features]
print(data_train)

x = data_train[selected_features]
y = data_train["duplicate"]

x_train, x_val, y_train, y_val = train_test_split(x,y, stratify=y, test_size=.2,random_state=0)



models =[]
for i in range(5):
     models.append(("RF" + str(i), RandomForestRegressor(n_estimators=150, random_state=22)))
# models.append(("SV0", svm.SVR()))
# models.append(("SV1", svm.SVR()))
# models.append(("GB0", HistGradientBoostingRegressor()))
# models.append(("GB1", HistGradientBoostingRegressor()))
# models.append(("L0", LassoCV()))
# models.append(("LC1", LassoCV()))
clf = StackingRegressor(estimators=models, final_estimator=svm.SVR())
# clf = RandomForestRegressor(n_estimators=100, random_state=0)
#clf = LassoCV()
clf.fit(x_train, y_train)
total = 0
total_false_positive = 0
total_true_positive = 0
total_false_negative = 0
total_true_negative = 0



y_pred = clf.predict(x_val)
y_val = y_val.astype(int)
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred, pos_label=1)
for i in range(len(fpr)):
    print(fpr[i], tpr[i], thresholds[i])

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

min = 1
save = -1
row = 0
opt = thresholds[np.argmin(abs(tpr-(1-fpr)))]
print("thresh: ", opt)

y_pred_labels = []
y_val_labels = []
for i in y_pred:
    if i > opt:
        y_pred_labels.append("T")
    else:
        y_pred_labels.append("F")
for i in y_val:
    if i > opt:
        y_val_labels.append("T")
    else:
        y_val_labels.append("F")
print(accuracy_score(y_val_labels, y_pred_labels))
print(classification_report(y_val_labels, y_pred_labels))
print(confusion_matrix(y_val_labels, y_pred_labels))
# print(total/val_size)
# print(clf.oob_score_)



