

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics, kernel_ridge
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.model_selection import learning_curve
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor


data_train = pd.read_csv(open("data_ssim.csv"))
for i, row in data_train.iterrows():
    if data_train.at[i, "duplicate"] == "F":
        data_train.at[i, "duplicate"] = 0
    if data_train.at[i, "duplicate"] == "T":
        data_train.at[i, "duplicate"] = 1

data_train = data_train.dropna()
data_train = data_train.sample(frac=1).reset_index(drop=True)
norm_df = (data_train-data_train.mean())/data_train.std()
data_train[["seconds", "trajectoy_distance"]]=norm_df[[ "seconds", "trajectoy_distance"]]
print(data_train)
val_size = 500
val = data_train.iloc[:val_size, :]
data_train = data_train.iloc[val_size:,:]

x = data_train[["ssim", "seconds", "trajectoy_distance"]]
y = data_train["duplicate"]

# model_n = 50
# models =[]
# for i in range(model_n):
#     models.append(("ab" + str(i), MLPRegressor(random_state=0, max_iter=200 )))
# clf = StackingRegressor(estimators=models)
clf = RandomForestRegressor(n_estimators=250, random_state=0)
# clf = R(weights='distance')
clf.fit(x, y)
total = 0
total_false_positive = 0
total_true_positive = 0
total_false_negative = 0
total_true_negative = 0

x_val = val[["ssim", "seconds", "trajectoy_distance"]]
y_val  = val["duplicate"]

y_pred = clf.predict(x_val)
y_val = y_val.astype(int)
fpr, tpr, thresholds = metrics.roc_curve(y_val, y_pred, pos_label=1)

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



