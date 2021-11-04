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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
#from xgboost.sklearn import XGBClassifier

data_train = pd.read_csv(open("data_ssim.csv"))

data_train = data_train.dropna()
data_train = data_train.sample(frac=1).reset_index(drop=True)
norm_df = (data_train-data_train.mean())/data_train.std()
data_train[["ssim", "trajectoy_distance"]]=norm_df[["ssim", "trajectoy_distance"]]
val_size = 500
val = data_train.iloc[:val_size, :]
data_train = data_train.iloc[val_size:,:]

x = data_train[["ssim", "trajectoy_distance"]]
y = data_train["duplicate"]

print(np.shape(x), np.shape(y))
clf = make_pipeline(StandardScaler(), SGDClassifier())
clf.fit(x, y)
total = 0
total_false_positive = 0
total_true_positive = 0
total_false_negative = 0
total_true_negative = 0
for i,row in enumerate(val.iterrows()):
    input = [[row[1][4], row[1][2]]]
    pred = clf.predict(input)
    if pred[0] == row[1][3]:
        total += 1
    if pred[0] == 'T':
        if row[1][3] == 'T':
            total_true_positive += 1
        if row[1][3] == 'F':
            total_false_positive += 1
    if pred[0] == 'F':
        if row[1][3] == 'T':
            total_false_negative += 1
        if row[1][3] == 'F':
            total_true_negative += 1
print(total/val_size)
print(total_true_positive, total_false_negative)
print(total_false_positive, total_true_negative)


data = csv.reader(open("Thomasville N.csv"))
last = None
saved = None
dataset_dup = []
dataset_not_dup = []


def parse_list(list):
    save = ()
    output = []
    x = ""
    for i, c in enumerate(list):
        if c == '[' or c == ']':
            continue
        if c != ',':
            x += c
        else:
            x = int(x)
            if len(save) == 2:
                output.append(save)
                save = ()
            else:
                save += (x,)
            x = ""

    return output


def trajectoy_distance(list1, list2):
    list1 = parse_list(list1)
    list2 = parse_list(list2)
    if len(list1) < len(list2):
        temp = list1
        list1 = list2
        list2 = temp
    total = 0
    for i, vec in enumerate(list1):
        if i >= len(list2):
            return total
        total += pow(vec[0] - list2[i][0], 2) + pow(vec[1] - list2[i][1], 2)
    return total


for row in data:
    if row[11] == 'Duplicate':
        saved = last
        dist = trajectoy_distance(saved[6], row[6])
        dataset_dup.append((saved[8], row[8], dist))
    elif row[8] != "clip":
        temp = last
        last = row
        if temp is not None:
            dist = trajectoy_distance(temp[6], last[6])
            dataset_not_dup.append((temp[8], last[8], dist))
#print(len(dataset_not_dup), len(dataset_dup))
dataset = []
for row in dataset_not_dup:
    dataset.append(row + ("F",))
for row in dataset_dup:
    dataset.append(row + ("T",))

# TODO Generate dataset
# size = 100
# output = []
# accuracy_total = 0
# total_dup = 0
# total_not_dup = 0
# true_total_dup = 0
# true_total_not_dup = 0
# table = [[0, 0], [0, 0]]
# for i in range(len(dataset)):
#     cap1 = cv.VideoCapture(dataset[i][0])
#     last_frame_num = cap1.get(cv.CAP_PROP_FRAME_COUNT)
#     cap1.set(cv.CAP_PROP_POS_FRAMES, last_frame_num - 3)
#     cap2 = cv.VideoCapture(dataset[i][1])
#     print(i)
#     total = 0
#     count = 0
#     while cap1.isOpened() and cap2.isOpened():
#         ret1, frame1 = cap1.read()
#         ret2, frame2 = cap2.read()
#         frame1 = frame1[25:1530, 635:]
#         frame2 = frame2[25:1530, 635:]
#         if frame1 is not None and frame2 is not None:
#             dist = sm.structural_similarity(frame1, frame2, multichannel=True)
#             print(dist, dataset[i][2])
#             dataset[i] += (dist,)
#         count += 1
#         if count > 0:
#             break
# with open('data_ssim.csv', 'w', newline='') as csvfile:
#     csvfile.truncate(0)
#
#     output = csv.writer(csvfile)
#     output.writerow(['video1', 'video2', "trajectoy_distance", 'duplicate', 'ssim'])
#     for row in dataset:
#         output.writerow(row)
