import argparse
import csv
import cv2 as cv
import skimage.metrics as sm
from sklearn.linear_model import LinearRegression

import ast
import math
import numpy as np
import pandas as pd

def getfile(args):
    path = args.data_path
    data = pd.read_csv(path)
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', type=str, default="Ramsey.xlsx", help='what data file to read')

    args = parser.parse_args()

data = getfile(args)

last = None
saved = None
dataset_dup = []
dataset_not_dup = []


def trajectoy_distance(list1, list2):
    list1 = ast.literal_eval(list1)
    list2 = ast.literal_eval(list2)
    if len(list1) > 0 and len(list2) > 0:
        start_dist = pow(list1[len(list1) - 1][0] - list2[0][0], 2) + pow(list1[len(list1) - 1][1] - list2[0][1], 2)
        if len(list1) < len(list2):
            temp = list1
            list1 = list2
            list2 = temp
        total = 0
        for i, vec in enumerate(list1):
            if i >= len(list2):
                return [max(total, .01), max(start_dist, .01)]
            total += pow(vec[0] - list2[i][0], 2) + pow(vec[1] - list2[i][1], 2)
        return [max(total, .01), max(start_dist, .01)]


def predict_path(list):
    list = np.array(ast.literal_eval(list))
    x = []
    y = []
    for tup in list:
        x.append([tup[0]])
        y.append([tup[1]])
    x = np.array(x)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return [reg.coef_, reg.intercept_]


is_last_duplicate = False
duplicate_count = 0
dataset = []
for index, row in data.iterrows():

    # print(row)
    if row['Explanation'] == 'Duplicate':
        temp = last
        last = row

        dist, start_dist = trajectoy_distance(temp["trajectory"], last["trajectory"])

        coefficient1, intercept1 = predict_path(temp["trajectory"])
        coefficient2, intercept2 = predict_path(last["trajectory"])

        time_dist = last["start_time"] - temp["end_time"]
        time_dist = time_dist.total_seconds()

        if is_last_duplicate:
            dataset.append((temp["clip"], last["clip"], dist, math.log(dist), start_dist, math.log(start_dist), time_dist,
                            coefficient1[0][0], intercept1[0], coefficient2[0][0], intercept2[0], 1, duplicate_count, "T"))
        else:
            dataset.append((temp["clip"], last["clip"], dist, math.log(dist), start_dist, math.log(start_dist), time_dist,
                            coefficient1[0][0], intercept1[0], coefficient2[0][0], intercept2[0], 0, duplicate_count, "T"))

        is_last_duplicate = True
        duplicate_count += 1
    elif row[8] != "clip":

        temp = last
        last = row
        if temp is not None:
            dist, start_dist = trajectoy_distance(temp["trajectory"], last["trajectory"])

            coefficient1, intercept1 = predict_path(temp["trajectory"])
            coefficient2, intercept2 = predict_path(last["trajectory"])

            time_dist = last["start_time"] - temp['end_time']
            time_dist = time_dist.total_seconds()

            if is_last_duplicate:
                dataset.append((temp["clip"], last["clip"], dist, math.log(dist), start_dist, math.log(start_dist), time_dist,
                                coefficient1[0][0], intercept1[0], coefficient2[0][0], intercept2[0], 1, duplicate_count, "F"))
            else:
                dataset.append((temp["clip"], last["clip"], dist, math.log(dist), start_dist, math.log(start_dist), time_dist,
                                coefficient1[0][0], intercept1[0], coefficient2[0][0], intercept2[0], 0, duplicate_count, "F"))
        is_last_duplicate = False
        duplicate_count = 0

size = 100
output = []
accuracy_total = 0
total_dup = 0
total_not_dup = 0
true_total_dup = 0
true_total_not_dup = 0
table = [[0, 0], [0, 0]]
for i in range(len(dataset)):
    cap1 = cv.VideoCapture(dataset[i][0])
    last_frame_num = cap1.get(cv.CAP_PROP_FRAME_COUNT)
    cap1.set(cv.CAP_PROP_POS_FRAMES, last_frame_num - 3)
    cap2 = cv.VideoCapture(dataset[i][1])
    print(100 * (i / float(len(dataset))))
    total = 0
    count = 0
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        # ret2, frame3 = cap2.read()
        # ret2, frame4 = cap2.read()
        if frame1 is not None and frame2 is not None:
            frame1 = frame1[10:581, 268:574]
            frame2 = frame2[10:581, 268:574]
            # frame3 = frame1[25:1530, 635:]
            # frame4 = frame2[25:1530, 635:]
            dist1 = sm.structural_similarity(frame1, frame2, multichannel=True)
            # dist2 = sm.structural_similarity(frame1, frame3, multichannel=True)
            # dist3 = sm.structural_similarity(frame1, frame4, multichannel=True)
            # print(dist, dataset[i][2], dataset[i][3], dataset[i][4])
            dataset[i] += (dist1,)
            print(dataset[i])
        count += 1
        if count > 0:
            break
with open('data.csv', 'w', newline='') as csvfile:
    csvfile.truncate(0)

    output = csv.writer(csvfile)
    output.writerow(
        ['video1', 'video2', "trajectory_distance", "ln_trajectory_distance", "start_distance", "ln_start_distance",
         'seconds', 'coefficient1', 'intercept1', 'coefficient2', 'intercept2', 'is_last_duplicate', 'duplicate_count',
         'duplicate', 'ssim'])
    for row in dataset:
        print(row)
        output.writerow(row)

