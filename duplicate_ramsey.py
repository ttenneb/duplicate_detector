import argparse
import csv
import pickle
from datetime import datetime

import cv2 as cv
import skimage.metrics as sm
from sklearn.linear_model import LinearRegression

import ast
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

threshold = 0.20666666666666667


def trajectory_distance(list1, list2):
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


def predict_path(list1):
    list1 = np.array(ast.literal_eval(list1))
    x = []
    y = []
    for tup in list1:
        x.append([tup[0]])
        y.append([tup[1]])
    x = np.array(x)
    y = np.array(y)
    reg = LinearRegression().fit(x, y)
    return [reg.coef_, reg.intercept_]


# 'seconds', "trajectory_distance", "ssim", "start_distance",
# 'coefficient1', 'intercept1', 'coefficient2', 'intercept2',
# 'duplicate_count'
def classify(clf_model, start_time, end_time, trajectory1, trajectory2, frame_start, frame_end, duplicate_count):
    input = pd.DataFrame()

    seconds = start_time - end_time
    input['seconds'] = [seconds.total_seconds()]

    trajectory_dist, start_dist = trajectory_distance(trajectory1, trajectory2)

    input['trajectory_distance'] = [trajectory_dist]

    input['ssim'] = [sm.structural_similarity(frame_start, frame_end, channel_axis=2)]

    input['start_distance'] = [start_dist]

    coefficient1, intercept1 = predict_path(trajectory1)
    coefficient2, intercept2 = predict_path(trajectory2)

    input['coefficient1'] = [coefficient1[0][0]]
    input['intercept1'] = [intercept1[0]]
    input['coefficient2'] = [coefficient2[0][0]]
    input['intercept2'] = [intercept2[0]]

    input['duplicate_count'] = [duplicate_count]

    print(input)
    output = clf_model.predict(input)

    return output[0], output[0] > threshold


def crop(img):
    return img[10:581, 268:574]


if __name__ == '__main__':
    clf = pickle.load(open("decision_tree_ramsey.sav", 'rb'))

    parser = argparse.ArgumentParser()

    parser.add_argument('--trajectory1', type=str, help='Trespassing object\'s trajectory')
    parser.add_argument('--trajectory2', type=str, help='Previous trespassing object\'s trajectory')
    parser.add_argument('--clip1', type=str, help='Trespassing event\'s clip')
    parser.add_argument('--clip2', type=str, help='Previous trespassing event\'s clip')
    parser.add_argument('--start-time', type=str, help='Event\'s start time')
    parser.add_argument('--end-time', type=str, help='Previous event\'s end time')
    parser.add_argument('--consecutive-duplicates', type=int, default=0, help='Number of consecutive duplicates '
                                                                              'before this event')
    args = parser.parse_args()

    cap1 = cv.VideoCapture(args.clip2)
    last_frame_num = cap1.get(cv.CAP_PROP_FRAME_COUNT)
    cap1.set(cv.CAP_PROP_POS_FRAMES, last_frame_num - 3)
    cap2 = cv.VideoCapture(args.clip1)

    ret1, frame1 = cap2.read()
    ret2, frame2 = cap1.read()

    if frame1 is not None and frame2 is not None:
        frame1 = crop(frame1)
        frame2 = crop(frame2)

        print(classify(clf, pd.to_datetime(args.start_time), pd.to_datetime(args.end_time), args.trajectory2,
                       args.trajectory1, frame1, frame2, args.consecutive_duplicates))
