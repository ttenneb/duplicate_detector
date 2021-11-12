import csv
import cv2 as cv
import skimage.metrics as sm
from datetime import datetime
import ast
import math

data = csv.reader(open("Thomasville N.csv"))
last = None
saved = None
dataset_dup = []
dataset_not_dup = []


def trajectoy_distance(list1, list2):
    list1 = ast.literal_eval(list1)
    list2 = ast.literal_eval(list2)
    if len(list1) > 0 and len(list2) > 0:
        start_dist = pow(list1[len(list1) - 1][0] - list2[0][0], 2) + pow(list1[len(list1) - 1][1] - list2[0][1], 2)
        if start_dist == 1263761:
            print(list1)
            print(list1[len(list1) - 1], list2[0], start_dist)
        if len(list1) < len(list2):
            temp = list1
            list1 = list2
            list2 = temp
        total = 0
        for i, vec in enumerate(list1):
            if i >= len(list2):
                return [max(total,.01), max(start_dist,.01)]
            total += pow(vec[0] - list2[i][0], 2) + pow(vec[1] - list2[i][1], 2)
        return [max(total,.01), max(start_dist,.01)]


for row in data:
    if last is not None and (row[8] == last[8] or row[8] == ""):
        continue
    if row[11] == 'Duplicate':
        saved = last
        dist, start_dist = trajectoy_distance(saved[6], row[6])
        time_dist = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S") - datetime.strptime(saved[3], "%Y-%m-%d %H:%M:%S")
        time_dist = time_dist.total_seconds()
        print(dist, start_dist)
        dataset_dup.append((saved[8], row[8], dist, math.log(dist), start_dist, math.log(start_dist), time_dist))
    elif row[8] != "clip":
        temp = last
        last = row
        if temp is not None:
            dist, start_dist = trajectoy_distance(temp[6], last[6])

            time_dist = datetime.strptime(last[2], "%Y-%m-%d %H:%M:%S") - datetime.strptime(temp[3],
                                                                                            "%Y-%m-%d %H:%M:%S")
            time_dist = time_dist.total_seconds()
            print(dist,start_dist)
            dataset_not_dup.append((temp[8], last[8], dist, math.log(dist), start_dist, math.log(start_dist), time_dist))
# print(len(dataset_not_dup), len(dataset_dup))
dataset = []
for row in dataset_dup:
    dataset.append(row + ("T",))
for row in dataset_not_dup:
    dataset.append(row + ("F",))
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
        ret2, frame3 = cap2.read()
        ret2, frame4 = cap2.read()
        if frame1 is not None and frame2 is not None:
            frame1 = frame1[25:1530, 635:]
            frame2 = frame2[25:1530, 635:]
            frame3 = frame1[25:1530, 635:]
            frame4 = frame2[25:1530, 635:]
            dist1 = sm.structural_similarity(frame1, frame2, multichannel=True)
            dist2 = sm.structural_similarity(frame1, frame3, multichannel=True)
            dist3 = sm.structural_similarity(frame1, frame4, multichannel=True)
            # print(dist, dataset[i][2], dataset[i][3], dataset[i][4])
            dataset[i] += (dist1,dist2,dist3,)
            print(dataset[i])
        count += 1
        if count > 0:
            break
with open('data_ssim.csv', 'w', newline='') as csvfile:
    csvfile.truncate(0)

    output = csv.writer(csvfile)
    output.writerow(['video1', 'video2', "trajectory_distance", "ln_trajectory_distance", "start_distance", "ln_start_distance",'seconds', 'duplicate', 'ssim'])
    for row in dataset:
        print(row)
        output.writerow(row)
