#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script converts rosbag containing raw IMU data into a text file.
# This allows to run ROS on a Raspberry Pi, then download the text file 
# to another commputer to analyse data.

import rospy
import rospkg
import rosbag

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import time

from mpu_6050_reader.msg import Array_2D
from geometry_msgs.msg import Vector3

use_acc = True
use_gyro = False

data = {
    "Time_in": np.zeros((1,1)),
    "Time_out": np.zeros((1,1)),
    "Acc_in": np.zeros((1,3)),
    "Acc_out": np.zeros((1,3))
}

if use_gyro:
    data["Gyro_in"]: np.zeros((1,3))
    data["Gyro_out"]: np.zeros((1,3))

# Read data from bag
rospack = rospkg.RosPack()
bag = rosbag.Bag(rospack.get_path('mpu_6050_reader') + '/log/log.bag')

for topic, msg, t in bag.read_messages(topics=['/acceleration_1']):
    data["Time_in"] = np.append(data["Time_in"], [[t.to_sec()]])
    data["Acc_in"] = np.append(data["Acc_in"], [[msg.x, msg.y, msg.z]], axis = 0)

if use_gyro:
    for topic, msg, t in bag.read_messages(topics=['/gyro_1']):
        data["Gyro_in"] = np.append(data["Gyro_in"], [[msg.x, msg.y, msg.z]], axis = 0)

for topic, msg, t in bag.read_messages(topics=['/acceleration_2']):
    data["Time_out"] = np.append(data["Time_out"], [[t.to_sec()]])
    data["Acc_out"] = np.append(data["Acc_out"], [[msg.x, msg.y, msg.z]], axis = 0)

if use_gyro:
    for topic, msg, t in bag.read_messages(topics=['/gyro_2']):
        data["Gyro_out"] = np.append(data["Gyro_out"], [[msg.x, msg.y, msg.z]], axis = 0)

#print("{} {} {} {} {} {} ".format(data["Time_in"].shape, data["Acc_in"].shape, data["Gyro_in"].shape, data["Time_out"].shape, data["Acc_out"].shape, data["Gyro_out"].shape))

# Clip data to fixed value and change intial time
data_len = min([max(data[key].shape) for key in data])
data["Time_in"] = data["Time_in"][1:data_len]-data["Time_in"][1]
data["Acc_in"] = data["Acc_in"][1:data_len, :]

data["Time_out"] = data["Time_out"][1:data_len]-data["Time_out"][1]
data["Acc_out"] = data["Acc_out"][1:data_len, :]

if use_gyro:
    data["Gyro_in"] = data["Gyro_in"][1:data_len, :]
    data["Gyro_out"] = data["Gyro_out"][1:data_len, :]

print("Experiment duration: {:.3f} | mean sampling rate: {:.3f} ms (SD: {:.3f})".format(max(data["Time_in"]), 1000*np.mean(np.diff(data["Time_in"])), 1000*np.std(np.diff(data["Time_in"]))))



# Write data to txt file
if use_gyro:
    data_file = np.vstack(( [data["Time_in"]], data["Acc_in"].T, data["Gyro_in"].T, [data["Time_out"]], data["Acc_out"].T, data["Gyro_out"].T )).T
else:
    data_file = np.vstack(( [data["Time_in"]], data["Acc_in"].T, [data["Time_out"]], data["Acc_out"].T )).T

log_filename = "log/log.txt"
np.savetxt(log_filename, data_file, delimiter=', ', fmt='%1.4e')
print("Logged saved to " + log_filename)


# For calibration dataset 
if False:
    data_file = np.vstack(( [data["Time_in"]], data["Acc_in"].T, data["Gyro_in"].T)).T
    np.savetxt("log/log1.txt", data_file, delimiter=', ', fmt='%1.4e')

    data_file = np.vstack(([data["Time_out"]], data["Acc_out"].T, data["Gyro_out"].T )).T
    np.savetxt("log/log2.txt", data_file, delimiter=', ', fmt='%1.4e')


