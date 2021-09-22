#!/usr/bin/env python
# license removed for brevity

import numpy as np
from scipy.fft import fft, fftfreq

import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

from geometry_msgs.msg import Accel
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

from mpu_6050_reader.msg import TwoDimensionalPlot
from mpu_6050_reader.msg import TwoDimensionalPlotDatapoint
from mpu_6050_reader.msg import Array_2D


import time
import board
import adafruit_mpu6050

use_acc = rospy.get_param("accelerometer")
use_gyro = rospy.get_param("gyroscope")
do_fft = rospy.get_param("real_time_fft")

IMU_calib = [
    {"acc_gain": np.array([0.9997, 0.9983, 1.0242]), "acc_bias": np.array([0.5608, -0.0070, -1.7242]),
     "gyro_gain": np.array([0.9871, 0.9907, 1.0292]), "gyro_bias": np.array([-0.0073, -0.0064, -0.0315])
    },

    {"acc_gain": np.array([0.9947, 1.0001, 1.0150]), "acc_bias": np.array([0.4469, 0.0836, -0.1983]),
     "gyro_gain": np.array([0.9666, 1.0254, 1.0084]), "gyro_bias": np.array([-0.0467, -0.0239, -0.0348])
    }
]

N_SAMPLE = 40
time_period = 8e-3

#frequ = np.linspace(0.0, FREQU_LENGTH, FREQU_LENGTH*FREQU_SPACING, dtype=np.float32)
frequ = fftfreq(N_SAMPLE, time_period)

i2c = board.I2C()  # uses board.SCL and board.SDA


class mpu_reader:

    def __init__(self, i2c_address, imu_number):

        # Class to access IMU data
        if i2c_address == 0x68 or i2c_address == 0x69:
            self.mpu = adafruit_mpu6050.MPU6050(i2c, i2c_address)

        else:
            self.mpu = None

        self.imu_number = imu_number

        # ------------- Data -------------

        # Raw acceleration and gyroscope data to be published directly
        self.accel_data = Vector3()
        self.gyro_data = Vector3()

        # Buffer of raw data
        self.accel_data_fft_buffer = Array_2D()
        self.gyro_data_fft_buffer = Array_2D()

        # Computed FFT from buffer data
        self.accel_data_fft = Array_2D()
        self.gyro_data_fft = Array_2D()

        # ------------- Publishers -------------

        # Raw data topics
        self.acceleration_pub = rospy.Publisher('acceleration_'+str(imu_number), Vector3, queue_size=10)
        self.gyro_pub = rospy.Publisher('gyro_'+str(imu_number), Vector3, queue_size=10)

        # FFT topics
        if do_fft:
            self.frequency_fft_pub = rospy.Publisher('freq_fft_'+str(imu_number), numpy_msg(Floats), queue_size=10)
            self.acceleration_fft_pub = rospy.Publisher('accel_fft_'+str(imu_number), numpy_msg(Array_2D), queue_size=10)
            self.gyroscope_fft_pub = rospy.Publisher('gyro_fft_'+str(imu_number), numpy_msg(Array_2D), queue_size=10)


    def handle_raw_data(self, i):

        # Get raw data and publish them
        if self.mpu != None:

            if use_acc:
                acceleration = np.array(self.mpu.acceleration)
                acceleration = (acceleration - IMU_calib[self.imu_number-1]["acc_bias"])/IMU_calib[self.imu_number-1]["acc_gain"]

                self.accel_data.x, self.accel_data.y, self.accel_data.z = tuple(acceleration)
                self.acceleration_pub.publish(self.accel_data)

            if use_gyro:
                gyro = np.array(self.mpu.gyro)
                gyro = (gyro - IMU_calib[self.imu_number-1]["gyro_bias"])/IMU_calib[self.imu_number-1]["gyro_gain"]

                self.gyro_data.x, self.gyro_data.y, self.gyro_data.z = tuple(gyro)
                self.gyro_pub.publish(self.gyro_data)
        
        
        # Fill buffer
        if do_fft:
            self.accel_data_fft_buffer.x[i], self.accel_data_fft_buffer.y[i], self.accel_data_fft_buffer.z[i] = (self.accel_data.x, self.accel_data.y, self.accel_data.z)
            self.gyro_data_fft_buffer.x[i], self.gyro_data_fft_buffer.y[i], self.gyro_data_fft_buffer.z[i] = (self.gyro_data.x, self.gyro_data.y, self.gyro_data.z)


    def handle_fft(self):

        if not do_fft:
            return

        # Compute FFT from buffer data
        self.accel_data_fft.x, self.accel_data_fft.y, self.accel_data_fft.z = ( np.abs(fft(self.accel_data_fft_buffer.x)) , np.abs(fft(self.accel_data_fft_buffer.y)), np.abs(fft(self.accel_data_fft_buffer.z)) )
        self.gyro_data_fft.x, self.gyro_data_fft.y, self.gyro_data_fft.z = ( np.abs(fft(self.gyro_data_fft_buffer.x)) , np.abs(fft(self.gyro_data_fft_buffer.y)), np.abs(fft(self.gyro_data_fft_buffer.z)) )

        # print(frequ)
        # print("")
        
        # Publish FFT
        self.frequency_fft_pub.publish( frequ )
        self.acceleration_fft_pub.publish( self.accel_data_fft )
        self.gyroscope_fft_pub.publish( self.gyro_data_fft )



def reader():

    IMU_IN = mpu_reader(0x69, 1)
    IMU_OUT = mpu_reader(0x68, 2)
    
    rospy.init_node('reader', anonymous=True)
    rate = rospy.Rate(1/time_period) # Sampling rate in Hz

    if do_fft:
        accel_tf = Array_2D()
        gyro_tf = Array_2D()

        accel_transfer_function_pub = rospy.Publisher('accel_tf', numpy_msg(Array_2D), queue_size=10)
        gyro_transfer_function_pub = rospy.Publisher('gyro_tf', numpy_msg(Array_2D), queue_size=10)


    # Infinite loop: continously sample, compute FFT and send data
    i = 0
    while not rospy.is_shutdown():

        IMU_IN.handle_raw_data(i)
        IMU_OUT.handle_raw_data(i)

        if do_fft:
            #Compute FFT only when buffer is full
            if i == N_SAMPLE-1:
                i = 0
                IMU_IN.handle_fft()
                IMU_OUT.handle_fft()

                accel_tf.x, accel_tf.y, accel_tf.z = ( IMU_OUT.accel_data_fft.x/IMU_IN.accel_data_fft.x, 
                                                    IMU_OUT.accel_data_fft.y/IMU_IN.accel_data_fft.y,
                                                    IMU_OUT.accel_data_fft.z/IMU_IN.accel_data_fft.z, )

                gyro_tf.x, gyro_tf.y, gyro_tf.z = ( IMU_OUT.gyro_data_fft.x/IMU_IN.gyro_data_fft.x, 
                                                    IMU_OUT.gyro_data_fft.y/IMU_IN.gyro_data_fft.y,
                                                    IMU_OUT.gyro_data_fft.z/IMU_IN.gyro_data_fft.z, )

                
                accel_transfer_function_pub.publish( accel_tf)
                gyro_transfer_function_pub.publish(gyro_tf)


        i += 1
        rate.sleep()

if __name__ == '__main__':
    try:
        reader()
    except rospy.ROSInterruptException:
        pass