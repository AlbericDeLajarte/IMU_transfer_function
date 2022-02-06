# IMU_transfer_function

This project is used for system identification using 2 IMU: one attached to the input of the system, the other to its output.
By comparing the input and output acceleration and angular rates, a transfer function can be derived to model the system.

A typical implementation uses a RaspberryPi with ROS to read and stream IMU data.

## Tools:

- ROS (Robot Operating System): Used as middleware to connect sensor reading, web dashboard, and logger

- Webviz: Web dashboard to visualize ROS messages. Accessible at https://webviz.io/app/?rosbridge-websocket-url=ws://albepi.local:8080 (replace `albepi.local` with your Raspberry Pi address)

- MPU6050: IMU from Adafruit. To install necessary libraries, go to : https://learn.adafruit.com/mpu6050-6-dof-accelerometer-and-gyro?view=all
