import matplotlib.pyplot as plt
import numpy as np 
from scipy import integrate

g =  9.806 # local gravity in Lausanne
SENSITIVITY_A = .5*g
SENSITIVITY_AD = 1e-4
SENSITIVITY_G = .05
SENSITIVITY_DG = 1e-5
KNOWN_ANGLE = 2*np.pi

def movmean(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


#read data
sensors = ['t', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
data = {}
#filename = input('filename?')#'C:\\Users\\anith\\Downloads\\data_IMU_test_2.txt' # #input('filename?\n')
#filename = 'C:\\Users\\anith\\OneDrive\\Documents\\MATLAB\\acceleration.txt'
filename = 'postProcess/calib_file/log2.txt'


for x in sensors:
    data[x] = []

# ------------- Get data -------------

filereader = open(filename, 'r')
lines = filereader.readlines()

for line in lines: 
    values = line.split(',')
    for i in range(0, len(sensors)):
        data[sensors[i]].append(float(values[i]))

times = data['t']

# ------------- Calibration -------------
print("------------- Starting calibration -------------\n")

# Find motionless periods of accelerometer
vals = {}
positives = {}
negatives = {}

# Find motionless periods of gyroscope
gyro_still = {}

a_bias = {}
a_gain = {}

#gyro bias 
g_bias = {}
g_gain = {}
#integrated gyro data
gyro_int = {}
for sensor in sensors: 
    if 'A' in sensor:
        vals[sensor] = []
        positives[sensor] = []
        negatives[sensor] = []

        # Detect motionless data around ± 1g
        data_gradient = abs(np.gradient(movmean(data[sensor], 50)))
        for i in range(0, len(data[sensor])-1):
            if (abs(-g-data[sensor][i]) < SENSITIVITY_A or abs(data[sensor][i] - g) < SENSITIVITY_A) and data_gradient[i]< SENSITIVITY_AD:
                    vals[sensor].append(i)
                    if abs(-g-data[sensor][i]) < SENSITIVITY_A:
                        negatives[sensor].append(i)
                    if abs(-g+data[sensor][i]) < SENSITIVITY_A:
                        positives[sensor].append(i)

        print("Detected {} valid samples in {}".format(len(vals[sensor]), sensor))

        # Use this data to compute first estimate of accelerometer gains and biases
        a_bias[sensor] = (sum([data[sensor][index] for index in positives[sensor]])/len([data[sensor][index] for index in positives[sensor]]) + sum([data[sensor][index] for index in negatives[sensor]])/len([data[sensor][index] for index in negatives[sensor]])) / 2
        a_gain[sensor] = (sum([data[sensor][index] for index in positives[sensor]])/len([data[sensor][index] for index in positives[sensor]]) - sum([data[sensor][index] for index in negatives[sensor]])/len([data[sensor][index] for index in negatives[sensor]])) / (2*g)
        print('Accelerometers {} bias: {:.4f} and gain: {:.4f} \n'.format(sensor, a_bias[sensor], a_gain[sensor]))

    if 'G' in sensor: 
        g_bias[sensor] = 0
        gyro_still[sensor] = []
        
        # Detect motionless data around 0 rad/s
        data_gradient = abs(np.gradient(movmean(data[sensor], 50)))
        for i in range(0, len(data[sensor])-1):
            if abs(data[sensor][i]) < SENSITIVITY_G and data_gradient[i]< SENSITIVITY_DG:
                gyro_still[sensor].append(i)
                g_bias[sensor] += data[sensor][i]

        # Bias is offset from 0 rad/s, and gain is ratio of integrated gyro (bias corrected) to full angle
        g_bias[sensor] /= len(gyro_still[sensor])
        
        gyro_int[sensor] = integrate.simps(data[sensor], data['t']) - g_bias[sensor]*max(data['t'])
        sign = 1 if abs(max(data[sensor])) > abs(min(data[sensor])) else -1
        g_gain[sensor] = gyro_int[sensor]/(sign*KNOWN_ANGLE)
        print('Gyroscope {} bias: {:.4f} and gain: {:.4f} \n'.format(sensor, g_bias[sensor], g_gain[sensor]))

        

#determine accelerometer scale factors
Uad = np.zeros((3,3))
theta = np.zeros((3,3))
W = np.zeros((3,3))
for i in range(0,3):
    for j in range(0,3):
        if i == j:
            Uad.itemset((i,j),  data[sensors[i+1]][positives[sensors[i+1]][j+1]]-data[sensors[i+1]][negatives[sensors[j+1]][i+1]]) 
            theta.itemset((i,j), KNOWN_ANGLE)
            W.itemset((i,j), gyro_int[sensors[i+4]])
a = 0
r  = np.matrix([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0,0,1]])
Uad = np.matmul(Uad, r)
print('accel scale factors')  
scaleFactors = (np.diag(np.matmul(Uad, Uad.T) / (4* (g**2)))**.5)
print(scaleFactors)
mult = np.matmul(W, np.linalg.inv(theta))
print("gyro scale factors")
print(np.diag(np.matmul(mult,mult.T))**.5)


# plot data 
fig, axs = plt.subplots(3, 3, figsize=(10,7))
fig.tight_layout()
for ax, sensor in zip(axs.flatten(), sensors[1:]):
    ax.plot(data['t'], data[sensor])

    if 'A' in sensor:
        ax.plot(data['t'], (np.array(data[sensor])-a_bias[sensor])/a_gain[sensor])
        ax.plot([data['t'][index] for index in vals[sensor]],[data[sensor][index] for index in vals[sensor]] ,'r+')       
        ax.axhline(g, color="green", linestyle=":")
        ax.axhline(-g, color="green", linestyle=":")

    if 'G' in sensor:
        ax.plot(data['t'], (np.array(data[sensor])-g_bias[sensor])/g_gain[sensor])
        ax.plot([data['t'][index] for index in gyro_still[sensor]],[data[sensor][index] for index in gyro_still[sensor]] ,'r+')
        ax.axhline(0.0, color="green", linestyle=":")

    ax.set_title(sensor)

for ax, sensor in zip(axs.flatten()[6:], sensors[4:]):
    ax.plot(data['t'], (180.0/np.pi)*integrate.cumtrapz(((np.array(data[sensor])-g_bias[sensor])/g_gain[sensor]), data['t'], initial=0))
    
    sign = 1 if abs(max(data[sensor])) > abs(min(data[sensor])) else -1
    for i in range(5):
        ax.axhline(sign*i*90, color="green", linestyle=":")

plt.show()
            