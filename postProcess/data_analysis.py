import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.fft import fft, fftfreq


data = np.loadtxt("log/log.txt", delimiter=", ").T

time_in = data[0]
acc_in = data[1:4, :]
gyro_in = data[5:8, :]

time_out = data[7]
acc_out = data[8:11, :]
gyro_out = data[12:, :]

# Compute FFT
time_period = 25e-3
min_frequ_fft = 0.1
N_sample = int((1/time_period)/min_frequ_fft)
frequ = fftfreq(N_sample, time_period)[:N_sample//2]

FFT_acc_in = np.array([ fft(acc_in[0]), fft(acc_in[1]), fft(acc_in[2]) ])[:, :N_sample//2]

FFT_acc_out = np.array([ fft(acc_out[0]), fft(acc_out[1]), fft(acc_out[2]) ])[:, :N_sample//2]

print(FFT_acc_in.shape)
#plt.plot(time_in, acc_in[0], time_out, acc_out[0])

plt.plot(frequ, abs(FFT_acc_out[0])/abs(FFT_acc_in[0]), frequ, abs(FFT_acc_out[1])/abs(FFT_acc_in[1]), frequ, abs(FFT_acc_out[2])/abs(FFT_acc_in[2]))

plt.show()
