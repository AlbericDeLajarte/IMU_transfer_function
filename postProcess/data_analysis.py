import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter

# ------------ Get data ------------

data = np.loadtxt("log/log.txt", delimiter=", ").T

time_in = data[0]
acc_in = data[1:4, :]
#gyro_in = data[4:7, :]

time_out = data[4]
acc_out = data[5:, :]

#time_out = data[7]
#acc_out = data[8:11, :]
#gyro_out = data[11:, :]

# ------------ Compute FFT ------------

time_period = np.mean(np.diff(time_in))
min_frequ_fft = 0.3
N_sample = int((1/time_period)/min_frequ_fft)
#N_sample = len(time_in)
#print(time_period)

FFT_acc_in = np.zeros((3, N_sample//2),dtype=np.complex_)
FFT_acc_out = np.zeros((3, N_sample//2),dtype=np.complex_)

frequ = fftfreq(N_sample, time_period)[:N_sample//2]

n_segment = int(len(time_in)/N_sample)

for i in range(n_segment):

    FFT_acc_in += np.array([ fft(acc_in[0, i*N_sample:(i+1)*N_sample]), fft(acc_in[1, i*N_sample:(i+1)*N_sample]), fft(acc_in[2, i*N_sample:(i+1)*N_sample]) ])[:, :N_sample//2]

    FFT_acc_out += np.array([ fft(acc_out[0, i*N_sample:(i+1)*N_sample]), fft(acc_out[1, i*N_sample:(i+1)*N_sample]), fft(acc_out[2, i*N_sample:(i+1)*N_sample]) ])[:, :N_sample//2]

#FFT_gyro_in = np.array([ fft(gyro_in[0]), fft(gyro_in[1]), fft(gyro_in[2]) ])[:, :N_sample//2]

#FFT_gyro_out = np.array([ fft(gyro_out[0]), fft(gyro_out[1]), fft(gyro_out[2]) ])[:, :N_sample//2]

TF_acc = FFT_acc_out/FFT_acc_in

bode_magnitude = 20*np.log(abs(TF_acc[2]))
bode_magnitude_raw = 20*np.log(abs(fft(acc_out[0, 0:N_sample])/fft(acc_in[0, 0:N_sample])))[0:N_sample//2]


# ------------ Plot results ------------

#plt.plot(time_in, acc_in[0], time_out, acc_out[0])

#plt.plot(frequ, abs(FFT_acc_in[0]), frequ, abs(FFT_acc_in[1]), frequ, abs(FFT_acc_in[2]))
#plt.plot(frequ, abs(FFT_acc_out[0])/abs(FFT_acc_in[0]), frequ, abs(FFT_acc_out[1])/abs(FFT_acc_in[1]), frequ, abs(FFT_acc_out[2])/abs(FFT_acc_in[2]))
#plt.plot(frequ, abs(FFT_gyro_out[0])/abs(FFT_gyro_in[0]), frequ, abs(FFT_gyro_out[1])/abs(FFT_gyro_in[1]), frequ, abs(FFT_gyro_out[2])/abs(FFT_gyro_in[2]))

#plt.plot(abs(FFT_acc_out[2, :]))
#plt.plot(frequ, bode_magnitude_raw)
# plt.plot(frequ, bode_magnitude)
# plt.plot(frequ, savgol_filter(bode_magnitude, 5, 3))

fig, axs = plt.subplots(2, 3)
for i, ax in enumerate(axs.flatten()):
    ax.plot(frequ, bode_magnitude)
    ax.plot(frequ, savgol_filter(bode_magnitude, 11 + 6*i, 3))

#plt.xscale('log')

plt.show()
