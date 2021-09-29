import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy import interpolate

import sippy as si
import control.matlab as cnt

def movmean(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

# ------------ Get data ------------

data = np.loadtxt("log/log.txt", delimiter=", ").T

# Parse raw data
time_in = data[0]
acc_in = data[1:4, :]

time_out = data[4]
acc_out = data[5:, :]

# Reinterpolate to get constant timestep
f_a_in = interpolate.interp1d(time_in, acc_in)
time_in = np.linspace(min(time_in), max(time_in), len(time_in))
acc_in = f_a_in(time_in)

f_a_out = interpolate.interp1d(time_out, acc_out)
time_out = np.linspace(min(time_out), max(time_out), len(time_out))
acc_out = f_a_out(time_out)

# Filter out high frequency noise
sos = signal.butter(2, 0.3, btype='low', analog=False, output='sos')
plt.plot(time_in, acc_in[0])
acc_in = signal.sosfiltfilt(sos, acc_in)
plt.plot(time_in, acc_in[0])
plt.show()
acc_out = signal.sosfiltfilt(sos, acc_out)

# Remove bias and add horizontal acceleleration in vector
acc_in -= np.array([acc_in.mean(axis=1)]).T
acc_in = np.vstack((acc_in, acc_in[1, :]+acc_in[2, :]))

acc_out -= np.array([acc_out.mean(axis=1)]).T
acc_out = np.vstack((acc_out, acc_out[1, :]+acc_out[2, :]))

#time_out = data[7]
#acc_out = data[8:11, :]
#gyro_out = data[11:, :]
#gyro_in = data[4:7, :]

print("Experiment duration: {:.3f} | mean sampling rate: {:.3f} ms (SD: {:.3f})".format(max(time_in), 1000*np.mean(np.diff(time_in)), 1000*np.std(np.diff(time_in))))

# ------------ Compute FFT ------------

time_period = np.mean(np.diff(time_in))
min_frequ_fft = 0.1
N_sample = int((1/time_period)/min_frequ_fft)
#N_sample = len(time_in)
#print(time_period)

FFT_acc_in = np.zeros((4, N_sample//2),dtype=np.complex_)
FFT_acc_out = np.zeros((4, N_sample//2),dtype=np.complex_)

frequ = fftfreq(N_sample, time_period)[:N_sample//2]

n_segment = int(len(time_in)/N_sample)

for i in range(n_segment):

    FFT_acc_in += np.array([ fft(acc_in[0, i*N_sample:(i+1)*N_sample]), fft(acc_in[1, i*N_sample:(i+1)*N_sample]), fft(acc_in[2, i*N_sample:(i+1)*N_sample]), fft(acc_in[3, i*N_sample:(i+1)*N_sample]) ])[:, :N_sample//2]

    FFT_acc_out += np.array([ fft(acc_out[0, i*N_sample:(i+1)*N_sample]), fft(acc_out[1, i*N_sample:(i+1)*N_sample]), fft(acc_out[2, i*N_sample:(i+1)*N_sample]), fft(acc_out[3, i*N_sample:(i+1)*N_sample]) ])[:, :N_sample//2]

#FFT_gyro_in = np.array([ fft(gyro_in[0]), fft(gyro_in[1]), fft(gyro_in[2]) ])[:, :N_sample//2]

#FFT_gyro_out = np.array([ fft(gyro_out[0]), fft(gyro_out[1]), fft(gyro_out[2]) ])[:, :N_sample//2]

TF_acc = FFT_acc_out/FFT_acc_in

# na_ord=[9, 9], nb_ord=[2, 2], nc_ord=[9, 9], delays=[1, 1],
TF_ARMAX = []
for i in range(4):
    TF_ARMAX.append(si.system_identification(acc_out[i], acc_in[i], 'ARMAX', IC='BIC', na_ord=[9, 9], nb_ord=[2, 2],
                                    nc_ord=[9, 9], delays=[1, 1], max_iterations=200, ARMAX_mod = 'ILLS', tsample=time_period))

#TF_ARMAX.append(si.system_identification(acc_out_horizontal, acc_in_horizontal, 'ARMAX', IC='BIC', na_ord=[15, 15], nb_ord=[15, 15],
 #                                   nc_ord=[15, 15], delays=[1, 1], max_iterations=200, ARMAX_mod = 'ILLS', tsample=time_period))
# ------------ Plot results ------------

#plt.plot(time_in, acc_in[0], time_out, acc_out[0])

#plt.plot(frequ, abs(FFT_acc_in[0]), frequ, abs(FFT_acc_in[1]), frequ, abs(FFT_acc_in[2]))
#plt.plot(frequ, abs(FFT_acc_out[0])/abs(FFT_acc_in[0]), frequ, abs(FFT_acc_out[1])/abs(FFT_acc_in[1]), frequ, abs(FFT_acc_out[2])/abs(FFT_acc_in[2]))
#plt.plot(frequ, abs(FFT_gyro_out[0])/abs(FFT_gyro_in[0]), frequ, abs(FFT_gyro_out[1])/abs(FFT_gyro_in[1]), frequ, abs(FFT_gyro_out[2])/abs(FFT_gyro_in[2]))

w_v = np.logspace(-1,3,num=701)
#w_v = np.linspace(min_frequ_fft, 4.0/time_period)

if False:
    plt.figure(1)
    for i in range(4):
        mag, fi, om = cnt.bode(TF_ARMAX[i].G, w_v, Hz = True, plot=True)
        plt.legend(["X", "Y", "Z", "plane"])
    plt.show()



if False:
    fig, axs = plt.subplots(1, 3, figsize=(12,5))
    fig.tight_layout()

    for i, ax in enumerate(axs.flatten()):
        ax.plot(time_out, acc_out[i])
        ax.plot(time_out, movmean(acc_out[i], 6))
        ax.plot(time_in, TF_ARMAX[i].Yid.T, "r+-")

    plt.show()

if False:
    fig, axs = plt.subplots(1, 3, figsize=(12,5))
    fig.tight_layout()
    for i, ax in enumerate(axs.flatten()):

        bode_magnitude = 20*np.log(abs(TF_acc[i]))
        bode_magnitude_raw = 20*np.log(abs(fft(acc_out[i, 0:N_sample])/fft(acc_in[i, 0:N_sample])))[0:N_sample//2]

        #ax.plot(frequ, 20*np.log(abs(FFT_acc_in[i, :])))


        ax.plot(frequ, bode_magnitude, label="averaged")
        #ax.plot(frequ, signal.savgol_filter(bode_magnitude, 51, 3), label="fit")

        # ax.plot(time_in, acc_out[i], label="output")
        # ax.plot(time_in, acc_in[i], label="input")

        # plt.plot(frequ, bode_magnitude_raw, label= "raw")
        # plt.plot(frequ, savgol_filter(bode_magnitude_raw, 51, 3))

        ax.legend()
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_title("Bode diagram for acceleration {}".format(["X", "Y", "Z"][i]))



    # fig, axs = plt.subplots(2, 3)
    # for i, ax in enumerate(axs.flatten()):
    #     ax.plot(frequ, bode_magnitude)
    #     ax.plot(frequ, savgol_filter(bode_magnitude, 11 + 6*i, 3))

    #plt.xscale('log')

    plt.show()
