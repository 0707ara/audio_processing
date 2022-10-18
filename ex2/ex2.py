import sounddevice as sd
import soundfile as sf

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
from scipy.fftpack import fftfreq

# 2.a
# When you use the hamming window, magnitude gets smaller with same frequency.
data, samplerate = sf.read('audio1.wav')
plt.plot(data)


#sample rate is 22050 100ms(0.1s)
for i in range(0,len(data),int(samplerate*0.1)):
    endindex = i + int(samplerate*0.1)
    data_subset = data[i:endindex]
    hamming_window = signal.windows.hamming(int(samplerate*0.1))
    hamming_multiplied = data_subset * hamming_window
    yf = fft(hamming_multiplied)
    xf = fftfreq(int(samplerate * 0.1) , 1 / samplerate)
    #plt.plot(xf, np.abs(yf1))
    #plt.show()


# 2.b
powerspectrum = []
j = 1
for i in range(0,len(data)-int(samplerate*0.1*0.5),int(samplerate*0.1*0.5)):
    endindex = i + int(samplerate*0.1)
    data_subset = data[i:endindex]
    hamming_window = signal.windows.hamming(int(samplerate*0.1))
    hamming_multiplied = data_subset * hamming_window

    yf1 = fft(data_subset)
    yf_hamming = fft(hamming_multiplied)
    yf_square = np.square(np.abs(yf_hamming))
    powerspectrum.append(yf_square)
    xf = fftfreq(int(samplerate * 0.1) , 1 / samplerate)

    j=j+1
powerspectrum_arr = np.asarray(powerspectrum)
powerspectrum_arr_T = powerspectrum_arr.T

plt.imshow(powerspectrum_arr_T,cmap='jet',aspect='auto',origin='lower')
plt.title('Power spectrum of my own')

plt.imshow(np.log(powerspectrum_arr_T),cmap='jet',aspect='auto',origin='lower')
plt.title('Power spectrum of my own (logarithm)')


# Problem 2.a) Spectrogram using librosa library
x, sr= librosa.load('audio1.wav',sr=None)
D = librosa.stft(x)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio 1')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# Problem 2.b) Spectrogram using librosa library with different window size
# audio1.wav
x1, sr= librosa.load('audio1.wav',sr=None)
x1_16ms = x1[2500:2500+int(sr*0.016)]
x1_32ms = x1[2500:2500+int(sr*0.032)]
x1_64ms = x1[2500:2500+int(sr*0.064)]
x1_128ms = x1[2500:2500+int(sr*0.128)]

D1 = librosa.stft(x1)
D1_16ms = librosa.stft(x1_16ms)
D1_32ms = librosa.stft(x1_32ms)
D1_64ms = librosa.stft(x1_64ms)
D1_128ms = librosa.stft(x1_128ms)
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 whole window')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_16ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 16ms')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_32ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 32ms')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 4)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_64ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 64ms')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 5)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_128ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 128ms')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# audio2.wav
x1, sr= librosa.load('audio2.wav',sr=None)
x1_16ms = x1[2500:2500+int(sr*0.016)]
x1_32ms = x1[2500:2500+int(sr*0.032)]
x1_64ms = x1[2500:2500+int(sr*0.064)]
x1_128ms = x1[2500:2500+int(sr*0.128)]

D1 = librosa.stft(x1)
D1_16ms = librosa.stft(x1_16ms)
D1_32ms = librosa.stft(x1_32ms)
D1_64ms = librosa.stft(x1_64ms)
D1_128ms = librosa.stft(x1_128ms)
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 whole window')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_16ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 16ms')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_32ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 32ms')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 4)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_64ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 64ms')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 5)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_128ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 128ms')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# output.wav
x1, sr= librosa.load('output.wav',sr=None)
x1_16ms = x1[2500:2500+int(sr*0.016)]
x1_32ms = x1[2500:2500+int(sr*0.032)]
x1_64ms = x1[2500:2500+int(sr*0.064)]
x1_128ms = x1[2500:2500+int(sr*0.128)]

D1 = librosa.stft(x1)
D1_16ms = librosa.stft(x1_16ms)
D1_32ms = librosa.stft(x1_32ms)
D1_64ms = librosa.stft(x1_64ms)
D1_128ms = librosa.stft(x1_128ms)
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 whole window')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_16ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 16ms')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_32ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 32ms')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 4)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_64ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 64ms')
plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 3, 5)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(D1_128ms),ref=np.max),y_axis='log', x_axis='time')
plt.title('Power spectrogram of audio1 128ms')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
