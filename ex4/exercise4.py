import numpy as np
import scipy
from scipy.io import wavfile
from numpy.fft import fft, ifft, fftshift
import librosa as lb

from scipy.signal import hamming, hann
from matplotlib import pyplot as plt


from audiolazy import lazy_lpc as lpc

# interactive plotting
plt.ion()



def est_predictor_gain(x, a, p):
    '''
    A function to compute gain of the residual signal in LP analyis.
    x:  signal 
    a: LPC coefficients
    p: order of the filter
    '''
    cor = np.correlate(x, x, mode='full')
    
    rr = cor[len(cor)//2: len(cor)//2+p+1]
    g = np.sqrt(np.sum(a*rr))
    return g

   
    
def reject_outliers(data, m=2):
    '''
    Function to reject outliers. All values beyond m standard deviations from means are excluded
    '''
    return data[abs(data - np.mean(data)) < m * np.std(data)]
    
    
# # read audio
audioIn, fs=lb.load('audio.wav', sr=None)   

# filter order
p = 4                    # has to be tuned

# number of DFT points
nfft = 1024

inInd =0
wLen = int(0.02*fs) # 20 ms window
win = hamming(wLen) # hamming window for example

cnt = 0
numframes = np.ceil( (len(audioIn)-wLen)/(wLen/2)) # number of franes 
formants  = []                                     # A placeholder for storing formants

# choose a representative frame of the vowel
plot_frame = int(numframes/2)  # middle of the vowel

# The analysis loop
while inInd< len(audioIn)-wLen:
    # audio frame
    frame = audioIn[inInd:inInd+wLen]* win
    
    
    # compute LPC and gain 
    
 
    
    # Compute the filter tansfer function
 
    
    # Compute DFT spectrum
    
    
    # Compute roots
       
       
    #  LPC coefficients are real-valued, the roots occur in complex conjugate pairs. Retain only the roots with +ve sign for the imaginary part 
    
    
    # compute formants from roots
    

    # convert to Hertz from angular frequencies
    angz = angz*(fs/(2*np.pi))

    # sort the formants in increasing order
    angz = np.sort(angz)
    
    # remove zero frequencies
    angz = angz[angz !=0]
    
    # First three formants
    formants.append(angz[:3]) 
    
    inInd = inInd + int(wLen/2) # frame advance
    
    cnt = cnt+1
    
    # plot the FFT spectrum and LPC spectrum here for chosen frame
    if cnt == plot_frame :
        # plot DFT spectrum (remember both in dB scale)
        # plot LPC spectrum
        plt.figure()
        

formants = np.array(formants)

print('------ The computed formants are :', np.mean(formants, 0))

# Refine formant estimations (optional)











