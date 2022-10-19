
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav

from audiolazy import lpc
from multiprocessing import Pool

class LPC:

    def __init__(self, _num_worker):
        self._num_worker = _num_worker

    def wav2lpc(self, wav_file: str) -> np.array:
        (rate, sig) = wav.read(wav_file)

        videorate = 24
        nw = int(round(rate  /  videorate)) # time
        inc = int(nw / 2) # 2* overlap
        winfunc = np.hanning(nw)
        sig = signal.detrend(sig, type= 'constant')

        frame = self.enframe(sig, nw, inc, winfunc)
        assert len(frame) >= 64

        win_size = 64
        K = 32 # number of coefficients
        win_count = int((len(frame)-win_size) / 2)+1 # len(frame) or frame.shape[0]
        lpc_feature = np.zeros(shape=(frame.shape[0], K))
        output = np.zeros(shape=(win_count, win_size, K))

        pool = Pool(self._num_worker)
        filt_coef = pool.map(self.lpc_K, frame)
        lpc_feature[:] = filt_coef

        for win in range(win_count):
            output[win] = lpc_feature[2*win : 2*win+win_size]

    def lpc_K(frame, order=32):
        filt = lpc.nautocor(frame, order=order)
        return filt.numerator[1:] # List of coefficients

    def enframe(self, signal, nw, inc, winfunc):
        """turn audio signal to frame.
        parameters:
        signal: original audio signal
        nw: length of each frame(audio samples = audio sample rate * time interval)
        inc: intervals of consecutive frames
        """
        signal_length=len(signal) #length of audio signal
        if signal_length<=nw:
            nf=1
        else: #otherwise, compute the length of audio frame
            nf=int(np.ceil((1.0*signal_length-nw+inc) / inc))
        pad_length=int((nf-1)*inc+nw) #length of flatten all frames
        zeros=np.zeros((pad_length-signal_length,)) #
        pad_signal=np.concatenate((signal,zeros)) #after padding
        indices=np.tile(np.arange(0,nw),(nf,1))+np.tile(np.arange(0,nf*inc,inc),(nw,1)).T
        indices=np.array(indices,dtype=np.int32) #turn indices to frames
        frames=pad_signal[indices] #get frames
        win=np.tile(winfunc,(nf,1))  #window function
        return frames*win   #return frame matrix