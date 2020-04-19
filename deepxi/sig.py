## AUTHOR:         Aaron Nicolson
## AFFILIATION:    Signal Processing Laboratory, Griffith University.
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.

from tensorflow.python.ops.signal import window_ops
import functools
import numpy as np
import scipy.special as spsp
import tensorflow as tf

class STFT:
	"""
	Short-time Fourier transform.
	"""
	def __init__(self, N_d, N_s, NFFT, f_s):
		"""
		Argument/s:
			N_d - window duration (samples).
			N_s - window shift (samples).
			NFFT - number of DFT bins.
			f_s - sampling frequency.
		"""
		self.N_d = N_d
		self.N_s = N_s
		self.NFFT = NFFT
		self.f_s = f_s
		self.W = functools.partial(window_ops.hamming_window, periodic=False)
		self.ten = tf.cast(10.0, tf.float32)

	def polar_analysis(self, x):
		"""
		Polar-form acoustic-domain analysis.

		Argument/s:
			x - waveform.

		Returns:
			Short-time magnitude and phase spectrums.
		"""
		STFT = tf.signal.stft(x, self.N_d, self.N_s, self.NFFT, window_fn=self.W, pad_end=True)
		return tf.abs(STFT), tf.math.angle(STFT)

	def polar_synthesis(self, STMS, STPS):
		"""
		Polar-form acoustic-domain synthesis.

		Argument/s:
			STMS - short-time magnitude spectrum.
			STPS - short-time phase spectrum.

		Returns:
			Waveform.
		"""
		STFT = tf.cast(STMS, tf.complex64)*tf.exp(1j*tf.cast(STPS, tf.complex64))
		return tf.signal.inverse_stft(STFT, self.N_d, self.N_s, self.NFFT, tf.signal.inverse_stft_window_fn(self.N_s, self.W))

class DeepXiInput(STFT):
	"""
	Input to Deep Xi.
	"""
	def __init__(self, N_d, N_s, NFFT, f_s, mu=None, sigma=None):
		super().__init__(N_d, N_s, NFFT, f_s)
		"""
		Argument/s
			N_d - window duration (samples).
			N_s - window shift (samples).
			NFFT - number of DFT bins.
			f_s - sampling frequency.
			mu - sample mean of each instantaneous a priori SNR (dB) frequency component.
			sigma - sample standard deviation of each instantaneous a priori SNR (dB) frequency component.
		"""
		self.mu = mu
		self.sigma = sigma

	def observation(self, x):
		"""
	    An observation for Deep Xi (noisy-speech STMS).

		Argument/s:
			x - noisy speech (dtype=tf.int32).
			x_len - noisy speech length without padding (samples).

		Returns:
			x_STMS - speech magnitude spectrum.
			x_STPS - speech phase spectrum.
		"""
		x = self.normalise(x)
		x_STMS, x_STPS = self.polar_analysis(x)
		return x_STMS, x_STPS

	def example(self, s, d, s_len, d_len, snr):
		"""
		Compute example for Deep Xi, i.e. observation (noisy-speech STMS)
		and target (mapped a priori SNR).

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			x_STMS - noisy-speech short-time magnitude spectrum.
			xi_bar - mapped a priori SNR.
			n_frames - number of time-domain frames.
		"""
		s_STMS, d_STMS, x_STMS, n_frames = self.mix(s, d, s_len, d_len, snr)
		mask = tf.expand_dims(tf.cast(tf.sequence_mask(n_frames), tf.float32), 2)
		xi_bar = tf.multiply(self.xi_bar(s_STMS, d_STMS), mask)
		return x_STMS, xi_bar, n_frames

	def mix(self, s, d, s_len, d_len, snr):
		"""
		Mix the clean speech and noise at SNR level, and then perform STFT analysis.

		Argument/s:
			s - clean speech (dtype=tf.int32).
			d - noise (dtype=tf.int32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.

		Returns:
			s_STMS - clean-speech short-time magnitude spectrum.
			d_STMS - noise short-time magnitude spectrum.
			x_STMS - noisy-speech short-time magnitude spectrum.
			n_frames - number of time-domain frames.
		"""
		s, d = self.normalise(s), self.normalise(d)
		n_frames = self.n_frames(s_len)
		(x, s, d) = self.add_noise_batch(s, d, s_len, d_len, snr)
		s_STMS, _ = self.polar_analysis(s)
		d_STMS, _ = self.polar_analysis(d)
		x_STMS, _ = self.polar_analysis(x)
		return s_STMS, d_STMS, x_STMS, n_frames

	def normalise(self, x):
		"""
		Convert waveform from int32 to float32 and normalise between [-1.0, 1.0].

		Argument/s:
			x - tf.int32 waveform.

		Returns:
			tf.float32 waveform between [-1.0, 1.0].
		"""
		return tf.truediv(tf.cast(x, tf.float32), 32768.0)

	def n_frames(self, N):
		"""
		Returns the number of frames for a given sequence length, and
		frame shift.

		Argument/s:
			N - sequence length (samples).

		Returns:
			Number of frames
		"""
		return tf.cast(tf.math.ceil(tf.truediv(tf.cast(N, tf.float32), tf.cast(self.N_s, tf.float32))), tf.int32)

	def add_noise_batch(self, s, d, s_len, d_len, snr):
		"""
		Creates noisy speech batch from clean speech, noise, and SNR batches.

		Argument/s:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR levels.

		Returns:
			tuple consisting of clean speech, noisy speech, and noise (x, s, d).
		"""
		return tf.map_fn(lambda z: self.add_noise_pad(z[0], z[1], z[2], z[3], z[4],
			tf.reduce_max(s_len)), (s, d, s_len, d_len, snr), dtype=(tf.float32, tf.float32,
			tf.float32), back_prop=False)

	def add_noise_pad(self, s, d, s_len, d_len, snr, pad_len):
		"""
		Calls add_noise() and pads the waveforms to the length given by 'pad_len'.
		Also normalises the waveforms.

		Argument/s:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level.
			pad_len - padded length.

		Returns:
			s - padded clean-speech waveform.
			x - padded noisy-speech waveform.
			d - truncated, scaled, and padded noise waveform.
		"""
		s, d = s[:s_len], d[:d_len]
		(x, d) = self.add_noise(s, d, s_len, d_len, snr)
		total_zeros = tf.subtract(pad_len, s_len)
		x = tf.pad(x, [[0, total_zeros]], "CONSTANT")
		s = tf.pad(s, [[0, total_zeros]], "CONSTANT")
		d = tf.pad(d, [[0, total_zeros]], "CONSTANT")
		return (x, s, d)

	def add_noise(self, s, d, s_len, d_len, snr):
		"""
		Adds noise to the clean speech at a specific SNR value. A random section
		of the noise waveform is used.

		Argument/s:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).
			s_len - clean-speech length without padding (samples).
			d_len - noise length without padding (samples).
			snr - SNR level (dB).

		Returns:
			x - noisy-speech waveform.
			d - truncated and scaled noise waveform.
		"""
		snr = tf.cast(snr, tf.float32)
		snr = tf.pow(self.ten, tf.truediv(snr, self.ten)) # inverse of dB.
		i = tf.random.uniform([1], 0, tf.add(1, tf.subtract(d_len, s_len)), tf.int32)
		d = tf.slice(d, [i[0]], [s_len])
		P_s = tf.reduce_mean(tf.math.square(s), 0) # average power of clean speech.
		P_d = tf.reduce_mean(tf.math.square(d), 0) # average power of noise.
		alpha = tf.math.sqrt(tf.truediv(P_s,
			tf.maximum(tf.multiply(P_d, snr), 1e-12))) # scaling factor.
		d =	tf.multiply(d, alpha)
		x = tf.add(s, d)
		return (x, d)

	def snr_db(self, s, d):
		"""
		Calculates the SNR (dB) between the speech and noise.

		Argument/s:
			s - clean speech (dtype=tf.float32).
			d - noise (dtype=tf.float32).

		Returns:
			SNR level (dB).
		"""
		P_s = tf.reduce_mean(tf.math.square(s), 0) # average power of clean speech.
		P_d = tf.reduce_mean(tf.math.square(d), 0) # average power of noise.
		return tf.multiply(self.ten, self.log_10(tf.truediv(P_s, P_d)))

	def log_10(self, x):
		"""
		log_10(x).

		Argument/s:
			x - input.

		Returns:
			log_10(x)
		"""
		return tf.truediv(tf.math.log(x), tf.math.log(self.ten))

	def xi(self, s_STMS, d_STMS):
		"""
		Instantaneous a priori SNR.

		Argument/s:
			s_STMS - clean-speech short-time magnitude spectrum.
			d_STMS - noise short-time magnitude spectrum.

		Returns:
			Instantaneous a priori SNR.
		"""
		return tf.truediv(tf.square(s_STMS), tf.maximum(tf.square(d_STMS), 1e-12))

	def xi_db(self, s_STMS, d_STMS):
		"""
		Instantaneous a priori SNR in dB.

		Argument/s:
			s_STMS - clean-speech short-time magnitude spectrum.
			d_STMS - noise short-time magnitude spectrum.

		Returns:
			Instantaneous a priori SNR in dB.
		"""
		return tf.multiply(10.0, self.log_10(tf.maximum(self.xi(s_STMS, d_STMS), 1e-12)))

	def xi_bar(self, s_STMS, d_STMS):
		"""
		Mapped a priori SNR in dB.

		Argument/s:
			s_STMS - clean-speech short-time magnitude spectrum.
			d_STMS - noise short-time magnitude spectrum.

		Returns:
			Mapped a priori SNR in dB.
		"""
		return tf.multiply(0.5, tf.add(1.0, tf.math.erf(tf.truediv(tf.subtract(self.xi_db(s_STMS, d_STMS), self.mu),
			tf.multiply(self.sigma, tf.sqrt(2.0))))))

	def xi_hat(self, xi_bar_hat):
		"""
		A priori SNR estimate.

		Argument/s:
			xi_bar_hat - mapped a priori SNR estimate.

		Returns:
			A priori SNR estimate.
		"""
		xi_db_hat = np.add(np.multiply(np.multiply(self.sigma, np.sqrt(2.0)),
			spsp.erfinv(np.subtract(np.multiply(2.0, xi_bar_hat), 1))), self.mu)
		return np.power(10.0, np.divide(xi_db_hat, 10.0))
