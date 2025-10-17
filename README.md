# Phone INS

this is partially complete INS for smartphone without root. Kalman and fft filters.

before running the code on your device you must log accelerometer data, and obtain fourier spectra in fft=exp(iR) representation. This will allow you to create fft mask (preserve negative freqs for mathematical symmetry).

such a spectrum is added - this screenshot


additional filtering of accelerometer - fft - was added.

use it for 3D scanning purposes only, but i know...

works on pydroid3
