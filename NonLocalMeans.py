# -*- coding: utf-8 -*-
"""301Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tZLLH7iPrISvyCmKYFGRbCxuFboeMGYL
"""

import cv2                                   ## not all used, I will remove some later
import numpy as np 
from skimage  import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
import scipy.io as sc
from scipy.io import savemat
from scipy import signal
import numpy as np
import matplotlib
import matplotlib.image as im
import matplotlib.pyplot as plt

import pywt

mario = im.imread('mariop.png')
imagem = mario
plt.imshow(imagem, interpolation='nearest', cmap='gray')
plt.title('Original Mario')
plt.show()
richb = im.imread('richb8.png')
imager = richb
plt.imshow(imager, interpolation='nearest', cmap='gray')
plt.title('Original Richb')
plt.show()
bad = im.imread('unclearp.png')
imageb = bad
plt.imshow(imageb, interpolation='nearest', cmap='gray')
plt.title('Original Weird Thing')
plt.show()
small = im.imread('smallmariop.png')
imagesm = small
plt.imshow(imagesm, interpolation='nearest', cmap='gray')
plt.title('Original Small Mario')
plt.show()
print(imagem.shape)

def add_gaussian_noise(image_in, noise_sigma):      #Adds gaussian noise with noise_sigma as standard deviation
    noisy = np.zeros(image_in.shape)
    n1,n2,n3 = image_in.shape
    for i in range(n3):
      noise = np.random.randn(n1, n2) * noise_sigma
      for j in range(n1):
        for k in range(n2):
          noisy[j,k,i] = image_in[j,k,i] + noise[j,k]
    
    return noisy

def find_mse(image1, image2):           # finds mse of two images with same dimensions
  n1,n2,n3 = image1.shape
  mse = 0.0
  for i in range(n3):
    for j in range(n1):
      for k in range(n2):
        mse= mse + (image2[j,k,i]-image1[j,k,i])**2
  mse = mse / (n1*n2*n3)
  return mse

noisy_imagem = add_gaussian_noise(imagem, 1)             #adding noise to big mario
plt.imshow(noisy_imagem, interpolation='nearest', cmap='gray')
plt.title('noisy mario')
plt.show()
print(find_mse(imagem, noisy_imagem))

noisy_imagesmall = add_gaussian_noise(imagesm, 1)     # adding noise to small mario
plt.imshow(noisy_imagesmall, interpolation='nearest', cmap='gray')
plt.title('noisy small mario')
plt.show()
print(find_mse(imagesm, noisy_imagesmall))

noisy_imager = add_gaussian_noise(imager, .1)              #adding noise to richb
plt.imshow(noisy_imager, interpolation='nearest', cmap='gray')
plt.title('noisyrichb')
plt.show()
print(find_mse(imager, noisy_imager))



noisy_imageb = add_gaussian_noise(imageb, .1)          #adding noise to unnatural image
plt.imshow(noisy_imageb, interpolation='nearest', cmap='gray')
plt.title('noisy weird thing')
plt.show()
print(find_mse(imageb, noisy_imageb))

img = img_as_float(noisy_imagem)                #first nlm denoising
sigma_est = np.mean(estimate_sigma(img, multichannel=False))
denoise_img = denoise_nl_means(img, h=1.*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

plt.imshow(denoise_img, interpolation='nearest', cmap='gray')
plt.title('recon mario')
plt.show()
print(find_mse(imagem, denoise_img))            #Checking performance

img = img_as_float(denoise_img)
sigma_est = np.mean(estimate_sigma(img, multichannel=False))
denoise_img = denoise_nl_means(img, h=1.*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

plt.imshow(denoise_img, interpolation='nearest', cmap='gray')
plt.title('recon mario')
plt.show()
print(find_mse(imagem, denoise_img))              #checking second nlm denoising

##plt.imshow(out.astype('uint8'))
img = img_as_float(noisy_imager)
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=1.*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

plt.imshow(denoise_img, interpolation='nearest', cmap='gray')
plt.title('recon mario')
plt.show()
print(find_mse(imager, denoise_img))

img = img_as_float(denoise_img)
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=1.*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

plt.imshow(denoise_img, interpolation='nearest', cmap='gray')
plt.title('recon mario')
plt.show()
print(find_mse(imager, denoise_img))

img = img_as_float(noisy_imageb)
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=1.*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

plt.imshow(denoise_img, interpolation='nearest', cmap='gray')
plt.title('recon mario')
plt.show()
print(find_mse(imageb, denoise_img))

img = img_as_float(denoise_img)
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=1.*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

plt.imshow(denoise_img, interpolation='nearest', cmap='gray')
plt.title('recon mario')
plt.show()
print(find_mse(imageb, denoise_img))

img = img_as_float(noisy_imagesmall)
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=1.*sigma_est, fast_mode=True, patch_size=5, patch_distance=10, multichannel=True)

plt.imshow(denoise_img, interpolation='nearest', cmap='gray')
plt.title('recon mario')
plt.show()
print(find_mse(imageb, denoise_img))

img = img_as_float(denoise_img)
sigma_est = np.mean(estimate_sigma(img, multichannel=True))
denoise_img = denoise_nl_means(img, h=1.*sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=True)

plt.imshow(denoise_img, interpolation='nearest', cmap='gray')
plt.title('recon mario')
plt.show()
print(find_mse(imageb, denoise_img))