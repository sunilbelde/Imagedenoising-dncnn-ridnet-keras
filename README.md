# Image denoising using deep learning model

Creating deep learning model that removes noise out of a given noisy image

## What is noise ?

  An additional unnecessary pixel values are added to a image causing the loss of information.The noise can be originated by many ways such as while capturing images in low-light situations, damage of electric circuits due to heat, sensor illumination levels of a digital camera or due to the faulty memory locations in hardware or bit errors in transmission of data over long distances.
  It is essential to remove the noise and recover the original image from the degraded images where getting the original image is important for robust performance or in cases where filling the missing information is very useful like the astronomical images that are taken from very distant objects.


## Solution :

Therea are many deep learning model that can be used for completing this task of image denoising. Now we will use Deep Convulutional Neural Network mode (DnCNN)

Architecture of the model : 

![Architecture](https://user-images.githubusercontent.com/47601858/115210654-6e807280-a11c-11eb-8456-b0930aa15c7c.JPG)


Given a noisy image 'y' the model will predict residual image 'R' and clean image 'x' can be obtained by 

x=y-R

Research paper : https://arxiv.org/pdf/1608.03981v1.pdf

## Dataset: 

We will be using publicly avaliable image and modify it according to our requirement 

dataset : https://github.com/BIDS/BSDS500

This Dataset is provided by Berkeley University of California which contains 500 natural images.

Now we create 85600 patches of size 40 x 40 that was created from 400 train images and
21400 patches of size 40 x 40 that was created from 100 test images 

## Training:

Model has been train for 30 epochs with Adam optimizer of learning rate=0.001 and with learning rate decay of 5% per epoch
.Mean Squared Error is used as loss function.

## Results :

For an noisy image with psnr of 20.530 obtained denoised image which has psnr of 31.193

Image showing the comparision of ground truth, noisy image and denoised image.

![result_images](https://user-images.githubusercontent.com/47601858/115210102-e732ff00-a11b-11eb-9881-92521a7e84a6.JPG)

Image showing patch wise noisy and denoised images.
![result_patches](https://user-images.githubusercontent.com/47601858/115210524-501a7700-a11c-11eb-8950-ca5897e61a72.JPG)

Below plot shows the model performance on different noise levels:
![results_dncnn](https://user-images.githubusercontent.com/47601858/115216274-f74ddd00-a121-11eb-8ecc-84bac484b3c4.JPG)

