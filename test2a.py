# -*- coding: utf-8 -*-
import cv2
import numpy as np

image = cv2.imread('original_imgs/point.jpg', 0)
N,M=image.shape

# This is a Laplacian Filter used for Point Detection
struct = np.asarray([[-1, -1, -1],
                     [-1, 8, -1],
                     [-1, -1,-1]])

M,N= image.shape

kRows=3
kCols=3

PaddedImage=np.asarray([[0.0 for column in range(N)] for row in range(M)])
#Here we performing 2D Convolution operation
kCenterX = int(kCols / 2);
kCenterY = int(kRows / 2);

for i in range(1, M-1):
     for j in range(1, N-1):
         x=0
         for m in range(0, kRows):
                     mm = kRows - 1 - m
                     for n in range(0, kCols):
                         nn = kCols - 1 - n
                         ii = i + (kCenterY - mm)
                         jj = j + (kCenterX - nn)
                         if (ii >= 0 and ii < N and jj >= 0 and jj < M):
                             x+=image[ii][jj] * struct[mm][nn]
                             PaddedImage[i][j]=abs(x)

#Here the threshold value is set based on the maximum intensity value in the image & is compared with the intensity values of each pixel in the image.
#If value of the pixel being compared is more than the threshold value, it is considered as a point & we set it to the maximum intensity value of 255.
MaxRvalue = np.max(PaddedImage)
Coordinates = []
Threshold = 0.9 * MaxRvalue
for row in range(0, len(PaddedImage)):
    for col in range(0, len(PaddedImage[0])):
        if PaddedImage[row][col] >= Threshold:
            PaddedImage[row][col] = 255
            Coordinates.append([row, col])
        else:
            PaddedImage[row][col] = 0
print("Co-Ordinate: ("+ str(Coordinates[0][1])+","+str(Coordinates[0][0])+")")

cv2.imwrite('images/Point_Detected.jpg',PaddedImage)

