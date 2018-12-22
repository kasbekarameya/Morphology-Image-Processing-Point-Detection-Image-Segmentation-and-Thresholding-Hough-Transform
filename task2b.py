#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

image = cv2.imread('original_imgs/segment.jpg', 0)
M,N= image.shape

#Here we plot the points in a histogram based on the intensity of the pixels & their occurences in the image.
arr=[]
for k in range(1, M-1):
        for l in range(1, N-1):
            arr.append(image[k][l])
freq = {}
for item in arr:
        if(item==0):
            freq[item] = 0
        elif (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

for x,y in freq.items():
    plt.bar(x, y)

plt.savefig("images/Histogram.png")

# Here we apply a optimal threshold value in order to remove the noise & segment only the pixels belonging to the bones.
for k in range(1, M-1):
    for l in range(1, N-1):
        if image[k][l]>=205:
            image[k][l]=255
        else:
            image[k][l]=0

cv2.imwrite('images/Segmented.jpg',image)

# These lines are used to generate bounding boxes around each of the segmented bone.
cv2.rectangle(image,(162,124),(203,165),(255,255,255),1)
cv2.rectangle(image,(250,76),(304,205),(255,255,255),1)
cv2.rectangle(image,(335,24),(365,287),(255,255,255),1)
cv2.rectangle(image,(387,41),(425 ,253),(255,255,255),1)

cv2.imwrite('images/Segmented_Bounded.jpg',image)