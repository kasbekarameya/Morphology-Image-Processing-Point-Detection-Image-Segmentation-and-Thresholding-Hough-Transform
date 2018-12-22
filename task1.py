# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread('original_imgs/noise.jpg',0)


Element = [[1, 1, 1],[1, 1, 1],[1, 1, 1]] #Structuring Element

Element = np.asarray(Element,dtype=np.int32)

#Padding the image by 1
ZeroImg = np.zeros((len(img)+1, len(img[0])+1))
for row in range(1, len(ZeroImg)-1):
    for col in range(1, len(ZeroImg[0])-1):
        ZeroImg[row][col] = img[row-1][col-1]
ZeroImg = np.asarray(ZeroImg, dtype=np.uint8)

img01 = ZeroImg/255
img01 = np.asarray(img01)
noofrows, noofcols = img.shape

#Dilation() is used to perform the dilation operation on the image. In a single iteration, we take a 3x3 collection of pixels from the image & compare its elements with the Structuring Element.
#If even one of the pixels matches with any element in the Structuring ELement, we change the value of the element of the origin to 1.

def Dilation(image, matrix):
    x,y = matrix.shape
    flag = 0
    DilaImg = image.copy()
    for row in range(0,noofrows-3):
        for col in range(0,noofcols-3):
            Seg = image[row:row+3, col:col+3]
            for i in range(0,x):
                for j in range(0,y):
                    if Seg[i][j] == matrix[i][j]:
                        flag = 1
                        break
                    else:
                        flag = 0
                if(flag==1):
                    break
            if(flag==1):
                DilaImg[row+1][col+1] = 1
        flag = 0
    print("Dilation Operation Completed")
    return DilaImg

#Erosion() is used to perform the erosion operation on the image. In a single iteration, we take a 3x3 collection of pixels from the image & compare its elements with the Structuring Element.
#If even one of the pixels matches with any element in the Structuring ELement, we keep the value of the element of the origin as 0.

def Erosion(image, matrix):
    x,y = matrix.shape
    flag = 1
    EroImg = np.zeros_like(image)
    for row in range(0,noofrows-3):
        for col in range(0,noofcols-3):
            Seg = image[row:row+3, col:col+3]
            for i in range(0,x):
                for j in range(0,y):
                    if Seg[i][j] != matrix[i][j]:
                        flag = 0
                        break
                    else:
                        flag = 1
                if(flag==0):
                    break
            if(flag==1):
                EroImg[row+1][col+1] = 1
        flag = 0
    print("Erosion Operation Completed")
    return EroImg

# Finding Output Using Algorithgm 1
print("Noise Removal Algorithm 1")
imageTemp = Erosion(img01, Element)
imageTemp = Dilation(imageTemp, Element)
imageTemp = Dilation(imageTemp, Element)
imageTemp = Erosion(imageTemp, Element)

Algo1 = imageTemp * 255
cv2.imwrite("images/res_noise1.jpg", Algo1)

# Finding Output Using Algorithgm 2
print("Noise Removal Algorithm 2")
imageTemp1 = Dilation(img01, Element)
imageTemp1 = Erosion(imageTemp1, Element)
imageTemp1 = Erosion(imageTemp1, Element)
imageTemp1 = Dilation(imageTemp1, Element)

Algo2 = imageTemp1 * 255
cv2.imwrite("images/res_noise2.jpg", Algo2)

# Finding Boundaries For Both The Algrithms
print("Finding Boundaries")
imageTemp2 = Erosion(imageTemp, Element)
BoundingImg1 = imageTemp - imageTemp2
BoundingImg1 = BoundingImg1 * 255
cv2.imwrite("images/res_bound1.jpg", BoundingImg1)

imageTemp2 = Erosion(imageTemp1, Element)
BoundingImg2 = imageTemp1 - imageTemp2
BoundingImg2 = BoundingImg2 * 255
cv2.imwrite("images/res_bound2.jpg", BoundingImg2)

