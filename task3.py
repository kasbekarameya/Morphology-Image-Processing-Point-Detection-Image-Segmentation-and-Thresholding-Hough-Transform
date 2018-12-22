'''
Reference used: https://gist.github.com/rishabhsixfeet/45cb32dd5c1485e273ab81468e531f09

Reference used: https://alyssaq.github.io/2014/understanding-hough-transform/
'''

import cv2
import numpy as np


# GetSobelFilter() is used to find out the edges of the image by applying a Sobel Filter over the image.
#Here along with the application of Sobel Filter, we have also applied Thresholding to accurately segment the coins & lines in the images.
def GetSobelFilter(img):
    sobely = np.asarray([(-1,0,1),
                         (-2,0,2),
                         (-1,0,1)])

    sobelx = np.asarray([(-1,-2,-1),
                         (0,0,0),
                         (1,2,1)])


    l, h = sobely.shape
    for i in range(0,l):
       for j in range(0,h-1):
           temp = sobely[i][j]
           sobely[i][j] = sobely[i][h-j-1]
           sobely[i][h-j-1] = temp

    for i in range(0,l-1):
        for j in range(0,h):
            temp = sobely[i][j]
            sobely[i][j] = sobely[l-i-1][j]
            sobely[l-i-1][j] = temp

    l, h = sobelx.shape
    for i in range(0,l):
        for j in range(0,h-1):
            temp = sobelx[i][j]
            sobelx[i][j] = sobelx[i][h-j-1]
            sobelx[i][h-j-1] = temp

    for i in range(0,l-1):
        for j in range(0,h):
            temp = sobelx[i][j]
            sobelx[i][j] = sobelx[l-i-1][j]
            sobelx[l-i-1][j] = temp

    Gx = img
    Gx = np.asarray([[0 for j in i] for i in Gx])
    Gy = img
    Gy = np.asarray([[0 for j in i] for i in Gy])
    pstimg = img
    pstimg = np.asarray([[0 for j in i] for i in pstimg])

    size = pstimg.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (sobelx[0][0]*img[i - 1][j - 1] + sobelx[0][1]*img[i][j - 1] + sobelx[0][2]*img[i + 1][j - 1] + sobelx[2][0]*img[i - 1][j + 1] + sobelx[2][1]*img[i][j + 1] + sobelx[2][2]*img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])

            Gx[i-1][j-1]=gx
            Gy[i-1][j-1]=gy
            pstimg[i][j] = min(255, (gx*gx + gy*gy)**0.5)

    Threshold = 100
    for row in range(0, len(Gx)):
        for col in range(0, len(Gx[0])):
            if Gx[row][col] >= Threshold:
                Gx[row][col] = 255
            else:
                Gx[row][col] = 0

    for row in range(0, len(Gy)):
        for col in range(0, len(Gy[0])):
            if Gy[row][col] >= Threshold:
                Gy[row][col] = 255
            else:
                Gy[row][col] = 0

    return Gx, Gy

# GenerateLinenCoinAccumulatorMatrix() is used to generate the Accumulator Matrix representing values based on Voting performed for each point.
def GenerateLinenCoinAccumulatorMatrix(image,dowhat):

    width,height = image.shape
    radius = 20
    len_diagonal = int(np.ceil(np.sqrt(width**2 + height**2)))
    rhos = np.arange(-len_diagonal, len_diagonal)
    thetal = np.deg2rad(np.arange(-90.0, 89.0))
    thetac = np.deg2rad(np.arange(0.0, 360.0))

    if(dowhat == "Line"):
        cos_theta = np.cos(thetal)
        sin_theta = np.sin(thetal)
        len_thetas = len(thetal)
        len_rhos = len(rhos)

        H = np.zeros((len_rhos, len_thetas), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(image)
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]
            for t_idxs in range(len_thetas):
                rho = int(round(x * cos_theta[t_idxs] + y * sin_theta[t_idxs]) + len_diagonal)
                H[rho, t_idxs] += 1

        return H, thetal, rhos

    else:
        cos_theta = np.cos(thetac)
        sin_theta = np.sin(thetac)
        len_thetas = len(thetac)

        H = np.zeros((len_diagonal, len_diagonal), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(image)

        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for t_idxs in range(len_thetas):
                a = int(round(x - radius * cos_theta[t_idxs]) )
                b = int(round(y - radius * sin_theta[t_idxs]) )
                H[a, b] += 1

        return H, thetac

# GetAllPeaks() is used to find out the peak values i.e. the Local Maxima in order to detect the points that are most prominent on the lines & circles to be detected.
def GetAllPeaks(H, num_peaks, dowhat):

    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
    peakVal = np.vstack(np.unravel_index(indices, H.shape)).T

    if(dowhat == "Line"):
        #Finding Peaks for Vertical Lines
        peakVert = peakVal[peakVal[:,1]>87]
        peakVert=  peakVert[peakVert[:,1]<89]
        peakVert = peakVert[:15,:]

        #Finding Peaks for Horizontal Lines
        peakHori = peakVal[peakVal[:,1]<55]
        peakHori = peakHori[peakHori[:,1]>53]
        peakHori = peakHori[peakHori[:,0]>707]
        peakHori = peakHori[:15,:]

        return peakVert, peakHori
    else:
        return peakVal

# GetDrawLinesnCoins() is a simple function used to draw all the lines & circles that match with the peak values in the Accumulator Matrix.
def GetDrawLinesnCoins(img, indicies, rhos, thetas, peaks, imgCoins, flag):
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if(flag==0):
        for i in range(len(peaks)):
            cx = peaks[i][0]
            cy = peaks[i][1]
            cv2.circle(imgCoins, (cx, cy), 20, (0, 255, 0), 1)

# This is the main function used to perform all the operations to implement Hough Transform
def RunIt():
    # Detecting All Lines & Coins

    imgOriginal = cv2.imread('original_imgs/hough.jpg')


    imgCoins = imgOriginal.copy()

    imgBW =  cv2.imread('original_imgs/hough.jpg',0)

    SobelX, SobelY = GetSobelFilter(imgBW)

    accLine, Theta, Rho = GenerateLinenCoinAccumulatorMatrix(SobelX,"Line")

    peakUps, peakSlant = GetAllPeaks(accLine,500,"Line")

    accCoin , ThetaC = GenerateLinenCoinAccumulatorMatrix(SobelY,"Coin")
    peakRound = GetAllPeaks(accCoin,150,"Coin")

    imgVert = imgOriginal.copy()
    GetDrawLinesnCoins(imgVert, peakUps, Rho, Theta, peakRound,imgCoins, 0)

    imgHori = imgOriginal.copy()
    GetDrawLinesnCoins(imgHori, peakSlant, Rho, Theta, peakRound,imgCoins, 1)

    cv2.imwrite("images/red_lines.jpg",imgVert)
    cv2.imwrite("images/blue_lines.jpg",imgHori)
    cv2.imwrite("images/coin.jpg",imgCoins)



if __name__ == '__main__':
    RunIt()