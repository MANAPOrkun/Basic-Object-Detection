import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('02-Corner-Detection/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)

gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

real_chess = cv2.imread('02-Corner-Detection/real_chessboard.jpeg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)

gray_real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)


# Harris Corner Detection

def harris(gray_img, img):
    # cornerHarris Function
    #
    # src Input single-channel 8-bit or floating-point image.
    # dst Image to store the Harris detector responses. It has the type CV_32FC1 and the same size as src .
    # blockSize Neighborhood size (see the details on #cornerEigenValsAndVecs ).
    # ksize Aperture parameter for the Sobel operator.
    # k Harris detector free parameter. See the formula in DocString
    # borderType Pixel extrapolation method. See #BorderTypes.

    # Convert Gray Scale Image to Float Values
    gray = np.float32(gray_img)

    # Corner Harris Detection

    # img - Input image, it should be grayscale and float32 type.
    # blockSize - It is the size of neighbourhood considered for corner detection
    # ksize - Aperture parameter of Sobel derivative used.
    # k - Harris detector free parameter in the equation.

    dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

    # result is dilated for marking the corners, not important to actual corner detection
    # this is just so we can plot out the points on the image shown
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [255, 0, 0]

    plt.imshow(img)


# Shi-Tomasi Corner Detector & Good Features to Track Paper

def shi_tomasi(gray_img, img, dots):
    corners = cv2.goodFeaturesToTrack(gray_img, dots, 0.1, 30)
    corners = np.int0(corners)

    for i in range(len(corners)):
        x, y = corners[i][0]
        cv2.circle(img, (x, y), 10, 255, -1)

    plt.imshow(img)



shi_tomasi(gray_real_chess, real_chess, 45)
# shi_tomasi(gray_flat_chess, flat_chess, 45)

plt.show()
