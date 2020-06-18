import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('01-Template-Matching/random_cat.jpg')

# Canny Edge Detection

# Calculate the median pixel value
med_val = np.median(img) 

# Lower bound is either 0 or 70% of the median value, whicever is higher
lower = int(max(0, 0.7* med_val))

# Upper bound is either 255 or 30% above the median value, whichever is lower
upper = int(min(255,1.3 * med_val))

blurred_img = cv2.blur(img,ksize=(5,5))

edges = cv2.Canny(image=blurred_img, threshold1=lower , threshold2=upper+80)

plt.imshow(edges)
plt.show()