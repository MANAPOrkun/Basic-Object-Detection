import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('04-Contour-Detection/contour.png',0)

# findContours

# function will return back contours in an image, and based on the RETR method called, you can get back external, internal, or both:

# cv2.RETR_EXTERNAL:Only extracts external contours
# cv2.RETR_CCOMP: Extracts both internal and external contours organized in a two-level hierarchy
# cv2.RETR_TREE: Extracts both internal and external contours organized in a tree graph
# cv2.RETR_LIST: Extracts all contours without any internal/external relationship

contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Create empty array to hold contours
empty_array = np.zeros(img.shape)


# Draw External Contours
def external(empty_array, contours, hierarchy):

    external_contours = empty_array

    # For every entry in contours
    for i in range(len(contours)):

        # last column in the array is -1 if an external contour (no contours inside of it)
        if hierarchy[0][i][3] == -1:

            # We can now draw the external contours from the list of contours
            cv2.drawContours(external_contours, contours, i, 255, -1)

    return external_contours


# Draw Internal Contours
def internal(empty_array, contours, hierarchy):

    image_internal = np.zeros(img.shape)

    # Iterate through list of contour arrays
    for i in range(len(contours)):
        # If third column value is NOT equal to -1 than its internal
        if hierarchy[0][i][3] != -1:

            # Draw the Contour
            cv2.drawContours(image_internal, contours, i, 255, -1)

    return image_internal



plt.imshow(internal(empty_array, contours, hierarchy),cmap='gray')

plt.show()
