import cv2
import numpy as np
import matplotlib.pyplot as plt

product = cv2.imread('06-Feature-Matching/product.png',0)     
products = cv2.imread('06-Feature-Matching/products.jpg',0)

def display(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    plt.show()

# Brute Force Detection with ORB Descriptors
    
def orb(product, products):

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(product,None)
    kp2, des2 = orb.detectAndCompute(products,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 25 matches.
    product_matches = cv2.drawMatches(product,kp1,products,kp2,matches[:25],None,flags=2)

    return product_matches


# Brute-Force Matching with SIFT Descriptors and Ratio Test
def sift(product, products):
    # Create SIFT Object
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(product,None)
    kp2, des2 = sift.detectAndCompute(products,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for match1,match2 in matches:
        if match1.distance < 0.75*match2.distance:
            good.append([match1])

    # cv2.drawMatchesKnn expects list of lists as matches.
    sift_matches = cv2.drawMatchesKnn(product,kp1,products,kp2,good,None,flags=2)

    return sift_matches 

def flann(product, products):

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(product,None)
    kp2, des2 = sift.detectAndCompute(products,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)  

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []

    # ratio test
    for i,(match1,match2) in enumerate(matches):
        if match1.distance < 0.7*match2.distance:
        
            good.append([match1])


    flann_matches = cv2.drawMatchesKnn(product,kp1,products,kp2,good,None,flags=0)

    return flann_matches

display(flann(product, products))

