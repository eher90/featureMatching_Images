# BRUTE FORCE METHOD using ORB

# Import necessary libraries
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

# Initialize the first two images
names = ['artemis', 'babylon', 'lighthouse', 'masoleum', 'rhodes', 'zeus']

for x in names:
    image1 = cv.imread(f'featureMatching_Images/images/{x}Drawing.jpg', cv.IMREAD_GRAYSCALE) # queryImage
    image2 = cv.imread(f'featureMatching_Images/images/{x}Professional.jpg', cv.IMREAD_GRAYSCALE) # train Image

    # Added ORB detector (algoritm)
    orb = cv.ORB_create()

    # find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Create BFMatcher object (distance measurement)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in order of distance
    matches = sorted(matches, key=lambda x:x.distance)

    img3 = cv.drawMatches(image1, kp1, image2, kp2, matches[:20], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3), plt.show()
    print(matches)


    # USING SIFT Descriptors
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img3 = cv.drawMatchesKnn(image1,kp1,image2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3),plt.show()