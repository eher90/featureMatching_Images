# Import necessary libraries
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt

# Initialize the first two images
names = ['artemis', 'babylon', 'lighthouse', 'masoleum', 'rhodes', 'zeus']

for x in names:
    image1 = cv.imread(f'featureMatching_Images/images/{x}Drawing.jpg', cv.IMREAD_GRAYSCALE) # queryImage
    image2 = cv.imread(f'featureMatching_Images/images/{x}Professional.jpg', cv.IMREAD_GRAYSCALE) # train Image

    # BRUTE FORCE METHOD using ORB
    #-----------------------------------------

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
    #------------------------------------------

    # Used SIFT detector
    sift = cv.SIFT_create()

    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1,None)
    kp2, des2 = sift.detectAndCompute(image2,None)

    # Created BFMatcher object
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Find the good values in the knn Matches
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img3 = cv.drawMatchesKnn(image1,kp1,image2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3),plt.show()
    print(matches)

    # SIFT and FLANN
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    
    img3 = cv.drawMatchesKnn(image1,kp1,image2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()

    # Feature Matching + Homography to find Objects
    #----------------------------------------------------------

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10


    if len(good)> MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = image1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(image2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    
    img3 = cv.drawMatches(image1,kp1,image2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()