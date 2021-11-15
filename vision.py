import numpy as np
import math
import cv2
from glob import glob
import copy
import random

def Homography(corr, ImageCoord1, ImageCoord2):     #calculates homography
    h = []
    for corr in corr:
        matrix = []
        i = 0
        for i in range(len(corr)):
            x1 = ImageCoord1[i][0]
            y1 = ImageCoord1[i][1]
            x2 = ImageCoord2[i][0]
            y2 = ImageCoord2[i][1]
            row1 = np.array([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
            row2 = np.array([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
            matrix.append(row1)
            matrix.append(row2)
        u, s, v = np.linalg.svd(matrix)
        h = v[8]
        h = np.array(h)
        h = h.reshape((3, 3))
        return h


def ransac(keypoints, matches):           #calculates ransac and calls homography function
    correspondence = []
    for k in matches:
        (x1, y1) = keypoints[0][k.queryIdx].pt
        (x2, y2) = keypoints[1][k.trainIdx].pt
        correspondence.append([x1, y1, x2, y2])

    cor = np.matrix(correspondence)
    maxInliers = []
    inliers = []
    finalHom = None
    for i in range(1500):
        co1 = cor[random.randrange(0, len(cor))]                #choose random numbers
        co2 = cor[random.randrange(0, len(cor))]
        co3 = cor[random.randrange(0, len(cor))]
        co4 = cor[random.randrange(0, len(cor))]
        random1 = np.vstack((co1, co2))
        random2 = np.vstack((co3, co4))
        randoms = np.vstack((random1, random2))
        ImageCoord1 = np.asarray(randoms)[:, 0:2]
        ImageCoord2 = np.asarray(randoms)[:, 2:4]
        h = Homography(cor, ImageCoord1, ImageCoord2)

        for j in range(len(cor)):
            p1 = np.transpose(np.matrix([cor[j][0].item(0), cor[j][0].item(1), 1]))
            p2 = np.transpose(np.matrix([cor[j][0].item(2), cor[j][0].item(3), 1]))
            d = np.dot(h, p1)
            d = (1 / d.item(2)) * d
            dd = p2 - d
            dist = np.linalg.norm(dd)
            if dist < 5:
                inliers.append(cor[j])
            if len(inliers) > len(maxInliers):
                maxInliers = inliers
                finalHom = h
        if len(maxInliers) > (len(cor)*2):
            break
    return finalHom


def features(img):          #finds features
    orb = cv2.ORB_create(nfeatures=1500)
    (keyps, features) = orb.detectAndCompute(img, None)
    img_ = cv2.drawKeypoints(img, keyps,np.array([]), color=(0, 255, 0), flags=0)
    cv2.imshow("keypoint", img_)
    return keyps, features

def resize(image):           # makes the image smaller
    image = cv2.resize(image, (0, 0), None, 0.5, 0.5)
    return image

def makearr(ori,img_src,img):           #makes array
    for i in range(len(ori)):
        img_src.append(ori[i])
        img.append(cv2.imread(img_src[i], 0))
    return img

#TO LOOK FOR DIFFERENT PANORAMAS YOU NEED TO CHANGE HERE!!
ori = glob(".\HW2_Dataset\pano1" + "/*.png")         #this is the addresses of the pano files you may need to change this part

img_src = []
img = []

makearr(ori, img_src, img)

for i in range(len(img)):               #in all of these for loops program finds the wanted

    kp1, des1 = features(img[i])
    kp2, des2 = features(img[i + 1])
    keypoints = [kp1, kp2]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matching_result = cv2.drawMatches(img[i], kp1, img[i + 1], kp2, matches[:50], None, flags=2)
    cv2.waitKey()
    cv2.imshow("Matching result", matching_result)
    H = ransac(keypoints, matches)
    print("Final homography: ", H)
    i = i + 2


cv2.waitKey()
cv2.destroyAllWindows()
