import os
from aiohttp import Fingerprint
import cv2

sample = cv2.imread("Altered/Altered-Hard/1_M_Left_index_finger_Elliot_Obl.bmp")
# sample = cv2.imread("Real/1_M_Left_index_finger_Elliot.bmp")


best_score = counter = 0
filename = image = kp1 = kp2 = mp = None
for file in os.listdir("Real")[1000:1200]:
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter += 1

    fingerprint_img = cv2.imread(
        "Real/" + file
    )
    sift = cv2.SIFT_create()
    keypoints_1, des1 = sift.detectAndCompute(sample, None)
    keypoints_2, des2 = sift.detectAndCompute(fingerprint_img, None)
    # print(kp1, des1)
    # print(kp2, des2)
    # print(len(kp1), len(kp2))
    # print(len(des1), len(des2))
    # print(des1[0], des2[0])\
    # fast library for approx best match KNN
    matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 10}, {}).knnMatch(
        des1, des2, k=2
    )

    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = 0
    if len(keypoints_1) <= len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)
    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_img
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points

print("Best match:  " + str(filename))
print("Best score:  " + str(best_score))

if len(match_points) > 0:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    result = cv2.resize(result, None, fx=5, fy=5)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
