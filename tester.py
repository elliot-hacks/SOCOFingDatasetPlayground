import os
import cv2

# Load and resize the sample fingerprint image
sample = cv2.imread("Altered/Altered-Easy/1__M_Left_index_finger_CR.BMP")
sample = cv2.resize(sample, None, fx=2.5, fy=2.5)

# Initialize variables for finding the best match
best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

counter = 0

# Loop through fingerprint images in the "Real" directory
for file in [file for file in os.listdir("Real")][:1000]:
    if counter % 10 == 0:
        print(counter)
        print(file)
    counter += 1
    fingerprint_image = cv2.imread("Real/" + file)
    sift = cv2.SIFT_create()

    keypoint_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoint_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    if descriptors_1 is None or descriptors_2 is None:
        continue  # Skip if no descriptors are found

    # Create the FLANN matcher object
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {})
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = min(len(keypoint_1), len(keypoint_2))

    if len(match_points) / keypoints * 100 > best_score:
        best_score = len(match_points) / keypoints * 100
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoint_1, keypoint_2, match_points

print("BEST MATCH: " + str(filename))
print("SCORE: " + str(best_score))

# Draw the matches on the result image
result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
