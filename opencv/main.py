from math import sqrt
from random import randrange

import cv2
import imutils
import numpy as np
from scipy.spatial.distance import cdist


def drawContours(title, cons, img):
    for c in cons:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (randrange(30, 255), randrange(30, 200), randrange(30, 200)),
            2,
        )
        cv2.imshow(title, img)


# def point_distance(x1,y1,x2,y2):
#     dx = x2 - x1
#     dy = y2 -y1

#     return sqrt(dx * dx + dy * dy)


def map_to_bounding_rects(cons):
    rects = []
    for idx, val in enumerate(cons):
        x, y, w, h = cv2.boundingRect(val)
        area = cv2.contourArea(val)
        if area > 0:
            rects.append(
                {
                    "name": f"rect_{idx}",
                    "xywh": (x, y, w, h),
                    "area": cv2.contourArea(val),
                    "contour": val,
                }
            )
    return rects


def is_near(current, target, used, max_distance):
    current_name = current["name"]
    current_rect = current["xywh"]
    current_area = current["area"]
    target_name = target["name"]
    target_rect = target["xywh"]
    target_area = target["area"]
    (tx, ty, tw, th) = target_rect

    if current_name in used:
        print(f"Current {current_name} is already used")
        return False

    if current_name == target_name:
        return False

    (cx, cy, cw, ch) = current_rect

    target_points = [
        [tx, ty],  # bottom left
        [tx + tw, ty + th],  # top right
        [tx + tw, ty],  # bottom right
        [tx, ty + th],  # top left
    ]
    current_points = [
        [cx, cy],
        [cx + cw, cy + ch],
        [cx + cw, cy],
        [cx, cy + ch],
    ]

    # calculates all distances between points
    distances = cdist(target_points, current_points, metric="cityblock")

    # find smallest distance
    min_distance = min(distances.flatten())

    print(f"Current: {current_name} rect={current_rect} area={current_area}")
    print(f"{target_name} ==> {current_name} min_distance: {min_distance}")

    if min_distance < max_distance:
        print(f'{current["name"]} is near')
        return True

    return False


def find_neighbors(rects, targets, used):
    neighbors = []
    target_names = [n["name"] for n in targets]
    print(f"Find neighbors of targets {target_names}")
    for target in targets:
        target_name = target["name"]
        target_rect = target["xywh"]
        target_area = target["area"]

        # if target_name in used:
        #     print(f"Target {target_name} is already used {used} ")
        #     continue

        print("Target", target_name, target_rect, target_area)

        for current in rects:

            is_neighbor = is_near(current, target, used, 5)

            if is_neighbor:
                neighbors.append(current)

    if len(neighbors) > 0:
        neighbors.append(target)

    return neighbors


def group_neighbors(rects):
    used = []
    completed = []

    for rect in rects:
        done = False
        neighbors = [rect]
        print(f'Find neighbors for {rect["name"]}, used={used}')
        while not done:
            # neighbors including self
            curr_neighbors = find_neighbors(rects, neighbors, used)

            if len(curr_neighbors) == 0:
                print(f'Rect {neighbors[0]["name"]} has no neigbors')
                done = True
                # completed.append(neighbors)
                # used.extend([n["name"] for n in neighbors])
            else:
                neighbors.extend(curr_neighbors)
                # save used refs to prevent multiple uses
                used.extend([n["name"] for n in curr_neighbors])

        # if len(neighbors) > 1:
        used.extend([n["name"] for n in neighbors])
        print(f'{rect["name"]} >> Grouped {[n["name"] for n in neighbors]}\n')
        completed.append(neighbors)
        # else:
        # print(f'{rect["name"]} >> No group\n')

        # return completed

    return completed


input_image = "logo.png"

# load image and resize
print(f"Loading image {input_image}")
base_image = cv2.imread(input_image)
resized = imutils.resize(base_image, width=300)
ratio = base_image.shape[0] / float(resized.shape[0])
print(f"Image has ratio {ratio}")
cv2.imshow("Original image", resized)

# remove colors
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale image", gray)

# blur image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blurred image", blurred)

# apply threshold to image to ignore fades etc
# see https://learnopencv.com/opencv-threshold-python-cpp/
threshold = 80
maxValue = 128
thresh = cv2.threshold(blurred, threshold, maxValue, cv2.THRESH_BINARY)[1]
cv2.imshow("Threshold image", thresh)

# detect edges
edged = cv2.Canny(thresh, 30, 200)
cv2.imshow("Edged image", edged)

# find all contours
all_contours, hierarchy = cv2.findContours(
    edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)

single_contours = resized.copy()

drawContours("All countours", all_contours, resized.copy())

# sort contours by size
sorted_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)


# enhance found contours with bounding rects and stuff
rects = map_to_bounding_rects(sorted_contours)

groups = group_neighbors(rects)

print(f"Found {len(groups)} groups")

img = resized.copy()

for idx, group in enumerate(groups):
    combined_contours = np.vstack([n["contour"] for n in group])
    x, y, w, h = cv2.boundingRect(combined_contours)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # drawContours(f'Group {idx}', combined_contours, img)

cv2.imshow("final_result image", img)

# first_rect = rects[0]
# combine contours to a larger one
# neighbors = find_neighbors(rects, first_rect , [])

# print("neighbors", neighbors)

# close_contours = [n["contour"] for n in neighbors]
# close_rects = [n["name"] for n in neighbors]
# print(f'Close rects for {first_rect["name"]} {close_rects}')
# drawContours("close contours", close_contours, resized.copy())

# unified_contours = np.vstack(close_contours)

# get bounding rect for combined contours
# x, y, w, h = cv2.boundingRect(unified_contours)
# unified_contours_img = resized.copy()
# cv2.rectangle(unified_contours_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
# cv2.imshow("union_contours rect", unified_contours_img)

# cv2.imshow('grouped_contours rect',grouped_contours)
# cv2.drawContours(final_result,contours,-1,(0,255,0),3)
# cv2.imshow('final_result image',final_result)


cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()
