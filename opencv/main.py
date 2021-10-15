import datetime
import os
from math import sqrt
from random import randrange

import cv2
import imutils
import numpy as np
from PIL import Image
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
    distances = cdist(target_points, current_points, metric="euclidean")

    # find smallest distance
    min_distance = min(distances.flatten())


    # This is hacky but seems to work. check if contour is within the other
    if cx > tx and cy > ty:
        if cx < tx + tw and cy < ty + th:
            return True

    # inverse condition
    if tx > cx and ty > cy:
        if tx < cx + cw and ty < cy + ch:
            return True

    # print(f"Current: {current_name} rect={current_rect} area={current_area}")
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

        print("Target", target_name, target_rect, target_area)

        for current in rects:

            is_neighbor = is_near(current, target, used, MAX_OBJECT_DISTANCE)

            if is_neighbor:
                neighbors.append(current)

    return neighbors


def group_neighbors(rects):
    used = []
    completed = []

    for rect in rects:
        done = False
        neighbors = [rect]

        name = rect["name"]

        if name in used:
            print(f"Target {name} is already used {used} ")
            continue

        print(f'Find neighbors for {rect["name"]}, used={used}')
        while not done:
            # neighbors including self
            curr_neighbors = find_neighbors(rects, neighbors, used)

            if len(curr_neighbors) == 0:
                print(f'Rect {neighbors[0]["name"]} has no neigbors')
                done = True
            else:
                neighbors.extend(curr_neighbors)
                # save used refs to prevent multiple uses
                used.extend([n["name"] for n in curr_neighbors])

        print(f'{rect["name"]} >> Grouped {[n["name"] for n in neighbors]}\n')
        completed.append(neighbors)

    return completed




# load image and resize
def get_bounded_groups(input_image, show_steps):
    print(f"Loading image {input_image}")
    base_image = cv2.imread(input_image)
    resized = imutils.resize(base_image, width=300)
    ratio = base_image.shape[0] / float(resized.shape[0])
    print(f"Image has ratio {ratio}")
    cv2.imshow(f'Original {input_image}', resized)

    # remove colors
    gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
    if show_steps:
        cv2.imshow(f'Grayscale {input_image}', gray)

    # blur image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if show_steps:
        cv2.imshow(f'Blurred {input_image}', blurred)

    # apply threshold to image to ignore fades etc
    # see https://learnopencv.com/opencv-threshold-python-cpp/
    # threshold = 127
    # maxValue = 0
    # thresh = cv2.threshold(blurred, threshold, maxValue, cv2.THRESH_BINARY)[1]
    # cv2.imshow("Threshold image", thresh)

    # detect edges
    edged = cv2.Canny(blurred, 30, 200)
    if show_steps:
        cv2.imshow(f'Edged {input_image}', edged)

    # find all contours
    all_contours, hierarchy = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if show_steps:
        drawContours(f'All contours {input_image}', all_contours, resized.copy())

    # sort contours by size
    sorted_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

    # enhance found contours with bounding rects and stuff
    rects = map_to_bounding_rects(sorted_contours)

    print(f'Found rects {[n["name"] for n in rects]}\n')

    groups = group_neighbors(rects)

    print(f"Found {len(groups)} groups")

    img = resized.copy()

    for idx, group in enumerate(groups):
        combined_contours = np.vstack([n["contour"] for n in group])
        x, y, w, h = cv2.boundingRect(combined_contours)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow(f'Result {input_image}', img)

    return groups


# this is the key component. it defines the maximum distance
MAX_OBJECT_DISTANCE = 12

filelist=os.listdir('input')
images = [f'input/{i}' for i in filelist]
# convert all png to jpg with gray bg
for image in images:
    if image.endswith(".png"):
        im = Image.open(image).convert("RGBA")
        new_image = Image.new("RGBA", im.size, "#e0e0e0")
        new_image.paste(im, mask=im)   
        rgb_im = new_image.convert('RGB')
        rgb_im.save(image.replace(".png",".jpg"))


filelist=os.listdir('input')
images = [f'input/{i}' for i in filelist]

# only handle jpg
for image in filter(lambda i: i.endswith(".jpg"), images):
    print(f'Process {image}')
    a = datetime.datetime.now()
    get_bounded_groups(image, False)
    b = datetime.datetime.now()
    c = b - a
    print("Finding groups took",c.microseconds / 1000, "ms")

cv2.waitKey(0)
# closing all open windows
cv2.destroyAllWindows()
