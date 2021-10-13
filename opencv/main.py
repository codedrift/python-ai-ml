import cv2
import imutils
import numpy as np

input_image = 'logo.jpg'

print(f'Loading image {input_image}')

image = cv2.imread(input_image)

resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

print(f'Image has ratio {ratio}')

cv2.imshow('Original image',resized)

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale image',gray)


blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('Blurred image',blurred)


# see https://learnopencv.com/opencv-threshold-python-cpp/
threshold = 200
maxValue = 255
thresh = cv2.threshold(blurred, threshold, maxValue, cv2.THRESH_BINARY)[1]
cv2.imshow('threshold image',thresh)


edged=cv2.Canny(thresh,30,200)
cv2.imshow('edged image',edged)

contours,hierarchy=cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

single_contours = resized.copy()

for c in contours:
    x,y,w,h=cv2.boundingRect(c)
    # print(f'got boundingrect {x} {y} {w} {h}')
    cv2.rectangle(single_contours,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Bounding rect',single_contours)



# sort contours by size
sorted_contours=sorted(contours, key=cv2.contourArea, reverse=True)

# pick largest contour
largest_contour=sorted_contours[1]


def drawContour(c):

    x,y,w,h=cv2.boundingRect(c)

    imgref = resized.copy()

    # draw combined contour
    cv2.rectangle(imgref,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('union_contours rect',imgref)



# bigger_contours = [] 

# for c in contours:
#     area = cv2.contourArea(c)
#     print("Contour area", area)
#     # ignore small contours
#     if area > 10:
#         bigger_contours.append(c)

# bigger_contours_img = resized.copy()

# for c in bigger_contours:
#     x,y,w,h=cv2.boundingRect(c)
#     # print(f'got boundingrect {x} {y} {w} {h}')
#     cv2.rectangle(bigger_contours_img,(x,y),(x+w,y+h),(0,0,255),2)
#     cv2.imshow('bigger contours rect',bigger_contours_img)




# rect = union(largest_contour,sorted_contours[2])

# combine contours to a larger one
rect = np.vstack([largest_contour,sorted_contours[2]])

# get bounding rect for combined contours
x,y,w,h=cv2.boundingRect(rect)

union_contours = resized.copy()

# draw combined contour
cv2.rectangle(union_contours,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('union_contours rect',union_contours)

# grouped_contours = resized.copy()

# print(closest_contour)

# cv2.drawContours(grouped_contours,[closest_contour],-1,(0,255,0),3)

# cv2.imshow('grouped_contours rect',grouped_contours)
# cv2.drawContours(final_result,contours,-1,(0,255,0),3)
# cv2.imshow('final_result image',final_result)


cv2.waitKey(0)
#closing all open windows 
cv2.destroyAllWindows() 
