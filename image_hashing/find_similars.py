import argparse
import os
import sys
import time

import cv2
from imutils import paths


def dhash(image, hashSize=8):
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hammingDistance(n1, n2) : 
  
    x = n1 ^ n2  
    setBits = 0
  
    while (x > 0) : 
        setBits += x & 1
        x >>= 1
      
    return setBits  

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--haystack", required=True,
                help="dataset of images to search through (i.e., the haytack)")
ap.add_argument("-n", "--needles", required=True,
                help="set of images we are searching for (i.e., needles)")
args = vars(ap.parse_args())


# grab the paths to both the haystack and needle images
print("[INFO] computing hashes for haystack...")
haystackPaths = list(paths.list_images(args["haystack"]))
needlePaths = list(paths.list_images(args["needles"]))
# remove the `\` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
    haystackPaths = [p.replace("\\", "") for p in haystackPaths]
    needlePaths = [p.replace("\\", "") for p in needlePaths]

# grab the base subdirectories for the needle paths, initialize the
# dictionary that will map the image hash to corresponding image,
# hashes, then start the timer
BASE_PATHS = set([p.split(os.path.sep)[-2] for p in needlePaths])
haystack = {}
needles = {}
start = time.time()


# loop over the haystack paths
for p in haystackPaths:
    # load the image from disk
    image = cv2.imread(p)
    # if the image is None then we could not load it from disk (so
    # skip it)
    if image is None:
        continue
    # convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(image)
    print(f'Haystack {p} => {imageHash}')
    # print(f'Hash is {imageHash}')
    # update the haystack dictionary
    l = haystack.get(imageHash, [])
    l.append(p)
    haystack[imageHash] = l

# show timing for hashing haystack images, then start computing the
# hashes for needle images
print("[INFO] processed {} images in {:.2f} seconds".format(
    len(haystack), time.time() - start))
print("[INFO] computing hashes for needles...")

# loop over the needle paths
for p in needlePaths:
    print(f'Examine needle image {p}')
    # load the image from disk
    image = cv2.imread(p)
    # if the image is None then we could not load it from disk (so
    # skip it)
    if image is None:
        continue
    # convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(image)
    print(f'Hash is {imageHash}')

    l = needles.get(imageHash, [])
    l.append(p)
    needles[imageHash] = l

    # grab all image paths that match the hash
    matchedPaths = haystack.get(imageHash, [])
    print(f'matching {matchedPaths}')
    # loop over all matched paths
    for matchedPath in matchedPaths:
        # extract the subdirectory from the image path
        b = p.split(os.path.sep)[-2]
        # if the subdirectory exists in the base path for the needle
        # images, remove it
        if b in BASE_PATHS:
            BASE_PATHS.remove(b)


for n in needles:
    for h in haystack:
        distance = hammingDistance(n, h)
        if distance <= 10 and distance >= 0:
            print(f'needle distance {n} {h} => {distance}')
            print(f'needle {needles[n]} matches {haystack[h]}')
