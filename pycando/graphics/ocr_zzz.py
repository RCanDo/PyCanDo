#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 11:11:12 2026

@author: arek
"""
# %%
from pathlib import Path

from PIL import Image, ExifTags, ImageEnhance, ImageFilter, PpmImagePlugin
from pdfplumber.pdf import PDF
import pdfplumber

from app.health_cert.ocr_correct_zzz import imshow, correct_angle, rotate_image, plot_rect, rotate_points
import matplotlib.pyplot as plt
import numpy as np

pth = Path("./data/health_cert/orzeczenia/orzeczenie_lekarskie-01.pdf")
pth.exists()

# %%
file: Image.Image | PDF = pdfplumber.open(pth)
type(file)  # pdfplumber.pdf.PDF
dir(file)

file.pages  # [<Page:1>]

# %%
from pdf2image import convert_from_path

pimg = convert_from_path(pth, first_page=1, last_page=1, dpi=300)
type(pimg)  # list
len(pimg)  # 1
type(pimg[0])  # PIL.PpmImagePlugin.PpmImageFile

img = pimg[0]

isinstance(img, Image.Image)  # True
isinstance(img, PpmImagePlugin.PpmImageFile)  # True


# %%
import cv2

img = np.array(img)
img.shape  # (3508, 2480, 3)
imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray.shape  # (3508, 2480)
imshow(gray, num='gray')

np.histogram(gray)
# (array([ 465763, 440  , 290, 5214  , 7295, 3924  , 7737, 28036  , 116956, 8064185         ]),
#  array([     10,  34.5,  59,   83.5,  108,  132.5,  157,   181.5,    206,     230.5, 255. ]))

# %% Detect skew angle
# Method: Hough transform on text lines

help(cv2.threshold)
# threshold(src, thresh, maxval, type[, dst]) -> retval, dst
# .   @brief Applies a fixed-level threshold to each array element.
# .
# .   The function applies fixed-level thresholding to a multiple-channel array. The function is typically
# .   used to get a bi-level (binary) image out of a grayscale image ( #compare could be also used for
# .   this purpose) or for removing a noise, that is, filtering out pixels with too small or too large
# .   values. There are several types of thresholding supported by the function. They are determined by
# .   type parameter.
# .
# .   Also, the special values #THRESH_OTSU or #THRESH_TRIANGLE may be combined with one of the
# .   above values. In these cases, the function determines the optimal threshold value using the Otsu's
# .   or Triangle algorithm and uses it instead of the specified thresh.
# .
# .   @note Currently, the Otsu's method is implemented only for CV_8UC1 and CV_16UC1 images,
# .   and the Triangle's method is implemented only for CV_8UC1 images.
# .
# .   @param src input array (multiple-channel, CV_8U, CV_16S, CV_16U, CV_32F or CV_64F).
# .   @param dst output array of the same size  and type and the same number of channels as src.
# .   @param thresh threshold value.
# .   @param maxval maximum value to use with the #THRESH_BINARY and #THRESH_BINARY_INV thresholding
# .   types.
# .   @param type thresholding type (see #ThresholdTypes).
# .   @return the computed threshold value if Otsu's or Triangle methods used.
# .
# .   @sa  thresholdWithMask, adaptiveThreshold, findContours, compare, min, max
#
# btw:
# {v: cv2.__dict__[v] for v in dcv if "THRESH_" in v}

# Binarize
bwl = cv2.threshold(gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
type(bwl)  # tuple
len(bwl)   # 2
bwl[0]  # 135.0  # 150
type(bwl[1])  # np.array
bw = bwl[1]
bw.shape  # 3508, 2480

bw[:10, :10]

imshow(bw, num="bw")

# Find coordinates of text pixels
wherewhite = np.where(bw > 0)
type(wherewhite)  # tuple
[v.shape for v in wherewhite]  # [(479545,), (479545,)]  == [(x coords,), (y coords,)] of non black pixels
wherewhite[0][:11]  # array([1027, 1027, 1027, 1027, 1027, 1027, 1027, 1027, 1027, 1027, 1027])
wherewhite[1][:11]  # array([465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475])

coords = np.column_stack(wherewhite)
coords.shape  # (479545, 2)
coords[:11, :]
# array([[1027,  465],
#        [1027,  466],
#        ...
#        [1027,  474],
#        [1027,  475]])
#          y      x

_ = plt.figure('bw')
plt.scatter(x=wherewhite[1], y=wherewhite[0], s=2, c='y')

# Get minimum area rectangle
minarect = cv2.minAreaRect(coords)
# (2264.86083984375, 1299.72998046875), (2464.9990234375, 1721.80078125), -0.3802761137485504)
#  y_center,         x_center,           height,          width         ,  angle in [-90, 0) degrees
# https://stackoverflow.com/questions/57967420/what-is-the-output-of-minarearectcontours
help(cv2.minAreaRect)
# minAreaRect(points) -> retval
# .   @brief Finds a rotated rectangle of the minimum area enclosing the input 2D point set.
# .
# .   The function calculates and returns the minimum-area bounding rectangle (possibly rotated) for a
# .   specified point set. The angle of rotation represents the angle between the line connecting the starting
# .   and ending points (based on the clockwise order with greatest index for the corner with greatest \f$y\f$)
# .   and the horizontal axis. This angle always falls between \f$[-90, 0)\f$ because, if the object
# .   rotates more than a rect angle, the next edge is used to measure the angle. The starting and ending points change
# .   as the object rotates.Developer should keep in mind that the returned RotatedRect can contain negative
# .   indices when data is close to the containing Mat element boundary.
# .
# .   @param points Input vector of 2D points, stored in std::vector\<\> or Mat
angle = minarect[-1]  # -0.3802761137485504

angle = correct_angle(angle)
print("Detected angle:", angle)  # 0.3802761137485504

_ = plt.figure("bw")
plot_rect(*minarect[0][::-1], *minarect[1][::-1], angle, 'r')

# %%  rotate image
rotated = rotate_image(bw, angle)
imshow(rotated, num="rot")
plot_rect(*minarect[0][::-1], *minarect[1][::-1], 0, 'r')

# %%  the same with extended background (black)
bw0 = np.r_[bw, np.zeros((200, 2480))]
imshow(bw0, num="bw0")
wherewhite0 = np.where(bw0 > 0)
coords0 = np.column_stack(wherewhite0)
minarect0 = cv2.minAreaRect(coords0)
angle0 = minarect0[-1]  # -0.3802761137485504  # the same!!!

angle0 = correct_angle(angle0)
_ = plt.figure("bw0")
plot_rect(*minarect[0][::-1], *minarect[1][::-1], angle, 'r')

rotated0 = rotate_image(bw0, angle0)
rotated0[:11, :11]

imshow(rotated0, num='rot0')
plot_rect(*minarect0[0][::-1], *minarect0[1][::-1], 0, 'r')

# %%
"""
Hence it works rather poorly :(

maybe try this
https://plainenglish.io/blog/how-to-extract-texts-from-rotated-skewed-text-images-using-craft-opencv-and-pytesseract
"""
