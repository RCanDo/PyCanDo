#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 08:41:26 2026

@author: arek
"""
# %%
from pathlib import Path

from PIL import Image, ExifTags, ImageEnhance, ImageFilter, PpmImagePlugin
from pdfplumber.pdf import PDF
import pdfplumber

from app.health_cert.ocr_correct_zzz import imshow  # , correct_angle, rotate_image
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

help(img.save)
img.save(open(pth.with_suffix(".png"), 'bw'))

# %%
# %%
import cv2
import pytesseract

# Load image
img = cv2.imread(pth.with_suffix(".png"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imshow(gray)

# --- 1. Detect text boxes using Tesseract (no OCR) ---

data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
# raw_text = pytesseract.image_to_string(gray, lang="pol")

type(data)  # dict
len(data)   # 12
data.keys()  # dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num',
             #            'left', 'top', 'width', 'height', 'conf', 'text'])

len(data["level"])  # 408

# %%
boxes = []
h, w = gray.shape

for i in range(len(data["level"])):
    x = data["left"][i]
    y = data["top"][i]
    bw = data["width"][i]
    bh = data["height"][i]
    conf = int(data["conf"][i])

    if conf > 40 and bw > 20 and bh > 10:
        boxes.append((x, y, bw, bh))

# --- 2. Detect horizontal lines ---

# Binarize
bw = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    31, 10
)

# Detect horizontal lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours of lines
contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

[c.shape for c in contours]

line_boxes = [cv2.boundingRect(c) for c in contours]

# --- 3. Intersection test ---

def intersects(box, line):
    bx, by, bw, bh = box
    lx, ly, lw, lh = line

    # Check vertical overlap
    return (
        lx < bx + bw and
        lx + lw > bx and
        ly < by + bh and
        ly + lh > by
    )

# --- 4. Mark striked text ---

result = img.copy()

for box in boxes:
    x, y, bw, bh = box
    is_striked = False

    for line in line_boxes:
        if intersects(box, line):
            is_striked = True
            break

    if is_striked:
        color = (0, 0, 255)   # red = striked
    else:
        color = (0, 255, 0)   # green = normal

    cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 2)

imshow(result)

# Save result
cv2.imwrite("./data/health_cert/orzeczenia/striked-01.png", result)


# %%
boxes0 = []
h, w = gray.shape

for i in range(len(data["level"])):
    x = data["left"][i]
    y = data["top"][i]
    bw = data["width"][i]
    bh = data["height"][i]
    conf = int(data["conf"][i])
    txt = data["text"][i]

    boxes0.append((x, y, bw, bh, conf, txt))

result0 = img.copy()

for box in boxes0:
    x, y, bw, bh, conf, txt = box
    is_striked = False

    for line in line_boxes:
        if intersects(box[:4], line):
            is_striked = True
            break

    if is_striked:
        color = (0, 0, 255)   # red = striked
    else:
        color = (0, 255, 0)   # green = normal

    cv2.rectangle(result0, (x, y), (x + bw, y + bh), color, 2)

imshow(result0)

# %%
