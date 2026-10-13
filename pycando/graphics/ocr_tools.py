#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
title: graphic tools for visualizing OCR preparations via cv2
file:
    - created: 2026-01-19
    - authors:
        - arek
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

from numpy.typing import NDArray
from typing import Literal

def imshow(
        image: np.ndarray,
        title: str = "",
        var: str = None,
        cmap: str | None = "gray",  # "gist_gray",  reversed: "Greys", "binary", "gist_yarg",
        figsize: tuple[int, int] = (4, 5),
        axis: bool = True,
        num: int | str = None,
):
    plt.figure(figsize=figsize, num=num)
    plt.style.use('dark_background')
    if not axis:
        plt.axis("off")
    plt.imshow(image, cmap=cmap)
    if var:
        title = title + "\n" + var
    plt.title(title)
    #plt.suptitle("image_parse-new")
    plt.tight_layout()


def correct_angle(angle: float) -> float:
    # Fix OpenCV angle convention – strange!
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def rotate_image(
        image: np.ndarray,
        angle: float,
        center: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    cv2 coordinates convention, i.e. [y, x] (vertical, horizontal), however
    it only relates to position of an element in the matrix of a picture (np.ndarray)
    e.g. `img[y, x]` means [y, x] element of the matrix (where numeric value of the color is stored);
    e.g. this translates to pixel at [x, y] in plt.imshow() terms
    (where y-axis goes downward: 0 = top, positive values = down, (negative values = up).).
    """
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


Point = NDArray[tuple[Literal[2]], float]


def rotate_points(
        points: NDArray[Point],
        center: Point,
        angle_rad: float,
) -> NDArray[Point]:
    """
    Standard coordinates convention, like e.g. for matplotlib.pyplot (plt), i.e. [x, y] (horizontal, vertical);
    positive values = right & top, negative values = left and bottom.
    """
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    R = np.array([[c, -s],
                  [s,  c]])

    return (points - center) @ R.T + center


def plot_rect(
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        angle: float,  # deg
        color: str,
) -> None:
    """
    Coordinates convention as for matplotlib.pyplot (plt), i.e. [x, y] (horizontal, vertical),
    and for plt.imshow() i.e. y-axis goes downward: 0 = top, positive values = down, (negative values = up).

    Notice that for cv2 convention is [y, x] (vertical, horizontal), however
    it only relates to position of an element in the matrix of a picture (np.ndarray)
    e.g. `img[y, x]` means [y, x] element of the matrix (where numeric value of the color is stored)
    and this translates to pixel at [x, y] in plt.imshow() terms.
    """
    left = x_center - width / 2
    right = x_center + width / 2
    top = y_center - height / 2
    bottom = y_center + height / 2

    center = np.array([x_center, y_center])
    corners = np.array([[left, top], [left, bottom], [right, bottom], [right, top], [left, top]])

    corners = rotate_points(corners, center, np.deg2rad(angle)) if angle else corners

    plt.plot(*corners.T, c=color)
    return corners
