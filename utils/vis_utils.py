import cv2
import numpy as np
from pathlib import Path

from typing import List, Union, Optional


def load_img(path: str, size: int) -> np.ndarray:
    """Load and center crop resize image

    Args:
        path (str): Path to image file
        size (int): Size to center crop and resize

    Returns:
        np.ndarray: Image array
    """

    assert Path(path).is_file(), path
    return center_crop(cv2.imread(str(path)), size)


def img_grid(images: List[Union[str, List[str]]], titles: List[str], size: int=512, num_cols: Optional[int]=None, font_scale: float=0.6, font_color=(255, 255, 255)) -> np.ndarray:
    """Creates a single image grid from images with text overlay (optional)

    Args:
        images (List[Union[str, List[str]]]): 1D or 2D list of image file paths. If 1D num_cols must be specified else num_cols must be None.
        titles (List[str]): Text to draw over each image
        size (int, optional): Size to crop and resize images to. Defaults to 512.
        num_cols (Optional[int], optional): Number of columns. Only pass if images is a 1D array. Defaults to None.
        font_scale (float, optional): Font scale. Defaults to 0.6.
        font_color (float, optional): Font color. Defaults to (255, 255, 255).

    Returns:
        np.ndarray: image grid
    """

    if num_cols is not None:
        # 1D -> 2D array
        num_rows = int(np.ceil(len(images) / num_cols))

        # add pad images
        num_cells = num_cols * num_rows
        if len(images) < num_cells:
            pad = num_cells - len(images)
            images = images + [np.zeros((size, size, 3), dtype=np.uint8)] * pad
            titles = titles + [""] * pad

        images = [images[r*num_cols:(r+1)*num_cols] for r in range(num_rows)]
        titles = [titles[r*num_cols:(r+1)*num_cols] for r in range(num_rows)]

    rows = []
    for row, row_titles in zip(images, titles):
        # load images for the current row
        row = [load_img(image, size=size) if isinstance(image, (str, Path)) else image for image in row]

        for image, title in zip(row, row_titles):
            # draw text over image
            lines = title.split("\n")
            y = 25
            for line in lines:
                cv2.putText(image, line, (0, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 2)
                y += 20

        # concat result
        row = np.concatenate(row, axis=1)
        rows.append(row)
    rows = np.concatenate(rows, axis=0)
    return rows


def center_crop(image: np.ndarray, size: int) -> np.ndarray:
    """Center crop image to given size (squared)

    Args:
        image (np.ndarray): Image to crop
        size (int): Size for cropping

    Returns:
        np.ndarray: Cropped image array
    """

    h, w = image.shape[:2]

    # find longer side and crop to square
    if w > h:
        x1 = (w - h) // 2
        x2 = w - h - x1

        if x2 > 0:
            image = image[:, x1:-x2]
    else:
        y1 = (h - w) // 2
        y2 = h - w - y1

        if y2 > 0:
            image = image[y1:-y2]
    
    # resize to target size
    image = cv2.resize(image, (size, size))
    return image