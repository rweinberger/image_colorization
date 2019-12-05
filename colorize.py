import cv2
import matplotlib.pyplot as plt
from math import floor, log
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from skimage.color import rgb2yiq, yiq2rgb

import sys
np.set_printoptions(threshold=sys.maxsize)

GRAYSCALE_IMG = "tinyexample.bmp"
MARKED_IMG = "tinyexample_marked.bmp"
OUT_IMG = f"marked_{GRAYSCALE_IMG}"
WINDOW_SZ = 3 # must be odd to be centered around pixel
VARIANCE_MULTIPLIER = 0.6
MIN_SIGMA = 0.000002

def preprocess(original, marked):
    """
    Preprocesses input images: the original grayscale image, and the
    version marked with colored scribbles. Converts images to YUV space
    and trims based on (???).
    """
    # read in images from files
    grayscale = cv2.imread(original) / 255.
    marked = cv2.imread(marked)
    marked = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    marked = marked / 255.

    # isolate colored markings
    marks = np.sum(np.abs(grayscale - marked), axis=2) > 0.01
    marks = marks.astype('double')

    # convert to YIQ
    gray_ntsc = rgb2yiq(grayscale)
    marked_ntsc = rgb2yiq(marked)

    # create image to be colorized
    h, w, c = gray_ntsc.shape
    im = np.empty((h, w, c))
    im[:,:,0] = gray_ntsc[:,:,0]
    im[:,:,1] = marked_ntsc[:,:,1]
    im[:,:,2] = marked_ntsc[:,:,2]

    # ###########################################
    # # not sure we need this section tbh
    # # wtf is it doing
    # # some sort of trimming of the images

    # max_d = floor(log(min(h, w)) / log(2) - 2)
    # iu = floor(h/(2^(max_d-1)))*(2^(max_d-1))
    # ju = floor(w/(2^(max_d-1)))*(2^(max_d-1))

    # # is it ok that the marks array doesn't have a 3rd axis?
    # # (Looks like it is)
    # marks = marks[0:iu, 0:ju]
    # im = im[0:iu, 0:ju, :]
    # ###########################################

    return (marks, im)

def get_neighbors(r, c, h, w):
    """
    Returns the set of valid neighbor indices for the pixel at (r, c). 
    Indices must be:
        - within WINDOW_SIZE of (r, c)
        - within the image bounds of (h, w)
    """
    assert r < h and r >= 0, f"get_neighbors: row {r} is out of bounds for height {h}"
    assert c < w and c >= 0, f"get_neighbors: col {c} is out of bounds for width {w}"

    half_window = WINDOW_SZ // 2

    # will be inclusive on lower bound
    min_r = max(0, r - half_window) 
    min_c = max(0, c - half_window)

    # will be exclusive on upper bound
    max_r = min(r + half_window + 1, h)
    max_c = min(c + half_window + 1, w)

    neighbors = [(n_r, n_c) for n_r in range(min_r, max_r) for n_c in range(min_c, max_c)]

    # don't count (r, c) itself as a neighbor
    neighbors.remove((r, c))

    return neighbors

def colorize(marks, im):
    """
    Colorizes im based on the colored markings given. Based on the algorithm
    found in this paper: 
    https://www.cs.huji.ac.il/~yweiss/Colorization/colorization-siggraph04.pdf
    """
    h, w, _ = im.shape
    im_sz = h * w
    window_len = WINDOW_SZ ** 2 # WINDOW_SZ x WINDOW_SZ
    full_len = im_sz * window_len # 1 window per pixel
    img_indices = np.arange(im_sz).reshape((h, w))

    absolute_idx = 0 # len
    pixel_idx = 0 # consts_len
    row_indices = np.zeros((full_len, 1))
    col_indices = np.zeros((full_len, 1))
    vals = np.zeros((full_len, 1))
    gvals = np.zeros((1, window_len))

    for r in range(h):
        for c in range(w):
            # if this pixel has not been colored by a scribble
            if not marks[r, c]:
                neighbor_idx = 0
                neighbors = get_neighbors(r, c, h, w)

                for (n_r, n_c) in neighbors:
                    row_indices[absolute_idx] = pixel_idx
                    col_indices[absolute_idx] = img_indices[n_r, n_c]
                    gvals[0, neighbor_idx] = im[n_r, n_c, 0]
                    absolute_idx += 1
                    neighbor_idx += 1

                current_pixel_val = im[r, c, 0]
                gvals[0, neighbor_idx] = current_pixel_val
                noninclusive_gvals = gvals[:, 0 : neighbor_idx]
                variance = np.mean((gvals[:, 0 : neighbor_idx + 1] - np.mean(gvals[:, 0 : neighbor_idx + 1])) ** 2)
                sigma = variance * VARIANCE_MULTIPLIER
                min_gvariance = np.min((noninclusive_gvals - current_pixel_val) ** 2)

                if sigma < -min_gvariance / log(0.01):
                    sigma = -min_gvariance / log(0.01)
                if sigma < MIN_SIGMA:
                    sigma = MIN_SIGMA

                something = np.exp(-(noninclusive_gvals - current_pixel_val) ** 2 / sigma)
                something = something / np.sum(something)
                vals[absolute_idx - neighbor_idx : absolute_idx] = - (something.T)

            row_indices[absolute_idx] = pixel_idx
            col_indices[absolute_idx] = img_indices[r, c]
            vals[absolute_idx] = 1
            pixel_idx += 1

    vals = (vals[0: absolute_idx + 1].T)[0]
    col_indices = (col_indices[0 : absolute_idx + 1].T)[0]
    row_indices = (row_indices[0 : absolute_idx + 1].T)[0]

    marked_indices = np.nonzero(marks)
    A = csr_matrix((vals, (row_indices, col_indices)), shape=(pixel_idx, im_sz))

    result = np.copy(im)
    for i in range(1, 3):
        current_slice = im[:, :, i]
        b = np.zeros((h, w))
        b[marked_indices] = current_slice[marked_indices]
        b = b.reshape((A.shape[0]))
        solution = lsqr(A, b)[0]
        result[:, :, i] = solution.reshape((h, w))

    return result

if __name__ == "__main__":
    marks, im = preprocess(GRAYSCALE_IMG, MARKED_IMG)
    result = colorize(marks, im)
    result = yiq2rgb(result)
    plt.imshow(result)
    plt.show()

