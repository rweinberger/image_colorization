import cv2
import matplotlib.pyplot as plt
from math import floor, log
import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from skimage.color import rgb2yiq, yiq2rgb
import time

import sys
np.set_printoptions(threshold=sys.maxsize, precision=1, linewidth=400)

DISPLAY = False
WINDOW_SZ = 3 # must be odd to be centered around pixel
VARIANCE_MULTIPLIER = 0.6
MIN_SIGMA = 0.000002
PRECISION = 0.01

IN_DIR = "./in/"
OUT_DIR = "./out/"

def infile(fn):
    return IN_DIR + fn + ".bmp"

def outfile(fn):
    return OUT_DIR + fn + ".png"

def preprocess(original, marked):
    """
    Preprocesses input images: the original grayscale image, and the
    version marked with colored scribbles. Converts images to YUV space
    and trims based on (???).
    """
    # read in images from files
    grayscale = cv2.imread(original)
    if grayscale is None:
        raise Exception(f"Could not read from image file '{original}'")

    marked = cv2.imread(marked)
    if marked is None:
        raise Exception(f"Could not read from image file '{marked}'")

    marked = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)

    # scale to float
    grayscale = grayscale / 255.
    marked = marked / 255.

    # isolate colored markings
    marks = np.sum(np.abs(grayscale - marked), axis=2) > PRECISION
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
    img_indices = np.arange(im_sz).reshape((h, w), order= "F")

    absolute_idx = 0 # len
    pixel_idx = 0 # consts_len
    row_indices = np.zeros((full_len, 1))
    col_indices = np.zeros((full_len, 1))
    vals = np.zeros((full_len, 1))

    for c in range(w):
        for r in range(h):
            gvals = np.zeros((1, window_len))
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

                inclusive_gvals = gvals[:, 0 : neighbor_idx + 1]
                noninclusive_gvals = gvals[:, 0 : neighbor_idx]

                variance = np.mean((inclusive_gvals - np.mean(inclusive_gvals)) ** 2)
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
            absolute_idx += 1

    vals = (vals[0: absolute_idx].T)[0]
    col_indices = (col_indices[0 : absolute_idx].T)[0]
    row_indices = (row_indices[0 : absolute_idx].T)[0]

    marked_indices = np.nonzero(marks)
    A = csr_matrix((vals, (row_indices, col_indices)), shape=(pixel_idx, im_sz))

    result = np.copy(im)
    for i in range(1, 3):
        current_slice = im[:, :, i]
        b = np.zeros((h, w))
        b[marked_indices] = current_slice[marked_indices]
        b = b.reshape((A.shape[0]), order="F")
        solution = lsqr(A, b)[0]
        result[:, :, i] = solution.reshape((h, w), order="F")

    return result

def get_colorized(in_bw, in_marked, out_name):
    start = time.time()
    print(f"Processing {in_bw} + {in_marked} -> {out_name}")

    marks, im = preprocess(in_bw, in_marked)
    result = colorize(marks, im)

    # convert to scaled RGB
    result = yiq2rgb(result)
    result = np.clip(result, 0, 1)

    # write to outfile
    plt.imsave(out_name, result)

    # optionally display colorized result
    if DISPLAY:
        plt.imshow(result)
        plt.show()

    print("runtime in sec: ", (time.time() - start) )

if __name__ == "__main__":
    qualifiers = {"reg", "thin", "thick", "dots"}
    argc = len(sys.argv)
    if argc > 1:
        assert argc == 4, "Must provide image name, qualifier, and run_all boolean"
        image_name, qualifier, run_all = sys.argv[1:]
        assert qualifier in qualifiers, f"Unrecognized qualifier '{qualifier}''"

        in_bw = infile(image_name)
        in_marked = infile(f"{image_name}_{qualifier}_marked")
        out = outfile(f"{image_name}_{qualifier}_colorized")

        get_colorized(in_bw, in_marked, out)

        if run_all == "true":
            for winsz in [5, 7, 9]:
                WINDOW_SZ = winsz
                new_out = outfile(f"{image_name}_{qualifier}_colorized_winsz{winsz}")
                get_colorized(in_bw, in_marked, new_out)
            WINDOW_SZ = 3

            for mult in [0.1, 0.3, 0.9]:
                VARIANCE_MULTIPLIER = mult
                new_out = outfile(f"{image_name}_{qualifier}_colorized_mult{mult}")
                get_colorized(in_bw, in_marked, new_out)
            VARIANCE_MULTIPLIER = 0.6

            for precision in [0.001, 0.1, 1.0]:
                PRECISION = precision
                new_out = outfile(f"{image_name}_{qualifier}_colorized_precision{precision}")
                get_colorized(in_bw, in_marked, new_out)
            PRECISION = 0.01
    else:
        print("Plz supply args")
        exit()

    

