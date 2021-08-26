import argparse
import sys
import torch
from argparse import ArgumentParser
#--output output\o6.jpg --sigma 3.0 --num-peaks 20 --background 255,255,255 --input input\download.jpg
import numpy as np
import cv2
#from skimage import cv2

from skimage.color import rgb2gray
from skimage.transform import rotate

#from deskew import determine_skew
from findskew import *
def main() -> None:
    parser: ArgumentParser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", default=None, help="Output file name")
    parser.add_argument("--sigma", default=3.0, help="The used sigma")
    parser.add_argument("--num-peaks", default=20, help="The used num peaks")
    parser.add_argument("--background", help="The used background color")
    parser.add_argument("--input", default=None, dest="input", help="Input file name")
    options = parser.parse_args()

    image = cv2.imread(options.input)
    #grayscale = rgb2gray(image)
    #angle = determine_skew(grayscale, sigma=options.sigma, num_peaks=options.num_peaks)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    #if options.output is None:
    print(f"Estimated angle: {angle}")
    #else:
    if options.background:
        try:
            background = [int(c) for c in options.background.split(",")]
        except:  # pylint: disable=bare-except
            print("Wrong background color, should be r,g,b")
            sys.exit(1)
        rotated = rotate(image, angle, resize=True, cval=-1) * 255
        pos = np.where(rotated == -255)
        rotated[pos[0], pos[1], :] = background
    else:
        rotated = rotate(image, angle, resize=True) * 255
    cv2.imwrite(options.output, rotated.astype(np.uint8))


if __name__ == "__main__":
    main()
