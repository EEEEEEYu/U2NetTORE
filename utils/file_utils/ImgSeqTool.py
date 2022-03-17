from optparse import OptionParser
import sys
import os
import glob
import cv2
import numpy as np
import math
import json
from tqdm import tqdm

ext: str = ""
"""
Assumption:
image sequence started with 0 and has sequential number file name

"""


def parseArgs():
    parser = OptionParser()

    parser.add_option("-i", "--input", type="string")
    parser.add_option("-o", "--output", type="string")
    parser.add_option("--remove_padding", default=0, type="int")

    parser.add_option("--batch_size", type="int", default=3000)
    parser.add_option("--fps", type="int", default=300)

    parser.add_option("-c", "--cleanup", action="store_true")
    (options, args) = parser.parse_args()

    return options


def sortByNumericalName(file: str):
    global ext
    return int(os.path.basename(file)[: -len(ext)])


def main():
    args = parseArgs()

    input_dir = args.input
    if not input_dir:
        raise Exception("No Input Provided")

    if not os.path.isdir(input_dir):
        raise Exception("Path: '{}' doesn't exist".format(input_dir))

    if args.remove_padding < 0:
        raise ValueError("Padding must be non-negative")

    output_dir = args.output
    if not output_dir:
        output_dir = input_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    file_count: int = 0
    valid_extensions = [".png", ".jpg", ".jpeg"]
    global ext

    for valid_extension in valid_extensions:
        file_count = len(glob.glob(os.path.join(input_dir, "*" + valid_extension)))

        if file_count != 0:
            ext = valid_extension
        break
    if file_count == 0:
        print("No image sequence found")
        quit()

    # sorting is too slow for such many files
    # files.sort(key=sortByNumericalName)

    # check if the file name is padded
    # e.g. 0001.png or 1.png

    padded_name: bool = not os.path.isfile(os.path.join(input_dir, "0" + ext))
    max_filename_length: int = int(math.log10(file_count - 1)) + 1

    batch_size: int = args.batch_size
    remove_padding: int = args.remove_padding

    print("Packing {} {} files into {} npz files".format(file_count, ext, math.ceil(file_count/args.batch_size)))
    # prepare a json to store the meta info

    meta_info = {
        "frame_count": file_count,
        "fps": args.fps,
        "batch_size": batch_size,
        "remove_padding": remove_padding,
        "file_list": {},
    }

    acc_data = []

    for frame_idx in tqdm(range(file_count), unit='fr'):

        file = "{}" + ext
        if padded_name:
            file = file.format(str(frame_idx).zfill(max_filename_length))
        else:
            file = file.format(str(frame_idx))

        file_path = os.path.join(input_dir, file)
        frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if remove_padding:
            frame = frame[
                    remove_padding:-remove_padding, remove_padding:-remove_padding
                    ]

        acc_data.append(frame)

        if args.cleanup:
            os.remove(file_path)

        if (frame_idx + 1) % batch_size == 0 or frame_idx >= file_count - 1:
            batch_idx: int = int(frame_idx / batch_size)
            meta_info["file_list"][str(batch_idx)] = str(batch_idx) + ".npz"

            np.savez_compressed(os.path.join(output_dir, str(batch_idx)), acc_data)
            acc_data = []

    with open(os.path.join(output_dir, "meta.json"), "w") as outfile:
        json.dump(meta_info, outfile)


if __name__ == "__main__":
    main()
