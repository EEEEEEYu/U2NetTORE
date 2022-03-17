import sys
import os
import math
from pathlib import Path


def RenameImgSeq(path: str):
    path = path.replace("\"", "").replace("'", "")

    files = os.listdir(path)

    valid_files = []

    valid_ext = [".jpg", ".jpeg", ".png"]

    ext = ""

    max_file_name = ""
    for file in files:
        if os.path.isdir(path+"\\"+file):
            continue
        if not ext:
            if os.path.splitext(path+"\\"+file)[1] in valid_ext:
                ext = os.path.splitext(path+"\\"+file)[1]

        if os.path.splitext(path+"\\"+file)[1] == ext:
            valid_files.append(file)
            if len(file) > len(max_file_name):
                max_file_name = file
            elif (file > max_file_name and len(file) == len(max_file_name)):
                max_file_name = file

    ext_length = len(ext)

    max_num: int = int(max_file_name[:-ext_length])

    digits: int = int(math.log10(max_num))+1

    for file in valid_files:
        new_name = file.zfill(digits+ext_length)
        Path(path+"\\"+file).rename(path+"\\"+new_name)


def main():
    if (len(sys.argv) > 1):
        RenameImgSeq(sys.argv[1])
    else:
        img_sqe_folder_path = str(input("Enter folder path: "))
        RenameImgSeq(img_sqe_folder_path)


if __name__ == "__main__":
    main()
