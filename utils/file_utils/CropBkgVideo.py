import os
import sys
from xmlrpc.client import Boolean
import ffmpeg

"""
Usage:
python CropBkgVideo input [aspect_ratio(width:height Default: 4:3)] [area_keep(left/middle/right Default:middle)] [output_path(Default: In place with '_cropped extension')]

Changing order not supported

or 

python CropBkgVideo
for interactive process


"""


def check_file_exist(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File: {} does not exist".format(file_path))


def check_video_file(video_path: str, video_name: str = "video") -> dict:
    if not video_path:
        raise Exception("No " + video_name + " provided, exit")
    try:
        ffmpeg.probe(video_path)
    except:
        raise Exception("Fail to read " + video_name + " file: " + video_path)

    video_tracks_info = ffmpeg.probe(video_path)["streams"]
    if len(video_tracks_info) > 1:
        print("More than 1 track in " + video_name + " file, using the first one by default. track info:")
        print(video_tracks_info[0])

    return video_tracks_info[0]


def main():
    src_video_path: str = ""
    aspect_ratio: str = "4:3"
    part_to_keep: str = ""

    if len(sys.argv) == 1:
        src_video_path = str(input("Enter file path: "))
        output_file_path = str(input("Enter output file path: (Default: In place with '_cropped extension')"))
        aspect_ratio = str(input("Enter desired aspect ratio (format: width:height Default: 4:3): "))
        part_to_keep = str(input("Enter the area want to keep (left/middle/right Default:middle)"))

    if len(sys.argv) > 1:
        src_video_path = sys.argv[1]
        if len(sys.argv) > 2:
            aspect_ratio = sys.argv[2]
        if len(sys.argv) > 3:
            part_to_keep = sys.argv[2]
        if len(sys.argv) > 4:
            output_file_path = sys.argv[3]

    src_video_path = src_video_path.replace("\"", "").replace("'", "")

    check_file_exist(src_video_path)

    video_info = check_video_file(src_video_path)

    if not aspect_ratio:
        aspect_ratio = "4:3"
    if not part_to_keep:
        part_to_keep = "middle"
    if not output_file_path:
        seperate_ext_name = os.path.splitext(src_video_path)
        output_file_path = seperate_ext_name[0] + "_cropped" + seperate_ext_name[1]

    try:
        width, height = aspect_ratio.split(':')
        width = int(width)
        height = int(height)
    except:
        exit("Invalid aspect ratio {}. Format: width:height".format(aspect_ratio))

    part_to_keep = part_to_keep.lower()
    if part_to_keep not in ["left", "right", "middle"]:
        print("Invalid keeping area {}. Acceptable Option: left/middle/right".format(part_to_keep))

    src_video_width = video_info['width']
    src_video_height = video_info['height']

    crop_horizontal: Boolean = True

    if src_video_width > (src_video_height / height * width):
        dest_video_width = src_video_height / height * width
        dest_video_height = src_video_height
    else:
        dest_video_width = src_video_width
        dest_video_height = src_video_width / width * height
        crop_horizontal = False

    left_upper_pixel = (0, 0)

    if crop_horizontal:
        if part_to_keep == "middle":
            left_upper_pixel = ((src_video_width - dest_video_width) / 2, 0)
        if part_to_keep == "right":
            left_upper_pixel = (src_video_width - dest_video_width, 0)
    else:
        if part_to_keep == "middle":
            left_upper_pixel = (0, (src_video_height - dest_video_height) / 2)
        if part_to_keep == "right":
            left_upper_pixel = (0, src_video_height - dest_video_height)

    cmd = "ffmpeg -i \"{src_video_path}\" -vf \"crop={dest_video_width}:{dest_video_height}:{crop_x}:{crop_y}\" -c:a copy \"{out_video_path}\"".format(
        src_video_path=src_video_path, dest_video_width=dest_video_width, dest_video_height=dest_video_height,
        crop_x=left_upper_pixel[0], crop_y=left_upper_pixel[1], out_video_path=output_file_path

    )

    os.system(cmd)


if __name__ == "__main__":
    main()
