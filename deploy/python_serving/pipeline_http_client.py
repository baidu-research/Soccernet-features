import argparse
import base64
import json
import os
import os.path as osp

import cv2
import numpy as np
import requests


def numpy_to_base64(array: np.ndarray) -> str:
    """numpy_to_base64

    Args:
        array (np.ndarray): input ndarray.

    Returns:
        bytes object: encoded str.
    """
    return base64.b64encode(array).decode('utf8')


def video_to_numpy(file_path: str) -> np.ndarray:
    """decode video with cv2 and return stacked frames
       as numpy.

    Args:
        file_path (str): video file path.

    Returns:
        np.ndarray: [T,H,W,C] in uint8.
    """
    cap = cv2.VideoCapture(file_path)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_frames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret is False:
            continue
        img = frame[:, :, ::-1]
        decoded_frames.append(img)
    decoded_frames = np.stack(decoded_frames, axis=0)
    return decoded_frames


def parse_file_paths(input_path: str) -> list:
    assert osp.exists(input_path), \
        f"{input_path} did not exists!"
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".avi") or file.endswith(".mp4"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files


def parse_args():
    # general params
    parser = argparse.ArgumentParser("PaddleVideo Web Serving model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/PP-TSM.yaml',
                        help='serving config file path')
    parser.add_argument('-ptn',
                        '--port_number',
                        type=int,
                        default=18080,
                        help='http port number')
    parser.add_argument('-i',
                        '--input_file',
                        type=str,
                        help='input file path or directory path')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    url = f"http://127.0.0.1:{args.port_number}/video/prediction"

    files_list = parse_file_paths(args.input_file)

    for file_path in files_list:
        # decoding video and get stacked frames as ndarray
        decoded_frames = video_to_numpy(file_path=file_path)

        # encode ndarray to base64 string for transportation.
        decoded_frames_base64 = numpy_to_base64(decoded_frames)

        # generate dict & convert to json.
        data = {
            "key": ["frames", "frames_shape"],
            "value": [decoded_frames_base64,
                      str(decoded_frames.shape)]
        }
        data = json.dumps(data)

        # transport to server & get get results.
        r = requests.post(url=url, data=data, timeout=100)

        # print result
        print(r.json())
