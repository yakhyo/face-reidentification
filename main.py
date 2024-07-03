import os
import cv2
import random
import warnings
import argparse
import onnxruntime
import numpy as np
import logging


from models import SCRFD, ArcFaceONNX
from utils.helpers import draw_fancy_bbox, compute_similarity
from typing import Union, List, Tuple


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection-and-Recognition")
    parser.add_argument(
        "--det-weight",
        type=str,
        default="./weights/det_10g.onnx",
        help="Path to detection model",
    )
    parser.add_argument(
        "--rec-weight",
        type=str,
        default="./weights/w600k_r50.onnx",
        help="Path to recognition model",
    )
    parser.add_argument(
        "--similarity-thresh",
        type=float,
        default=0.4,
        help="Similarity threshold between faces",
    )
    parser.add_argument(
        "--confidence-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for face detection",
    )
    parser.add_argument(
        "--faces-dir", type=str, default="./faces", help="Path to faces stored dir"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video file or video camera source. i.e 0 - webcam",
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=10,
        help="Maximum number of face detections from a frame",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def build_targets(
    detector, recognizer, params: argparse.Namespace
) -> Union[Tuple[np.ndarray, str]]:
    """Builds targets using face detection and recognition.

    Args:
        detector (SCRFD): Face detector model.
        recognizer (ArcFaceONNX): Face recognizer model.
        image_folder (str): Path to the folder containing images.

    Returns:
        List[Tuple[np.ndarray, str]]: A list where each tuple contains a feature vector and the corresponding image name.
    """
    targets = []
    for filename in os.listdir(params.faces_dir):
        name = filename[:-4]
        image_path = os.path.join(params.faces_dir, filename)

        image = cv2.imread(image_path)
        bboxes, kpss = detector.detect(
            image, input_size=(640, 640), thresh=params.confidence_thresh, max_num=1
        )

        if len(kpss) == 0:
            logging.warning(f"No face detected in {image_path}. Skipping...")
            continue

        feature_vector = recognizer(image, kpss[0])
        targets.append((feature_vector, name))

    return targets


def frame_processor(
    frame: np.ndarray,
    detector,
    recognizer,
    targets: List[Tuple[np.ndarray, str]],
    colors: dict,
    params,
) -> np.ndarray:
    bboxes, kpss = detector.detect(
        frame,
        input_size=(640, 640),
        thresh=params.confidence_thresh,
        max_num=params.max_num,
    )

    for bbox, kps in zip(bboxes, kpss):
        x1, y1, x2, y2, score = bbox.astype(np.int32)
        embedding = recognizer(frame, kps)

        max_similarity = 0
        best_match_name = "Unknown"
        for target, name in targets:
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity = similarity
                best_match_name = name

        if best_match_name != "Unknown":
            color = colors[best_match_name]
            draw_fancy_bbox(
                frame,
                bbox,
                similarity=max_similarity,
                name=best_match_name,
                color=color,
            )
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

    return frame


def main(params):
    setup_logging(params.log_level)

    detector = SCRFD(params.det_weight)
    recognizer = ArcFaceONNX(params.rec_weight)

    targets = build_targets(detector, recognizer, params)
    colors = {
        name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        for _, name in targets
    }

    cap = cv2.VideoCapture(params.source)
    if not cap.isOpened():
        raise Exception("Could not open video or webcam")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(
        "output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = frame_processor(
            frame=frame,
            detector=detector,
            recognizer=recognizer,
            targets=targets,
            colors=colors,
            params=params,
        )

        out.write(frame)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()

    if args.source.isdigit():
        args.source = int(args.source)

    main(args)
