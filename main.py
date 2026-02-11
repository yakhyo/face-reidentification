import os
import cv2
import random
import time
import warnings
import argparse
import logging
import numpy as np

from database import FaceDatabase
from models import SCRFD, ArcFace
from utils.logging import setup_logging
from utils.helpers import draw_bbox_info, draw_bbox


warnings.filterwarnings("ignore")
setup_logging(log_to_file=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Face Detection-and-Recognition with FAISS")

    parser.add_argument("--det-weight", type=str, default="./weights/det_10g.onnx", help="Path to detection model")
    parser.add_argument("--rec-weight", type=str, default="./weights/w600k_mbf.onnx", help="Path to recognition model")
    parser.add_argument("--similarity-thresh", type=float, default=0.4, help="Similarity threshold between faces")
    parser.add_argument("--confidence-thresh", type=float, default=0.5, help="Confidence threshold for face detection")
    parser.add_argument("--faces-dir", type=str, default="./assets/faces", help="Path to faces stored dir")
    parser.add_argument("--source", type=str, default="./assets/in_video.mp4", help="Video file or webcam source")
    parser.add_argument("--max-num", type=int, default=0, help="Maximum number of face detections from a frame")
    parser.add_argument(
        "--db-path",
        type=str,
        default="./database/face_database",
        help="path to vector db and metadata"
    )
    parser.add_argument("--update-db", action="store_true", help="Force update of the face database")
    parser.add_argument("--output", type=str, default="output_video.mp4", help="Output path for annotated video")

    return parser.parse_args()


def build_face_database(
    detector: SCRFD,
    recognizer: ArcFace,
    params: argparse.Namespace,
    force_update: bool = False,
) -> FaceDatabase:
    face_db = FaceDatabase(db_path=params.db_path)

    if not force_update and face_db.load():
        logging.info("Loaded face database from disk.")
        return face_db

    logging.info("Building face database from images...")

    if not os.path.exists(params.faces_dir):
        logging.warning(f"Faces directory {params.faces_dir} does not exist. Creating empty database.")
        face_db.save()
        return face_db

    embeddings: list[np.ndarray] = []
    names: list[str] = []

    for filename in sorted(os.listdir(params.faces_dir)):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        name = filename.rsplit(".", 1)[0]
        image_path = os.path.join(params.faces_dir, filename)
        image = cv2.imread(image_path)

        if image is None:
            logging.warning(f"Could not read image: {image_path}")
            continue

        try:
            bboxes, kpss = detector.detect(image, max_num=1)

            if len(kpss) == 0:
                logging.warning(f"No face detected in {image_path}. Skipping...")
                continue

            embedding = recognizer.get_embedding(image, kpss[0])
            embeddings.append(embedding)
            names.append(name)
            logging.info(f"Added face for: {name}")
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            continue

    if embeddings:
        face_db.add_faces_batch(embeddings, names)

    face_db.save()
    return face_db


def frame_processor(
    frame: np.ndarray,
    detector: SCRFD,
    recognizer: ArcFace,
    face_db: FaceDatabase,
    colors: dict,
    params: argparse.Namespace,
) -> np.ndarray:
    try:
        bboxes, kpss = detector.detect(frame, params.max_num)

        if len(bboxes) == 0:
            return frame

        # Collect embeddings and bounding boxes for all detected faces.
        embeddings: list[np.ndarray] = []
        processed_bboxes: list[list[int]] = []

        for bbox, kps in zip(bboxes, kpss):
            try:
                *bbox_coords, _ = bbox.astype(np.int32)
                embedding = recognizer.get_embedding(frame, kps)
                embeddings.append(embedding)
                processed_bboxes.append(bbox_coords)
            except Exception as e:
                logging.warning(f"Error processing face embedding: {e}")
                continue

        if not embeddings:
            return frame

        # Single FAISS batch call for all faces in the frame.
        results = face_db.batch_search(embeddings, params.similarity_thresh)

        # Draw results -- order is preserved by batch_search.
        for bbox, (name, similarity) in zip(processed_bboxes, results):
            if name != "Unknown":
                if name not in colors:
                    colors[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw_bbox_info(frame, bbox, similarity=similarity, name=name, color=colors[name])
            else:
                draw_bbox(frame, bbox, (255, 0, 0))

    except Exception as e:
        logging.error(f"Error in frame processing: {e}")

    return frame


def main(params):
    try:
        detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
        recognizer = ArcFace(params.rec_weight)
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        return

    face_db = build_face_database(detector, recognizer, params, force_update=params.update_db)
    colors: dict[str, tuple[int, int, int]] = {}

    try:
        cap = cv2.VideoCapture(params.source if not isinstance(params.source, int) else int(params.source))
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {params.source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(params.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            frame = frame_processor(frame, detector, recognizer, face_db, colors, params)
            elapsed = time.time() - start

            # Draw FPS on the frame.
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame,
                f"FPS: {current_fps:.1f}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
            )

            out.write(frame)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1
            logging.debug(f"Frame {frame_count}, FPS: {current_fps:.2f}")

        logging.info(f"Processed {frame_count} frames.")

    except Exception as e:
        logging.error(f"Error during video processing: {e}")
    finally:
        if "cap" in locals():
            cap.release()
        if "out" in locals():
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    try:
        args.source = int(args.source)
    except ValueError:
        pass
    main(args)
