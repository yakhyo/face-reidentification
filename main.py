import argparse
import logging
import os
import random
import time
import warnings

import cv2
import numpy as np

from database import FaceDatabase
from models import SCRFD, ArcFace
from utils.helpers import draw_bbox, draw_bbox_info
from utils.logging import setup_logging

# Suppress only known noisy warnings from third-party libraries.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"onnxruntime.*")

setup_logging(log_to_file=True)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for face re-identification pipeline."""
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
        help="path to vector db and metadata",
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
    """Build or load the FAISS face database from reference images.

    Args:
        detector: Face detection model.
        recognizer: Face recognition model (provides embedding_size).
        params: CLI arguments namespace.
        force_update: If True, rebuild even when a saved database exists.

    Returns:
        Populated FaceDatabase instance.
    """
    face_db = FaceDatabase(embedding_size=recognizer.embedding_size, db_path=params.db_path)

    if not force_update and face_db.load():
        logger.info("Loaded face database from disk.")
        return face_db

    logger.info("Building face database from images...")

    if not os.path.exists(params.faces_dir):
        logger.warning(f"Faces directory {params.faces_dir} does not exist. Creating empty database.")
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
            logger.warning(f"Could not read image: {image_path}")
            continue

        try:
            bboxes, kpss = detector.detect(image, max_num=1)

            if len(kpss) == 0:
                logger.warning(f"No face detected in {image_path}. Skipping...")
                continue

            embedding = recognizer.get_embedding(image, kpss[0])
            embeddings.append(embedding)
            names.append(name)
            logger.info(f"Added face for: {name}")
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
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
    colors: dict[str, tuple[int, int, int]],
    params: argparse.Namespace,
) -> np.ndarray:
    """Detect faces, extract embeddings, search the database, and annotate the frame.

    Args:
        frame: Input BGR video frame.
        detector: Face detection model.
        recognizer: Face recognition model.
        face_db: FAISS face database to query.
        colors: Mutable colour map — new identities are assigned a random colour.
        params: CLI arguments namespace.

    Returns:
        Annotated frame with bounding boxes and identity labels.
    """
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
                logger.warning(f"Error processing face embedding: {e}")
                continue

        if not embeddings:
            return frame

        # Single FAISS batch call for all faces in the frame.
        results = face_db.batch_search(embeddings, params.similarity_thresh)

        # Draw results — order is preserved by batch_search.
        for bbox, (name, similarity) in zip(processed_bboxes, results):
            if name != "Unknown":
                if name not in colors:
                    colors[name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw_bbox_info(frame, bbox, similarity=similarity, name=name, color=colors[name])
            else:
                draw_bbox(frame, bbox, (255, 0, 0))

    except Exception as e:
        logger.error(f"Error in frame processing: {e}")

    return frame


def _resolve_source(source: str) -> int | str:
    """Convert source string to an integer webcam index when appropriate.

    A pure-digit string (e.g. "0", "1") is treated as a webcam index.
    All other strings are returned as-is (file paths, URLs, etc.).
    """
    if source.isdigit():
        return int(source)
    return source


def main(params: argparse.Namespace) -> None:
    """Run the face re-identification video pipeline.

    Args:
        params: Parsed CLI arguments.
    """
    try:
        detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
        recognizer = ArcFace(params.rec_weight)
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return

    face_db = build_face_database(detector, recognizer, params, force_update=params.update_db)
    colors: dict[str, tuple[int, int, int]] = {}

    cap: cv2.VideoCapture | None = None
    out: cv2.VideoWriter | None = None
    try:
        source = _resolve_source(params.source)
        cap = cv2.VideoCapture(source)
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
            logger.debug(f"Frame {frame_count}, FPS: {current_fps:.2f}")

        logger.info(f"Processed {frame_count} frames.")

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(parse_args())
