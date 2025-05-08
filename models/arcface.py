import cv2
import numpy as np
from logging import getLogger
from typing import List, Tuple, Union
from onnxruntime import InferenceSession

from utils.helpers import face_alignment

__all__ = ["ArcFace"]

logger = getLogger(__name__)


class ArcFace:
    """
    ArcFace Model for Face Recognition

    This class implements a face encoder using the ArcFace architecture,
    loading a pre-trained model from an ONNX file.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the ArcFace face encoder model.

        Args:
            model_path (str): Path to ONNX model file.

        Raises:
            RuntimeError: If model initialization fails.
        """
        self.model_path = model_path
        self.input_size = (112, 112)  # Standard size for face recognition models
        self.normalization_mean = 127.5  # Normalization parameters
        self.normalization_scale = 127.5

        logger.info(f"Initializing ArcFace model from {self.model_path}")

        try:
            # Initialize model session with available providers
            self.session = InferenceSession(
                self.model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Extract model input configuration
            input_config = self.session.get_inputs()[0]
            self.input_name = input_config.name

            # Verify input dimensions
            input_shape = input_config.shape
            model_input_size = tuple(input_shape[2:4][::-1])  # (width, height)
            if model_input_size != self.input_size:
                logger.warning(
                    f"Model input size {model_input_size} differs from configured size {self.input_size}"
                )

            # Get output configuration
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.output_shape = self.session.get_outputs()[0].shape
            self.embedding_size = self.output_shape[1]

            assert len(self.output_names) == 1, "Expected only one output node."
            logger.info(
                f"Successfully initialized face encoder from {self.model_path} "
                f"(embedding size: {self.embedding_size})"
            )

        except Exception as e:
            logger.error(f"Failed to load face encoder model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize model session for '{self.model_path}'") from e

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess the face image: resize, normalize, and convert to the required format.

        Args:
            face_image (np.ndarray): Input face image in BGR format.

        Returns:
            np.ndarray: Preprocessed image blob ready for inference.
        """
        # Resize image to the required input size
        resized_face = cv2.resize(face_image, self.input_size)

        if isinstance(self.normalization_scale, (list, tuple)):
            # Handle per-channel normalization
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB).astype(np.float32)

            # Apply normalization
            mean_array = np.array(self.normalization_mean, dtype=np.float32)
            scale_array = np.array(self.normalization_scale, dtype=np.float32)
            normalized_face = (rgb_face - mean_array) / scale_array

            # Change to NCHW format (batch, channels, height, width)
            transposed_face = np.transpose(normalized_face, (2, 0, 1))  # CHW
            face_blob = np.expand_dims(transposed_face, axis=0)  # NCHW
        else:
            # Single-value normalization using cv2.dnn
            face_blob = cv2.dnn.blobFromImage(
                resized_face,
                scalefactor=1.0 / self.normalization_scale,
                size=self.input_size,
                mean=(self.normalization_mean, self.normalization_mean, self.normalization_mean),
                swapRB=True  # Convert BGR to RGB
            )
        return face_blob

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from an image using facial landmarks for alignment.

        Args:
            image (np.ndarray): Input image in BGR format.
            landmarks (np.ndarray): 5-point facial landmarks for alignment.

        Returns:
            np.ndarray: Face embedding vector (normalized feature vector).

        Raises:
            ValueError: If inputs are invalid.
        """
        if image is None or landmarks is None:
            raise ValueError("Image and landmarks must not be None")

        try:
            # Align face using landmarks
            aligned_face, _ = face_alignment(image, landmarks)

            # Preprocess and get embedding
            face_blob = self.preprocess(aligned_face)
            embedding = self.session.run(self.output_names, {self.input_name: face_blob})[0]

            # L2 normalization of embedding
            embedding_norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            normalized_embedding = embedding / embedding_norm

            return normalized_embedding.flatten()

        except Exception as e:
            logger.error(f"Error extracting face embedding: {str(e)}")
            raise
