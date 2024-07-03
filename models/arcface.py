import cv2
import numpy as np
import onnxruntime

from utils.helpers import norm_crop_image


class ArcFaceONNX:
    def __init__(self, model_path: str = None, session=None) -> None:
        self.session = session
        self.input_mean = 127.5
        self.input_std = 127.5
        self.taskname = "recognition"

        if session is None:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape

        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape

        outputs = self.session.get_outputs()
        output_names = []
        for output in outputs:
            output_names.append(output.name)

        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def get_feat(self, images: np.ndarray) -> np.ndarray:
        if not isinstance(images, list):
            images = [images]

        input_size = self.input_size
        blob = cv2.dnn.blobFromImages(
            images,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})[0]
        return outputs

    def __call__(self, image, kps):
        aligned_image = norm_crop_image(image, landmark=kps)
        embedding = self.get_feat(aligned_image).flatten()
        return embedding
