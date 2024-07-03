import os
import cv2
import numpy as np
import onnxruntime

from utils.helpers import distance2bbox, distance2kps


class SCRFD:
    """
    Title: "Sample and Computation Redistribution for Efficient Face Detection"
    Paper: https://arxiv.org/abs/2105.04714
    """

    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        self.taskname = "detection"
        self.batched = False

        if self.session is None:
            assert self.model_file is not None
            assert os.path.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(
                self.model_file,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        self.mean = 127.5
        self.std = 128.0

        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        # print(input_shape)
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])

        input_name = input_cfg.name
        self.input_shape = input_shape

        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True

        output_names = []
        for output in outputs:
            output_names.append(output.name)

        self.input_name = input_name
        self.output_names = output_names

        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def forward(self, image, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(image.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.std,
            input_size,
            (self.mean, self.mean, self.mean),
            swapRB=True
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:  # If model support batch dim, take first output
                scores = outputs[idx][0]
                bbox_preds = outputs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = outputs[idx + fmc * 2][0] * stride
            else:  # If model doesn't support batching take output as is
                scores = outputs[idx]
                bbox_preds = outputs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = outputs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, image, input_size=None, thresh=None, max_num=0, metric="default"):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / image.shape[0]
        resized_image = cv2.resize(image, (new_width, new_height))
        det_image = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized_image
        det_thresh = thresh if thresh is not None else self.det_thresh

        scores_list, bboxes_list, kpss_list = self.forward(det_image, det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale

        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, thresh=self.nms_thresh)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            image_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - image_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - image_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (area - offset_dist_squared * 2.0)  # some extra weight on the centering
            bindex = np.argsort(values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= thresh)[0]
            order = order[indices + 1]

        return keep


if __name__ == "__main__":
    detector = SCRFD(model_file="./weights/det_10g.onnx")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not cap.isOpened():
            break

        boxes_list, points_list = detector.detect(frame, input_size=(640, 640))

        if points_list is not None:
            print(points_list.shape)

        for boxes, points in zip(boxes_list, points_list):
            x1, y1, x2, y2, score = boxes.astype(np.int32)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if points_list is not None:
                for point in points:
                    point = point.astype(np.int32)
                    cv2.circle(frame, tuple(point), 1, (0, 0, 255), 2)

        cv2.imshow("FaceDetection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
