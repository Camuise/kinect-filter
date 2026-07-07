"""
Compute depth maps for images or live frames.

Modified by Issac to work continuously in a loop rather than just on a single image.
"""
import os
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from . import utils
from .transforms import Resize, NormalizeImage, PrepareForNet

tf.compat.v1.disable_eager_execution()

_DEFAULT_ESTIMATOR = None

MODEL_PATH = str(Path(__file__).resolve().parent / "models" / "model-f6b98070.pb")

class DepthMapEstimator:
    def __init__(self, model_path, model_type="large"):
        if model_type == "large":
            self.net_w, self.net_h = 384, 384
        elif model_type == "small":
            self.net_w, self.net_h = 256, 256
        else:
            raise ValueError(f"model_type '{model_type}' not implemented, use: large or small")

        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

            model_operations = self.graph.get_operations()
            self.input_tensor = self.graph.get_tensor_by_name("0:0")
            self.output_tensor = self.graph.get_tensor_by_name(model_operations[-1].name + ":0")

        self.sess = tf.compat.v1.Session(graph=self.graph)

        # Fail fast on CPU for NCHW models
        try:
            self.predict(np.zeros((self.net_h, self.net_w, 3), dtype=np.float32))
        except tf.errors.UnimplementedError as e:
            raise RuntimeError(
                "This MiDaS model uses NCHW and cannot run on TensorFlow CPU. "
                "Use a GPU backend or a CPU-compatible NHWC model."
            ) from e

        resize_image = Resize(
            self.net_w,
            self.net_h,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )

        def compose2(f1, f2):
            return lambda x: f2(f1(x))

        self.transform = compose2(resize_image, PrepareForNet())

    def predict(self, rgb_frame):
        img_input = self.transform({"image": rgb_frame})["image"]
        # Use tensor name as dict key to avoid unhashable Tensor objects
        feed_dict = {self.input_tensor.name: [img_input]}
        prediction = self.sess.run(self.output_tensor, feed_dict)
        prediction = prediction.reshape(self.net_h, self.net_w)
        prediction = cv2.resize(
            prediction,
            (rgb_frame.shape[1], rgb_frame.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        return prediction.astype(np.float32)

    def close(self):
        self.sess.close()


def get_default_estimator(model_path=MODEL_PATH, model_type="large"):
    global _DEFAULT_ESTIMATOR
    if _DEFAULT_ESTIMATOR is None:
        _DEFAULT_ESTIMATOR = DepthMapEstimator(model_path, model_type)
    return _DEFAULT_ESTIMATOR


def create_depth_map(rgb_frame, model_path=MODEL_PATH, model_type="large"):
    """Return a depth map array for a single RGB frame."""
    return get_default_estimator(model_path, model_type).predict(rgb_frame)


def run(input_path, output_path, model_path, model_type="large"):
    """Run depth estimation on files in a folder."""
    estimator = DepthMapEstimator(model_path, model_type)

    img_names = glob.glob(os.path.join(input_path, "*"))
    os.makedirs(output_path, exist_ok=True)

    try:
        for ind, img_name in enumerate(img_names):
            print(f"  processing {img_name} ({ind + 1}/{len(img_names)})")
            img = utils.read_image(img_name)  # RGB
            prediction = estimator.predict(img)

            filename = os.path.join(
                output_path, os.path.splitext(os.path.basename(img_name))[0]
            )
            utils.write_depth(filename, prediction, bits=2)
    finally:
        estimator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", default="input")
    parser.add_argument("-o", "--output_path", default="output")
    parser.add_argument("-m", "--model_weights", default=MODEL_PATH)
    parser.add_argument("-t", "--model_type", default="large")

    args = parser.parse_args()
    run(args.input_path, args.output_path, args.model_weights, args.model_type)