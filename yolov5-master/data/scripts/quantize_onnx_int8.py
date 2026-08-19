"""Create a static INT8 ONNX model using representative training images for calibration."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.augmentations import letterbox

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ImageCalibrationReader(CalibrationDataReader):
    """Feed letterboxed RGB images to ONNX Runtime static quantization calibration."""

    def __init__(self, model_path, image_directory, image_size, image_limit):
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = session.get_inputs()[0].name
        self.image_size = image_size
        self.image_paths = sorted(path for path in image_directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)[
            :image_limit
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No calibration images found in {image_directory}")
        self.index = 0

    def get_next(self):
        """Return one normalized NCHW image, or None after all calibration images are consumed."""
        if self.index >= len(self.image_paths):
            return None
        image_path = self.image_paths[self.index]
        self.index += 1
        image = cv2.imread(str(image_path))
        if image is None:
            return self.get_next()
        image = letterbox(image, new_shape=self.image_size, auto=False)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        return {self.input_name: image[None]}


def detect_head_nodes(model_path):
    """Return the final YOLO Detect-head and decode nodes to retain in FP32."""
    graph = onnx.load(model_path).graph
    start_index = next(
        (
            index
            for index, node in enumerate(graph.node)
            if any(input_name.startswith("model.25.m.") for input_name in node.input)
        ),
        None,
    )
    if start_index is None:
        raise ValueError("Could not identify YOLO Detect-head nodes in the ONNX graph.")
    return [node.name for node in graph.node[start_index:] if node.name]


def layer_weight_nodes(model_path, layer_indices):
    """Return convolution nodes whose trained weights belong to selected YOLO macro-layers."""
    prefixes = tuple(f"model.{index}." for index in layer_indices)
    return [
        node.name
        for node in onnx.load(model_path).graph.node
        if node.name and any(input_name.startswith(prefixes) for input_name in node.input)
    ]


def parse_opt():
    """Parse static quantization options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="FP32 ONNX model")
    parser.add_argument("--calibration-dir", required=True, type=Path, help="representative training images")
    parser.add_argument("--output", required=True, type=Path, help="INT8 ONNX output")
    parser.add_argument("--imgsz", type=int, default=640, help="letterbox image size")
    parser.add_argument("--calibration-images", type=int, default=128, help="number of calibration images")
    parser.add_argument("--protect-detect", action="store_true", help="retain Detect-head and decode nodes in FP32")
    parser.add_argument(
        "--protect-layers", nargs="+", type=int, default=[], help="YOLO macro-layer Conv nodes to retain in FP32"
    )
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_opt()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageCalibrationReader(options.model, options.calibration_dir, options.imgsz, options.calibration_images)
    nodes_to_exclude = []
    if options.protect_detect:
        nodes_to_exclude.extend(detect_head_nodes(options.model))
    if options.protect_layers:
        nodes_to_exclude.extend(layer_weight_nodes(options.model, options.protect_layers))
    nodes_to_exclude = list(dict.fromkeys(nodes_to_exclude)) or None
    quantize_static(
        str(options.model),
        str(options.output),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=nodes_to_exclude,
    )
    protected = f"; retained {len(nodes_to_exclude)} selected nodes in FP32" if nodes_to_exclude else ""
    print(f"INT8 ONNX saved to {options.output} using {len(reader.image_paths)} calibration images{protected}")
