"""Create a static INT8 ONNX model using representative training images for calibration."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
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


def parse_opt():
    """Parse static quantization options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="FP32 ONNX model")
    parser.add_argument("--calibration-dir", required=True, type=Path, help="representative training images")
    parser.add_argument("--output", required=True, type=Path, help="INT8 ONNX output")
    parser.add_argument("--imgsz", type=int, default=640, help="letterbox image size")
    parser.add_argument("--calibration-images", type=int, default=128, help="number of calibration images")
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_opt()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    reader = ImageCalibrationReader(options.model, options.calibration_dir, options.imgsz, options.calibration_images)
    quantize_static(
        str(options.model),
        str(options.output),
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"INT8 ONNX saved to {options.output} using {len(reader.image_paths)} calibration images")
