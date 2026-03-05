"""
Layer-by-layer inference with intermediate tensor capture.

Runs ONNX inference on a test image and captures the input/output activation
tensors at every Conv layer boundary. These serve as ground truth for verifying
that the FPGA produces correct results.
"""

import json
import os
import sys

import cv2
import numpy as np
import onnx
from onnx import shape_inference
import onnxruntime as ort


ONNX_PATH = "yolo11n.onnx"
DEBUG_ONNX_PATH = "yolo11n_debug.onnx"
CONV_LAYERS_PATH = "conv_layers.json"
OUTPUT_PATH = "intermediate_tensors.npz"
INPUT_SIZE = 640


def preprocess_image(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None, None

    original_shape = img.shape[:2]

    h, w = img.shape[:2]
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)

    top = (INPUT_SIZE - new_h) // 2
    left = (INPUT_SIZE - new_w) // 2

    canvas[top:top + new_h, left:left + new_w] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)

    return blob, original_shape


def build_debug_model():

    print(f"Loading {ONNX_PATH}")
    model = onnx.load(ONNX_PATH)

    print("Running shape inference...")
    model = shape_inference.infer_shapes(model)

    # Collect conv input/output tensor names
    conv_tensor_names = set()

    for node in model.graph.node:
        if node.op_type == "Conv":
            conv_tensor_names.add(node.input[0])
            conv_tensor_names.add(node.output[0])

    value_info_map = {vi.name: vi for vi in model.graph.value_info}
    graph_input_map = {inp.name: inp for inp in model.graph.input}

    existing_outputs = {o.name for o in model.graph.output}

    added = 0

    for name in conv_tensor_names:

        if name in existing_outputs:
            continue

        if name in value_info_map:
            model.graph.output.append(value_info_map[name])
            added += 1

        elif name in graph_input_map:

            inp = graph_input_map[name]
            tensor_type = inp.type.tensor_type
            elem_type = tensor_type.elem_type

            shape = []
            for d in tensor_type.shape.dim:
                if d.dim_value:
                    shape.append(int(d.dim_value))
                else:
                    shape.append(1)

            vi = onnx.helper.make_tensor_value_info(
                name,
                elem_type,
                shape
            )

            model.graph.output.append(vi)
            added += 1

        else:
            # fallback if shape inference didn't record tensor
            vi = onnx.helper.make_tensor_value_info(
                name,
                onnx.TensorProto.FLOAT,
                None
            )

            model.graph.output.append(vi)
            added += 1

    print(f"Added {added} intermediate outputs")

    onnx.save(model, DEBUG_ONNX_PATH)
    print(f"Saved debug model: {DEBUG_ONNX_PATH}")

    return model


def run_inference(onnx_path, input_tensor):

    session = ort.InferenceSession(onnx_path)

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    outputs = session.run(output_names, {input_name: input_tensor})

    return dict(zip(output_names, outputs))


def main():

    print("=" * 60)
    print("LAYER-BY-LAYER INFERENCE - Intermediate Tensor Capture")
    print("=" * 60)

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

    if not os.path.exists(ONNX_PATH):
        print(f"ERROR: {ONNX_PATH} not found. Run export_onnx.py first.")
        return

    print(f"\n[1/4] Preprocessing image: {image_path}")

    input_tensor, original_shape = preprocess_image(image_path)

    if input_tensor is None:
        return

    print(f"Input tensor shape: {input_tensor.shape}")

    print("\n[2/4] Running baseline inference")

    baseline_outputs = run_inference(ONNX_PATH, input_tensor)

    baseline_output_name = list(baseline_outputs.keys())[0]
    baseline_tensor = baseline_outputs[baseline_output_name]

    print(f"Baseline output shape: {baseline_tensor.shape}")

    print("\n[3/4] Building debug model")

    build_debug_model()

    print("Running debug inference")

    debug_outputs = run_inference(DEBUG_ONNX_PATH, input_tensor)

    print(f"Captured {len(debug_outputs)} tensors")

    debug_final = debug_outputs.get(baseline_output_name)

    if debug_final is not None:

        max_diff = np.max(np.abs(baseline_tensor - debug_final))

        print(f"Verification diff = {max_diff}")

    print("\n[4/4] Saving tensors")

    save_dict = {name: tensor for name, tensor in debug_outputs.items()}

    save_dict["preprocessed_input"] = input_tensor

    np.savez(OUTPUT_PATH, **save_dict)

    print(f"Saved tensors -> {OUTPUT_PATH}")

    print("\nINTERMEDIATE TENSOR CAPTURE COMPLETE")


if __name__ == "__main__":
    main()
