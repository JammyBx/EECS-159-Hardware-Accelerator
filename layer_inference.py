"""
Layer-by-layer inference with intermediate tensor capture.

Runs ONNX inference on a test image and captures the input/output activation
tensors at every Conv layer boundary. These serve as ground truth for verifying
that the FPGA produces correct results.

Requires: yolo11n.onnx (run export_onnx.py first)

Outputs:
  yolo11n_debug.onnx       - ONNX model with intermediate outputs exposed
  intermediate_tensors.npz - all captured activation tensors

Usage: python3 layer_inference.py [image_path]
       Defaults to test_image.jpg if no argument given.
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
    """
    Preprocess image the same way YOLO does:
    letterbox resize to 640x640, normalize to [0,1], HWC -> NCHW.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ERROR: Could not load image: {image_path}")
        return None, None

    original_shape = img.shape[:2]  # (height, width)

    # Letterbox resize: scale to fit 640x640 while maintaining aspect ratio
    h, w = img.shape[:2]
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to 640x640 (gray padding)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    top = (INPUT_SIZE - new_h) // 2
    left = (INPUT_SIZE - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    # BGR -> RGB, HWC -> CHW, normalize to [0, 1], add batch dimension
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)  # [1, 3, 640, 640]

    return blob, original_shape


def build_debug_model():
    """
    Add all Conv input/output tensor names as graph outputs so onnxruntime
    will return their values during inference.
    """
    print(f"  Loading {ONNX_PATH}...")
    model = onnx.load(ONNX_PATH)

    # Run shape inference to get type info for intermediate tensors
    print("  Running shape inference...")
    model = shape_inference.infer_shapes(model)

    # Collect tensor names we want to observe (conv inputs and outputs)
    conv_tensor_names = set()
    for node in model.graph.node:
        if node.op_type == "Conv":
            conv_tensor_names.add(node.input[0])   # activation input
            conv_tensor_names.add(node.output[0])   # activation output

    # Build lookup of value_info by name
    value_info_map = {vi.name: vi for vi in model.graph.value_info}

    # Also include graph inputs (the model input is a conv input for the first layer)
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
            # Graph input tensor — create a ValueInfoProto from the input
            inp = graph_input_map[name]
            vi = onnx.helper.make_value_info_proto(name, inp.type)
            model.graph.output.append(vi)
            added += 1

    print(f"  Added {added} intermediate outputs to graph.")

    onnx.save(model, DEBUG_ONNX_PATH)
    print(f"  Saved debug model: {DEBUG_ONNX_PATH}")
    return model


def run_inference(onnx_path, input_tensor):
    """Run inference and return all outputs as a name->tensor dict."""
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, {input_name: input_tensor})
    return dict(zip(output_names, outputs))


def main():
    print("=" * 60)
    print("LAYER-BY-LAYER INFERENCE - Intermediate Tensor Capture")
    print("=" * 60)

    # Determine image path
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

    if not os.path.exists(ONNX_PATH):
        print(f"\n  ERROR: {ONNX_PATH} not found. Run export_onnx.py first.")
        return

    # Step 1: Preprocess image
    print(f"\n[1/4] Preprocessing image: {image_path}")
    input_tensor, original_shape = preprocess_image(image_path)
    if input_tensor is None:
        return
    print(f"  Input tensor shape: {input_tensor.shape}")
    print(f"  Value range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

    # Step 2: Run baseline inference (unmodified model)
    print(f"\n[2/4] Running baseline inference on unmodified model...")
    baseline_outputs = run_inference(ONNX_PATH, input_tensor)
    baseline_output_name = list(baseline_outputs.keys())[0]
    baseline_tensor = baseline_outputs[baseline_output_name]
    print(f"  Baseline output: {baseline_output_name} -> shape {baseline_tensor.shape}")

    # Step 3: Build debug model and run inference with intermediate outputs
    print(f"\n[3/4] Building debug model with intermediate outputs...")
    build_debug_model()

    print(f"\n  Running debug inference...")
    debug_outputs = run_inference(DEBUG_ONNX_PATH, input_tensor)
    print(f"  Captured {len(debug_outputs)} output tensors.")

    # Verify final output matches baseline
    debug_final = debug_outputs.get(baseline_output_name)
    if debug_final is not None:
        max_diff = np.max(np.abs(baseline_tensor - debug_final))
        print(f"\n  Verification: final output max diff = {max_diff:.2e}")
        if max_diff < 1e-5:
            print("  PASS: Debug model matches baseline.")
        else:
            print("  WARNING: Outputs differ. Check model export.")
    else:
        print(f"  WARNING: Could not find baseline output '{baseline_output_name}' in debug outputs.")

    # Step 4: Map tensors to Conv layers and save
    print(f"\n[4/4] Mapping tensors to Conv layers...")

    # Load conv layer metadata if available
    conv_metadata = None
    if os.path.exists(CONV_LAYERS_PATH):
        with open(CONV_LAYERS_PATH) as f:
            conv_metadata = json.load(f)

    # Print tensor summary for Conv layers
    if conv_metadata:
        print(f"\n  {'Layer':<6} {'Input Tensor':<35} {'Input Shape':<25} {'Output Shape':<25}")
        print("  " + "-" * 95)

        for layer in conv_metadata:
            inp_name = layer["input_name"]
            out_name = layer["output_name"]
            inp_tensor = debug_outputs.get(inp_name)
            out_tensor = debug_outputs.get(out_name)

            inp_shape = str(inp_tensor.shape) if inp_tensor is not None else "NOT CAPTURED"
            out_shape = str(out_tensor.shape) if out_tensor is not None else "NOT CAPTURED"

            print(f"  {layer['layer_idx']:<6} {inp_name:<35} {inp_shape:<25} {out_shape:<25}")

    # Save all intermediate tensors
    # Filter to only save tensors we actually captured (not None)
    save_dict = {name: tensor for name, tensor in debug_outputs.items()}
    # Also save the preprocessed input
    save_dict["preprocessed_input"] = input_tensor

    np.savez(OUTPUT_PATH, **save_dict)
    file_size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\n  Saved {len(save_dict)} tensors to {OUTPUT_PATH} ({file_size:.1f} MB)")

    # Print value range summary for each Conv output
    if conv_metadata:
        print(f"\n  Conv output value ranges:")
        for layer in conv_metadata:
            out_name = layer["output_name"]
            tensor = debug_outputs.get(out_name)
            if tensor is not None:
                print(f"    Layer {layer['layer_idx']:>2}: min={tensor.min():>10.4f}  "
                      f"max={tensor.max():>10.4f}  mean={tensor.mean():>10.4f}  "
                      f"shape={tensor.shape}")

    print("\n" + "=" * 60)
    print("INTERMEDIATE TENSOR CAPTURE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
