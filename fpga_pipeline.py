"""
Hybrid CPU/FPGA inference pipeline for YOLO11n.

Routes Conv layers to FPGA via UART for hardware-accelerated computation,
runs all other operations (SiLU, Add, Concat, Resize, MaxPool) on CPU
via onnxruntime. Compares results against a pure-CPU baseline.

Usage:
  python3 fpga_pipeline.py [image_path]
  Defaults to test_image.jpg if no argument given.
"""

import json
import math
import os
import sys
import time

import cv2
import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from onnx import shape_inference
import onnxruntime as ort

from quantize import quantize_activation, quantize_weights, dequantize_output
from uart_comm import FPGAUart, FPGAUartLoopback


ONNX_PATH = "yolo11n.onnx"
CONV_LAYERS_PATH = "conv_layers.json"
CONV_WEIGHTS_PATH = "conv_weights.npz"
CONV_WEIGHTS_INT8_PATH = "conv_weights_int8.npz"
CONV_SCALES_PATH = "conv_scales.json"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# COCO class names (80 classes)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


def preprocess(image_path):
    """Letterbox resize, normalize, HWC -> NCHW. Same as YOLO preprocessing."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    original_shape = img.shape[:2]
    h, w = original_shape
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    top = (INPUT_SIZE - new_h) // 2
    left = (INPUT_SIZE - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, axis=0)

    pad_info = (top, left, scale)
    return blob, original_shape, pad_info


def postprocess(raw_output, original_shape, pad_info):
    """
    Decode YOLO output to detections.

    Args:
        raw_output: np.ndarray of shape (1, 84, 8400)
        original_shape: (original_h, original_w)
        pad_info: (pad_top, pad_left, scale)

    Returns:
        list of (class_name, confidence, x1, y1, x2, y2)
    """
    output = raw_output[0]  # (84, 8400)
    output = output.T       # (8400, 84)

    # Split into boxes and class scores
    boxes_xywh = output[:, :4]         # (8400, 4) — cx, cy, w, h
    class_scores = output[:, 4:]       # (8400, 80)

    # Get best class per detection
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    # Filter by confidence
    mask = confidences > CONF_THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_xywh) == 0:
        return []

    # Convert xywh -> xyxy
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

    # Undo letterbox padding and scaling
    pad_top, pad_left, scale = pad_info
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_left) / scale
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_top) / scale
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_left) / scale
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_top) / scale

    # Clip to image bounds
    orig_h, orig_w = original_shape
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    # NMS
    keep = nms(boxes_xyxy, confidences, IOU_THRESHOLD)

    detections = []
    for i in keep:
        name = COCO_NAMES[class_ids[i]] if class_ids[i] < len(COCO_NAMES) else f"class_{class_ids[i]}"
        detections.append((
            name,
            float(confidences[i]),
            float(boxes_xyxy[i, 0]),
            float(boxes_xyxy[i, 1]),
            float(boxes_xyxy[i, 2]),
            float(boxes_xyxy[i, 3]),
        ))

    return detections


def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


class HybridPipeline:
    """
    Runs YOLO11n inference with Conv layers on FPGA and everything else on CPU.
    """

    def __init__(self, onnx_path=ONNX_PATH, uart_port=None, use_loopback=True,
                 verbose=False):
        self.verbose = verbose

        # Load ONNX model
        print("  Loading ONNX model...")
        self.onnx_model = onnx.load(onnx_path)
        self.onnx_model = shape_inference.infer_shapes(self.onnx_model)

        # Build initializer lookup
        self.initializers = {}
        for init in self.onnx_model.graph.initializer:
            self.initializers[init.name] = numpy_helper.to_array(init)

        # Identify Conv nodes
        self.conv_node_indices = set()
        self.conv_nodes = []
        for i, node in enumerate(self.onnx_model.graph.node):
            if node.op_type == "Conv":
                self.conv_node_indices.add(i)
                self.conv_nodes.append((i, node))

        print(f"  Found {len(self.conv_nodes)} Conv layers.")

        # Load quantized weights and scales
        print("  Loading quantized weights...")
        self.int8_weights = {}
        self.weight_scales = {}

        if os.path.exists(CONV_WEIGHTS_INT8_PATH) and os.path.exists(CONV_SCALES_PATH):
            int8_data = np.load(CONV_WEIGHTS_INT8_PATH)
            with open(CONV_SCALES_PATH) as f:
                scales_data = json.load(f)

            for conv_idx, (node_idx, node) in enumerate(self.conv_nodes):
                w_key = f"layer_{conv_idx}_weight"
                if w_key in int8_data:
                    self.int8_weights[conv_idx] = int8_data[w_key]
                s_key = str(conv_idx)
                if s_key in scales_data:
                    self.weight_scales[conv_idx] = np.array(scales_data[s_key], dtype=np.float32)
        else:
            print("  WARNING: Quantized weights not found. Run quantize.py first.")
            print("  Will quantize on-the-fly (slower).")

        # CPU inference session (for baseline and non-Conv ops)
        self.cpu_session = ort.InferenceSession(onnx_path)
        self.input_name = self.cpu_session.get_inputs()[0].name

        # Build debug session with intermediate outputs
        self._build_debug_session(onnx_path)

        # Initialize UART
        if uart_port:
            self.uart = FPGAUart(port=uart_port, verbose=verbose)
        elif use_loopback:
            self.uart = FPGAUartLoopback(verbose=verbose)
        else:
            self.uart = None

        # Pre-load weights to FPGA
        if self.uart and self.int8_weights:
            self._preload_weights()

    def _build_debug_session(self, onnx_path):
        """Build an onnxruntime session that exposes all intermediate tensors."""
        model = onnx.load(onnx_path)
        model = shape_inference.infer_shapes(model)

        value_info_map = {vi.name: vi for vi in model.graph.value_info}
        existing = {o.name for o in model.graph.output}

        for node in model.graph.node:
            for output_name in node.output:
                if output_name not in existing and output_name in value_info_map:
                    model.graph.output.append(value_info_map[output_name])

        debug_path = onnx_path.replace('.onnx', '_pipeline_debug.onnx')
        onnx.save(model, debug_path)

        self.debug_session = ort.InferenceSession(debug_path)
        self.debug_output_names = [o.name for o in self.debug_session.get_outputs()]

    def _preload_weights(self):
        """Send all Conv layer weights to FPGA once at startup."""
        print("  Pre-loading weights to FPGA...")
        for conv_idx, (node_idx, node) in enumerate(self.conv_nodes):
            if conv_idx not in self.int8_weights:
                continue

            # Get layer config
            config = self._get_conv_config(node)
            self.uart.send_layer_config(
                conv_idx,
                kernel_shape=config['kernel_shape'],
                strides=config['strides'],
                pads=config['pads'],
                group=config['group'],
                in_channels=config['in_channels'],
                out_channels=config['out_channels'],
            )
            self.uart.send_weights(conv_idx, self.int8_weights[conv_idx])

        print(f"  Pre-loaded {len(self.int8_weights)} weight tensors.")

    def _get_conv_config(self, node):
        """Extract Conv attributes from an ONNX node."""
        def get_attr(name, default):
            for attr in node.attribute:
                if attr.name == name:
                    if attr.type == onnx.AttributeProto.INTS:
                        return list(attr.ints)
                    elif attr.type == onnx.AttributeProto.INT:
                        return attr.i
            return default

        weight = self.initializers[node.input[1]]
        out_ch = weight.shape[0]
        group = get_attr('group', 1)
        in_ch = weight.shape[1] * group

        return {
            'kernel_shape': get_attr('kernel_shape', [1, 1]),
            'strides': get_attr('strides', [1, 1]),
            'pads': get_attr('pads', [0, 0, 0, 0]),
            'group': group,
            'in_channels': in_ch,
            'out_channels': out_ch,
        }

    def run_cpu(self, input_tensor):
        """Run full inference on CPU (baseline)."""
        output_names = [o.name for o in self.cpu_session.get_outputs()]
        outputs = self.cpu_session.run(output_names, {self.input_name: input_tensor})
        return outputs[0]

    def run_hybrid(self, input_tensor):
        """
        Run inference with Conv on FPGA, everything else on CPU.

        Uses a per-layer comparison approach: runs the full model on CPU to get
        intermediate tensors, then replaces each Conv layer's output with the
        FPGA-computed (quantized) result for comparison.

        Returns the FPGA-computed conv outputs alongside the CPU baseline output.
        """
        if self.uart is None:
            raise RuntimeError("No UART connection. Use use_loopback=True or provide uart_port.")

        # Get all intermediate tensors from CPU
        debug_outputs = self.debug_session.run(
            self.debug_output_names,
            {self.input_name: input_tensor}
        )
        tensor_map = dict(zip(self.debug_output_names, debug_outputs))

        # For each Conv layer, compute on FPGA and compare
        fpga_results = {}
        for conv_idx, (node_idx, node) in enumerate(self.conv_nodes):
            inp_name = node.input[0]
            out_name = node.output[0]

            # Get CPU activation input for this conv
            if inp_name == self.input_name:
                act_float = input_tensor
            elif inp_name in tensor_map:
                act_float = tensor_map[inp_name]
            else:
                if self.verbose:
                    print(f"    Skipping layer {conv_idx}: input '{inp_name}' not available")
                continue

            # Quantize activation
            act_int8, act_scale = quantize_activation(act_float)

            # Get or compute quantized weights
            if conv_idx in self.int8_weights:
                w_int8 = self.int8_weights[conv_idx]
                w_scales = self.weight_scales[conv_idx]
            else:
                w_float = self.initializers[node.input[1]]
                w_int8, w_scales = quantize_weights(w_float)

            # Send to FPGA and compute
            config = self._get_conv_config(node)
            self.uart.send_activation(conv_idx, act_int8)
            self.uart.compute(conv_idx)

            # Compute expected output shape
            pads = config['pads']
            strides = config['strides']
            kh, kw = config['kernel_shape']
            _, _, hi, wi = act_float.shape
            ho = (hi + pads[0] + pads[2] - kh) // strides[0] + 1
            wo = (wi + pads[1] + pads[3] - kw) // strides[1] + 1
            out_shape = (act_float.shape[0], config['out_channels'], ho, wo)

            # Receive result
            result_int32 = self.uart.receive_result(conv_idx, out_shape)

            # Dequantize
            result_float = dequantize_output(result_int32, act_scale, w_scales)

            # Add bias if present
            if len(node.input) > 2 and node.input[2] in self.initializers:
                bias = self.initializers[node.input[2]]
                result_float += bias.reshape(1, -1, 1, 1)

            # Store and compare
            cpu_output = tensor_map.get(out_name)
            fpga_results[conv_idx] = {
                'fpga_output': result_float,
                'cpu_output': cpu_output,
                'output_name': out_name,
            }

            if cpu_output is not None and result_float.shape == cpu_output.shape:
                max_err = np.max(np.abs(cpu_output - result_float))
                rmse = np.sqrt(np.mean((cpu_output - result_float) ** 2))
                if self.verbose:
                    print(f"    Layer {conv_idx}: max_err={max_err:.6f}, rmse={rmse:.6f}")

        # Return CPU baseline final output (FPGA doesn't change it in this mode)
        cpu_final = self.run_cpu(input_tensor)
        return cpu_final, fpga_results

    def verify(self, image_path):
        """
        Run both CPU and hybrid inference, compare results per Conv layer.
        Returns a summary dict.
        """
        input_tensor, original_shape, pad_info = preprocess(image_path)

        print(f"\n  Running CPU baseline...")
        t0 = time.time()
        cpu_output = self.run_cpu(input_tensor)
        cpu_time = time.time() - t0
        print(f"  CPU inference: {cpu_time:.3f}s")

        print(f"\n  Running hybrid (FPGA conv layers)...")
        t0 = time.time()
        _, fpga_results = self.run_hybrid(input_tensor)
        hybrid_time = time.time() - t0
        print(f"  Hybrid inference: {hybrid_time:.3f}s")

        # Per-layer error summary
        print(f"\n  {'Layer':<6} {'Max Err':<12} {'RMSE':<12} {'Shape'}")
        print("  " + "-" * 50)

        layer_errors = []
        for conv_idx in sorted(fpga_results.keys()):
            r = fpga_results[conv_idx]
            if r['cpu_output'] is not None and r['fpga_output'].shape == r['cpu_output'].shape:
                max_err = float(np.max(np.abs(r['cpu_output'] - r['fpga_output'])))
                rmse = float(np.sqrt(np.mean((r['cpu_output'] - r['fpga_output']) ** 2)))
                print(f"  {conv_idx:<6} {max_err:<12.6f} {rmse:<12.6f} {r['fpga_output'].shape}")
                layer_errors.append({'layer': conv_idx, 'max_err': max_err, 'rmse': rmse})

        # Compare final detections
        cpu_detections = postprocess(cpu_output, original_shape, pad_info)

        print(f"\n  CPU detections ({len(cpu_detections)}):")
        for name, conf, x1, y1, x2, y2 in cpu_detections:
            print(f"    {name}: {conf:.1%} at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

        return {
            'cpu_time': cpu_time,
            'hybrid_time': hybrid_time,
            'layer_errors': layer_errors,
            'cpu_detections': cpu_detections,
        }


def main():
    print("=" * 60)
    print("HYBRID CPU/FPGA INFERENCE PIPELINE")
    print("=" * 60)

    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"

    # Check prerequisites
    for path in [ONNX_PATH]:
        if not os.path.exists(path):
            print(f"\n  ERROR: {path} not found. Run export_onnx.py first.")
            return

    if not os.path.exists(image_path):
        print(f"\n  ERROR: {image_path} not found.")
        return

    # Initialize pipeline with loopback (no FPGA hardware needed)
    print("\n[1/2] Initializing pipeline...")
    pipeline = HybridPipeline(
        onnx_path=ONNX_PATH,
        use_loopback=True,
        verbose=True,
    )

    # Run verification
    print("\n[2/2] Running verification...")
    results = pipeline.verify(image_path)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  CPU inference time:    {results['cpu_time']:.3f}s")
    print(f"  Hybrid inference time: {results['hybrid_time']:.3f}s")

    if results['layer_errors']:
        avg_rmse = np.mean([e['rmse'] for e in results['layer_errors']])
        max_max_err = max(e['max_err'] for e in results['layer_errors'])
        print(f"  Avg RMSE across layers:  {avg_rmse:.6f}")
        print(f"  Worst max error:         {max_max_err:.6f}")
        high_err = [e for e in results['layer_errors'] if e['rmse'] > 0.1]
        if high_err:
            print(f"  Layers with RMSE > 0.1:  {[e['layer'] for e in high_err]}")
        else:
            print(f"  All layers within acceptable error range.")

    print(f"  Detections found: {len(results['cpu_detections'])}")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    pipeline.uart.close()


if __name__ == "__main__":
    main()
