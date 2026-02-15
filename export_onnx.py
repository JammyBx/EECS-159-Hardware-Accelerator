"""
Export YOLO11n to ONNX format.

Converts the PyTorch model to an explicit computational graph where every
operation (Conv, Sigmoid, Add, etc.) is visible as a node. BatchNorm layers
are fused into Conv layers during simplification, reducing the number of
distinct operations the FPGA needs to handle.

Output: yolo11n.onnx in the current directory.

Usage: python3 export_onnx.py
"""

import os
import sys

import onnx
from ultralytics.models import YOLO


def main():
    print("=" * 60)
    print("ONNX EXPORT - YOLO11n")
    print("=" * 60)

    # Load the pre-trained YOLO11n model
    print("\n[1/3] Loading YOLO11n model...")
    try:
        model = YOLO("yolo11n.pt")
        print("  Model loaded.")
    except Exception as e:
        print(f"  ERROR: Could not load model: {e}")
        return

    # Export to ONNX
    # simplify=True fuses BatchNorm into Conv (fewer nodes for FPGA)
    # dynamic=False fixes input shape to [1, 3, 640, 640]
    print("\n[2/3] Exporting to ONNX...")
    try:
        export_path = model.export(format="onnx", simplify=True, dynamic=False, imgsz=640)
        print(f"  Exported to: {export_path}")
    except Exception as e:
        print(f"  ERROR: Export failed: {e}")
        return

    # Verify and summarize the ONNX model
    print("\n[3/3] Verifying ONNX model...")
    onnx_path = export_path if os.path.exists(str(export_path)) else "yolo11n.onnx"

    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX model is valid.")
    except Exception as e:
        print(f"  ERROR: Verification failed: {e}")
        return

    # Print model summary
    graph = onnx_model.graph

    # Input/output shapes
    for inp in graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"\n  Input:  {inp.name} -> shape {dims}")
    for out in graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} -> shape {dims}")

    # Count operations
    total_nodes = len(graph.node)
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print(f"\n  Total nodes: {total_nodes}")
    print(f"  Conv nodes:  {op_counts.get('Conv', 0)}")
    print(f"\n  All operation types:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op}: {count}")

    print(f"\n  ONNX file: {onnx_path}")
    print(f"  File size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
