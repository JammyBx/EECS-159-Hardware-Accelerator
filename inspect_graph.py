"""
Inspect the ONNX graph and extract all Conv layer metadata and weights.

Parses yolo11n.onnx to catalog every Conv node: kernel size, stride, padding,
groups, channels, weight/bias tensors. This is the specification sheet for the
FPGA implementation.

Outputs:
  conv_layers.json  - metadata for every Conv layer (no tensor data)
  conv_weights.npz  - weight and bias tensors keyed by layer index

Usage: python3 inspect_graph.py
"""

import json
import os
import sys

import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper


ONNX_PATH = "yolo11n.onnx"
BRAM_BUDGET_BYTES = 225 * 1024  # Basys3: ~225 KB block RAM


def get_attribute(node, name, default=None):
    """Extract a named attribute from an ONNX node."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.FLOATS:
                return list(attr.floats)
    return default


def main():
    print("=" * 60)
    print("ONNX GRAPH INSPECTION - Conv Layer Extraction")
    print("=" * 60)

    if not os.path.exists(ONNX_PATH):
        print(f"\n  ERROR: {ONNX_PATH} not found.")
        print("  Run export_onnx.py first.")
        return

    print(f"\n  Loading {ONNX_PATH}...")
    model = onnx.load(ONNX_PATH)
    graph = model.graph

    # Build initializer lookup: name -> numpy array
    initializers = {}
    for init in graph.initializer:
        initializers[init.name] = numpy_helper.to_array(init)

    # Extract Conv layer info
    conv_layers = []
    weight_tensors = {}
    total_params = 0

    print(f"\n  Scanning {len(graph.node)} nodes for Conv operations...\n")
    print(f"  {'Idx':<5} {'Name':<30} {'Kernel':<8} {'Stride':<8} {'Pad':<12} "
          f"{'Group':<6} {'In Ch':<7} {'Out Ch':<7} {'Params':<10} {'Weight KB':<10} {'BRAM?'}")
    print("  " + "-" * 115)

    for node_idx, node in enumerate(graph.node):
        if node.op_type != "Conv":
            continue

        layer_idx = len(conv_layers)

        # Extract attributes
        kernel_shape = get_attribute(node, "kernel_shape", [1, 1])
        strides = get_attribute(node, "strides", [1, 1])
        pads = get_attribute(node, "pads", [0, 0, 0, 0])
        group = get_attribute(node, "group", 1)
        dilations = get_attribute(node, "dilations", [1, 1])

        # Get weight tensor
        weight_name = node.input[1]
        weight = initializers.get(weight_name)
        if weight is None:
            print(f"  WARNING: No weight initializer found for {weight_name}")
            continue

        # Weight shape: [out_channels, in_channels/groups, kH, kW]
        out_channels = weight.shape[0]
        in_channels = weight.shape[1] * group
        param_count = weight.size

        # Get bias tensor (optional)
        bias = None
        bias_name = None
        if len(node.input) > 2 and node.input[2] != "":
            bias_name = node.input[2]
            bias = initializers.get(bias_name)
            if bias is not None:
                param_count += bias.size

        total_params += param_count

        # Weight size in bytes (int8 after quantization)
        weight_bytes = weight.size  # 1 byte per int8
        fits_bram = "YES" if weight_bytes <= BRAM_BUDGET_BYTES else "NO"

        # Determine if depthwise
        is_depthwise = group > 1 and group == out_channels

        # Node name (may be empty in some exports)
        name = node.name if node.name else f"node_{node_idx}"

        layer_info = {
            "layer_idx": layer_idx,
            "node_idx": node_idx,
            "node_name": name,
            "input_name": node.input[0],
            "output_name": node.output[0],
            "weight_name": weight_name,
            "bias_name": bias_name,
            "kernel_shape": kernel_shape,
            "strides": strides,
            "pads": pads,
            "group": group,
            "dilations": dilations,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "weight_shape": list(weight.shape),
            "bias_shape": list(bias.shape) if bias is not None else None,
            "param_count": param_count,
            "weight_bytes_int8": weight_bytes,
            "is_depthwise": is_depthwise,
            "fits_bram": weight_bytes <= BRAM_BUDGET_BYTES,
        }
        conv_layers.append(layer_info)

        # Store weight (and bias) tensors
        weight_tensors[f"layer_{layer_idx}_weight"] = weight
        if bias is not None:
            weight_tensors[f"layer_{layer_idx}_bias"] = bias

        # Print summary row
        dw_flag = " (DW)" if is_depthwise else ""
        print(f"  {layer_idx:<5} {name:<30} {str(kernel_shape):<8} {str(strides):<8} "
              f"{str(pads):<12} {group:<6} {in_channels:<7} {out_channels:<7} "
              f"{param_count:<10} {weight_bytes/1024:<10.1f} {fits_bram}{dw_flag}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total Conv layers:    {len(conv_layers)}")
    print(f"  Total Conv params:    {total_params:,}")
    print(f"  Total weight size:    {sum(l['weight_bytes_int8'] for l in conv_layers) / 1024:.1f} KB (int8)")

    depthwise_count = sum(1 for l in conv_layers if l["is_depthwise"])
    standard_count = len(conv_layers) - depthwise_count
    print(f"  Standard Conv layers: {standard_count}")
    print(f"  Depthwise Conv layers: {depthwise_count}")

    bram_overflow = [l for l in conv_layers if not l["fits_bram"]]
    if bram_overflow:
        print(f"\n  WARNING: {len(bram_overflow)} layers exceed Basys3 BRAM budget ({BRAM_BUDGET_BYTES // 1024} KB):")
        for l in bram_overflow:
            print(f"    Layer {l['layer_idx']}: {l['weight_bytes_int8'] / 1024:.1f} KB")
    else:
        print(f"\n  All Conv weight tensors fit in Basys3 BRAM ({BRAM_BUDGET_BYTES // 1024} KB).")

    # Save metadata to JSON
    json_path = "conv_layers.json"
    with open(json_path, "w") as f:
        json.dump(conv_layers, f, indent=2)
    print(f"\n  Saved metadata: {json_path}")

    # Save weight tensors to npz
    npz_path = "conv_weights.npz"
    np.savez(npz_path, **weight_tensors)
    print(f"  Saved weights:  {npz_path} ({os.path.getsize(npz_path) / 1024:.1f} KB)")

    # Print all operation types in the graph for reference
    print(f"\n  All operation types in graph:")
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op}: {count}")

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
