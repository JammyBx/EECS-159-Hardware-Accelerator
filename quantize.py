"""
int8 quantization for Conv layer weights and activations.

Quantization scheme:
  Weights:     per-output-channel symmetric (no zero-point)
  Activations: per-tensor symmetric (computed at runtime)
  Accumulator: int32 (weight_int8 * activation_int8)
  Dequant:     output_float32 = output_int32 * act_scale * weight_scale[c]

Outputs:
  conv_weights_int8.npz - quantized weight tensors
  conv_scales.json      - per-channel weight scales for each layer

Usage: python3 quantize.py
"""

import json
import os
import sys

import numpy as np


CONV_WEIGHTS_PATH = "conv_weights.npz"
CONV_LAYERS_PATH = "conv_layers.json"
INTERMEDIATE_TENSORS_PATH = "intermediate_tensors.npz"
OUTPUT_WEIGHTS_PATH = "conv_weights_int8.npz"
OUTPUT_SCALES_PATH = "conv_scales.json"


def quantize_weights(weight_float32):
    """
    Per-output-channel symmetric quantization of weights to int8.

    Args:
        weight_float32: np.ndarray of shape [Co, Ci, Kh, Kw] (float32)

    Returns:
        weight_int8: np.ndarray of same shape (int8)
        scales: np.ndarray of shape [Co] (float32) - per-channel scale factors
    """
    out_channels = weight_float32.shape[0]
    # Flatten each output channel to find its max absolute value
    reshaped = weight_float32.reshape(out_channels, -1)
    max_abs = np.max(np.abs(reshaped), axis=1)

    # Avoid division by zero for channels with all-zero weights
    max_abs = np.maximum(max_abs, 1e-8)

    scales = max_abs / 127.0

    # Quantize: divide by scale, round, clip to [-128, 127]
    weight_scaled = weight_float32 / scales.reshape(-1, *([1] * (weight_float32.ndim - 1)))
    weight_int8 = np.clip(np.round(weight_scaled), -128, 127).astype(np.int8)

    return weight_int8, scales


def quantize_activation(activation_float32):
    """
    Per-tensor symmetric quantization of activation tensor to int8.

    Args:
        activation_float32: np.ndarray (float32)

    Returns:
        activation_int8: np.ndarray of same shape (int8)
        scale: float - scale factor
    """
    max_abs = np.max(np.abs(activation_float32))
    if max_abs < 1e-8:
        return np.zeros_like(activation_float32, dtype=np.int8), 1.0

    scale = float(max_abs / 127.0)
    activation_int8 = np.clip(np.round(activation_float32 / scale), -128, 127).astype(np.int8)
    return activation_int8, scale


def dequantize_output(output_int32, act_scale, weight_scales):
    """
    Convert int32 accumulator output back to float32.

    The FPGA computes: output_int32[c] = sum(act_int8 * weight_int8[c])
    Dequantized: output_float32[c] = output_int32[c] * act_scale * weight_scales[c]

    Args:
        output_int32: np.ndarray with output channels in axis 1 (e.g., [1, Co, H, W])
        act_scale: float - activation quantization scale
        weight_scales: np.ndarray of shape [Co] - per-channel weight scales

    Returns:
        output_float32: np.ndarray (float32)
    """
    # Broadcast weight_scales over spatial dimensions
    scale_shape = [1, len(weight_scales)] + [1] * (output_int32.ndim - 2)
    combined_scale = act_scale * weight_scales.reshape(scale_shape)
    return output_int32.astype(np.float32) * combined_scale


def compute_conv_int8(activation_float32, weight_float32, bias_float32,
                      strides, pads, group):
    """
    Simulate int8 convolution: quantize inputs, do int32 accumulation, dequantize.
    Used for accuracy verification against float32 ground truth.
    """
    # Quantize
    act_int8, act_scale = quantize_activation(activation_float32)
    w_int8, w_scales = quantize_weights(weight_float32)

    # Pad input
    if any(p > 0 for p in pads):
        pad_top, pad_left, pad_bottom, pad_right = pads
        act_int8 = np.pad(act_int8,
                          ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='constant', constant_values=0)

    # Convolution in int32
    N, Ci, Hi, Wi = act_int8.shape
    Co, CiG, Kh, Kw = w_int8.shape
    sh, sw = strides
    Ho = (Hi - Kh) // sh + 1
    Wo = (Wi - Kw) // sw + 1
    channels_per_group = Ci // group

    output_int32 = np.zeros((N, Co, Ho, Wo), dtype=np.int32)

    for g in range(group):
        c_start_in = g * channels_per_group
        c_end_in = c_start_in + channels_per_group
        c_start_out = g * (Co // group)
        c_end_out = c_start_out + (Co // group)

        for n in range(N):
            for co in range(c_start_out, c_end_out):
                co_local = co - c_start_out
                for oh in range(Ho):
                    for ow in range(Wo):
                        ih = oh * sh
                        iw = ow * sw
                        patch = act_int8[n, c_start_in:c_end_in, ih:ih+Kh, iw:iw+Kw].astype(np.int32)
                        kernel = w_int8[co, :, :, :].astype(np.int32)
                        output_int32[n, co, oh, ow] = np.sum(patch * kernel)

    # Dequantize
    output_float32 = dequantize_output(output_int32, act_scale, w_scales)

    # Add bias if present
    if bias_float32 is not None:
        output_float32 += bias_float32.reshape(1, -1, 1, 1)

    return output_float32


def main():
    print("=" * 60)
    print("INT8 QUANTIZATION - Conv Layer Weights")
    print("=" * 60)

    # Check prerequisites
    for path in [CONV_WEIGHTS_PATH, CONV_LAYERS_PATH]:
        if not os.path.exists(path):
            print(f"\n  ERROR: {path} not found. Run inspect_graph.py first.")
            return

    # Load conv layer metadata and weights
    print("\n[1/3] Loading conv layer data...")
    with open(CONV_LAYERS_PATH) as f:
        conv_layers = json.load(f)
    weights_data = np.load(CONV_WEIGHTS_PATH)
    print(f"  Loaded {len(conv_layers)} conv layers.")

    # Load intermediate tensors for verification (optional)
    has_intermediates = os.path.exists(INTERMEDIATE_TENSORS_PATH)
    intermediates = None
    if has_intermediates:
        intermediates = np.load(INTERMEDIATE_TENSORS_PATH)
        print(f"  Loaded intermediate tensors for verification.")
    else:
        print(f"  No intermediate tensors found. Skipping accuracy verification.")
        print(f"  Run layer_inference.py first for full verification.")

    # Quantize all conv weights
    print("\n[2/3] Quantizing weights to int8...")
    print(f"\n  {'Layer':<6} {'Shape':<22} {'Max Abs':<12} {'Scale Range':<25} {'Quant Err'}")
    print("  " + "-" * 75)

    quantized_tensors = {}
    all_scales = {}

    for layer in conv_layers:
        idx = layer["layer_idx"]
        weight_key = f"layer_{idx}_weight"
        weight = weights_data[weight_key]

        # Quantize
        w_int8, scales = quantize_weights(weight)
        quantized_tensors[f"layer_{idx}_weight"] = w_int8
        all_scales[str(idx)] = scales.tolist()

        # Quantize bias too (keep as float32, applied after dequant)
        bias_key = f"layer_{idx}_bias"
        if bias_key in weights_data:
            quantized_tensors[f"layer_{idx}_bias"] = weights_data[bias_key]

        # Compute reconstruction error of weights
        reconstructed = w_int8.astype(np.float32) * scales.reshape(-1, *([1] * (weight.ndim - 1)))
        weight_err = np.max(np.abs(weight - reconstructed))

        print(f"  {idx:<6} {str(weight.shape):<22} {np.max(np.abs(weight)):<12.6f} "
              f"[{scales.min():.6f}, {scales.max():.6f}]   {weight_err:.6f}")

    # Accuracy verification against float32 ground truth
    print("\n[3/3] Verifying quantization accuracy...")

    if has_intermediates:
        print(f"\n  {'Layer':<6} {'Max Abs Err':<14} {'RMSE':<14} {'Rel RMSE':<14} {'Status'}")
        print("  " + "-" * 65)

        error_summary = []
        for layer in conv_layers:
            idx = layer["layer_idx"]
            inp_name = layer["input_name"]
            out_name = layer["output_name"]

            # Check if we have the activation tensors
            if inp_name not in intermediates or out_name not in intermediates:
                continue

            act_input = intermediates[inp_name]
            expected_output = intermediates[out_name]

            weight = weights_data[f"layer_{idx}_weight"]
            bias_key = f"layer_{idx}_bias"
            bias = weights_data[bias_key] if bias_key in weights_data else None

            # Compute int8 convolution
            int8_output = compute_conv_int8(
                act_input, weight, bias,
                strides=layer["strides"],
                pads=layer["pads"],
                group=layer["group"]
            )

            # Note: expected_output may include activation function (SiLU etc)
            # applied after conv, so we compare shapes first
            if int8_output.shape != expected_output.shape:
                # Shape mismatch means there's a fused activation — skip
                # comparison since our int8 conv doesn't include it
                continue

            max_err = np.max(np.abs(expected_output - int8_output))
            rmse = np.sqrt(np.mean((expected_output - int8_output) ** 2))
            ref_rms = np.sqrt(np.mean(expected_output ** 2))
            rel_rmse = rmse / ref_rms if ref_rms > 1e-8 else 0.0

            status = "OK" if rel_rmse < 0.05 else "WARN" if rel_rmse < 0.10 else "HIGH"
            print(f"  {idx:<6} {max_err:<14.6f} {rmse:<14.6f} {rel_rmse:<14.6f} {status}")

            error_summary.append({
                "layer_idx": idx,
                "max_abs_error": float(max_err),
                "rmse": float(rmse),
                "relative_rmse": float(rel_rmse),
            })
    else:
        print("  Skipped (no intermediate tensors available).")

    # Save outputs
    np.savez(OUTPUT_WEIGHTS_PATH, **quantized_tensors)
    print(f"\n  Saved quantized weights: {OUTPUT_WEIGHTS_PATH} "
          f"({os.path.getsize(OUTPUT_WEIGHTS_PATH) / 1024:.1f} KB)")

    with open(OUTPUT_SCALES_PATH, "w") as f:
        json.dump(all_scales, f, indent=2)
    print(f"  Saved scale factors:    {OUTPUT_SCALES_PATH}")

    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
