# YOLO11n Vision System with FPGA Acceleration

Real-time object detection using YOLO11n on a Raspberry Pi, with convolutional layer acceleration on a Xilinx FPGA via UART. Detects apples vs oranges (and 78 other COCO object classes) using a pre-trained model.

## Project Structure

```
setup_pi.py / setupPi.py                           One-time environment setup
vision_system_static.py / visionSystemStatic.py     Static image detection
vision_system.py / visionSystem.py                  Live webcam detection
export_onnx.py                                      PyTorch to ONNX conversion
inspect_graph.py                                    ONNX graph parsing and Conv extraction
layer_inference.py                                  Intermediate tensor capture
quantize.py                                         int8 quantization
uart_comm.py                                        UART protocol for FPGA communication
fpga_pipeline.py                                    Hybrid CPU/FPGA inference pipeline
```

Each original script has two versions: a `snake_case` version with detailed comments and a `camelCase` version that is more concise. They are functionally identical.

## Prerequisites

- Python 3.8+
- Raspberry Pi with USB webcam (for live detection)
- Xilinx FPGA board (Basys3/Zynq) connected via UART (for FPGA acceleration)

Run the setup script first to install all dependencies:

```
python3 setup_pi.py
```

This installs: `opencv-python`, `ultralytics`, `onnx`, `onnxruntime`, `pyserial`.

## Run Order

```
python3 setup_pi.py               # 1. Install dependencies (run once)
python3 vision_system_static.py   # 2. Test detection on a photo (no camera needed)
python3 vision_system.py          # 3. Test live camera detection
python3 export_onnx.py            # 4. Export model to ONNX
python3 inspect_graph.py          # 5. Extract Conv layer specs and weights
python3 layer_inference.py        # 6. Capture intermediate tensors
python3 quantize.py               # 7. Quantize weights to int8
python3 uart_comm.py              # 8. Self-test UART protocol (loopback)
python3 fpga_pipeline.py          # 9. Run full hybrid pipeline verification
```

---

## File-by-File Walkthrough

### 1. setup_pi.py -- One-Time Environment Setup

**Run once:** `python3 setup_pi.py`

This script prepares the Raspberry Pi before any detection or FPGA work.

**Flow:**

1. **Check Python version** -- verifies Python 3.8+ is installed (YOLO11 requires it). If too old, stops immediately.
2. **Install pip packages** -- runs `pip install --upgrade` for each dependency: `opencv-python`, `ultralytics`, `onnx`, `onnxruntime`, `pyserial`. If any fails, stops.
3. **Test camera** -- opens `cv2.VideoCapture(0)` to check if a USB webcam responds. Reports resolution if found. Non-blocking -- failure here just means you can still test with static images.
4. **Download sample image** -- downloads `bus.jpg` from ultralytics.com and saves it as `test_image.jpg`. Skips if the file already exists.

---

### 2. vision_system_static.py -- Static Image Detection

**Run:** `python3 vision_system_static.py`

Tests YOLO11n object detection on a single photo. No camera needed.

**Flow:**

1. **Load YOLO11n model** -- `YOLO('yolo11n.pt')`. Downloads the ~6MB weights file on first run.
2. **Load test image** -- `cv2.imread('test_image.jpg')`. Reads the image as a NumPy matrix of BGR pixel values. Reports dimensions (height x width). Can change the image by modifying the `image_path` variable.
3. **Run inference** -- `model(frame, verbose=False)`. This single line does everything: resizes the image to 640x640 internally, runs the neural network forward pass, and returns bounding boxes with class IDs and confidence scores.
4. **Draw results** -- `results[0].plot()` takes the original photo and draws colored bounding boxes around detected objects, plus text labels like "person 95%" next to each box. The result is called the "annotated image" -- your original photo with detection boxes overlaid on top.
5. **Display and save** -- `cv2.imshow()` opens a window on screen to display the annotated image. Waits for any keypress to close. Saves the result to `detection_result.jpg`.
6. **Print detections** -- loops through `results[0].boxes`, extracts each detection's class name (e.g., "person", "car") and confidence score (e.g., 95.3%), and prints them.

No machine learning happens in this file. The model was already trained by Ultralytics on millions of images. This script only does inference -- feeding a new image through the finished model to get predictions.

---

### 3. vision_system.py -- Live Webcam Detection

**Run:** `python3 vision_system.py`

Same detection pipeline as the static version, but running continuously on live video.

**Flow:**

1. **Load model** -- same `YOLO('yolo11n.pt')`.
2. **Open camera** -- `cv2.VideoCapture(0)`, requests 640x480 resolution, reads back the actual resolution the camera provides.
3. **Main loop** (`while True`):
   - **Capture frame** -- `cap.read()` grabs one frame from the camera. A video is just a sequence of still images shown rapidly (like a flip book). If the camera disconnects, breaks out of the loop.
   - **Run inference** -- `model(frame, verbose=False)` -- same as the static version, but now running on every frame (~30 times per second).
   - **Annotate** -- `results[0].plot()` draws bounding boxes and labels on the frame.
   - **Info overlay** -- `add_info_overlay()` draws a semi-transparent black box in the corner showing "System: ACTIVE", the current frame count, and the model name.
   - **Display** -- `cv2.imshow()` shows the live annotated video feed.
   - **Keyboard check** -- `cv2.waitKey(1)` pauses for 1 millisecond and checks if a key was pressed. Without this call, the video window would not render at all (OpenCV requires `waitKey` to process window events). The controls are:
     - `q` -- breaks out of the loop, ending the program
     - `s` -- saves the current annotated frame as a JPEG screenshot
     - `i` -- toggles the info overlay on/off
4. **Cleanup** -- `cap.release()` tells the OS to release the camera hardware. If you skip this, the camera stays locked by your program even after it exits, and the next time you try to open it you get "camera in use" errors. `cv2.destroyAllWindows()` closes the display windows. The `finally:` block ensures cleanup happens even if the program crashes.

---

### 4. export_onnx.py -- Convert PyTorch Model to ONNX

**Run:** `python3 export_onnx.py`

**Requires:** `onnx` package installed (handled by `setup_pi.py`).

The PyTorch `.pt` model is a black box -- you call `model(frame)` and get results, but you cannot see the individual operations inside. ONNX (Open Neural Network Exchange) cracks it open into an explicit computational graph where every operation is a visible node.

**Flow:**

1. **Load model** -- `YOLO('yolo11n.pt')`.
2. **Export** -- `model.export(format='onnx', simplify=True, dynamic=False, imgsz=640)`.
   - `simplify=True` is critical: it fuses BatchNorm into Conv layers. Normally BatchNorm is a separate operation after Conv that normalizes activations. Fusing it bakes the normalization constants directly into the Conv weights and biases, so the FPGA only needs to implement Conv, not Conv + BatchNorm separately.
   - `dynamic=False` locks the input shape to `[1, 3, 640, 640]` (batch=1, 3 color channels, 640x640 pixels).
3. **Verify** -- loads the `.onnx` file back and runs `onnx.checker.check_model()` to validate the graph structure is well-formed.
4. **Print summary** -- reports input shape `[1, 3, 640, 640]`, output shape `[1, 84, 8400]`, total node count, and a breakdown of every operation type in the graph (Conv, Sigmoid, Mul, Add, Concat, MaxPool, Resize, etc.).

**Output:** `yolo11n.onnx` -- the full computational graph as a file.

---

### 5. inspect_graph.py -- Extract Every Conv Layer's Specs and Weights

**Run:** `python3 inspect_graph.py`

**Requires:** `yolo11n.onnx` (run `export_onnx.py` first).

This is the FPGA specification sheet generator. It parses the ONNX graph and extracts everything the FPGA needs to know about each convolutional layer.

**Flow:**

1. **Load ONNX model**, build an initializer lookup. The "initializers" are all the pre-trained weight tensors stored in the ONNX file. Creates a dictionary mapping `tensor_name -> numpy_array`.
2. **Scan every node** in the graph. For each `Conv` node:
   - Extract attributes: `kernel_shape` (e.g., [3,3] or [1,1]), `strides` (e.g., [1,1] or [2,2]), `pads` (e.g., [1,1,1,1]), `group` (1 for standard convolution, greater than 1 for depthwise), `dilations`.
   - Look up the weight tensor from initializers via `node.input[1]`. Shape is `[out_channels, in_channels/groups, Kh, Kw]` -- this is the actual trained convolution kernel.
   - Look up the bias tensor from `node.input[2]` if present.
   - Record the input and output tensor names -- these are the edges connecting this Conv to the rest of the graph. Essential for knowing which tensor to feed in and where the output goes next.
   - Flag depthwise convolutions (where `group == out_channels`) -- these are computed differently and may require special FPGA logic.
   - Check if the weight tensor fits in Basys3 BRAM (225 KB budget).
3. **Print summary table** -- shows every Conv layer's index, kernel shape, stride, padding, group, input/output channels, parameter count, weight size in KB, and whether it fits in BRAM.
4. **Save outputs:**
   - `conv_layers.json` -- metadata only (no tensor data). Every Conv layer's kernel shape, stride, padding, channel counts, tensor names, BRAM fit status, etc.
   - `conv_weights.npz` -- the actual float32 weight and bias arrays, keyed as `layer_0_weight`, `layer_0_bias`, `layer_1_weight`, etc.

---

### 6. layer_inference.py -- Capture Every Intermediate Tensor

**Run:** `python3 layer_inference.py [image_path]` (defaults to `test_image.jpg`)

**Requires:** `yolo11n.onnx` (run `export_onnx.py` first). Optionally uses `conv_layers.json` from `inspect_graph.py` for richer output.

This creates ground truth data -- the exact tensor values flowing into and out of every Conv layer for a specific test image. These are used later to verify that the FPGA produces correct results.

**Flow:**

1. **Preprocess image** -- performs the same preprocessing that YOLO does internally:
   - Letterbox resize: scale the image to fit 640x640 while maintaining aspect ratio, pad the remaining space with gray (pixel value 114).
   - BGR to RGB color channel reordering.
   - HWC to CHW (height/width/channels to channels/height/width) -- neural networks expect channels-first format.
   - Normalize pixel values from [0, 255] to [0.0, 1.0] float32.
   - Add batch dimension: final shape `[1, 3, 640, 640]`.
2. **Run baseline inference** -- runs the unmodified ONNX model in onnxruntime. Saves the final output tensor for comparison.
3. **Build debug model** -- ONNX Runtime normally only returns the graph's final output. To see intermediate values:
   - Runs `shape_inference` on the ONNX model to determine the type and shape of every intermediate tensor.
   - Adds every Conv input/output tensor name to the graph's output list.
   - Saves as `yolo11n_debug.onnx`.
   - Now when you run inference, onnxruntime returns all those intermediate tensors alongside the final output.
4. **Run debug inference** -- runs the modified model. Returns ~100+ tensors instead of just 1.
5. **Verify** -- checks that the final output of the debug model exactly matches the baseline. It should, since we only added outputs without changing any computation.
6. **Print value ranges** -- for each Conv layer output, reports min, max, and mean values. Useful for understanding the dynamic range the FPGA needs to handle.
7. **Save** -- `intermediate_tensors.npz` containing every captured tensor, plus the preprocessed input image.

---

### 7. quantize.py -- Convert float32 Weights to int8

**Run:** `python3 quantize.py`

**Requires:** `conv_weights.npz` and `conv_layers.json` (run `inspect_graph.py` first). Optionally uses `intermediate_tensors.npz` (from `layer_inference.py`) for accuracy verification.

The FPGA works with fixed-point integers, not floating-point numbers. This file converts all Conv weights from float32 to int8 and provides functions for runtime activation quantization.

**Quantization scheme:**

- **Weights** (per-output-channel symmetric): For each output channel, find the max absolute value. `scale = max_abs / 127`. `weight_int8 = round(weight / scale)`, clipped to [-128, 127]. Each channel gets its own scale factor.
- **Activations** (per-tensor symmetric, computed at runtime): Same idea but one scale factor for the entire tensor, since activation value ranges are not known until you actually run inference on an image.
- **Dequantization**: The FPGA produces int32 accumulator results (int8 x int8 summed). To convert back: `output_float32 = output_int32 * activation_scale * weight_scale[channel]`.

**Key functions (also imported by other files):**

- `quantize_weights()` -- quantize a float32 weight tensor to int8, return int8 weights + per-channel scales.
- `quantize_activation()` -- quantize a float32 activation tensor to int8, return int8 tensor + scale.
- `dequantize_output()` -- convert int32 FPGA output back to float32 using the scale factors.
- `compute_conv_int8()` -- pure NumPy simulation of what the FPGA will do. Quantizes inputs, performs convolution with int8 x int8 -> int32 accumulation using explicit nested loops, then dequantizes. Used for accuracy verification.

**When run as main:**

1. Loads all float32 weights from `conv_weights.npz`.
2. Quantizes each layer's weights to int8. Reports the quantization reconstruction error (how much precision is lost just from rounding the weights).
3. If `intermediate_tensors.npz` exists, runs the full int8 convolution simulation for each layer and compares the result against the float32 ground truth. Reports max absolute error, RMSE, and relative RMSE per layer. Flags layers above 5% relative error.
4. Saves `conv_weights_int8.npz` (quantized weight tensors) and `conv_scales.json` (per-channel scale factors for each layer).

---

### 8. uart_comm.py -- UART Protocol for FPGA Communication

**Run:** `python3 uart_comm.py` (self-test with loopback, no hardware needed)

Defines the binary protocol for sending int8 tensor data between the Raspberry Pi and Xilinx FPGA over UART.

**Packet format:**

```
| SYNC (2B) | CMD (1B) | LAYER_ID (2B) | ROWS (2B) | COLS (2B) | CH (2B) |
| PAYLOAD_LEN (4B) | CHUNK_IDX (2B) | TOTAL_CHUNKS (2B) | PAYLOAD (N) | CRC16 (2B) |
```

- SYNC: `0xAA 0x55` -- magic bytes for frame alignment (receiver scans for these to find packet boundaries)
- CMD: command type (SEND_CONFIG, SEND_WEIGHTS, SEND_ACTIVATION, COMPUTE, REQUEST_RESULT, RESULT_DATA, ACK, ERROR)
- CRC16: CRC-16/CCITT checksum covering header + payload for data integrity

**Chunking:** Tensors can be hundreds of KB. UART buffers are small. So tensors are split into 1024-byte chunks, each sent as a separate packet. The receiver ACKs each chunk. If no ACK within 500ms, the sender retries (up to 3 times).

**Communication flow for one Conv layer:**

```
Pi -> FPGA: SEND_CONFIG (kernel shape, stride, padding, channels)
FPGA -> Pi: ACK

Pi -> FPGA: SEND_WEIGHTS (chunked int8 weight tensor) -- only on first frame
FPGA -> Pi: ACK per chunk

Pi -> FPGA: SEND_ACTIVATION (chunked int8 input activation)
FPGA -> Pi: ACK per chunk

Pi -> FPGA: COMPUTE
FPGA -> Pi: ACK (when computation is complete)

Pi -> FPGA: REQUEST_RESULT
FPGA -> Pi: RESULT_DATA (chunked int32 output tensor)
Pi -> FPGA: ACK per chunk
```

**Two classes with the same API:**

- `FPGAUart` -- real UART communication via pyserial. Opens a serial port (default `/dev/ttyS0` at 115200 baud, 8N1).
- `FPGAUartLoopback` -- mock FPGA that computes convolutions in NumPy instead of sending data over a wire. Allows full pipeline testing without any FPGA hardware connected.

**Self-test:** When run directly, creates a tiny test convolution (1 input channel, 2 output channels, 3x3 kernel), runs it through the loopback mock, tests packet build/parse with CRC verification, and estimates transfer time for a large tensor at 115200 baud.

---

### 9. fpga_pipeline.py -- Hybrid CPU/FPGA Inference Pipeline

**Run:** `python3 fpga_pipeline.py [image_path]` (defaults to `test_image.jpg`)

**Requires:** `yolo11n.onnx` (from `export_onnx.py`). Optionally uses `conv_weights_int8.npz` and `conv_scales.json` (from `quantize.py`) for pre-quantized weights.

This ties everything together into a single end-to-end system. Conv layers run on the FPGA, everything else runs on the CPU.

**HybridPipeline class initialization:**

1. Loads the ONNX model and identifies all Conv nodes in the graph.
2. Loads pre-quantized int8 weights and scale factors (or quantizes on-the-fly if not available).
3. Creates two onnxruntime sessions: one normal session (for CPU baseline), one debug session (with all intermediate tensors exposed as outputs).
4. Initializes UART communication (real serial or loopback mock).
5. Pre-loads weights to the FPGA -- sends every Conv layer's configuration and int8 weights at startup. Weights are static and only need to be sent once, not per frame.

**run_cpu():**

Runs full inference on CPU via onnxruntime. This is the baseline to compare against.

**run_hybrid():**

The core of the FPGA acceleration:

1. Runs the debug session on CPU to get all intermediate tensors (needed to get the correct Conv input activations).
2. For each Conv layer in the graph:
   - Gets the input activation tensor from the CPU debug run.
   - Quantizes the activation to int8.
   - Sends it to the FPGA via UART.
   - Triggers COMPUTE.
   - Receives the int32 result from the FPGA.
   - Dequantizes back to float32.
   - Adds bias (bias is kept in float32, applied after dequantization).
   - Compares the FPGA result against the CPU-computed result for that layer.
3. Returns both the CPU baseline final output and per-layer FPGA comparison results.

**preprocess():**

Same letterbox resize as `layer_inference.py`. Also returns padding info needed to undo the transform when mapping output boxes back to original image coordinates.

**postprocess():**

Decodes the raw model output `(1, 84, 8400)` into actual object detections:

- The output contains 8400 candidate detection boxes.
- Each candidate has 4 box coordinates (center_x, center_y, width, height) plus 80 class scores (one per COCO class).
- Filters candidates by confidence threshold (0.25) -- discards low-confidence guesses.
- Converts center-format box coordinates to corner-format (x1, y1, x2, y2).
- Undoes the letterbox padding and scaling to map boxes back to original image pixel coordinates.
- Runs NMS (non-maximum suppression) -- when multiple overlapping boxes detect the same object, keeps only the most confident one. Uses an IoU (intersection over union) threshold of 0.45.

**verify():**

Runs both CPU and hybrid inference on the same image, prints a per-layer error comparison table (max error, RMSE for each Conv layer), and prints the final detected objects.

---

## End-to-End Data Flow

```
test_image.jpg
    |
    v
[preprocess] --- letterbox resize, normalize, HWC->NCHW
    |
    |  float32 [1, 3, 640, 640]
    v
+---------------------------------------------+
|  ONNX Graph (node by node, topologically)   |
|                                             |
|  Non-Conv node -> CPU (onnxruntime)         |
|  Conv node -> quantize act -> UART -> FPGA  |
|              FPGA computes int8 conv        |
|              UART <- int32 result           |
|              dequantize -> float32          |
|              + bias -> next node            |
+---------------------------------------------+
    |
    |  float32 [1, 84, 8400]
    v
[postprocess] --- decode boxes, filter confidence, NMS
    |
    v
Detections: [("apple", 0.92, x1, y1, x2, y2), ("orange", 0.87, ...)]
```

## Generated Output Files

| File | Created By | Description |
|------|-----------|-------------|
| yolo11n.onnx | export_onnx.py | Full ONNX computational graph |
| conv_layers.json | inspect_graph.py | Conv layer metadata (kernel, stride, channels, etc.) |
| conv_weights.npz | inspect_graph.py | Float32 weight/bias tensors for all Conv layers |
| yolo11n_debug.onnx | layer_inference.py | ONNX model with intermediate outputs exposed |
| intermediate_tensors.npz | layer_inference.py | Ground truth activation tensors for verification |
| conv_weights_int8.npz | quantize.py | Quantized int8 weight tensors |
| conv_scales.json | quantize.py | Per-channel weight scale factors for dequantization |

## Performance Note

At 115200 baud UART, transferring the largest activation tensor (~800 KB for a 128x80x80 int8 tensor) takes approximately 70 seconds. This phase of the project is for correctness verification, not real-time performance. Future optimizations include higher baud rates (921600+), SPI or AXI interfaces, and on Zynq boards, AXI DMA for GB/s throughput.
