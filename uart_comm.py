"""
UART communication module for FPGA tensor transfer.

Protocol for sending/receiving int8 tensors between Raspberry Pi and Xilinx FPGA.
Includes chunked transfer, CRC-16 integrity checks, and ACK/retry logic.

Packet format:
  | SYNC (2B) | CMD (1B) | LAYER_ID (2B) | ROWS (2B) | COLS (2B) | CH (2B) |
  | PAYLOAD_LEN (4B) | CHUNK_IDX (2B) | TOTAL_CHUNKS (2B) | PAYLOAD (N) | CRC16 (2B) |

Classes:
  FPGAUart         - real UART communication via pyserial
  FPGAUartLoopback - mock that computes conv in NumPy (no hardware needed)

Usage:
  from uart_comm import FPGAUart, FPGAUartLoopback
  comm = FPGAUartLoopback()  # or FPGAUart('/dev/ttyS0')
"""

import math
import struct
import time

import numpy as np


# Protocol constants
SYNC_BYTES = b'\xAA\x55'
CHUNK_SIZE = 1024  # bytes of payload per packet
HEADER_SIZE = 19   # fixed header bytes (before payload)
ACK_TIMEOUT = 0.5  # seconds
MAX_RETRIES = 3

# Command types
CMD_SEND_CONFIG = 0x01
CMD_SEND_WEIGHTS = 0x02
CMD_SEND_ACTIVATION = 0x03
CMD_COMPUTE = 0x04
CMD_REQUEST_RESULT = 0x05
CMD_RESULT_DATA = 0x06
CMD_ACK = 0x07
CMD_ERROR = 0xFF

CMD_NAMES = {
    CMD_SEND_CONFIG: "SEND_CONFIG",
    CMD_SEND_WEIGHTS: "SEND_WEIGHTS",
    CMD_SEND_ACTIVATION: "SEND_ACTIVATION",
    CMD_COMPUTE: "COMPUTE",
    CMD_REQUEST_RESULT: "REQUEST_RESULT",
    CMD_RESULT_DATA: "RESULT_DATA",
    CMD_ACK: "ACK",
    CMD_ERROR: "ERROR",
}


def crc16_ccitt(data):
    """Compute CRC-16/CCITT checksum."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def build_packet(cmd, layer_id, tensor_shape, payload, chunk_idx, total_chunks):
    """
    Assemble a protocol packet.

    Args:
        cmd: command byte
        layer_id: conv layer index
        tensor_shape: (rows, cols, channels) or (0,0,0) for non-tensor commands
        payload: bytes payload (up to CHUNK_SIZE)
        chunk_idx: current chunk index
        total_chunks: total number of chunks

    Returns:
        bytes: complete packet including sync and CRC
    """
    rows, cols, ch = tensor_shape
    payload_len = len(payload)

    # Pack header (everything after SYNC, before payload)
    header = struct.pack('<BHHHHIHH',
                         cmd,
                         layer_id,
                         rows, cols, ch,
                         payload_len,
                         chunk_idx,
                         total_chunks)

    # CRC covers header + payload
    crc_data = header + payload
    crc = crc16_ccitt(crc_data)

    return SYNC_BYTES + header + payload + struct.pack('<H', crc)


def parse_packet(data):
    """
    Parse a received packet.

    Args:
        data: bytes starting from after SYNC detection

    Returns:
        dict with cmd, layer_id, rows, cols, ch, payload, chunk_idx, total_chunks
        or None if CRC fails
    """
    if len(data) < HEADER_SIZE - 2:  # -2 for SYNC already consumed
        return None

    header_bytes = data[:17]  # 17 bytes of header after SYNC
    cmd, layer_id, rows, cols, ch, payload_len, chunk_idx, total_chunks = \
        struct.unpack('<BHHHHIHH', header_bytes)

    if len(data) < 17 + payload_len + 2:
        return None

    payload = data[17:17 + payload_len]
    received_crc = struct.unpack('<H', data[17 + payload_len:17 + payload_len + 2])[0]

    # Verify CRC
    expected_crc = crc16_ccitt(header_bytes + payload)
    if received_crc != expected_crc:
        return None

    return {
        'cmd': cmd,
        'layer_id': layer_id,
        'rows': rows,
        'cols': cols,
        'ch': ch,
        'payload': payload,
        'chunk_idx': chunk_idx,
        'total_chunks': total_chunks,
    }


class FPGAUart:
    """Real UART communication with FPGA via pyserial."""

    def __init__(self, port='/dev/ttyS0', baudrate=115200, timeout=1.0, verbose=False):
        import serial
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE)
        self.verbose = verbose
        print(f"  UART opened: {port} @ {baudrate} baud")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def send_layer_config(self, layer_id, kernel_shape, strides, pads, group,
                          in_channels, out_channels):
        """Send conv layer configuration to FPGA."""
        # Pack config into payload
        payload = struct.pack('<HHHHHHHHH',
                              kernel_shape[0], kernel_shape[1],
                              strides[0], strides[1],
                              pads[0], pads[1],  # only top/left (symmetric assumed)
                              group,
                              in_channels, out_channels)
        packet = build_packet(CMD_SEND_CONFIG, layer_id, (0, 0, 0), payload, 0, 1)
        self._send_and_ack(packet, layer_id, "CONFIG")

    def send_weights(self, layer_id, weights_int8):
        """Send quantized weight tensor in chunks."""
        self._send_tensor_chunked(CMD_SEND_WEIGHTS, layer_id, weights_int8)

    def send_activation(self, layer_id, activation_int8):
        """Send quantized input activation tensor in chunks."""
        self._send_tensor_chunked(CMD_SEND_ACTIVATION, layer_id, activation_int8)

    def compute(self, layer_id):
        """Trigger convolution computation on FPGA, wait for completion."""
        packet = build_packet(CMD_COMPUTE, layer_id, (0, 0, 0), b'', 0, 1)
        self._send_and_ack(packet, layer_id, "COMPUTE")

    def receive_result(self, layer_id, expected_shape):
        """Request and receive output tensor (int32) from FPGA."""
        # Send request
        packet = build_packet(CMD_REQUEST_RESULT, layer_id, (0, 0, 0), b'', 0, 1)
        self.ser.write(packet)

        # Receive chunked result
        total_bytes = int(np.prod(expected_shape)) * 4  # int32 = 4 bytes
        total_chunks = math.ceil(total_bytes / CHUNK_SIZE)
        received_data = bytearray()

        for chunk_idx in range(total_chunks):
            pkt = self._receive_packet_from_serial()
            if pkt is None:
                raise RuntimeError(f"Failed to receive chunk {chunk_idx}/{total_chunks}")
            received_data.extend(pkt['payload'])

            # Send ACK for this chunk
            ack = build_packet(CMD_ACK, layer_id, (0, 0, 0), b'\x00', chunk_idx, total_chunks)
            self.ser.write(ack)

        result = np.frombuffer(received_data[:total_bytes], dtype=np.int32)
        return result.reshape(expected_shape)

    def _send_tensor_chunked(self, cmd, layer_id, tensor):
        """Send a tensor in chunks with per-chunk ACK."""
        raw = tensor.tobytes()
        total_chunks = math.ceil(len(raw) / CHUNK_SIZE)
        shape = tensor.shape

        # Determine tensor_shape tuple for header (rows, cols, channels)
        if len(shape) == 4:
            tensor_shape = (shape[2], shape[3], shape[1])  # H, W, C
        elif len(shape) == 1:
            tensor_shape = (shape[0], 1, 1)
        else:
            tensor_shape = (shape[0] if len(shape) > 0 else 0,
                            shape[1] if len(shape) > 1 else 0,
                            shape[2] if len(shape) > 2 else 0)

        if self.verbose:
            print(f"    Sending {CMD_NAMES.get(cmd, '?')} layer {layer_id}: "
                  f"{len(raw)} bytes in {total_chunks} chunks")

        for i in range(total_chunks):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(raw))
            chunk = raw[start:end]
            packet = build_packet(cmd, layer_id, tensor_shape, chunk, i, total_chunks)
            self._send_and_ack(packet, layer_id, f"chunk {i}/{total_chunks}")

    def _send_and_ack(self, packet, layer_id, desc=""):
        """Send a packet and wait for ACK with retries."""
        for attempt in range(MAX_RETRIES):
            self.ser.write(packet)
            ack = self._receive_packet_from_serial()
            if ack and ack['cmd'] == CMD_ACK:
                return
            if self.verbose:
                print(f"    Retry {attempt + 1}/{MAX_RETRIES} for {desc}")
        raise RuntimeError(f"No ACK received for layer {layer_id} ({desc})")

    def _receive_packet_from_serial(self):
        """Read a complete packet from serial, starting with SYNC detection."""
        # Wait for sync bytes
        deadline = time.time() + ACK_TIMEOUT
        while time.time() < deadline:
            b = self.ser.read(1)
            if b == b'\xAA':
                b2 = self.ser.read(1)
                if b2 == b'\x55':
                    # Read header (17 bytes after SYNC)
                    header = self.ser.read(17)
                    if len(header) < 17:
                        continue
                    # Parse payload length
                    payload_len = struct.unpack('<I', header[9:13])[0]
                    # Read payload + CRC
                    rest = self.ser.read(payload_len + 2)
                    if len(rest) < payload_len + 2:
                        continue
                    return parse_packet(header + rest)
        return None


class FPGAUartLoopback:
    """
    Mock FPGA that computes convolutions in NumPy.
    Allows full pipeline testing without FPGA hardware.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.configs = {}      # layer_id -> config dict
        self.weights = {}      # layer_id -> int8 weight tensor
        self.activations = {}  # layer_id -> int8 activation tensor
        print("  UART Loopback mode (no hardware)")

    def close(self):
        pass

    def send_layer_config(self, layer_id, kernel_shape, strides, pads, group,
                          in_channels, out_channels):
        self.configs[layer_id] = {
            'kernel_shape': kernel_shape,
            'strides': strides,
            'pads': pads,
            'group': group,
            'in_channels': in_channels,
            'out_channels': out_channels,
        }
        if self.verbose:
            print(f"    [Loopback] Config layer {layer_id}: "
                  f"k={kernel_shape} s={strides} p={pads} g={group}")

    def send_weights(self, layer_id, weights_int8):
        self.weights[layer_id] = weights_int8.copy()
        if self.verbose:
            print(f"    [Loopback] Weights layer {layer_id}: shape={weights_int8.shape}")

    def send_activation(self, layer_id, activation_int8):
        self.activations[layer_id] = activation_int8.copy()
        if self.verbose:
            print(f"    [Loopback] Activation layer {layer_id}: shape={activation_int8.shape}")

    def compute(self, layer_id):
        """Compute convolution in NumPy, simulating what the FPGA would do."""
        if self.verbose:
            print(f"    [Loopback] Computing conv layer {layer_id}...")

    def receive_result(self, layer_id, expected_shape):
        """Compute and return the convolution result."""
        config = self.configs[layer_id]
        w = self.weights[layer_id]
        act = self.activations[layer_id]

        # Apply padding
        pads = config['pads']
        if any(p > 0 for p in pads):
            act = np.pad(act,
                         ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])),
                         mode='constant', constant_values=0)

        N, Ci, Hi, Wi = act.shape
        Co, CiG, Kh, Kw = w.shape
        sh, sw = config['strides']
        group = config['group']
        Ho = (Hi - Kh) // sh + 1
        Wo = (Wi - Kw) // sw + 1
        channels_per_group = Ci // group

        output = np.zeros((N, Co, Ho, Wo), dtype=np.int32)

        for g in range(group):
            c_in_start = g * channels_per_group
            c_in_end = c_in_start + channels_per_group
            c_out_start = g * (Co // group)
            c_out_end = c_out_start + (Co // group)

            for n in range(N):
                for co in range(c_out_start, c_out_end):
                    for oh in range(Ho):
                        for ow in range(Wo):
                            ih = oh * sh
                            iw = ow * sw
                            patch = act[n, c_in_start:c_in_end, ih:ih+Kh, iw:iw+Kw].astype(np.int32)
                            kernel = w[co].astype(np.int32)
                            output[n, co, oh, ow] = np.sum(patch * kernel)

        if self.verbose:
            print(f"    [Loopback] Result layer {layer_id}: shape={output.shape}")

        return output


def estimate_transfer_time(tensor, baudrate=115200):
    """Estimate UART transfer time for a tensor."""
    num_bytes = tensor.nbytes
    effective_bps = baudrate / 10  # 8N1 encoding: 10 bits per byte
    overhead = math.ceil(num_bytes / CHUNK_SIZE) * (HEADER_SIZE + 2)  # headers + CRC
    total_bytes = num_bytes + overhead
    seconds = total_bytes / effective_bps
    return seconds


if __name__ == "__main__":
    # Quick self-test with loopback
    print("=" * 60)
    print("UART MODULE SELF-TEST (Loopback)")
    print("=" * 60)

    comm = FPGAUartLoopback(verbose=True)

    # Create a small test conv: 1 input channel, 2 output channels, 3x3 kernel
    layer_id = 0
    comm.send_layer_config(layer_id,
                           kernel_shape=[3, 3], strides=[1, 1],
                           pads=[1, 1, 1, 1], group=1,
                           in_channels=1, out_channels=2)

    # Random test data
    weights = np.random.randint(-128, 127, (2, 1, 3, 3), dtype=np.int8)
    activation = np.random.randint(-128, 127, (1, 1, 8, 8), dtype=np.int8)

    comm.send_weights(layer_id, weights)
    comm.send_activation(layer_id, activation)
    comm.compute(layer_id)
    result = comm.receive_result(layer_id, (1, 2, 8, 8))

    print(f"\n  Input shape:  {activation.shape}")
    print(f"  Weight shape: {weights.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Output dtype: {result.dtype}")
    print(f"  Output range: [{result.min()}, {result.max()}]")

    # Test packet build/parse
    print("\n  Testing packet build/parse...")
    pkt = build_packet(CMD_SEND_WEIGHTS, 5, (8, 8, 1), weights.tobytes()[:CHUNK_SIZE], 0, 1)
    parsed = parse_packet(pkt[2:])  # skip SYNC
    assert parsed is not None, "CRC verification failed"
    assert parsed['cmd'] == CMD_SEND_WEIGHTS
    assert parsed['layer_id'] == 5
    print("  PASS: Packet build/parse OK")

    # Estimate transfer time
    big_tensor = np.zeros((1, 128, 80, 80), dtype=np.int8)
    t = estimate_transfer_time(big_tensor)
    print(f"\n  Transfer estimate for (1,128,80,80) int8 @ 115200 baud: {t:.1f}s")

    print("\n" + "=" * 60)
    print("SELF-TEST COMPLETE")
    print("=" * 60)
