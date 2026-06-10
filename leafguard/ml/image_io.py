import math
import random
import struct
from io import BytesIO


def _leaf_mask(x, y, size):
    cx = cy = (size - 1) / 2
    nx = (x - cx) / (size * 0.34)
    ny = (y - cy) / (size * 0.46)
    return nx * nx + ny * ny <= 1.0


def generate_leaf_bitmap(label, seed=0, size=48):
    rng = random.Random(seed + sum(ord(ch) for ch in label))
    pixels = []
    for y in range(size):
        row = []
        for x in range(size):
            if not _leaf_mask(x, y, size):
                row.append((236, 232, 210))
                continue
            base = [48 + rng.randint(-8, 8), 128 + rng.randint(-12, 12), 45 + rng.randint(-8, 8)]
            cx = x - size / 2
            cy = y - size / 2
            vein = abs(cx) < 1.2 or abs(cy - 0.35 * math.sin(cx / 3) * size / 10) < 0.9
            if vein:
                base = [94, 165, 82]
            if label == "rust":
                value = (x * 13 + y * 17 + seed * 19) % 41
                if value < 9:
                    base = [139 + rng.randint(-10, 10), 82 + rng.randint(-8, 8), 32]
            elif label == "blight":
                dx = x - size * 0.62
                dy = y - size * 0.42
                if dx * dx + dy * dy < (size * 0.18) ** 2 or (x + y + seed) % 37 < 5:
                    base = [62, 48, 29]
            elif label == "leaf_spot":
                for sx, sy in [(0.35, 0.36), (0.58, 0.62), (0.46, 0.52)]:
                    dx = x - size * sx - rng.randint(-1, 1)
                    dy = y - size * sy - rng.randint(-1, 1)
                    if dx * dx + dy * dy < (size * 0.055) ** 2:
                        base = [45, 36, 24]
            elif label == "senescent_leaf":
                base = [
                    176 + rng.randint(-18, 24),
                    132 + rng.randint(-16, 20),
                    42 + rng.randint(-10, 18),
                ]
                if (x * 7 + y * 11 + seed) % 29 < 4:
                    base = [120 + rng.randint(-12, 16), 82 + rng.randint(-10, 12), 28]
            row.append(tuple(max(0, min(255, channel)) for channel in base))
        pixels.append(row)
    return pixels


def write_bmp(path, pixels):
    path.write_bytes(write_bmp_bytes(pixels))


def write_bmp_bytes(pixels):
    height = len(pixels)
    width = len(pixels[0])
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_data_size = (width * 3 + row_padding) * height
    file_size = 54 + pixel_data_size
    handle = BytesIO()
    handle.write(b"BM")
    handle.write(struct.pack("<IHHI", file_size, 0, 0, 54))
    handle.write(struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, pixel_data_size, 2835, 2835, 0, 0))
    for row in reversed(pixels):
        for red, green, blue in row:
            handle.write(bytes((blue, green, red)))
        handle.write(b"\x00" * row_padding)
    return handle.getvalue()


def read_bmp(path):
    return read_bmp_bytes(path.read_bytes())


def read_bmp_bytes(data):
    handle = BytesIO(data)
    if handle.read(2) != b"BM":
        raise ValueError("Not a BMP file")
    handle.seek(10)
    offset = struct.unpack("<I", handle.read(4))[0]
    header_size = struct.unpack("<I", handle.read(4))[0]
    if header_size != 40:
        raise ValueError("Unsupported BMP header")
    width, height, planes, bits = struct.unpack("<iiHH", handle.read(12))
    if planes != 1 or bits != 24:
        raise ValueError("Only 24-bit BMP files are supported")
    handle.seek(offset)
    row_padding = (4 - (width * 3) % 4) % 4
    rows = []
    for _ in range(height):
        row = []
        for _ in range(width):
            blue, green, red = handle.read(3)
            row.append((red, green, blue))
        handle.read(row_padding)
        rows.append(row)
    return list(reversed(rows))
