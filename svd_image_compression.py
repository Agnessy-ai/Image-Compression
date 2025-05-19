import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def compress_image_svd(image_array, k):
    """Compress a single-channel image using SVD."""
    U, S, VT = np.linalg.svd(image_array, full_matrices=False)
    S_k = np.diag(S[:k])
    compressed = np.dot(U[:, :k], np.dot(S_k, VT[:k, :]))
    return compressed

def compress_rgb_image(image_path, k):
    """Apply SVD compression to each RGB channel."""
    original = Image.open(image_path)
    original = original.convert("RGB")
    img_array = np.array(original, dtype=float)

    compressed_channels = []
    for channel in range(3):
        compressed = compress_image_svd(img_array[:, :, channel], k)
        compressed_channels.append(compressed)

    compressed_img = np.stack(compressed_channels, axis=2).astype(np.uint8)
    return original, Image.fromarray(compressed_img)

def show_comparison(original, compressed, k):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Compressed Image (k = {k})")
    plt.imshow(compressed)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ----------- RUN HERE ----------
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    # Ask user to select image file
    tk.Tk().withdraw()
    image_path = filedialog.askopenfilename(title="Select an Image")

    if image_path:
        k = int(input("Enter number of singular values to retain (e.g., 50): "))
        original, compressed = compress_rgb_image(image_path, k)

        # Show comparison
        show_comparison(original, compressed, k)

        # Save output
        save_path = os.path.splitext(image_path)[0] + f"_compressed_k{k}.png"
        compressed.save(save_path)
        print(f"✅ Compressed image saved as: {save_path}")
    else:
        print("No image selected.")
