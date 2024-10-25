import numpy as np  
import pywt
from PIL import Image
from scipy.fftpack import dct, idct 
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim 
import cv2

model = 'haar'
level = 1

# Loading images
img1 = Image.open('images.jpg').resize((2048, 2048), Image.ANTIALIAS)
img = img1.convert('L')
image_array = np.array(img.getdata(), dtype=float).reshape((2048, 2048)) 

size = 128
watermark1 = Image.open('watermark.jpg').resize((size, size), Image.ANTIALIAS)
watermark = watermark1.convert('L')
watermark_array = np.array(watermark.getdata(), dtype=float).reshape((size, size))

# Wavelet decomposition
coeffs = pywt.wavedec2(data=image_array, wavelet=model, level=level)
coeffs_H = list(coeffs)

# Apply DCT to image
def apply_dct(image_array):
    size = len(image_array[0])
    all_subdct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct
    return all_subdct

# Embed watermark in DCT coefficients
def embed_watermark(watermark_array, orig_image):
    watermark_flat = watermark_array.ravel()
    ind = 0
    for x in range(0, len(orig_image), 8):
        for y in range(0, len(orig_image), 8): 
            if ind < len(watermark_flat):
                subdct = orig_image[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[ind]
                orig_image[x:x+8, y:y+8] = subdct
                ind += 1
    return orig_image 

# Inverse DCT
def inverse_dct(all_subdct):
    size = len(all_subdct[0])
    all_subidct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct
    return all_subidct

# Apply DCT and watermark
dct_array = apply_dct(coeffs_H[0])
dct_array = embed_watermark(watermark_array, dct_array) 
coeffs_H[0] = inverse_dct(dct_array)

# Reconstruct image
image_array_H = pywt.waverec2(coeffs_H, model)
image_array_copy = image_array_H.clip(0, 255).astype("uint8")
watermarked_img = Image.fromarray(image_array_copy)

# Display images
plt.figure(figsize=(10, 6))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Host image')
plt.axis('off')

plt.subplot(132)
plt.imshow(watermark, cmap='gray')
plt.title('Original Watermark')
plt.axis('off')

plt.subplot(133)
plt.imshow(watermarked_img, cmap='gray')
plt.title('Output Watermarked')
plt.axis('off')
plt.tight_layout()
plt.show()

# PSNR, MSE, and Entropy calculation
host_image = cv2.imread('images.jpg', cv2.IMREAD_GRAYSCALE)
host_image = cv2.resize(host_image, (256, 256))
watermarked_img = np.array(watermarked_img.resize((256, 256), Image.ANTIALIAS))

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_entropy(image):
    histogram = np.histogram(image, bins=256, range=(0, 256))[0]
    histogram = histogram / histogram.sum()
    entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
    return entropy

psnr_value = psnr(host_image, watermarked_img)
print(f"PSNR: {psnr_value} dB")

mse_value = np.mean((host_image - watermarked_img) ** 2)
print(f"MSE: {mse_value}")

entropy_value = calculate_entropy(watermarked_img)
print(f"Entropy: {entropy_value}")
