import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Fetch the image from the URL using requests
url = 'https://images.unsplash.com/photo-1702839836164-b2061b755bbd?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3Ds';
response = requests.get(url)
image = np.asarray(bytearray(response.content), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image
blurred_image = cv2.GaussianBlur(gray_image, (47, 47), 0)

# Detect edges using the Canny edge detection algorithm
edges = cv2.Canny(gray_image, 100, 200)

# Display the original and processed images : 
plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Blurred Image')
plt.imshow(blurred_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()