# Compressed-image-using-Principle-Component-Analysis
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = Image.open("C:/Users/start/Desktop/5226523_orig.jpg")
plt.imshow(img)
plt.show()
image_array = np.array(img)

img_meaned = image_array - np.mean(image_array, axis=0)
print(img_meaned)

print(image_array.shape)
print(img_meaned.shape)

# Reshape the img_meaned to make it 2D
img_meaned_2d = img_meaned.reshape(image_array.shape[0], -1)

print(img_meaned_2d.shape)

cov_mat = np.cov(img_meaned_2d, rowvar=False)
print(cov_mat.shape)
print(cov_mat)

eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

print(eigen_vectors.shape)
print(eigen_values)

# Sort the eigenvalues in descending order
sorted_index = np.argsort(eigen_values)[::-1]
print(sorted_index)
sorted_eigenvalues = eigen_values[sorted_index]
# Similarly sort the eigenvectors
sorted_eigenvectors = eigen_vectors[:, sorted_index]

n_components = 2  # You can select any number of components.
eigenvector_subset = sorted_eigenvectors[:, 0:n_components]

print(eigenvector_subset)

X_reduced = np.dot(eigenvector_subset.transpose(), img_meaned_2d.transpose()).transpose()
print(X_reduced)
compression_ratio = 0.5
n_components = int(compression_ratio * len(sorted_eigenvalues))

selected_eigenvectors = sorted_eigenvectors[:, :n_components]

X_reduced = np.dot(img_meaned_2d, selected_eigenvectors)

reconstructed_array = np.dot(X_reduced, selected_eigenvectors.T) + np.mean(image_array, axis=0)

reconstructed_image = reconstructed_array.reshape(image_array.shape)

compressed_image = Image.fromarray(reconstructed_image.astype(np.uint8))

compressed_image.save("compressed_image.jpg")
compressed_image.show()
