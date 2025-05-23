# Workflow of Solving ImageProcessingTask

1. Using `Nibabel` library to read the 3D image, and visualizing the slices.

2. Use of thresholding to segment the bones

3. Using dilation to expand the contour by 2mm

4. Use the bottom surfce with non zero to get the lower bone coordinates and sorting to get the medial-most and lateral-most points.
