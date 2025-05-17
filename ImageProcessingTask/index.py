import nibabel as nib 
import numpy as np 
import matplotlib.pyplot as plt 
from nilearn import plotting

img = nib.load('3702_left_knee.nii')
data = img.get_fdata()

slice = data[:, :, 59]

# Display a middle slice of the first axis
plt.imshow(slice, cmap='gray')
plt.title('Middle Slice')
plt.axis('off')
plt.show()

# visualizing different slices 
for i in range(0, 10):
	slice = data[:, :, 5+i]
	plt.subplot(2, 5, i+1)
	plt.imshow(slice, cmap='gray')
	plt.title(f'Slice {i}')
	plt.axis('off')
plt.show()

