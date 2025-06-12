import SimpleITK as sitk
import numpy as np 
from skimage.segmentation import clear_border

import matplotlib.pyplot as plt 
import cv2 as cv
from scipy.ndimage import (
    generate_binary_structure,
    binary_opening,
    binary_propagation,
    label,
    sum as nd_sum
)


def get_largest_object_in_region(region):
    # labels in the region, in the binary mask
    labeled_region, num_objects = label(region)
    if num_objects == 0:
        return np.zeros_like(region, dtype=np.uint8)
    
    object_sizes = nd_sum(region, labeled_region, index=np.arange(1, num_objects + 1))
    largest_object_label = np.argmax(object_sizes) + 1
    
    return (labeled_region == largest_object_label).astype(np.uint8)

def read_img():
    img_sitk = sitk.ReadImage('3702_left_knee.nii') 
    image = sitk.GetArrayFromImage(img_sitk) 

    return image, img_sitk


def get_segmented_bone(image):
    axial_index = image.shape[0] // 2      
    coronal_index = image.shape[1] // 2    
    sagittal_index = image.shape[2] // 2   

    weak_threshold = 80
    strong_threshold = 300
    minimum_bone_component_voxels = 4000 # Removes small objects.
    slice_shape = image.shape[0]


    intensity_clipped_volume = np.clip(image, -1000, 1500)
    print('clipped')
    vol_min = np.min(intensity_clipped_volume)
    vol_max = np.max(intensity_clipped_volume)
    normalized_float_volume = ((intensity_clipped_volume - vol_min) / (vol_max - vol_min)).astype(np.float32)
    print(f'normalized')
    normalized_8bit_volume = (normalized_float_volume * 255).astype(np.uint8)

    print("noise reduction filters to each slice")
    denoised_volume_float = np.zeros_like(normalized_float_volume, dtype=np.float32)
    for slice_index in range(slice_shape):
        gaussian_slice = cv.GaussianBlur(normalized_8bit_volume[slice_index, :, :], (5, 5), 3)
        bilateral_slice = cv.bilateralFilter(gaussian_slice, d=9, sigmaColor=75, sigmaSpace=75)
        denoised_volume_float[slice_index, :, :] = bilateral_slice.astype(np.float32) / 255.0

    difference = vol_max - vol_min
    filtered_hu_volume = (denoised_volume_float * difference) + vol_min

    high_confidence_seed_mask = filtered_hu_volume > strong_threshold
    potential_bone_region_mask = (filtered_hu_volume > weak_threshold) & (filtered_hu_volume <= strong_threshold)

    # masking all around the bones, to segment all of it.
    connectivity_structure = generate_binary_structure(3, 3)
    propagated_segmentation = binary_propagation(high_confidence_seed_mask, mask=potential_bone_region_mask, structure=connectivity_structure)
    comprehensive_bone_mask = np.logical_or(high_confidence_seed_mask, propagated_segmentation)

    morphologically_opened_mask = binary_opening(comprehensive_bone_mask, structure=np.ones((3, 3, 3)))

    labeled_components, num_components = label(morphologically_opened_mask)
    component_sizes = nd_sum(morphologically_opened_mask, labeled_components, index=np.arange(num_components + 1))

    # removing small objects
    final_refined_mask = np.isin(labeled_components, np.where(component_sizes > minimum_bone_component_voxels)[0])

    vertical_midpoint = final_refined_mask.shape[0] // 2

    # to store the mask
    labeled_output_mask = np.zeros(final_refined_mask.shape, dtype=np.uint8)

    # upto the point
    upper_half_region = final_refined_mask[:vertical_midpoint, :, :]
    femur_component = get_largest_object_in_region(upper_half_region)

    # from that point onwards
    lower_half_region = final_refined_mask[vertical_midpoint:, :, :]
    tibia_component = get_largest_object_in_region(lower_half_region)

    labeled_output_mask[:vertical_midpoint, :, :][femur_component > 0] = 1
    labeled_output_mask[vertical_midpoint:, :, :][tibia_component > 0] = 2

    return labeled_output_mask