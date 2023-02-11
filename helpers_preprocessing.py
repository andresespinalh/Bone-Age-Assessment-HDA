import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from scipy.stats import mode

## Plotting Functions
# Workaround to print images inside the Jupyter Notebook (cv.imshow method seems to be incompatible with Jupyter)
def display_img(image):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Print images side to side for comparison inside the Jupyter Notebook
def display_img_comp(images, labels):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 5))   
    
    for i, img in enumerate(images):
        axs[i].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        axs[i].set_title(labels[i])
        axs[i].axis('off')

    fig.tight_layout()
    plt.show()

## Image Manipulation Functions
# Enhance contrast using Constrast Limited Adaptive Histogram Equalization
def enhance_contrast_img(img, clipLimit=1, tileGridSize=(8,8)):
    # Apply CLAHE
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_p = clahe.apply(img)

    return img_p
    
# Resize image
def resize_img(img, width, height):
    img_p = cv.resize(img, (width, height))
    return img_p

def resize_img_fill(img, width, height):
    # Height & Width of the original image
    h, w = img.shape[:2]
    # Maximum box that can hold the image
    max_side = max(h, w)
    # Black image of the maximum box size
    square_img = np.zeros((max_side, max_side), np.uint8)
    # Position the image
    start_row = int((max_side - h) / 2)
    start_col = int((max_side - w) / 2)
    square_img[start_row:start_row + h, start_col:start_col + w] = img

    # Final resize to intended resolution
    img_p = resize_img(square_img, width, height)
    return img_p

def img_fill(img):
    # Height & Width of the original image
    h, w = img.shape[:2]
    # Maximum box that can hold the image
    max_side = max(h, w)
    # Black image of the maximum box size
    square_img = np.zeros((max_side, max_side), np.uint8)
    # Position the image
    start_row = int((max_side - h) / 2)
    start_col = int((max_side - w) / 2)
    square_img[start_row:start_row + h, start_col:start_col + w] = img
    img_p = square_img

    return img_p

# Crop Image
def crop_img(img):
    img_e = enhance_contrast_img(img, clipLimit=2, tileGridSize=(50, 50))
    # First perform Binary Thresholding through the Otsu method
    tc_o, img_thr_o = cv.threshold(img_e, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) 

    # Apply a Gaussian Blur
    img_blur_o = cv.GaussianBlur(img_thr_o, (25, 25), 5)

    # Find all contours on the blurred image
    contours_o, hier_o = cv.findContours(img_blur_o, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Select only the outermost contour
    largest_contour = max(contours_o, key=cv.contourArea)

    # Get the proportion of the total area of the image from the outermost contour
    img_tot_pixels = img.shape[0]*img.shape[1]
    contour_perc = cv.contourArea(largest_contour)/img_tot_pixels
    
    # If the contour found is composed by a percentage bigger than the threshold, perform the crop. Otherwise leave the image as it is
    if(contour_perc > 0.10):
        # Get a bounding box representing the outermost rectangle
        x, y, w, h = cv.boundingRect(largest_contour)
        # Crop the image
        img_p = img[y:y+h, x:x+w]
    else:
        img_p = img

    return img_p

# Crop Image Angle Aware
def crop_img_angled(img):
    # First perform Binary Thresholding through the Otsu method
    tc_o, img_thr_o = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Apply a Gaussian Blur
    img_blur_o = cv.GaussianBlur(img_thr_o, (71, 71), 333)

    # Find all contours on the blurred image
    contours_o, hier_o = cv.findContours(img_blur_o, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Select only the outermost contour
    largest_contour = max(contours_o, key=cv.contourArea)

    # Get the rotated bounding box of the largest contour
    rect = cv.minAreaRect(largest_contour)
    box = cv.boxPoints(rect)

    # Get the proportion of the total area of the image from the outermost contour
    img_tot_pixels = img.shape[0]*img.shape[1]
    contour_perc = cv.contourArea(largest_contour)/img_tot_pixels

    # If the contour found is composed by a percentage bigger than the threshold, perform the crop. Otherwise leave the image as it is
    if(contour_perc > 0.10):
        # Get a bounding box representing the outermost rectangle
        x, y, w, h = cv.boundingRect(box)

        # Get the center of the bounding box
        center = (x + w // 2, y + h // 2)

        # Define a set of points that represent the x-axis and y-axis in the bounding box
        # NOTE: Indexing of the rectangle starts on the bottom left corner (Index 0) and ends on the top left corner (Index 3)
        p1_x, p2_x, p1_y, p2_y = box[0], box[1], box[0], box[3]

        # Calculate the distance between the points for the horizontal and vertical axis (This is the resolution of the box)
        res_x = int(np.sqrt(np.sum((p2_x - p1_x)**2)))
        res_y = int(np.sqrt(np.sum((p2_y - p1_y)**2)))

        size = (res_x, res_y)
        

        # Get the rotation matrix for rotating the image
        M = cv.getPerspectiveTransform(box, np.array([[0, 0], [res_x, 0], [res_x, res_y], [0, res_y]], dtype=np.float32))

        # Crop the image based on the perspective transform matrix
        img_p = cv.warpPerspective(img, M, size) 
    else:
        img_p = img

    return img_p

## Experiments
# The whole pre-processing pipeline for a single image
def preprocess_image(img, codename):
    if(codename=='clahe1_r250'):
        img_p = enhance_contrast_img(img, clipLimit=2, tileGridSize=(20, 20))
        img_p = resize_img(img_p, 250, 250)
    elif(codename=='clahe2_r250'):
        img_p = enhance_contrast_img(img, clipLimit=5, tileGridSize=(50, 50))
        img_p = resize_img(img_p, 250, 250)
    elif(codename=='clahe3_r250'):
        img_p = enhance_contrast_img(img, clipLimit=2, tileGridSize=(10, 10))
        img_p = resize_img(img_p, 250, 250)
    elif(codename=='claheadapt1_r250'):
        img_p = enhance_contrast_img(img, clipLimit=2, tileGridSize=(10, 10))
        img_p = cv.adaptiveThreshold(img_p, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 7)
        img_p = resize_img(img_p, 250, 250)
    elif(codename=='claheadapt2_r250'):
        img_p = enhance_contrast_img(img, clipLimit=2, tileGridSize=(10, 10))
        img_p = cv.adaptiveThreshold(img_p, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 27, 7)  # Adaptive Thresholding Gaussian C
        img_p = resize_img(img_p, 250, 250)
    elif(codename=='clahecrop_r250'):
        img_p = enhance_contrast_img(img, clipLimit=2, tileGridSize=(10, 10))
        img_p = crop_img(img_p)
        img_p = resize_img(img_p, 250, 250)
    elif(codename=='clahecrop2_r250'):
        img_p = enhance_contrast_img(img, clipLimit=5, tileGridSize=(50, 50))
        img_p = crop_img(img_p)
        img_p = resize_img(img_p, 250, 250) 
    elif(codename=='clahecropfix_r250'):
        img_p = enhance_contrast_img(img, clipLimit=5, tileGridSize=(50, 50))
        img_p = crop_img(img_p)
        img_p = resize_img(img_p, 250, 250) 
    elif(codename=='clahecropangle_r250'):
        # Enhance the image to better detect finger borders
        img_p = enhance_contrast_img(img, clipLimit=2, tileGridSize=(10, 10))   
        img_p = crop_img_angled(img_p)
        img_p = resize_img(img_p, 250, 250) 
    elif(codename=='clahecropanglefill_r250'):
        # Enhance the image to better detect finger borders
        img_p = enhance_contrast_img(img, clipLimit=2, tileGridSize=(10, 10))   
        img_p = crop_img_angled(img_p)
        img_p = resize_img_fill(img_p, 250, 250) 
    elif(codename=='clahecropfill2_r250'):
        img_p = enhance_contrast_img(img, clipLimit=3, tileGridSize=(50, 50))
        img_p = crop_img(img)
        img_p = resize_img_fill(img_p, 250, 250) 
    elif(codename=='clahecrop3_r300'):
        img_p = crop_img(img)
        img_p = resize_img(img_p, 300, 300) 
        img_p = enhance_contrast_img(img_p, clipLimit=3, tileGridSize=(15, 15))
    elif(codename=='clahecrop3fill_r300'):
        # img_p = enhance_contrast_img(img, clipLimit=5, tileGridSize=(3, 3))
        img_p = crop_img(img)
        img_p = resize_img_fill(img_p, 300, 300) 
        img_p = enhance_contrast_img(img_p, clipLimit=3, tileGridSize=(15, 15))
    # Final Experiments
    elif(codename=='original_r250p'):
        img_p = resize_img_fill(img, 250, 250) 
    elif(codename=='preprocessed_r250p'):
        img_p = crop_img(img)
        img_p = img_fill(img_p) 
        img_p = enhance_contrast_img(img_p, clipLimit=2.5, tileGridSize=(5, 10)) 
        img_p = resize_img(img_p, 250, 250) 
    elif(codename=='preprocessed'):
        img_p = crop_img(img)
        img_p = img_fill(img_p) 
        img_p = enhance_contrast_img(img_p, clipLimit=2.5, tileGridSize=(5, 10)) 
    return img_p

# Function that writes the outputs of the processed images to disk, based on an experiment codename
def export_experiment(img_rel_paths_dict, img_paths_dest_dict, codename):
    # For each split (Training, Test, Validation sets)
    for split in img_rel_paths_dict.keys():
        # Get the source paths of images in the split, and the path of their destination
        img_paths_analysis = img_rel_paths_dict[split]  
        path_split_dest = img_paths_dest_dict[split]

        # Create a folder with the datasetname inside that path, if it doesn't exist yet
        path_dest = '{0}\{1}'.format(path_split_dest, codename)
        path_exists = os.path.exists(path_dest)
        
        if not path_exists:
            # Create a new directory because it does not exist
            os.makedirs(path_dest)

        # For each image belonging to the split, process the image and write the result to the provided folder
        for img_key in img_paths_analysis.keys():
            path_p = '{0}\{1}.png'.format(path_dest, img_key)
            path_src = img_paths_analysis[img_key]
            img_src = cv.imread(path_src)
            
            # Convert the image to 8-bit unsigned single channel
            img_src = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)

            # Function to apply to the image before writing
            img_p = preprocess_image(img_src, codename)

            cv.imwrite(path_p, img_p)

            # print(path_p)

            # Clean the memory for that image, as it is no longer needed
            del img_src, img_p