import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def random_crop(img, mask, min_size=100):
    """
    Crops a random subset of the picture with a standard minimum size of 80 x 80 pixels

    Parameters
    ----------
    img: ndarray
        Satellite image
    mask: ndarray
        Ground truth of the corresponding satellite image
    min_size: int
        Minimum number of rows/columns in the cropped picture. If nothing is inputted, this is set to 80.

    Returns
    -------
    sliced_img: ndarray
        Random slice of input image
    sliced_mask : ndarray
        Random slice of ground truth mask
    """
    # Select a random subset of rows >= min size
    row_end = np.random.randint(min_size,img.shape[0]+1)
    row_start = np.random.randint(0,row_end - min_size+1)

    # Select a random subset of columns >= min size
    col_end = np.random.randint(min_size,img.shape[1]+1)
    col_start = np.random.randint(0, col_end - min_size+1)

    # Slice image
    sliced_img = img[row_start:row_end, col_start:col_end]
    sliced_mask = mask[row_start:row_end, col_start:col_end]

    return sliced_img, sliced_mask


def rescale(img, mask, row_num=256, col_num=256):
    """
    Rescales the image into size 256 x 256

    Parameters
    ----------
    img: ndarray
        Satellite image
    mask: ndarray
        Ground truth of the corresponding satellite image
    row_num: int
        Number of rows the rescaled input should have 
    row_num: int
        Number of columns the rescaled input should have 

    Returns
    -------
    rescaled_image: ndarray
        Rescaled version of input image to shape (row_num x col_num)
    rescaled_mask : ndarray
        Rescaled version of input mask to shape (row_num x col_num)
    """
    rescaled_image = cv2.resize(img, dsize=(row_num, col_num))
    rescaled_mask = cv2.resize(mask, dsize=(row_num, col_num))
    
    return rescaled_image, rescaled_mask


def crop_and_rescale(img, mask):
    """
    Makes a random crop and rescales the image back into size 256 x 256
    
    Parameters
    ----------
    img: ndarray
        Satellite image
    mask: ndarray
        Ground truth of the corresponding satellite image

    Returns
    -------
    crop_rescale_img: ndarray
        Cropped and rescaled version of input image 
    crop_rescale_mask : ndarray
        Cropped and rescaled version of input mask
    """

    crop_img, crop_mask = random_crop(img, mask)
    crop_rescale_img, crop_rescale_mask = rescale(crop_img, crop_mask)
    
    return crop_rescale_img, crop_rescale_mask


def rotate_and_rescale(img, mask):
    """
    Makes a random rotation of the inputs, zooms in to crop out black fields after rotation and rescales the image back into size 256 x 256
    
    Parameters
    ----------
    img: ndarray
        Satellite image
    mask: ndarray
        Ground truth of the corresponding satellite image

    Returns
    -------
    rot_rescale_img: ndarray
        Rotated and rescaled version of input image 
    rot_rescale_mask : ndarray
        Rotated and rescaled version of input mask
    """

    # Pick random angle of rotation
    deg = np.random.randint(0,361)

    # Determine coordinates of inner square to crop out black fields after any rotation. 
    angle = deg
    while angle > 180:
        angle = angle - 180
    s = 400 / (np.cos(angle*np.pi/360) + (np.sin(angle*np.pi/360)))
    start = int((400 - s) // 2) + 25
    end = int(400 - start)

    # Rotate
    M = cv2.getRotationMatrix2D(((img.shape[0] // 2, img.shape[1] // 2)), deg, 1.0)
    rotated_image = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))
    rotated_mask = cv2.warpAffine(mask, M, (mask.shape[0], mask.shape[1]))
    
    # Rescale
    rot_rescale_img, rot_rescale_mask = rescale(rotated_image[start:end, start:end], rotated_mask[start:end, start:end])

    return rot_rescale_img, rot_rescale_mask


def apply_gaussian_blur(img):
    """
    Applies a gaussian blur to an image with a random kernel standard deviation
    
    Parameters
    ----------
    img: ndarray
        Satellite image

    Returns
    -------
    img: ndarray
        Rotated and rescaled version of input image 
    """
    
    # Get random gaussian kernel standard deviation between 0 and 3
    std = np.random.uniform(0,3)

    # apply gaussian blur
    img = cv2.GaussianBlur(img,(15,15), std)

    return img


def apply_color_filter(img):
    """
    Applies a color filter either making colors warmer or colder. 
    
    Parameters
    ----------
    img: ndarray
        Satellite image

    Returns
    -------
    img: ndarray
        Color filtered image
    """
    # Get random number to determine which filter to apply
    rand = np.random.randint(1,3)

    if rand == 1:
        # Make colors warmer
        img_color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        # Make colors colder
        img_color = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
    
    return img_color


def filter_color_and_blur(img):
    """
    Applies a random blur and color filter
    
    Parameters
    ----------
    img: ndarray
        Satellite image

    Returns
    -------
    img: ndarray
        Color filtered and gaussian blurred image
    """
    img = apply_gaussian_blur(img)
    img = apply_color_filter(img)
    
    return img


def augment(img, mask, crop_num=4, rot_num=2, filter_num=1):
    """
    Given image and corresponding mask, creates crop_num * rot_num * (filter_num + 1) augmentations

    Parameters
    ----------
    img: ndarray
        Satellite image
    mask: ndarray
        Ground truth of the corresponding satellite image
    crop_num: int
        Number of crops to create
    rot_num: int
        Number of rotations
    filter_num: int
        Number of filtered images for each crop_num

    Returns
    -------
    img_list: ndarray
        The image augmentations for the given satellite image
    mask_list : ndarray
        The image augmentations for the given ground truth. 
    """
    # Instantiate variables to hold all augmentations of the picture
    img_list = []
    mask_list = []

    # Make a given number of random crops 
    for i in range(crop_num):
        img_crop, mask_crop = crop_and_rescale(img, mask)
        
        # Store pictures
        img_list.append(img_crop)
        mask_list.append(mask_crop)
        
        # Create a filtered version (only applied to image and not mask)
        for i in range(filter_num):
            img_list.append(filter_color_and_blur(img_crop))
            mask_list.append(mask_crop)

        # Rotate each picture a given number of times
        for j in range(rot_num):
            img_rot, mask_rot = rotate_and_rescale(img, mask)
            img_list.append(img_rot)
            mask_list.append(mask_rot)
        
    return img_list, mask_list


def augment_dir(dir_images, dir_masks, destination_dir_images, destination_dir_masks):
    """
    Goes through the two directories and creates a folder in each of the directories with the feature augmented data set. 

    Parameters
    ----------
    dir_images: str
        Directory of the satellite images
    dir_masks: str
        Directory of the corresponding ground truth images
    destination_dir_images: str
        Directory of where to store the augmented sattellite images
    destination_dir_masks: str
        Directory of where to store the corresponding augmented ground truths
    """
    # Load the images
    images = os.listdir(dir_images)
    masks = os.listdir(dir_masks)
    assert len(images) == len(masks)

    for i in range(len(images)):
        # Only load pictures
        if images[i][-3:] != "png" or masks[i][-3:] != "png":
            continue
        
        # Create folders to store augmentations
        folders = ["90-split","10-split"]
        for folder_name in folders:
            if not os.path.exists(os.path.join(destination_dir_images, folder_name)):
                os.makedirs(os.path.join(destination_dir_images, folder_name))
            if not os.path.exists(os.path.join(destination_dir_masks, folder_name)):
                os.makedirs(os.path.join(destination_dir_masks, folder_name))
        
        # store in 10-split folder if the picture is one of the last 10
        id = int(images[i][-7:-4])
        print(id)
        if id > 90:
            folder = folders[1]
        else:
            folder = folders[0]

        # Load the images
        img = cv2.imread(os.path.join(dir_images, images[i]))
        mask = cv2.imread(os.path.join(dir_masks, masks[i]))

        # Create augmentations
        img_list, mask_list = augment(img, mask)

        # Save each augmentation
        for j in range(len(img_list)):
           
            # Create filenames and save
            img_filename = os.path.join(destination_dir_images, folder, images[i])[0:-4] + f"_Aug{str(j).zfill(2)}" + ".png"
            mask_filename = os.path.join(destination_dir_masks, folder, masks[i])[0:-4] + f"_Aug{str(j).zfill(2)}" + ".png"
            cv2.imwrite(img_filename, img_list[j])
            cv2.imwrite(mask_filename, mask_list[j])
        
        # Save original resized picture 
        img_filename = os.path.join(destination_dir_images, folder, images[i])
        mask_filename = os.path.join(destination_dir_masks, folder, masks[i])
        img_rescaled, mask_rescaled = rescale(img, mask, row_num=256, col_num=256)
        cv2.imwrite(img_filename, img_rescaled)
        cv2.imwrite(mask_filename, mask_rescaled)

if __name__ == "__main__":
    dir_images = "/Users/nikolaibeckjensen/Desktop/training/images" 
    dir_masks = "/Users/nikolaibeckjensen/Desktop/training/groundtruth"
    destination_dir_images = "/Users/nikolaibeckjensen/Desktop/ml-project-2-vikings_ml/data/training/images"
    destination_dir_masks = "/Users/nikolaibeckjensen/Desktop/ml-project-2-vikings_ml/data/training/groundtruth"
    
    augment_dir(dir_images, dir_masks, destination_dir_images, destination_dir_masks)
    
