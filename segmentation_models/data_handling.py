def extract_data(folderpath):
    """ (ETH) Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].

    Parameters
    ----------
    filename: string
        The name of the image file
    num_images: int
        The number of images that should be extracted

    Returns
    -------
    data: ndarray
        A numpy array containting the images
    """
    files = os.listdir(folderpath)
    n = len(files)
    imgs = [(mpimg.imread(folderpath+files[i])) for i in range(n)]
    data = np.asarray(imgs)
    return data

def extract_data_test(folderpath):
    """ (ETH) Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].

    Parameters
    ----------
    filename: string
        The name of the image file
    num_images: int
        The number of images that should be extracted

    Returns
    -------
    data: ndarray
        A numpy array containting the images
    """
    imgs=[]
    for i in range(1,51):
      img = mpimg.imread(folderpath+'test_%d.png'%i)
      imgs.append(img)
    data = np.asarray(imgs)
    return data


def extract_labels(folderpath):
    """ (ETH) Extract the labels into a 1-hot matrix [image index, label index].
    
    Parameters
    ----------
    filename: string
        The name of the image file
    num_images: int
        The number of images
    
    Returns
    --------
    labels: ndarray
        1-hot matrix [image index, label index]
    """
    gt_imgs = []
    files = os.listdir(folderpath)
    n = len(files)
    for i in range(n):
        img = mpimg.imread(folderpath+files[i])
        try:
            gt_imgs.append(img[:,:,0])
        except:
            gt_imgs.append(img)

    return np.asarray(gt_imgs)