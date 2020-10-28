import matplotlib.pyplot as plt

def imshow(*arg):
    """
    Short cut to show multiple images.

    Parameters
    ----------
    *arg : tuple
        A tuple of 2d array.

    Returns
    -------
    Show the images in a sequence.

    Examples
    --------
    import numpy as np
    from cellquantifier.io.imshow import imshow
    img1 = np.random.rand(2,2)
    img2 = np.random.rand(5,5)
    img3 = np.random.rand(10,10)
    imshow(img1, img2, img3)
    """

    plt.figure(figsize=(16,12))
    for i in range(len(arg)):
        plt.subplot(1,len(arg),i+1)
        plt.imshow(arg[i])
        plt.axis('off')

    plt.show()

def imshow_gray(*arg):
    """
    Short cut to show multiple images.

    Parameters
    ----------
    *arg : tuple
        A tuple of 2d array.

    Returns
    -------
    Show the images in a sequence.

    Examples
    --------
    import numpy as np
    from cellquantifier.io.imshow import imshow_gray
    img1 = np.random.rand(2,2)
    img2 = np.random.rand(5,5)
    img3 = np.random.rand(10,10)
    imshow_gray(img1, img2, img3)
    """

    plt.figure(figsize=(16,12))
    for i in range(len(arg)):
        plt.subplot(1,len(arg),i+1)
        plt.imshow(arg[i], cmap='gray')
        plt.axis('off')

    plt.show()
