import numpy as np
from src.optical_flow import get_displacements


def split(array, n_rows, n_cols):
    """
    Given a frame image this function returns the gridblocks.

    E.g the image is the following:

    abcd
    efgh
    ijkl
    mnop

    and the grid is 2x2 then it returns the following blocks:

    ab  cd  ij  kl
    ef, gh, mn, op

    Parameters
    ----------
    frame: numpy array
        2D numpy array
    n_rows: int,
        width of the grid used to estimate optical flow. E.g. 7 means that the image will be split into
        7 equal parts vertically.
    n_cols
        height of the grid used to estimate optical flow. E.g. 7 means that the image will be split into
        7 equal parts horizontally.
    Raises
    ------
        ValueError: if the frame cannot be divided into the given grid accurately.

    Returns
    -------
        the image blocks as numpy arrays with size [n_rows*n_cols,h // n_rows, w // n_cols]
    """
    h, w = array.shape
    assert h % n_rows == 0, "{} rows is not evenly divisble by {}".format(h, n_rows)
    assert w % n_cols == 0, "{} cols is not evenly divisble by {}".format(w, n_cols)

    blocks = []
    for y in range(0, h, h // n_rows):
        for x in range(0, w, w // n_cols):
            blocks.append(array[y:y + h // n_rows, x:x + w // n_cols])

    return np.array(blocks)


def get_grid_flow(image1, image2, n_rows, n_cols):
    """ Calculate an optical flow for each image block defind by grid.

    Parameters
    ----------
    image1 : numpy array
        image1 a grayscale image
    image2 : numpy array
        image2 a grayscale image
    n_rows : int
        number of rows in the grid
    n_cols : int
        number of columns in the grid

    Return
    ------
        A numpy array of shape (n_rows, n_cols, 2) where each image block is assigned with
        a 2D vector (relative to is centre) representing its average displacement.
    """
    # Get image blocks
    image1_blocks = split(image1, n_rows, n_cols)
    image2_blocks = split(image2, n_rows, n_cols)

    # Compute optical flow for each block
    block_flow = [*map(lambda x: get_displacements(*x), zip(image1_blocks, image2_blocks))]
    # Subtract new positions from the origins to get the displacement for each block
    block_displacements = map(lambda x: np.subtract(x[1], x[0]), block_flow)
    # Calculate the mean displacements for each block
    mean_block_dispalcements = np.array([*map(lambda d: np.mean(d, axis=0), block_displacements)])
    # Reshape the displacements so it has the grid like shape.

    mean_block_dispalcements = mean_block_dispalcements.reshape(n_rows, n_cols, 2).astype(np.int)
    origins = get_grid_centres(*image1.shape[0:2], n_rows, n_cols)

    return origins, mean_block_dispalcements+origins

def get_grid_centres(h, w, n_rows, n_cols):
    """Calculate a grid block centres on a given canvas size.

    Note that canvas size must me divisible by grid.

    Parameters
    ----------
    canvas_size : Tuple(int),
        canvas_size (width, height)
    n_rows : int
        number of rows in the grid
    n_cols : int
        number of columns in the grid

    Return
    ------
        A numpy array of size (n_rows, n_cols, 2) where a 2D coordinate is assigned to
        each grid block representing its centre. 
    """
    assert h % n_rows == 0, "{} rows is not divisble by {}".format(
        h, n_rows
    )
    assert w % n_cols == 0, "{} cols is not divisble by {}".format(
        w, n_cols
    )
    block_height = h // n_rows
    block_width = w // n_cols
    x = np.linspace(0, w, n_cols, endpoint=False)
    y = np.linspace(0, h, n_rows, endpoint=False)
    x += block_width // 2
    y += block_height // 2
    return np.rollaxis(np.rollaxis(np.array(np.meshgrid(x, y)),-1),-1).astype(np.int)