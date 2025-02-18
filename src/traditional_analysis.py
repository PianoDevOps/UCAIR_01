import numpy as np
from skimage import measure, morphology
from skimage.metrics import hausdorff_distance


def extract_largest_component(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Finds the largest component in the mask, returns a new mask of the component and
    its centroid

    Args:
        mask (NDArray): Input mask

    Returns:
        tuple[NDArray, tuple[int,int]]: new_mask, centroid
    """

    # Run connected component analysis
    labeled_mask = measure.label(mask)

    properties = measure.regionprops(labeled_mask)

    biggest_component = max(properties, key=lambda x: x.area)

    _centroid = biggest_component.centroid
    _row = int(np.round(_centroid[0], decimals=0))
    _column = int(np.round(_centroid[1], decimals=0))
    centroid = (_row, _column)

    largest_component_mask = (labeled_mask == biggest_component.label).astype(np.uint8)

    return largest_component_mask, centroid


def split_into_endo_epi_masks(
    mask: np.ndarray, centroid: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a mask of the largest component to separate endo- and epicardium masks

    Args:
        mask (NDArray): Mask of the largest component
        centroid (tuple[int,int]): (row, column)

    Returns:
        tuple[NDArray, NDArray]: epi_mask, endo_mask
    """

    epicardium_mask = morphology.convex_hull_image(mask, offset_coordinates=False)
    endocaridum_mask = morphology.flood(mask, centroid)
    return epicardium_mask.astype(np.uint8), endocaridum_mask.astype(np.uint8)


def get_endo_epi_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts endo- and epicardium masks from a general mask

    Args:
        mask (NDArray): General input mask

    Returns:
        tuple[NDArray, NDArray]: epicardium_mask, endocardium_mask
    """

    largest_component, centroid = extract_largest_component(mask)

    epi_mask, endo_mask = split_into_endo_epi_masks(largest_component, centroid)

    return epi_mask, endo_mask


def get_perimeter(mask: np.ndarray) -> np.ndarray:
    """
    Returns binary mask of the perimeter of the input mask

    Args:
        mask (NDArray): Input mask

    Returns:
        NDArray: Perimeter of input mask
    """

    return mask - morphology.binary_erosion(mask)


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    """
    Converts the mask of a simple connected (topology) shape to polygons

    Args:
        mask (NDArray): Input binary mask

    Returns:
        list[list[float]]: polygon
    """

    assert mask.ndim == 2, f"Mask must be 2D, not {mask.ndim}D"

    contours = measure.find_contours(mask)
    polygons = []
    for contour in contours:
        contour = np.flip(contour, axis=1)  # flip to get (x, y)
        polygons.append(contour.ravel().tolist())

    return polygons

