from typing import cast

import numpy as np
from numpy.typing import NDArray


def gaussian_heatmap(
    xy: NDArray[np.float64], shape: tuple[int, int], sigma: float = 1.0
) -> NDArray[np.float64]:
    """
    Create a single channel heatmap, with a single gaussian peak.

    Parameters:
        xy: The image location of the peak.
        shape: The shape (height, width) of the heatmap.

    Returns:
        The heatmap numpy array (float64).
    """
    h, w = shape
    y_grid, x_grid = np.ogrid[0:h, 0:w]

    x, y = xy
    return cast(
        NDArray[np.float64],
        np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2.0 * sigma**2)),
    )


def peak_location(heatmap: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """
    Find the peak location in a heatmap.

    Parameters:
        heatmap: The heatmap (single channel).

    Returns:
        Tuple with subpixel peak location and peak value.
    """
    _, w = heatmap.shape

    i = np.argmax(heatmap)
    xy = np.array([i % w, i // w])
    try:
        # Iterative subpixel might come up with a singular matrix.
        # Most often near borders.
        xy = iterative_subpixel(heatmap=heatmap, xy=xy)
    except np.linalg.LinAlgError:
        xy = weighted_subpixel(heatmap=heatmap, xy=xy)

    return xy, interpolate_quad(heatmap=heatmap, x=xy[0], y=xy[1])


def peak_sidelobe_ratio(
    heatmap: NDArray[np.float64], xy: NDArray[np.float64], radius: int = 5
) -> float:
    """
    Calculate Peak/Sidelobe ratio for the detected peak. How strong is the peak,
    measured in standard deviations.

    Parameters:
        heatmap: The heatmap (single channel).
        xy: The peak estimation.

    Returns:
        The PSR value.
    """
    h, w = heatmap.shape
    x, y = np.round(xy).astype(int)

    starty = max(y - radius, 0)
    endy = min(y + radius, h - 1)
    startx = max(x - radius, 0)
    endx = min(x + radius, w - 1)

    patch = heatmap[starty : endy + 1, startx : endx + 1]

    max_value = float(np.max(patch))
    mean_value = float(np.mean(patch))
    std_value = max(float(np.std(patch)), 1e-15)
    return (max_value - mean_value) / std_value


def iterative_subpixel(
    heatmap: NDArray[np.float64],
    xy: NDArray[np.float64],
    iterations: int = 3,
    step: float = 1e-03,
) -> NDArray[np.float64]:
    """
    Try to improve a pixel peak estimation through iterative solving.

    Parameters:
        heatmap: The heatmap (single channel).
        xy: The initial peak estimation.
        iterations: The maximum number of iterations.
        step: The step size for derivation.

    Returns:
        The improved peak estimation.
    """
    if len(heatmap.shape) != 2 or len(xy) != 2:
        raise ValueError

    error = (evaluate_peak(heatmap, xy) ** 2).sum()

    for _ in range(iterations):
        J = Jacobian(heatmap, xy, step)
        values = evaluate_peak(heatmap, xy)
        new_xy = xy - np.linalg.solve(J.T @ J, J.T @ values)
        new_values = evaluate_peak(heatmap, new_xy)
        new_error = (new_values**2).sum()

        if new_error < error:
            xy = new_xy
            error = new_error
        else:
            break

    return xy


def weighted_subpixel(
    heatmap: NDArray[np.float64], xy: NDArray[np.float64], window: int = 3
) -> NDArray[np.float64]:
    """
    Try to improve a pixel peak estimation through weighting.

    Parameters:
        heatmap: The heatmap (single channel).
        xy: The initial peak.
        window: The window size to work on.

    Returns:
        The improved estimation.
    """
    h, w = heatmap.shape

    x, y = int(xy[0]), int(xy[1])

    half_w = window // 2
    x_min = max(x - half_w, 0)
    x_max = min(x + half_w + 1, w)
    y_min = max(y - half_w, 0)
    y_max = min(y + half_w + 1, h)

    patch = heatmap[y_min:y_max, x_min:x_max]

    weight = patch.sum()

    x_grid, y_grid = np.meshgrid(
        np.arange(x_min, x_max), np.arange(y_min, y_max), indexing="xy"
    )

    x_new = (patch * x_grid).sum() / weight
    y_new = (patch * y_grid).sum() / weight

    return np.array([x_new, y_new])


def Jacobian(
    heatmap: NDArray[np.float64], xy: NDArray[np.float64], step: float
) -> NDArray[np.float64]:
    """
    Calculate the Jacobian.

    Parameters:
        heatmap: The heatmap.
        xy: Array with x, y values.
        step: Step size for derivation.

    Returns:
        Jacobian matrix J.
    """
    step_x = np.array([step, 0.0])
    x_plus = evaluate_peak(heatmap, xy + step_x)
    x_minus = evaluate_peak(heatmap, xy - step_x)
    dx = (x_plus - x_minus) / (2.0 * step)

    step_y = np.array([0.0, step])
    y_plus = evaluate_peak(heatmap, xy + step_y)
    y_minus = evaluate_peak(heatmap, xy - step_y)
    dy = (y_plus - y_minus) / (2.0 * step)

    return np.hstack((dx.reshape(2, 1), dy.reshape(2, 1)))


def evaluate_peak(
    heatmap: NDArray[np.float64], xy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Evaluate a peak hypothesis. The sharper the mid value is compared
    to surrounding values, the lower the returned score value will be.

    Parameters:
        heatmap: The heatmap.
        xy: Array with x, y values.

    Returns:
        Array with score in x, y.
    """
    x, y = float(xy[0]), float(xy[1])

    step = 0.75
    mid = interpolate_quad(heatmap, x, y)
    left = interpolate_quad(heatmap, x - step, y)
    right = interpolate_quad(heatmap, x + step, y)
    up = interpolate_quad(heatmap, x, y - step)
    down = interpolate_quad(heatmap, x, y + step)

    values = np.zeros(2)
    if 2.0 * mid - left - right > 0.0:
        values[0] = left - right

    if 2.0 * mid - up - down > 0.0:
        values[1] = up - down

    return values


def interpolate_quad(heatmap: NDArray[np.float64], x: float, y: float) -> float:
    """
    Interpolate a neighbourhood of 4x4 pixels using cubic splines.

    Parameters:
        heatmap: The heatmap.
        x: Subpixel coordinate.
        y: Subpixel coordinate.

    Returns:
        Interpolated value.
    """
    h, w = heatmap.shape

    xi, yi = min(int(x), w - 1), min(int(y), h - 1)
    if xi > 0 and xi < w - 2 and yi > 0 and yi < h - 2:
        xf, yf = x - xi, y - yi
        sx, sy = cubic_spline(xf), cubic_spline(yf)

        patch = heatmap[yi - 1 : yi + 3, xi - 1 : xi + 3]

        rows = (patch * sx).sum(axis=1)
        return float((rows * sy).sum())
    else:
        return float(heatmap[yi, xi])


def cubic_spline(x: float) -> NDArray[np.float64]:
    """
    Generate a cubic spline.
    """
    x2 = x * x
    x3 = x2 * x

    s0 = -0.5 * x3 + 1.0 * x2 - 0.5 * x
    s1 = 1.5 * x3 - 2.5 * x2 + 1.0
    s2 = -1.5 * x3 + 2.0 * x2 + 0.5 * x
    s3 = 0.5 * x3 - 0.5 * x2

    return np.array([s0, s1, s2, s3])
