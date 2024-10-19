import numpy as np


def power_method(
    A: np.ndarray,
    x0: np.ndarray | None = None,
    eps: float = 1e-7,
    max_iter: int = 10000,
) -> tuple[float, np.ndarray]:
    """
    Compute the dominant eigenvalue and corresponding eigenvector of a matrix using the Power Method.

    Parameters:
    -----------
    A : np.ndarray
        A 2D square matrix (n x n) for which the dominant eigenvalue and eigenvector are to be computed.
    x0 : np.ndarray, optional
        Initial guess for the eigenvector. If None, a vector of ones is used by default.
    eps : float, optional
        Convergence criterion. The algorithm stops when the change between iterations is less than eps. Default is 1e-7.
    max_iter : int, optional
        Maximum number of iterations. The algorithm stops if this limit is reached. Default is 10,000.

    Returns:
    --------
    tuple[float, np.ndarray]
        A tuple containing the dominant eigenvalue (float) and the corresponding normalized eigenvector (np.ndarray).

    Raises:
    -------
    ValueError
        If `A` is not a 2D matrix or if `x0` has an invalid shape.
    """
    if len(A.shape) != 2:
        raise ValueError("`A` must be a 2D matrix.")
    if x0 is None:
        x0 = np.ones(shape=(A.shape[1]))
    else:
        x0 = np.squeeze(x0)
        if len(x0.shape) != 1:
            raise ValueError("The shape of `x0` must be (n, 1) or (n,).")

    x_i = x0
    for _ in range(max_iter):
        y_i = A @ x_i
        x_ip1 = y_i / np.linalg.norm(y_i)
        error = np.linalg.norm(x_ip1 - x_i)
        x_i = x_ip1
        if error < eps:
            break

    eigenvalue = float(np.max(A @ x_i / x_i))
    return eigenvalue, x_i


if __name__ == "__main__":
    A = np.array([[1, 2, 1], [1, 1, 2], [3, 1, 1]])

    alpha, _ = power_method(A, np.ones(3))
    alpha_exact = float(np.linalg.eig(A)[0][0].real)
    print(f"{alpha = } vs {alpha_exact = }")
