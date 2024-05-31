import numbers

import numpy as np


def embed_rows_diagonally(matrix):
    """
    Convert a 2D matrix into a 3D matrix where each row of the input matrix is placed along the diagonal
    of each 2D slice in the resulting 3D matrix.

    Parameters:
    matrix (numpy.ndarray): The input 2D matrix to be transformed.

    Returns:
    numpy.ndarray: A 3D matrix where each 2D slice along the first dimension has its diagonal filled with
    the corresponding row of the original matrix.

    Raises:
    TypeError: If the input is not a NumPy array.
    ValueError: If the input is a scalar.

    Example:
    >>> import numpy as np
    >>> input_matrix = np.array([[1, 2], [3, 4], [5, 6]])
    >>> output_matrix = embed_rows_diagonally(input_matrix)
    >>> print(output_matrix)
    [[[1. 0.]
      [0. 2.]]
    <BLANKLINE>
     [[3. 0.]
      [0. 4.]]
    <BLANKLINE>
     [[5. 0.]
      [0. 6.]]]
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError('Input must be a NumPy array')

    d = np.ndim(matrix)

    if d == 0:
        raise ValueError('Input cannot be a scalar')
    z = matrix
    if d == 1:
        z = np.expand_dims(matrix, axis=0)

    _, n = z.shape
    identity_reshaped = np.eye(n).reshape((1, n, n))
    return np.einsum('ij,ajk->ijk', z, identity_reshaped)


def multiply_2D_with_3D(matrix_2D, matrix_3D):
    """
    Multiply a 2D matrix by a 3D matrix, ensuring matching dimensions.

    This function performs matrix multiplication of a 2D matrix by a 3D matrix.
    Each row of the 2D matrix is multiplied by a 2D slice of the 3D matrix.
    It ensures that the first two dimensions of the 3D matrix match the shape of
    the 2D matrix, and raises a ValueError if they don't.

    Parameters:
    matrix_2D (numpy.ndarray): The 2D matrix.
    matrix_3D (numpy.ndarray): The 3D matrix. Its first two dimensions must match the shape of matrix_2D.

    Returns:
    numpy.ndarray: The result of the multiplication.

    Raises:
    ValueError: If the first two dimensions of matrix_3D do not match the shape of matrix_2D.
    """

    if matrix_2D.shape[:2] != matrix_3D.shape[:2]:
        raise ValueError("The first two dimensions of matrix_3D must match the shape of matrix_2D.")

    return np.einsum('ij, ijl -> il', matrix_2D, matrix_3D)


def float_formatter(value, alignment='<', width=10, precision=4):
    if 10.0 ** -(precision - 1) <= abs(value) <= 10.0 ** (width - precision - 2):
        return f"{value:{alignment}{width}.{precision}f}"
    else:
        return f"{value:{alignment}{width}.{precision}e}"


def is_scalar_vector(value) -> bool:
    """
        Determines if the input value is a scalar, a 1-dimensional array, or a 2-dimensional array
        with at least one dimension of size 1 (effectively a scalar or vector).

        Args:
            value: The input value to be checked. It can be a number or a numpy array.

        Returns:
            bool: True if the input is a scalar, a 1-dimensional array, or a 2-dimensional array
                  with at least one dimension of size 1. False otherwise.

        Notes:
            - Scalars include any instances of numbers.Number.
            - 1-dimensional arrays (vectors) are considered as scalar vectors.
            - 2-dimensional arrays are considered scalar vectors if one of their dimensions is of size 1.

        Examples:
            >>> is_scalar_vector(5)
            True
            >>> is_scalar_vector(np.array(5))
            True
            >>> is_scalar_vector(np.array([1, 2, 3]))
            True
            >>> is_scalar_vector(np.array([[1], [2], [3]]))
            True
            >>> is_scalar_vector(np.array([[1, 2, 3], [4, 5, 6]]))
            False
        """
    if isinstance(value, numbers.Number):
        return True
    if not isinstance(value, np.ndarray):
        return False
    else:
        if np.ndim(value) in [0, 1]:
            return True
        elif np.ndim(value) == 2:
            if any(z == 1 for z in np.shape(value)):
                return True
            else:
                return False
        else:
            return False
