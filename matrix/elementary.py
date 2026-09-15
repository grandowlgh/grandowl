```python
import torch


def rowswap(matrix, source, target):
    """
    Swap two rows of a matrix.

    Parameters:
        matrix: PyTorch tensor
        source: index of the source row
        target: index of the target row

    Returns:
        PyTorch tensor with the two rows swapped.
    """
    result = matrix.clone()

    temp = result[source].clone()
    result[source] = result[target]
    result[target] = temp

    return result


def rowscale(matrix, source, factor):
    """
    Scale a row by a given factor.

    Parameters:
        matrix: PyTorch tensor
        source: index of the row to scale
        factor: scaling factor

    Returns:
        PyTorch tensor with the specified row scaled.
    """
    result = matrix.clone()

    result[source] = result[source] * factor

    return result


def rowreplacement(matrix, first_row, second_row, j, k):
    """
    Perform the elementary row operation:

        jR_i + kR_j

    where first_row is R_i and second_row is R_j.

    Parameters:
        matrix: PyTorch tensor
        first_row: index of the first row
        second_row: index of the second row
        j: scaling factor for the first row
        k: scaling factor for the second row

    Returns:
        PyTorch tensor after the row replacement.
    """
    result = matrix.clone()

    result[first_row] = (
        j * result[first_row] +
        k * result[second_row]
    )

    return result


def rref(matrix):
    """
    Convert a matrix into row echelon form according to the
    assignment requirements.

    Each pivot is made equal to 1, and all entries below
    each pivot are made equal to 0.

    Parameters:
        matrix: PyTorch tensor

    Returns:
        PyTorch tensor in row echelon form.
    """
    result = matrix.clone().to(torch.float64)

    rows, cols = result.shape
    pivot_row = 0

    for col in range(cols):

        # Stop if all rows have been processed
        if pivot_row >= rows:
            break

        # Find a row with a nonzero entry in the current column
        pivot = pivot_row

        while pivot < rows and result[pivot, col] == 0:
            pivot += 1

        # If there is no pivot in this column, move to next column
        if pivot == rows:
            continue

        # Swap the pivot row into the correct position
        if pivot != pivot_row:
            result = rowswap(result, pivot, pivot_row)

        # Make the pivot equal to 1
        pivot_value = result[pivot_row, col]

        if pivot_value != 1:
            result = rowscale(
                result,
                pivot_row,
                1.0 / pivot_value
            )

        # Make all entries below the pivot equal to zero
        for row in range(pivot_row + 1, rows):

            value = result[row, col]

            if value != 0:
                result = rowreplacement(
                    result,
                    row,
                    pivot_row,
                    1,
                    -value
                )

        pivot_row += 1

    return result


if __name__ == "__main__":

    # Test matrix from the homework
    matrix = torch.tensor([
        [1, 3, 0, 0, 3],
        [0, 0, 1, 0, 9],
        [0, 0, 0, 1, -4]
    ], dtype=torch.float64)

    print("Original matrix:")
    print(matrix)

    # R1 <-> R2
    matrix = rowswap(matrix, 0, 1)

    print("\nAfter R1 <-> R2:")
    print(matrix)

    # (1/3)R1
    matrix = rowscale(matrix, 0, 1 / 3)

    print("\nAfter (1/3)R1:")
    print(matrix)

    # R3 = -3R1 + R3
    #
    # first_row = 2 (R3)
    # second_row = 0 (R1)
    # j = 1
    # k = -3
    matrix = rowreplacement(matrix, 2, 0, 1, -3)

    print("\nAfter R3 = -3R1 + R3:")
    print(matrix)

    # Test rref
    print("\nRREF / row echelon result:")
    print(rref(matrix))
```
