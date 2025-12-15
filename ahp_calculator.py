import numpy as np

def calculate_ahp_weights(matrix):
    """
    Calculate weights from a pairwise comparison matrix using the Analytic Hierarchy Process (AHP).
    This uses the simplified method of normalizing columns and averaging rows, which is a good
    approximation of the principal eigenvector method.

    Args:
        matrix (np.ndarray or list of lists): A square, reciprocal pairwise comparison matrix.
                                              Example for 3 factors:
                                              [[1, 3, 5],
                                               [1/3, 1, 3],
                                               [1/5, 1/3, 1]]

    Returns:
        np.ndarray: A 1D array of calculated weights for each factor.
        float: The consistency ratio (CR). A value > 0.1 may indicate inconsistent judgments.
    """
    matrix = np.array(matrix, dtype=float)
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        raise ValueError("Comparison matrix must be square.")

    # 1. Normalize the matrix by column
    col_sums = matrix.sum(axis=0)
    # Avoid division by zero if a column is all zeros
    col_sums[col_sums == 0] = 1 
    normalized_matrix = matrix / col_sums

    # 2. Calculate weights by averaging rows
    weights = normalized_matrix.mean(axis=1)

    # 3. Perform a consistency check
    # Calculate lambda_max
    weighted_sum_vector = np.dot(matrix, weights)
    lambda_max = np.mean(weighted_sum_vector / weights)
    
    # Calculate Consistency Index (CI)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    
    # Random Index (RI) from Saaty's table for n=1 to 10
    ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_table.get(n, 1.51) # Default for n > 10, though less reliable

    # Calculate Consistency Ratio (CR)
    cr = ci / ri if ri != 0 else 0

    return weights, cr
