package math

type Matrix [][]float64

func AllocateMatrix(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
	}

	return matrix
}

// m x n
func (m Matrix) Multiply(n Matrix) Matrix {
	if len(m) == 0 || len(n) == 0 || len(m[0]) == 0 || len(n[0]) == 0 {
		return nil
	}

	if len(m[0]) != len(n) {
		return nil
	}

	rows := len(m)
	cols := len(n[0])
	result := AllocateMatrix(rows, cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum := 0.0

			for k := 0; k < len(n); k++ {
				sum += m[i][k] * n[k][j]
			}

			result[i][j] = sum
		}
	}

	return result
}

// m + n
func (m Matrix) Add(n Matrix) Matrix {
	if (len(m) != len(n)) || (len(m[0]) != len(n[0])) {
		return nil
	}

	rows := len(m)
	cols := len(m[0])
	result := AllocateMatrix(rows, cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = m[i][j] + n[i][j]
		}
	}

	return result
}
