package utils

import m "github.com/0xmukesh/neural-networks/math"

func RandMatrix(rows, cols int, rng func() float64) m.Matrix {
	matrix := m.AllocateMatrix(rows, cols)

	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = rng()
		}
	}

	return matrix
}

func ZeroMatrix(rows, cols int) m.Matrix {
	matrix := m.AllocateMatrix(rows, cols)

	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = 0
		}
	}

	return matrix
}