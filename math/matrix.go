package math

import (
	"fmt"
)

type Matrix []Vector

func AllocateMatrix(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make(Vector, cols)
	}

	return matrix
}

func (m Matrix) Rows() int {
	return len(m)
}

func (m Matrix) Cols() int {
	return len(m[0])
}

// convert a column matrix into a vector
func (m Matrix) ToVector() Vector {
	if m.Cols() != 1 {
		panic("not a column matrix")
	}

	vector := make(Vector, m.Rows())

	for i := range m {
		for j := range m[i] {
			vector[i] = m[i][j]
		}
	}

	return vector
}

// m_(axb) + n_(bxc)
func (m Matrix) Multiply(n Matrix) Matrix {
	if m.Rows() == 0 || n.Rows() == 0 || m.Cols() == 0 || n.Cols() == 0 {
		panic("invalid dimensions")
	}

	if m.Cols() != n.Rows() {
		panic(fmt.Errorf("invalid operation, tried to multiply %dx%d matrix with %dx%d matrix", m.Rows(), m.Cols(), n.Rows(), n.Cols()))
	}

	result := AllocateMatrix(m.Rows(), n.Cols())

	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < n.Cols(); j++ {
			sum := 0.0

			for k := 0; k < n.Rows(); k++ {
				sum += m[i][k] * n[k][j]
			}

			result[i][j] = sum
		}
	}

	return result
}

// m_(axb) + n_(axb)
func (m Matrix) Add(n Matrix) Matrix {
	if (m.Rows() != n.Rows()) || (m.Cols() != n.Cols()) {
		panic("invalid dimensions")
	}

	result := AllocateMatrix(m.Rows(), n.Cols())

	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < n.Cols(); j++ {
			result[i][j] = m[i][j] + n[i][j]
		}
	}

	return result
}

func (m Matrix) Subtract(n Matrix) Matrix {
	return m.Add(n.ForEach(func(f float64) float64 {
		return f * -1
	}))
}

// m_(axb) + v_(a); adds vector `v` to each row vector of matrix `m`
func (m Matrix) AddVector(v Vector) Matrix {
	if m.Rows() != v.Size() {
		panic("invalid dimensions")
	}

	result := AllocateMatrix(m.Rows(), m.Cols())

	for i := range m {
		result[i] = m[i].Add(v)
	}

	return result
}

// m_(axb) -> n_(bxa)
func (m Matrix) Transpose() Matrix {
	result := AllocateMatrix(m.Cols(), m.Rows())

	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			result[j][i] = m[i][j]
		}
	}

	return result
}

// applies function `f` on every single element in matrix `m`
func (m Matrix) ForEach(f func(float64) float64) Matrix {
	output := AllocateMatrix(m.Rows(), m.Cols())

	for i := range m {
		for j := range m[i] {
			output[i][j] = f(m[i][j])
		}
	}

	return output
}
