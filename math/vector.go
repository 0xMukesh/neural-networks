package math

import "fmt"

type Vector []float64

func (v Vector) Size() int {
	return len(v)
}

func (v Vector) ToRowMatrix() Matrix {
	m := AllocateMatrix(1, len(v))

	for i := range m {
		for j := range m[i] {
			m[i][j] = v[j]
		}
	}

	return m
}

func (v Vector) Add(u Vector) Vector {
	if v.Size() != u.Size() {
		panic("invalid dimensions")
	}

	vector := make(Vector, v.Size())

	for i := range v {
		vector[i] = v[i] + u[i]
	}

	return vector
}

func (v Vector) Dot(u Vector) float64 {
	if len(v) != len(u) {
		panic(fmt.Errorf("invalid dimensions. dim(v) != dim(u); dim(v) = %d and dim(u) = %d", len(v), len(u)))
	}

	sum := 0.0

	for i := range v {
		sum += v[i] * u[i]
	}

	return sum
}

func (v Vector) Diagonalize() Matrix {
	m := AllocateMatrix(len(v), len(v))

	for i := range m {
		for j := range m[i] {
			if i == j {
				m[i][j] = v[i]
			}
		}
	}

	return m
}
