package utils

import (
	m "nn/math"

	"golang.org/x/exp/constraints"
)

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

type Number interface {
	constraints.Integer | constraints.Float
}

func Mean[T Number](s []T) float64 {
	sum := Reduce(s, T(0.0), func(acc, curr T) T {
		return acc + curr
	})

	return float64(sum) / float64(len(s))
}

func LinSpace(start, end float64, num int) m.Vector {
	result := make(m.Vector, num)
	step := (end - start) / float64(num-1)

	for i := range result {
		result[i] = start + float64(i)*step
	}

	return result
}
