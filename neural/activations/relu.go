package activations

import (
	"math"

	m "nn/math"
)

type Relu struct {
	Input m.Matrix
}

func NewRelu() *Relu {
	return &Relu{}
}

func (r *Relu) Forward(input m.Matrix) m.Matrix {
	output := m.AllocateMatrix(input.Rows(), input.Cols())

	for i := range input {
		for j := range input[i] {
			output[i][j] = math.Max(0, input[i][j])
		}
	}

	r.Input = input

	return output
}

func (r *Relu) Backward(dvalues m.Matrix) m.Matrix {
	drelu := dvalues

	for i := range r.Input {
		for j := range r.Input[i] {
			if r.Input[i][j] <= 0 {
				drelu[i][j] = 0
			}
		}
	}

	return drelu
}
