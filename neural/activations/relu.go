package activations

import (
	"math"

	m "nn/math"
)

type Relu struct {
	Input, Output m.Matrix
	DInput        m.Matrix
}

func NewRelu() *Relu {
	return &Relu{}
}

func (r *Relu) Forward(input m.Matrix) {
	r.Output = m.AllocateMatrix(input.Rows(), input.Cols())

	for i := range input {
		for j := range input[i] {
			r.Output[i][j] = math.Max(0, input[i][j])
		}
	}

	r.Input = input
}

func (r *Relu) Backward(dvalues m.Matrix) {
	r.DInput = dvalues

	for i := range r.Input {
		for j := range r.Input[i] {
			if r.Input[i][j] <= 0 {
				r.DInput[i][j] = 0
			}
		}
	}
}
