package activations

import (
	"math"

	m "github.com/0xmukesh/neural-networks/math"
)

type Relu struct {
	Input m.Matrix
}

func NewRelu(input m.Matrix) Relu {
	return Relu{
		Input: input,
	}
}

// f(x) = max(0, x)
func (r Relu) Forward() m.Matrix {
	input := r.Input
	output := m.AllocateMatrix(input.Rows(), input.Cols())

	for i := range input {
		for j := range input[i] {
			output[i][j] = math.Max(0, input[i][j])
		}
	}

	return output
}
