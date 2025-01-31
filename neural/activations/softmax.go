package activations

import (
	"math"
	"slices"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/utils"
)

type Softmax struct {
	Input m.Matrix
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

// softmax activation function takes in unnormalized data from ReLU and converts it into normalized i.e. the values lie in between [0, 1] and the values are inclusive i.e. all of the always add up to 1 as they are probablities/confidence scores
func (s *Softmax) Forward(input m.Matrix) m.Matrix {
	output := m.AllocateMatrix(input.Rows(), input.Cols())

	for i := range input {
		max := slices.Max(input[i])
		exps := []float64{}

		for j := range input[i] {
			exp := math.Exp(input[i][j] - max)
			exps = append(exps, exp)
		}

		sum := utils.Reduce(exps, 0, func(acc, curr float64) float64 {
			return acc + curr
		})

		for j := range input[i] {
			output[i][j] = exps[j] / sum
		}
	}

	s.Input = input

	return output
}

// (S_(i, j) * (delta)_(j, k)) - (S_(i, j) * S_(i, k))
func (s *Softmax) Backward(outputs, dvalues m.Matrix) m.Tensor {
	tensor := m.Tensor{}

	for _, output := range outputs {
		singleOutput := make(m.Vector, len(output))
		copy(singleOutput, output)

		jacobianMatrix := m.AllocateMatrix(len(output), len(output))

		for i := range singleOutput {
			jacobianMatrix[i][i] = singleOutput[i]
		}

		for i := range singleOutput {
			for j := range singleOutput {
				jacobianMatrix[i][j] -= singleOutput[i] * singleOutput[j]
			}
		}

		tensor = append(tensor, jacobianMatrix)
	}

	return tensor
}
