package activations

import (
	"math"
	"slices"

	m "nn/math"
	"nn/utils"
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
func (s *Softmax) Backward(outputs, dvalues m.Matrix) m.Matrix {
	tensor := make(m.Tensor, len(outputs))

	for i, output := range outputs {
		jacobianMatrix := m.AllocateMatrix(len(output), len(output))

		for i := range output {
			for j := range output {
				kroneckerDelta := 0.0
				if i == j {
					kroneckerDelta = 1.0
				}

				jacobianMatrix[i][j] = output[i] * (kroneckerDelta - output[j])
			}
		}

		tensor[i] = jacobianMatrix
	}

	output := m.AllocateMatrix(outputs.Rows(), outputs.Cols())

	for i := range output {
		for j := range output[i] {
			output[i][j] = dvalues[i].Dot(tensor[i][j])
		}
	}

	return output
}
