package activations

import (
	"math"
	"slices"

	m "nn/math"
	"nn/utils"
)

type Softmax struct {
	Input, Output m.Matrix
	DInput        m.Matrix
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Forward(input m.Matrix) {
	s.Input = input
	s.Output = m.AllocateMatrix(input.Rows(), input.Cols())

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
			s.Output[i][j] = exps[j] / sum
		}
	}
}

func (s *Softmax) Backward(dvalues m.Matrix) {
	tensor := make(m.Tensor, len(s.Output))

	for i, output := range s.Output {
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

	s.DInput = m.AllocateMatrix(s.Output.Rows(), s.Output.Cols())

	for i := range s.DInput {
		for j := range s.DInput[i] {
			s.DInput[i][j] = dvalues[i].Dot(tensor[i][j])
		}
	}
}

func (s *Softmax) BackwardWithCcel(targetClasses m.Matrix) {
	s.DInput = m.AllocateMatrix(s.Output.Rows(), s.Output.Cols())

	for i := range s.Output {
		for j := range s.Output[i] {
			s.DInput[i][j] = (s.Output[i][j] - targetClasses[i][j]) / float64(len(s.Output))
		}
	}
}
