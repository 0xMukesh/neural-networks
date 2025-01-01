package neural

import (
	"math"
	"slices"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/utils"
)

// ReLU is used for applying non-linear transformation on the network
func ReluActivationFunc(input m.Matrix) m.Matrix {
	output := m.AllocateMatrix(input.Rows(), input.Cols())

	for i := range input {
		for j := range input[i] {
			output[i][j] = math.Max(0, input[i][j])
		}
	}

	return output
}

// softmax activation function takes in unnormalized data from ReLU and converts it into normalized i.e. the values lie in between [0, 1] and the values are inclusive i.e. all of the always add up to 1 as they are probablities/confidence scores
func SoftmaxActivationFunc(input m.Matrix) m.Matrix {
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

	return output
}
