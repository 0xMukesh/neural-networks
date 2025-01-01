package neural

import (
	"math"

	m "github.com/0xmukesh/neural-networks/math"
)

// `targetClasses` is a matrix of one-hot vectors
// `input` is the result from softmax activation function
func CategoricalCrossEntropyLoss(targetClasses, input m.Matrix) m.Vector {
	predicatedClassesIdxs := []int{}
	losses := []float64{}

	for i := range targetClasses {
		for j := range targetClasses[i] {
			if targetClasses[i][j] == 1 {
				predicatedClassesIdxs = append(predicatedClassesIdxs, j)
			}
		}
	}

	for i, v := range predicatedClassesIdxs {
		predicatedValue := input[i][v]
		losses = append(losses, -math.Log(predicatedValue))
	}

	return losses
}
