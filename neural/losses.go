package neural

import (
	"math"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/utils"
)

type LossFn interface {
	Loss(m.Matrix, m.Matrix) m.Vector
}

func CalculateLoss(targetClasses, input m.Matrix, lossFn LossFn) float64 {
	losses := lossFn.Loss(targetClasses, input)
	return utils.Mean(losses)
}

func CalculateAccuracy(targetClasses, input m.Matrix) float64 {
	predicatedClassesIdx := []int{}

	for i := range targetClasses {
		for j := range targetClasses[i] {
			if targetClasses[i][j] == 1 {
				predicatedClassesIdx = append(predicatedClassesIdx, j)
			}
		}
	}

	accuracies := []int{}

	for i, v := range predicatedClassesIdx {
		maxElemIdx := utils.MaxElemIdx(input[i])
		predicatedClassIdx := v

		if maxElemIdx == predicatedClassIdx {
			accuracies = append(accuracies, 1)
		} else {
			accuracies = append(accuracies, 0)
		}
	}

	return utils.Mean(accuracies)
}

// categorical cross entropy is a common choice for loss function whenever softmax activation function is used on the output layer
type CategoricalCrossEntropyLoss struct{}

// `targetClasses` is a matrix of one-hot vectors
// TODO: check if `targetClasses` is _actually_ a matrix of one-hot vectors
// `input` is the result from softmax activation function
func (l CategoricalCrossEntropyLoss) Loss(targetClasses, input m.Matrix) m.Vector {
	// techinically, one-hot vectors can either be an array containing 0s and 1s where 1 indicates the desired predication ([0, 0, 1, 0] - predication at index = 2 is the desired predication)
	// but they can also be sparse i.e. contain index of the desired predications
	// `CategoricalCrossEntropyLoss`function doesn't support sparse one-hot vectors at the moment but according to the defination, `predicatedClassesIdxs` act like one-hot vector
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
