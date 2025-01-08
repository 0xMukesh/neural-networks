package losses

import (
	"math"

	m "github.com/0xmukesh/neural-networks/math"
)

// categorical cross entropy is a common choice for loss function whenever softmax activation function is used on the output layer
type CategoricalCrossEntropyLoss struct{}

// `targetClasses` is a matrix of one-hot vectors
// TODO: check if `targetClasses` is _actually_ a matrix of one-hot vectors
// `input` is the result from softmax activation function
func (l CategoricalCrossEntropyLoss) Loss(targetClasses, input m.Matrix) m.Vector {
	// techinically, one-hot vectors can either be an array containing 0s and 1s where 1 indicates the desired predication ([0, 0, 1, 0] - predication at index = 2 is the desired predication)
	// but they can also be sparse i.e. contain index of the desired predications
	// `CategoricalCrossEntropyLoss`function doesn't support sparse one-hot vectors at the moment but according to the defination
	// `predicatedClassesIdxs` acts like one-hot vector
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

// partial derivate of cross entropy loss function is -y/(y_hat)
func (l CategoricalCrossEntropyLoss) Backward(targetClasses, dvalues m.Matrix) m.Matrix {
	output := m.AllocateMatrix(targetClasses.Rows(), targetClasses.Cols())

	for i := range targetClasses {
		for j := range targetClasses[i] {
			if targetClasses[i][j] == 1 {
				output[i][j] = -targetClasses[i][j] / dvalues[i][j]
			}
		}
	}

	// normalizing the gradient
	output = output.ForEach(func(f float64) float64 {
		return f / float64(len(dvalues))
	})

	return output
}
