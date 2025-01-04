package losses

import (
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
