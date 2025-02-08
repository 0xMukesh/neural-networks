package losses

import (
	m "nn/math"
	"nn/utils"
)

type LossFn interface {
	Loss(targetClasses, output m.Matrix) m.Vector
}

func CalculateAccuracy(targetClasses, networkOutput m.Matrix) float64 {
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
		maxElemIdx := utils.MaxElemIdx(networkOutput[i])
		predicatedClassIdx := v

		if maxElemIdx == predicatedClassIdx {
			accuracies = append(accuracies, 1)
		} else {
			accuracies = append(accuracies, 0)
		}
	}

	return utils.Mean(accuracies)
}
