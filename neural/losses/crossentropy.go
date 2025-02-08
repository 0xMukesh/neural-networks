package losses

import (
	"math"

	m "nn/math"
)

type CategoricalCrossEntropyLoss struct {
	NetworkOutput, TargetClasses m.Matrix
	DInput                       m.Matrix
}

func NewCcel() *CategoricalCrossEntropyLoss {
	return &CategoricalCrossEntropyLoss{}
}

func (l *CategoricalCrossEntropyLoss) Loss(targetClasses, networkOutput m.Matrix) m.Vector {
	l.NetworkOutput = networkOutput
	l.TargetClasses = targetClasses

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
		predicatedValue := networkOutput[i][v]

		if predicatedValue == 0 {
			losses = append(losses, 0)
		} else {
			losses = append(losses, -math.Log(predicatedValue))
		}
	}

	return losses
}

func (l *CategoricalCrossEntropyLoss) Backward() {
	l.DInput = m.AllocateMatrix(l.TargetClasses.Rows(), l.TargetClasses.Cols())

	for i := range l.TargetClasses {
		for j := range l.TargetClasses[i] {
			l.DInput[i][j] = -l.TargetClasses[i][j] / l.NetworkOutput[i][j]
		}
	}

	l.DInput = l.DInput.ForEach(func(f float64) float64 {
		return f / float64(len(l.NetworkOutput))
	})
}
