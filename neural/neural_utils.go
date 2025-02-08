package neural

import (
	m "nn/math"
)

func SoftmaxCcelBackwardPass(softmaxOutputs, targetClasses m.Matrix) m.Matrix {
	output := m.AllocateMatrix(softmaxOutputs.Rows(), softmaxOutputs.Cols())

	for i := range softmaxOutputs {
		for j := range softmaxOutputs[i] {
			output[i][j] = softmaxOutputs[i][j] - targetClasses[i][j]
		}
	}

	return output
}
