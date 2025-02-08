package optimizers

import "nn/neural"

type SGD struct {
	LearningRate float64
}

func NewSGD(learningRate float64) *SGD {
	return &SGD{
		LearningRate: learningRate,
	}
}

func (sgd *SGD) UpdateParams(dl *neural.DenseLayer) {
	for i := range dl.Weights {
		for j := range dl.Weights[i] {
			dl.Weights[i][j] -= sgd.LearningRate * dl.DWeight[i][j]
		}
	}

	for i := range dl.Bias {
		for j := range dl.Bias[i] {
			// dl.Bias is a row matrix
			// dl.DBias is a column matrix
			dl.Bias[i][j] -= sgd.LearningRate * dl.DBias.Transpose()[i][j]
		}
	}
}
