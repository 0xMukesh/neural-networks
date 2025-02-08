package optimizers

import "nn/neural"

type SGD struct {
	InitialLR, CurrentLR, LRDecayRate, NIterations float64
}

func NewSGD(learningRate, learningRateDecay float64) *SGD {
	return &SGD{
		InitialLR:   learningRate,
		CurrentLR:   learningRate,
		LRDecayRate: learningRateDecay,
		NIterations: 0,
	}
}

func (sgd *SGD) PreUpdateParams() {
	sgd.CurrentLR = sgd.InitialLR / (1.0 + (sgd.LRDecayRate * sgd.NIterations))
}

func (sgd *SGD) UpdateParams(dl *neural.DenseLayer) {
	for i := range dl.Weights {
		for j := range dl.Weights[i] {
			dl.Weights[i][j] -= sgd.CurrentLR * dl.DWeight[i][j]
		}
	}

	for i := range dl.Bias {
		for j := range dl.Bias[i] {
			// dl.Bias is a row matrix
			// dl.DBias is a column matrix
			dl.Bias[i][j] -= sgd.CurrentLR * dl.DBias.Transpose()[i][j]
		}
	}
}

func (sgd *SGD) PostUpdateParams() {
	sgd.NIterations++
}
