package optimizers

import "nn/neural"

// w_(t + 1) = w_(t) - (LR) * (gradient)
type SGD struct {
	InitialLR, CurrentLR, LRDecayRate, NIterations float64
}

func NewSGD(lr, lrDecay float64) *SGD {
	return &SGD{
		InitialLR:   lr,
		CurrentLR:   lr,
		LRDecayRate: lrDecay,
		NIterations: 0,
	}
}

func (sgd *SGD) PreUpdateParams() {
	sgd.CurrentLR = sgd.InitialLR / (1.0 + (sgd.LRDecayRate * sgd.NIterations))
}

func (sgd *SGD) UpdateParams(dl *neural.DenseLayer) {
	dl.Weights = dl.Weights.Subtract(dl.DWeight).ForEach(func(f float64) float64 {
		return f * sgd.CurrentLR
	})

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
