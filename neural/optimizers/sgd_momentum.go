package optimizers

import (
	m "nn/math"
	"nn/neural"
)

// v_(t) = (momentum) * v_(t - 1) + (LR) * gradient
// w_(t + 1) = w_(t) - v_(t)
// momentum = 0.9 (generally)
type SGDMomentum struct {
	InitialLR, CurrentLR, LRDecayRate, NIterations, Momentum float64
}

func NewSGDMomentum(lr, lrDecayRate, momentum float64) *SGDMomentum {
	return &SGDMomentum{
		InitialLR:   lr,
		CurrentLR:   lr,
		LRDecayRate: lrDecayRate,
		Momentum:    momentum,
		NIterations: 0,
	}
}

func (sgd *SGDMomentum) PreUpdateParams() {
	sgd.CurrentLR = sgd.InitialLR / (1.0 + (sgd.LRDecayRate * sgd.NIterations))
}

func (sgd *SGDMomentum) UpdateParams(dl *neural.DenseLayer) {
	if dl.WeightMomentum == nil || dl.BiasMomentum == nil {
		dl.WeightMomentum = m.AllocateMatrix(dl.Weights.Rows(), dl.Weights.Cols())
		dl.BiasMomentum = m.AllocateMatrix(dl.Bias.Rows(), dl.Bias.Cols())
	}

	for i := range dl.Weights {
		for j := range dl.Weights[i] {
			dl.WeightMomentum[i][j] = sgd.Momentum*dl.WeightMomentum[i][j] + sgd.CurrentLR*dl.DWeight[i][j]
			dl.Weights[i][j] = dl.Weights[i][j] - dl.WeightMomentum[i][j]
		}
	}

	for i := range dl.Bias {
		for j := range dl.Bias[i] {
			dl.BiasMomentum[i][j] = sgd.Momentum*dl.BiasMomentum[i][j] + sgd.CurrentLR*dl.DBias.Transpose()[i][j]
			dl.Bias[i][j] = dl.Bias[i][j] - dl.BiasMomentum[i][j]
		}
	}

}

func (sgd *SGDMomentum) PostUpdateParams() {
	sgd.NIterations++
}
