package neural

import (
	"math/rand"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/utils"
)

type DenseLayer struct {
	Weights m.Matrix
	Bias    m.Vector
}

func NewDenseLayer(nInputs, nNeurons int) DenseLayer {
	dl := DenseLayer{}

	// TODO: use gaussian distribution to generate random matrices with given size
	dl.Weights = utils.RandMatrix(nNeurons, nInputs, func() float64 {
		// transforms range of rand.Float64() from [0, 1) to [-1, 1)
		return rand.Float64()*2 - 1
	})
	dl.Bias = utils.ZeroMatrix(nNeurons, 1).ToVector()

	return dl
}

func (dl DenseLayer) Forward(input m.Matrix) m.Matrix {
	output := input.Multiply(dl.Weights.Transpose()).AddVector(dl.Bias)
	return output
}
