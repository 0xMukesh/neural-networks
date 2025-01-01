package neural

import (
	"math/rand"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/utils"
)

type DenseLayer struct {
	Input   m.Matrix
	Weights m.Matrix
	Bias    m.Vector
}

func NewDenseLayer(inputs m.Matrix, nInputs, nNeurons int) DenseLayer {
	dl := DenseLayer{}

	dl.Input = inputs
	// TODO: use gaussian distribution to generate random matrices with given size
	dl.Weights = utils.RandMatrix(nNeurons, nInputs, func() float64 {
		// transforms range of rand.Float64() from [0, 1) to [-1, 1)
		return 0.1 * (rand.Float64()*2 - 1)
	})
	dl.Bias = utils.ZeroMatrix(nNeurons, 1).ToVector()

	return dl
}

func (dl DenseLayer) Forward() m.Matrix {
	output := dl.Input.Multiply(dl.Weights.Transpose()).AddVector(dl.Bias)
	return output
}
