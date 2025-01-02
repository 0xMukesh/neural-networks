package neural

import (
	"math/rand"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/utils"
)

type DenseLayer struct {
	Weights           m.Matrix
	Bias              m.Vector
	NInputs, NNeurons int
}

func NewDenseLayer(nInputs, nNeurons int) DenseLayer {
	dl := DenseLayer{}

	// TODO: use gaussian distribution to generate random matrices with given size
	dl.Weights = utils.RandMatrix(nNeurons, nInputs, func() float64 {
		// transforms range of rand.Float64() from [0, 1) to [-1, 1)
		return rand.Float64()*2 - 1
	})
	dl.Bias = utils.ZeroMatrix(nNeurons, 1).ToVector()
	dl.NInputs = nInputs
	dl.NNeurons = nNeurons

	return dl
}

func (dl DenseLayer) Forward(input m.Matrix) m.Matrix {
	output := m.Matrix{}

	for _, v := range input {
		output = append(output, dl.Weights.Multiply(v.ToMatrix()).ToVector().Add(dl.Bias))
	}

	return output
}
