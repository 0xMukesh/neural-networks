package neural

import (
	"math"
	"math/rand"

	m "nn/math"
	"nn/utils"
)

type DenseLayer struct {
	Input, Output, Weights, Bias m.Matrix
	DWeight, DInput, DBias       m.Matrix
	NInputs, NNeurons            int
}

func NewDenseLayer(nInputs, nNeurons int) *DenseLayer {
	dl := &DenseLayer{}

	stddev := math.Sqrt(2.0 / float64(nInputs))
	dl.Weights = utils.RandMatrix(nInputs, nNeurons, func() float64 {
		return rand.NormFloat64() * stddev
	})
	dl.Bias = m.AllocateMatrix(1, nNeurons)

	dl.NInputs = nInputs
	dl.NNeurons = nNeurons

	return dl
}

func (dl *DenseLayer) Forward(input m.Matrix) {
	dl.Input = input
	dl.Output = m.AllocateMatrix(len(input), dl.NNeurons)

	for i, v := range input {
		dl.Output[i] = v.ToRowMatrix().Multiply(dl.Weights).Add(dl.Bias).ToFloatSlice()
	}
}

func (dl *DenseLayer) Backward(dvalues m.Matrix) {
	dl.DWeight = dl.Input.Transpose().Multiply(dvalues)
	dl.DInput = dvalues.Multiply(dl.Weights.Transpose())

	J := m.AllocateMatrix(dl.Bias.Cols(), 1)

	for i := range J {
		for j := range J[i] {
			J[i][j] = 1
		}
	}

	dl.DBias = dvalues.Multiply(J)
}
