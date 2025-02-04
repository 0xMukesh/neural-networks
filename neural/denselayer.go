package neural

import (
	"math/rand"

	m "nn/math"
	"nn/utils"
)

type DenseLayer struct {
	Input, Weights    m.Matrix
	Bias              m.Vector
	NInputs, NNeurons int
}

func NewDenseLayer(nInputs, nNeurons int) *DenseLayer {
	dl := &DenseLayer{}

	dl.Weights = utils.RandMatrix(nNeurons, nInputs, func() float64 {
		return rand.NormFloat64() * 0.1
	})
	dl.Bias = utils.ZeroMatrix(nNeurons, 1).ToVector()
	dl.NInputs = nInputs
	dl.NNeurons = nNeurons

	return dl
}

func (dl *DenseLayer) Forward(input m.Matrix) m.Matrix {
	output := m.Matrix{}

	for _, v := range input {
		output = append(output, dl.Weights.Multiply(v.ToMatrix()).ToVector().Add(dl.Bias))
	}

	dl.Input = input

	return output
}

func (dl DenseLayer) Backward(dvalues m.Matrix) (m.Matrix, m.Matrix, m.Matrix) {
	// dvalues is next layer gradient
	// dw is partial derivative with respect to weights
	// di is partial derivative with respect to inputs
	// db is partial derivative with respect to biases

	// shape of dvalues is (n_samples, n_neurons)
	// shape of input is (n_samples, n_inputs)
	// shape of weight is (n_inputs, n_neurons)
	dw := dl.Input.Transpose().Multiply(dvalues)
	di := dvalues.Multiply(dl.Weights.Transpose())

	J := m.AllocateMatrix(len(dl.Bias), 1)

	for i := range J {
		for j := range J[i] {
			J[i][j] = 1
		}
	}

	db := dvalues.Multiply(J)

	return dw, di, db
}
