package neural

import (
	"math/rand"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/utils"
)

type DenseLayer struct {
	Input, Weights    m.Matrix
	Bias              m.Vector
	NInputs, NNeurons int
}

func NewDenseLayer(nInputs, nNeurons int) *DenseLayer {
	dl := &DenseLayer{}

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

func (dl *DenseLayer) Forward(input m.Matrix) m.Matrix {
	output := m.Matrix{}

	for _, v := range input {
		output = append(output, dl.Weights.Multiply(v.ToMatrix()).ToVector().Add(dl.Bias))
	}

	dl.Input = input

	return output
}

func (dl DenseLayer) Backward(dvalues m.Matrix) (dw m.Matrix, di m.Matrix, db m.Vector) {
	// dvalues is next layer gradient
	// dw is partial derivative with respect to weights
	// di is partial derivative with respect to inputs
	// db is partial derivative with respect to biases

	// shape of dvalues is (n_samples, n_neurons)
	// shape of input is (n_samples, n_inputs)
	// shape of weight is (n_inputs, n_neurons)
	dw = dl.Input.Transpose().Multiply(dvalues)
	di = dvalues.Multiply(dl.Weights.Transpose())
	db = m.Vector{}

	// adding row-wise
	for _, v := range dvalues.Transpose() {
		sum := utils.Reduce(v, 0.0, func(acc, curr float64) float64 {
			return acc + curr
		})

		db = append(db, sum)
	}

	return
}
