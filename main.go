package main

import (
	"fmt"
	"strings"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/neural/activations"
	"github.com/0xmukesh/neural-networks/neural/losses"
)

func main() {
	softmaxOutputs := m.Matrix{
		{0.7, 0.1, 0.2},
		{0.1, 0.5, 0.4},
		{0.02, 0.9, 0.08},
	}
	classTargets := m.Matrix{
		{1, 0, 0},
		{0, 1, 0},
		{0, 1, 0},
	}

	ccel := losses.CategoricalCrossEntropyLoss{}
	softmax := activations.NewSoftmax()

	dvalues1 := ccel.Backward(classTargets, softmaxOutputs)
	dvalues2 := softmax.Backward(softmaxOutputs, dvalues1)

	for _, v := range dvalues2 {
		for _, u := range v {
			fmt.Println(u)
		}
		fmt.Println(strings.Repeat("=", 10))
	}
}
