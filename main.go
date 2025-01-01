package main

import (
	"fmt"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/neural"
	"github.com/0xmukesh/neural-networks/neural/activations"
)

func main() {
	inputs := m.Matrix{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	}

	denselayer := neural.NewDenseLayer(inputs, 4, 3)
	dlResult := denselayer.Forward()

	for _, v := range dlResult {
		fmt.Println(v)
	}

	reluActivation := activations.NewRelu(dlResult)
	reluResult := reluActivation.Forward()

	for _, v := range reluResult {
		fmt.Println(v)
	}
}
