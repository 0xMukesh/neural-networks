package main

import (
	"fmt"

	m "github.com/0xmukesh/neural-networks/math"
)

func main() {
	inputs := m.Matrix{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	}
	weights := m.Matrix{
		{0.2, 0.8, -0.5, 1.0},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87},
	}
	biases := m.Vector{2.0, 3.0, 0.5}

	result := (inputs.Multiply(weights.Transpose())).AddVector(biases)

	for _, v := range result {
		fmt.Println(v)
	}
}
