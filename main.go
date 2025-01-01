package main

import (
	"fmt"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/neural"
)

func main() {
	inputs := m.Matrix{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	}

	denselayer := neural.NewDenseLayer(4, 3)
	result := denselayer.Forward(inputs)

	for _, v := range result {
		fmt.Println(v)
	}
}
