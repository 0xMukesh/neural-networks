package main

import (
	"fmt"

	m "github.com/0xmukesh/neural-networks/math"
)

func main() {
	inputs := m.Matrix([][]float64{
		{1},
		{2},
		{3},
		{4},
	})
	weights := m.Matrix([][]float64{
		{1, 2, 3, 4},
		{1, 2, 3, 4},
		{1, 2, 3, 4},
	})
	bias := m.Matrix([][]float64{
		{1},
		{2},
		{3},
	})

	// result = bias + (weights * inputs)
	result := bias.Add(weights.Multiply(inputs))

	for _, v := range result {
		for _, w := range v {
			fmt.Println(w)
		}
	}
}
