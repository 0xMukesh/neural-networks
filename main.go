package main

import (
	"fmt"
	"strings"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/neural"
)

func main() {
	inputs := m.Matrix{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	}

	denselayer1 := neural.NewDenseLayer(4, 3)
	denselayer2 := neural.NewDenseLayer(3, 3)

	dl1Result := denselayer1.Forward(inputs)
	reluResult := neural.ReluActivationFunc(dl1Result)
	dl2Result := denselayer2.Forward(reluResult)
	softmaxResult := neural.SoftmaxActivationFunc(dl2Result)

	for _, v := range dl1Result {
		fmt.Println(v)
	}

	fmt.Println(strings.Repeat("=", 10))

	for _, v := range reluResult {
		fmt.Println(v)
	}

	fmt.Println(strings.Repeat("=", 10))

	for _, v := range dl2Result {
		fmt.Println(v)
	}

	fmt.Println(strings.Repeat("=", 10))

	for _, v := range softmaxResult {
		fmt.Println(v)
	}
}
