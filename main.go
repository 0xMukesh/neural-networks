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

	targetClasses := m.Matrix{
		{0, 1, 0},
		{1, 0, 0},
		{0, 0, 1},
	}

	ccel := neural.CategoricalCrossEntropy{}
	losses := ccel.Loss(targetClasses, softmaxResult)

	for _, v := range softmaxResult {
		fmt.Println(v)
	}

	fmt.Println(strings.Repeat("=", 10))

	for _, v := range losses {
		fmt.Println(v)
	}

	avgLoss := neural.CalculateLoss(targetClasses, softmaxResult, ccel)
	fmt.Printf("avg loss - %f\n", avgLoss)
	accuracy := neural.CalculateAccuracy(targetClasses, softmaxResult)
	fmt.Printf("accuracy - %f\n", accuracy)
}
