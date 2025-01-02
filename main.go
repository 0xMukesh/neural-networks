package main

import (
	"fmt"
	"math"
	"math/rand/v2"

	m "github.com/0xmukesh/neural-networks/math"
	"github.com/0xmukesh/neural-networks/neural"
	"github.com/0xmukesh/neural-networks/utils"
)

func main() {
	inputs := m.Matrix{
		{1.0, 2.0, 3.0, 2.5},
		{2.0, 5.0, -1.0, 2.0},
		{-1.5, 2.7, 3.3, -0.8},
	}

	targetClasses := m.Matrix{
		{0, 1, 0},
		{1, 0, 0},
		{0, 0, 1},
	}

	// 4 input neurons and 3 output neurons
	// weights = 3x4
	// biases = 3x1
	dl1 := neural.NewDenseLayer(4, 3)
	// 3 input neurons and 3 output neurons
	// weights = 3x3
	// biases = 3x1
	dl2 := neural.NewDenseLayer(3, 3)

	lowestLoss := math.MaxFloat64
	optimalDl1Biases := dl1.Bias
	optimalDl1Weights := dl1.Weights
	optimalDl2Biases := dl2.Bias
	optimalDl2Weights := dl2.Weights

	// iterating 100 times as the input data is very small
	for i := 1; i <= 100; i++ {
		dl1.Weights = dl1.Weights.Add(utils.RandMatrix(3, 4, func() float64 {
			return 0.05 * ((rand.Float64() * 2) - 1)
		}))
		dl1.Bias = dl1.Bias.Add(utils.RandMatrix(3, 1, func() float64 {
			return 0.05 * ((rand.Float64() * 2) - 1)
		}).ToVector())

		dl2.Weights = dl2.Weights.Add(utils.RandMatrix(3, 3, func() float64 {
			return 0.05 * ((rand.Float64() * 2) - 1)
		}))
		dl2.Bias = dl2.Bias.Add(utils.RandMatrix(3, 1, func() float64 {
			return 0.05 * ((rand.Float64() * 2) - 1)
		}).ToVector())

		dl1Result := dl1.Forward(inputs)
		reluResult := neural.ReluActivationFunc(dl1Result)
		dl2Result := dl2.Forward(reluResult)
		softmaxResult := neural.SoftmaxActivationFunc(dl2Result)

		ccel := neural.CategoricalCrossEntropyLoss{}
		avgLoss := neural.CalculateLoss(targetClasses, softmaxResult, ccel)

		if avgLoss < lowestLoss {
			optimalDl1Biases = dl1.Bias
			optimalDl1Weights = dl1.Weights
			optimalDl2Biases = dl2.Bias
			optimalDl2Weights = dl2.Weights

			lowestLoss = avgLoss

			accuracy := neural.CalculateAccuracy(targetClasses, softmaxResult)

			fmt.Printf("avg loss - %f and accuracy - %f\n", lowestLoss, accuracy)
		} else {
			// if the loss is greater than the lowest loss then revert back to the **current** optimal biases and weights for DL1 and DL2
			dl1.Weights = optimalDl1Weights
			dl1.Bias = optimalDl1Biases
			dl2.Weights = optimalDl2Weights
			dl2.Bias = optimalDl2Biases
		}
	}

	fmt.Printf("dl1 biases - %v", optimalDl1Biases)
	fmt.Printf("dl2 biases - %v", optimalDl2Biases)
	fmt.Printf("dl1 weights - %v", optimalDl1Weights)
	fmt.Printf("dl2 weights - %v", optimalDl2Weights)
}
