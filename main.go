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
		{1.0, 2.0, 3.0, 2.5, 0.9, 1.1},
		{2.0, 5.0, -1.0, 2.0, 2.1, 5.5},
		{-1.5, 2.7, 3.3, -0.8, -0.9, -0.89},
		{-1.23, 0.85, -2.14, 1.67, -0.93, 1.44},
		{0.76, -1.89, 2.31, -0.92, 1.55, -2.11},
		{-1.67, 2.23, -0.84, 1.92, -1.43, 0.95},
		{2.14, -1.32, 0.88, -2.45, 1.76, -0.91},
		{-0.95, 1.78, -2.34, 0.87, -1.65, 2.21},
	}

	targetClasses := m.Matrix{
		{0, 1, 0},
		{1, 0, 0},
		{0, 0, 1},
	}

	// 6 input neurons and 5 output neurons
	// weights = 5x6
	// biases = 5x1
	dl1 := neural.NewDenseLayer(6, 5)
	// 5 input neurons and 5 output neurons
	// weights = 5x5
	// biases = 5x1
	dl2 := neural.NewDenseLayer(5, 5)

	lowestLoss := math.MaxFloat64
	optimalDl1Biases := dl1.Bias
	optimalDl1Weights := dl1.Weights
	optimalDl2Biases := dl2.Bias
	optimalDl2Weights := dl2.Weights

	// iterating 500 times as the input data is very small
	for i := 1; i <= 500; i++ {
		dl1.Weights = dl1.Weights.Add(utils.RandMatrix(5, 6, func() float64 {
			return 0.05 * ((rand.Float64() * 2) - 1)
		}))
		dl1.Bias = dl1.Bias.Add(utils.RandMatrix(5, 1, func() float64 {
			return 0.05 * ((rand.Float64() * 2) - 1)
		}).ToVector())

		dl2.Weights = dl2.Weights.Add(utils.RandMatrix(5, 5, func() float64 {
			return 0.05 * ((rand.Float64() * 2) - 1)
		}))
		dl2.Bias = dl2.Bias.Add(utils.RandMatrix(5, 1, func() float64 {
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

	fmt.Printf("dl1 biases - %v\n", optimalDl1Biases)
	fmt.Printf("dl2 biases - %v\n", optimalDl2Biases)
	fmt.Printf("dl1 weights - %v\n", optimalDl1Weights)
	fmt.Printf("dl2 weights - %v\n", optimalDl2Weights)
}
