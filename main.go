package main

import (
	"fmt"
	"nn/datasets"
	"nn/neural"
	"nn/neural/activations"
	"nn/neural/losses"
	"nn/neural/optimizers"
	"nn/utils"
	"os"
)

func main() {
	cmd := os.Args[len(os.Args)-1]

	if cmd == "generate" {
		if err := datasets.GenerateAndSaveSpiralDataset("data/spiral_dataset", 200, 3); err != nil {
			fmt.Printf("an error occurred while generating the dataset - %s\n", err.Error())
			os.Exit(1)
		}

		return
	}

	X, Y, err := datasets.LoadSpiralDataset("data/spiral_dataset_train.csv")
	if err != nil {
		fmt.Printf("an error occurred while loading the dataset - %s\n", err.Error())
		os.Exit(1)
	}

	dl1 := neural.NewDenseLayer(2, 64)
	relu := activations.NewRelu()
	dl2 := neural.NewDenseLayer(64, 3)
	softmax := activations.NewSoftmax()
	ccel := losses.NewCcel()
	sgdMomentum := optimizers.NewSGDMomentum(0.4, 0.01, 0.9)

	for i := range 10_001 {
		dl1.Forward(X)
		relu.Forward(dl1.Output)
		dl2.Forward(relu.Output)
		softmax.Forward(dl2.Output)

		networkLosses := ccel.Loss(Y, softmax.Output)
		meanLoss := utils.Mean(networkLosses)
		meanAccuracy := losses.CalculateAccuracy(Y, softmax.Output)

		if i%100 == 0 {
			fmt.Printf("epoch - %d\n", i)
			fmt.Printf("mean loss - %f\n", meanLoss)
			fmt.Printf("mean accuracy - %f\n", meanAccuracy)
		}

		softmax.BackwardWithCcel(Y)
		dl2.Backward(softmax.DInput)
		relu.Backward(dl2.DInput)
		dl1.Backward(relu.DInput)

		sgdMomentum.PreUpdateParams()
		sgdMomentum.UpdateParams(dl1)
		sgdMomentum.UpdateParams(dl2)
		sgdMomentum.PostUpdateParams()
	}
}
