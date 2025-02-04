package main

import (
	"fmt"
	"log"
	"nn/parsers"
)

func main() {
	parser := parsers.NewMnistParser("datasets/mnist/train-images", "datasets/mnist/train-labels")

	_, err := parser.ReadImages()
	if err != nil {
		log.Fatal(err.Error())
	}

	labels, err := parser.ReadLabels()
	if err != nil {
		log.Fatal(err.Error())
	}

	for _, v := range labels {
		fmt.Println(v)
	}
}
