package datasets

import (
	"encoding/binary"
	m "nn/math"
	"os"
)

type MNISTParser struct {
	Images string
	Labels string
}

type MNISTImage struct {
	Data [28][28]uint8
}

type MNISTLabel uint8

// starting with 600 images
const DATASET_SIZE = 600

func NewMnistParser(images, labels string) MNISTParser {
	return MNISTParser{
		Images: images,
		Labels: labels,
	}
}

func (p MNISTParser) ReadImages() (m.Matrix, error) {
	file, err := os.Open(p.Images)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, count, rows, cols uint32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &count)
	binary.Read(file, binary.BigEndian, &rows)
	binary.Read(file, binary.BigEndian, &cols)

	imagesTensor := make([][28][28]uint8, count)

	for i := 0; i < DATASET_SIZE; i++ {
		var imgData [28][28]uint8
		binary.Read(file, binary.BigEndian, &imgData)
		imagesTensor[i] = imgData
	}

	imagesMatrix := m.AllocateMatrix(DATASET_SIZE, 28*28)

	for i := 0; i < DATASET_SIZE; i++ {
		for j := range imagesTensor[i] {
			for k := range imagesTensor[i][j] {
				imagesMatrix[i][28*j+k] = float64(imagesTensor[i][j][k])
			}
		}
	}

	return imagesMatrix, nil
}

func (p MNISTParser) ReadLabels() (m.Matrix, error) {
	file, err := os.Open(p.Labels)

	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic, count uint32
	binary.Read(file, binary.BigEndian, &magic)
	binary.Read(file, binary.BigEndian, &count)

	labels := make([]MNISTLabel, count)
	binary.Read(file, binary.BigEndian, &labels)

	oneHotEncoded := m.AllocateMatrix(DATASET_SIZE, 10)

	for i := 0; i < DATASET_SIZE; i++ {
		oneHotEncoded[i][labels[i]] = 1
	}

	return oneHotEncoded, nil
}
