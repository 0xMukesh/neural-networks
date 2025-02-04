package parsers

import (
	"encoding/binary"
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

func NewMnistParser(images, labels string) MNISTParser {
	return MNISTParser{
		Images: images,
		Labels: labels,
	}
}

func (p MNISTParser) ReadImages() ([]MNISTImage, error) {
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

	images := make([]MNISTImage, count)
	for i := 0; i < int(count); i++ {
		var img MNISTImage
		binary.Read(file, binary.BigEndian, &img.Data)
		images[i] = img
	}

	return images, nil
}

func (p MNISTParser) ReadLabels() ([][]MNISTLabel, error) {
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

	oneHotEncoded := make([][]MNISTLabel, count)

	for i := range oneHotEncoded {
		oneHotEncoded[i] = make([]MNISTLabel, 10)
	}

	for i := range labels {
		oneHotEncoded[i][labels[i]] = 1
	}

	return oneHotEncoded, nil
}
