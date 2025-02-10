package datasets

import (
	"encoding/csv"
	"math"
	"math/rand/v2"
	m "nn/math"
	"nn/utils"
	"os"
	"strconv"
)

func NewSpiralData(numberOfPoints, numberOfClasses int) (m.Matrix, m.Matrix) {
	X := m.AllocateMatrix(numberOfPoints*numberOfClasses, 2)
	Y := m.AllocateMatrix(numberOfPoints*numberOfClasses, numberOfClasses)

	for c := range numberOfClasses {
		radius := utils.LinSpace(0, 1, numberOfPoints)
		t := utils.LinSpace(float64(c*4), float64((c+1)*4), numberOfPoints)

		for i := range t {
			t[i] += 0.2 * rand.NormFloat64()
		}

		for i := range numberOfPoints {
			X[c*numberOfPoints+i][0] = radius[i] * math.Sin(t[i]*2.5)
			X[c*numberOfPoints+i][1] = radius[i] * math.Cos(t[i]*2.5)
			Y[c*numberOfPoints+i][c] = 1
		}
	}

	return X, Y
}

func SaveSpiralDataset(filename string, X, Y m.Matrix, trainRatio float64) error {
	if trainRatio <= 0 || trainRatio >= 1 {
		trainRatio = 0.8
	}

	totalSamples := len(X)
	trainSamples := int(float64(totalSamples) * trainRatio)

	indices := make([]int, totalSamples)
	for i := range indices {
		indices[i] = i
	}
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	trainFile, err := os.Create(filename + "_train.csv")
	if err != nil {
		return err
	}
	defer trainFile.Close()

	testFile, err := os.Create(filename + "_test.csv")
	if err != nil {
		return err
	}
	defer testFile.Close()

	trainWriter := csv.NewWriter(trainFile)
	testWriter := csv.NewWriter(testFile)
	defer trainWriter.Flush()
	defer testWriter.Flush()

	header := []string{"x1", "x2"}
	for i := range Y[0] {
		header = append(header, "class_"+strconv.Itoa(i))
	}
	if err := trainWriter.Write(header); err != nil {
		return err
	}
	if err := testWriter.Write(header); err != nil {
		return err
	}

	for i := 0; i < trainSamples; i++ {
		row := []string{
			strconv.FormatFloat(X[indices[i]][0], 'f', 6, 64),
			strconv.FormatFloat(X[indices[i]][1], 'f', 6, 64),
		}
		for j := range Y[indices[i]] {
			row = append(row, strconv.FormatFloat(Y[indices[i]][j], 'f', 6, 64))
		}
		if err := trainWriter.Write(row); err != nil {
			return err
		}
	}

	for i := trainSamples; i < totalSamples; i++ {
		row := []string{
			strconv.FormatFloat(X[indices[i]][0], 'f', 6, 64),
			strconv.FormatFloat(X[indices[i]][1], 'f', 6, 64),
		}
		for j := range Y[indices[i]] {
			row = append(row, strconv.FormatFloat(Y[indices[i]][j], 'f', 6, 64))
		}
		if err := testWriter.Write(row); err != nil {
			return err
		}
	}

	return nil
}

func LoadSpiralDataset(filename string) (X, Y m.Matrix, err error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	records = records[1:]

	X = m.AllocateMatrix(len(records), 2)
	Y = m.AllocateMatrix(len(records), len(records[0])-2)

	for i, record := range records {
		X[i][0], _ = strconv.ParseFloat(record[0], 64)
		X[i][1], _ = strconv.ParseFloat(record[1], 64)

		for j := 2; j < len(record); j++ {
			Y[i][j-2], _ = strconv.ParseFloat(record[j], 64)
		}
	}

	return X, Y, nil
}

func GenerateAndSaveSpiralDataset(filename string, points, classes int) error {
	X, Y := NewSpiralData(points, classes)
	return SaveSpiralDataset(filename, X, Y, 0.8)
}
