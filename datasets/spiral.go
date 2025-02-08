package datasets

import (
	"math"
	"math/rand/v2"
	m "nn/math"
	"nn/utils"
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
