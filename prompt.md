Project Path: neural-networks

Source Tree:

```
neural-networks
├── data
├── main.go
├── math
│   ├── matrix.go
│   ├── tensor.go
│   └── vector.go
├── datasets
│   ├── spiral.go
│   └── mnist.go
├── neural
│   ├── losses
│   │   ├── crossentropy.go
│   │   └── losses.go
│   ├── neural_utils.go
│   ├── optimizers
│   │   └── sgd.go
│   ├── activations
│   │   ├── relu.go
│   │   └── softmax.go
│   └── denselayer.go
├── go.sum
├── utils
│   ├── utils.go
│   └── math_utils.go
├── README.md
└── go.mod

```

`/home/mukesh/dev/from-scratch/neural-networks/main.go`:

```go
package main

import (
	"fmt"
	"nn/datasets"
	"nn/neural"
	"nn/neural/activations"
	"nn/neural/losses"
	"nn/neural/optimizers"
	"nn/utils"
)

func main() {
	X, Y := datasets.NewSpiralData(100, 3)

	dl1 := neural.NewDenseLayer(2, 64)
	relu := activations.NewRelu()
	dl2 := neural.NewDenseLayer(64, 3)
	softmax := activations.NewSoftmax()
	ccel := losses.NewCcel()
	sgd := optimizers.NewSGD(0.90)

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

		sgd.UpdateParams(dl1)
		sgd.UpdateParams(dl2)
	}
}

```

`/home/mukesh/dev/from-scratch/neural-networks/math/matrix.go`:

```go
package math

import (
	"fmt"
)

type Matrix []Vector

func AllocateMatrix(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make(Vector, cols)
	}

	return matrix
}

func (m Matrix) Rows() int {
	return len(m)
}

func (m Matrix) Cols() int {
	return len(m[0])
}

// convert a row matrix into a slice of floats
func (m Matrix) ToFloatSlice() []float64 {
	if len(m) != 1 {
		panic("not a row matrix")
	}

	return m[0]
}

// m_(axb) + n_(bxc)
func (m Matrix) Multiply(n Matrix) Matrix {
	if m.Rows() == 0 || n.Rows() == 0 || m.Cols() == 0 || n.Cols() == 0 {
		panic("invalid dimensions")
	}

	if m.Cols() != n.Rows() {
		panic(fmt.Errorf("invalid operation, tried to multiply %dx%d matrix with %dx%d matrix", m.Rows(), m.Cols(), n.Rows(), n.Cols()))
	}

	result := AllocateMatrix(m.Rows(), n.Cols())

	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < n.Cols(); j++ {
			sum := 0.0

			for k := 0; k < n.Rows(); k++ {
				sum += m[i][k] * n[k][j]
			}

			result[i][j] = sum
		}
	}

	return result
}

// m_(axb) + n_(axb)
func (m Matrix) Add(n Matrix) Matrix {
	if (m.Rows() != n.Rows()) || (m.Cols() != n.Cols()) {
		panic("invalid dimensions")
	}

	result := AllocateMatrix(m.Rows(), n.Cols())

	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < n.Cols(); j++ {
			result[i][j] = m[i][j] + n[i][j]
		}
	}

	return result
}

func (m Matrix) Subtract(n Matrix) Matrix {
	return m.Add(n.ForEach(func(f float64) float64 {
		return f * -1
	}))
}

// m_(axb) + v_(a); adds vector `v` to each row vector of matrix `m`
func (m Matrix) AddVector(v Vector) Matrix {
	if m.Rows() != v.Size() {
		panic("invalid dimensions")
	}

	result := AllocateMatrix(m.Rows(), m.Cols())

	for i := range m {
		result[i] = m[i].Add(v)
	}

	return result
}

// m_(axb) -> n_(bxa)
func (m Matrix) Transpose() Matrix {
	result := AllocateMatrix(m.Cols(), m.Rows())

	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			result[j][i] = m[i][j]
		}
	}

	return result
}

// applies function `f` on every single element in matrix `m`
func (m Matrix) ForEach(f func(float64) float64) Matrix {
	output := AllocateMatrix(m.Rows(), m.Cols())

	for i := range m {
		for j := range m[i] {
			output[i][j] = f(m[i][j])
		}
	}

	return output
}

```

`/home/mukesh/dev/from-scratch/neural-networks/math/tensor.go`:

```go
package math

type Tensor []Matrix

func AllocateTensor(depth, rows, cols int) Tensor {
	tensor := make(Tensor, depth)

	for i := range tensor {
		tensor[i] = AllocateMatrix(rows, cols)
	}

	return tensor
}

```

`/home/mukesh/dev/from-scratch/neural-networks/math/vector.go`:

```go
package math

import "fmt"

type Vector []float64

func (v Vector) Size() int {
	return len(v)
}

func (v Vector) ToRowMatrix() Matrix {
	m := AllocateMatrix(1, len(v))

	for i := range m {
		for j := range m[i] {
			m[i][j] = v[j]
		}
	}

	return m
}

func (v Vector) Add(u Vector) Vector {
	if v.Size() != u.Size() {
		panic("invalid dimensions")
	}

	vector := make(Vector, v.Size())

	for i := range v {
		vector[i] = v[i] + u[i]
	}

	return vector
}

func (v Vector) Dot(u Vector) float64 {
	if len(v) != len(u) {
		panic(fmt.Errorf("invalid dimensions. dim(v) != dim(u); dim(v) = %d and dim(u) = %d", len(v), len(u)))
	}

	sum := 0.0

	for i := range v {
		sum += v[i] * u[i]
	}

	return sum
}

func (v Vector) Diagonalize() Matrix {
	m := AllocateMatrix(len(v), len(v))

	for i := range m {
		for j := range m[i] {
			if i == j {
				m[i][j] = v[i]
			}
		}
	}

	return m
}

```

`/home/mukesh/dev/from-scratch/neural-networks/datasets/spiral.go`:

```go
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

```

`/home/mukesh/dev/from-scratch/neural-networks/datasets/mnist.go`:

```go
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

```

`/home/mukesh/dev/from-scratch/neural-networks/neural/losses/crossentropy.go`:

```go
package losses

import (
	"math"

	m "nn/math"
)

type CategoricalCrossEntropyLoss struct {
	NetworkOutput, TargetClasses m.Matrix
	DInput                       m.Matrix
}

func NewCcel() *CategoricalCrossEntropyLoss {
	return &CategoricalCrossEntropyLoss{}
}

func (l *CategoricalCrossEntropyLoss) Loss(targetClasses, networkOutput m.Matrix) m.Vector {
	l.NetworkOutput = networkOutput
	l.TargetClasses = targetClasses

	predicatedClassesIdxs := []int{}
	losses := []float64{}

	for i := range targetClasses {
		for j := range targetClasses[i] {
			if targetClasses[i][j] == 1 {
				predicatedClassesIdxs = append(predicatedClassesIdxs, j)
			}
		}
	}

	for i, v := range predicatedClassesIdxs {
		predicatedValue := networkOutput[i][v]

		if predicatedValue == 0 {
			losses = append(losses, 0)
		} else {
			losses = append(losses, -math.Log(predicatedValue))
		}
	}

	return losses
}

func (l *CategoricalCrossEntropyLoss) Backward() {
	l.DInput = m.AllocateMatrix(l.TargetClasses.Rows(), l.TargetClasses.Cols())

	for i := range l.TargetClasses {
		for j := range l.TargetClasses[i] {
			l.DInput[i][j] = -l.TargetClasses[i][j] / l.NetworkOutput[i][j]
		}
	}

	l.DInput = l.DInput.ForEach(func(f float64) float64 {
		return f / float64(len(l.NetworkOutput))
	})
}

```

`/home/mukesh/dev/from-scratch/neural-networks/neural/losses/losses.go`:

```go
package losses

import (
	m "nn/math"
	"nn/utils"
)

type LossFn interface {
	Loss(targetClasses, output m.Matrix) m.Vector
}

func CalculateAccuracy(targetClasses, networkOutput m.Matrix) float64 {
	predicatedClassesIdx := []int{}

	for i := range targetClasses {
		for j := range targetClasses[i] {
			if targetClasses[i][j] == 1 {
				predicatedClassesIdx = append(predicatedClassesIdx, j)
			}
		}
	}

	accuracies := []int{}

	for i, v := range predicatedClassesIdx {
		maxElemIdx := utils.MaxElemIdx(networkOutput[i])
		predicatedClassIdx := v

		if maxElemIdx == predicatedClassIdx {
			accuracies = append(accuracies, 1)
		} else {
			accuracies = append(accuracies, 0)
		}
	}

	return utils.Mean(accuracies)
}

```

`/home/mukesh/dev/from-scratch/neural-networks/neural/neural_utils.go`:

```go
package neural

import (
	m "nn/math"
)

func SoftmaxCcelBackwardPass(softmaxOutputs, targetClasses m.Matrix) m.Matrix {
	output := m.AllocateMatrix(softmaxOutputs.Rows(), softmaxOutputs.Cols())

	for i := range softmaxOutputs {
		for j := range softmaxOutputs[i] {
			output[i][j] = softmaxOutputs[i][j] - targetClasses[i][j]
		}
	}

	return output
}

```

`/home/mukesh/dev/from-scratch/neural-networks/neural/optimizers/sgd.go`:

```go
package optimizers

import "nn/neural"

type SGD struct {
	LearningRate float64
}

func NewSGD(learningRate float64) *SGD {
	return &SGD{
		LearningRate: learningRate,
	}
}

func (sgd *SGD) UpdateParams(dl *neural.DenseLayer) {
	for i := range dl.Weights {
		for j := range dl.Weights[i] {
			dl.Weights[i][j] -= sgd.LearningRate * dl.DWeight[i][j]
		}
	}

	for i := range dl.Bias {
		for j := range dl.Bias[i] {
			// dl.Bias is a row matrix
			// dl.DBias is a column matrix
			dl.Bias[i][j] -= sgd.LearningRate * dl.DBias.Transpose()[i][j]
		}
	}
}

```

`/home/mukesh/dev/from-scratch/neural-networks/neural/activations/relu.go`:

```go
package activations

import (
	"math"

	m "nn/math"
)

type Relu struct {
	Input, Output m.Matrix
	DInput        m.Matrix
}

func NewRelu() *Relu {
	return &Relu{}
}

func (r *Relu) Forward(input m.Matrix) {
	r.Output = m.AllocateMatrix(input.Rows(), input.Cols())

	for i := range input {
		for j := range input[i] {
			r.Output[i][j] = math.Max(0, input[i][j])
		}
	}

	r.Input = input
}

func (r *Relu) Backward(dvalues m.Matrix) {
	r.DInput = dvalues

	for i := range r.Input {
		for j := range r.Input[i] {
			if r.Input[i][j] <= 0 {
				r.DInput[i][j] = 0
			}
		}
	}
}

```

`/home/mukesh/dev/from-scratch/neural-networks/neural/activations/softmax.go`:

```go
package activations

import (
	"math"
	"slices"

	m "nn/math"
	"nn/utils"
)

type Softmax struct {
	Input, Output m.Matrix
	DInput        m.Matrix
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

func (s *Softmax) Forward(input m.Matrix) {
	s.Input = input
	s.Output = m.AllocateMatrix(input.Rows(), input.Cols())

	for i := range input {
		max := slices.Max(input[i])
		exps := []float64{}

		for j := range input[i] {
			exp := math.Exp(input[i][j] - max)
			exps = append(exps, exp)
		}

		sum := utils.Reduce(exps, 0, func(acc, curr float64) float64 {
			return acc + curr
		})

		for j := range input[i] {
			s.Output[i][j] = exps[j] / sum
		}
	}
}

func (s *Softmax) Backward(dvalues m.Matrix) {
	tensor := make(m.Tensor, len(s.Output))

	for i, output := range s.Output {
		jacobianMatrix := m.AllocateMatrix(len(output), len(output))

		for i := range output {
			for j := range output {
				kroneckerDelta := 0.0
				if i == j {
					kroneckerDelta = 1.0
				}

				jacobianMatrix[i][j] = output[i] * (kroneckerDelta - output[j])
			}
		}

		tensor[i] = jacobianMatrix
	}

	s.DInput = m.AllocateMatrix(s.Output.Rows(), s.Output.Cols())

	for i := range s.DInput {
		for j := range s.DInput[i] {
			s.DInput[i][j] = dvalues[i].Dot(tensor[i][j])
		}
	}
}

func (s *Softmax) BackwardWithCcel(targetClasses m.Matrix) {
	s.DInput = m.AllocateMatrix(s.Output.Rows(), s.Output.Cols())

	for i := range s.Output {
		for j := range s.Output[i] {
			s.DInput[i][j] = (s.Output[i][j] - targetClasses[i][j]) / float64(len(s.Output))
		}
	}
}

```

`/home/mukesh/dev/from-scratch/neural-networks/neural/denselayer.go`:

```go
package neural

import (
	"math/rand"

	m "nn/math"
	"nn/utils"
)

type DenseLayer struct {
	Input, Output, Weights, Bias m.Matrix
	DWeight, DInput, DBias       m.Matrix
	NInputs, NNeurons            int
}

func NewDenseLayer(nInputs, nNeurons int) *DenseLayer {
	dl := &DenseLayer{}

	dl.Weights = utils.RandMatrix(nInputs, nNeurons, func() float64 {
		return rand.NormFloat64() * 0.1
	})
	dl.Bias = m.AllocateMatrix(1, nNeurons)

	dl.NInputs = nInputs
	dl.NNeurons = nNeurons

	return dl
}

func (dl *DenseLayer) Forward(input m.Matrix) {
	dl.Input = input
	dl.Output = m.AllocateMatrix(len(input), dl.NNeurons)

	for i, v := range input {
		dl.Output[i] = v.ToRowMatrix().Multiply(dl.Weights).Add(dl.Bias).ToFloatSlice()
	}
}

func (dl *DenseLayer) Backward(dvalues m.Matrix) {
	dl.DWeight = dl.Input.Transpose().Multiply(dvalues)
	dl.DInput = dvalues.Multiply(dl.Weights.Transpose())

	J := m.AllocateMatrix(dl.Bias.Cols(), 1)

	for i := range J {
		for j := range J[i] {
			J[i][j] = 1
		}
	}

	dl.DBias = dvalues.Multiply(J)
}

```

`/home/mukesh/dev/from-scratch/neural-networks/go.sum`:

```sum
golang.org/x/exp v0.0.0-20241217172543-b2144cdd0a67 h1:1UoZQm6f0P/ZO0w1Ri+f+ifG/gXhegadRdwBIXEFWDo=
golang.org/x/exp v0.0.0-20241217172543-b2144cdd0a67/go.mod h1:qj5a5QZpwLU2NLQudwIN5koi3beDhSAlJwa67PuM98c=

```

`/home/mukesh/dev/from-scratch/neural-networks/utils/utils.go`:

```go
package utils

import (
	"cmp"
	"fmt"
	m "nn/math"
	"strings"
)

func Reduce[T any](s []T, initial T, f func(T, T) T) T {
	result := initial
	for _, v := range s {
		result = f(result, v)
	}
	return result
}

func MaxElemIdx[T cmp.Ordered](s []T) int {
	idx := 0
	max := s[idx]

	for i := range s {
		if s[i] > max {
			idx = i
		}
	}

	return idx
}

func PrintMatrix(label string, matrix m.Matrix) {
	fmt.Printf("%s:\n", label)
	for _, v := range matrix {
		fmt.Printf("%+v\n", v)
	}
	fmt.Println(strings.Repeat("=", 10))
}

```

`/home/mukesh/dev/from-scratch/neural-networks/utils/math_utils.go`:

```go
package utils

import (
	m "nn/math"

	"golang.org/x/exp/constraints"
)

func RandMatrix(rows, cols int, rng func() float64) m.Matrix {
	matrix := m.AllocateMatrix(rows, cols)

	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = rng()
		}
	}

	return matrix
}

func ZeroMatrix(rows, cols int) m.Matrix {
	matrix := m.AllocateMatrix(rows, cols)

	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = 0
		}
	}

	return matrix
}

type Number interface {
	constraints.Integer | constraints.Float
}

func Mean[T Number](s []T) float64 {
	sum := Reduce(s, T(0.0), func(acc, curr T) T {
		return acc + curr
	})

	return float64(sum) / float64(len(s))
}

func LinSpace(start, end float64, num int) m.Vector {
	result := make(m.Vector, num)
	step := (end - start) / float64(num-1)

	for i := range result {
		result[i] = start + float64(i)*step
	}

	return result
}

```

`/home/mukesh/dev/from-scratch/neural-networks/README.md`:

```md
# neural-networks

trying to implement a neural network from scratch in golang

```

`/home/mukesh/dev/from-scratch/neural-networks/go.mod`:

```mod
module nn

go 1.23.0

require golang.org/x/exp v0.0.0-20241217172543-b2144cdd0a67

```