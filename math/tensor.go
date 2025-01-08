package math

type Tensor []Matrix

func AllocateTensor(depth, rows, cols int) Tensor {
	tensor := make(Tensor, depth)

	for i := range tensor {
		tensor[i] = AllocateMatrix(rows, cols)
	}

	return tensor
}
