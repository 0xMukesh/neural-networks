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
