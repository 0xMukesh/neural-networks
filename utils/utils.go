package utils

import "cmp"

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
