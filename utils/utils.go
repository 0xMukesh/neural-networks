package utils

func Reduce[T any](s []T, initial T, f func(T, T) T) T {
	result := initial
	for _, v := range s {
		result = f(result, v)
	}
	return result
}
