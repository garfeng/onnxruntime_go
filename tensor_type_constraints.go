package onnxruntime_go

// This file contains definitions for the generic tensor data types we support.

// #include "onnxruntime_wrapper.h"
import "C"

import (
	"errors"
	"math"
	"reflect"
)

type FloatData interface {
	~float32 | ~float64
}

type IntData interface {
	~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64
}

// This is used as a type constraint for the generic Tensor type.
type TensorData interface {
	FloatData | IntData
}

// Returns the ONNX enum value used to indicate TensorData type T.
func GetTensorElementDataType[T TensorData](option ...*TensorDataOption) C.ONNXTensorElementDataType {
	// Sadly, we can't do type assertions to get underlying types, so we need
	// to use reflect here instead.
	var v T
	kind := reflect.ValueOf(v).Kind()
	switch kind {
	case reflect.Float64:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
	case reflect.Float32:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
	case reflect.Int8:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
	case reflect.Uint8:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
	case reflect.Int16:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
	case reflect.Uint16:
		// save float16 in uint16
		if len(option) > 0 && option[0].UseFp16 {
			return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
		}
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
	case reflect.Int32:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
	case reflect.Uint32:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
	case reflect.Int64:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
	case reflect.Uint64:
		return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
	}
	return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
}

type Float16 = uint16

func float32ToFloat16(f float32) Float16 {
	fbits := math.Float32bits(f)
	sign := Float16((fbits >> 16) & 0x00008000)
	rv := (fbits & 0x7fffffff) + 0x1000
	if rv >= 0x47800000 {
		if (fbits & 0x7fffffff) >= 0x47800000 {
			if rv < 0x7f800000 {
				return sign | 0x7c00
			}
			return sign | 0x7c00 | uint16((fbits&0x007fffff)>>13)
		}
		return sign | 0x7bff
	}
	if rv >= 0x38800000 {
		return sign | uint16((rv-0x38000000)>>13)
	}
	if rv < 0x33000000 {
		return sign
	}
	rv = (fbits & 0x7fffffff) >> 23
	return sign | uint16(((fbits&0x7fffff)|0x800000)+(0x800000>>(rv-102))>>(126-rv)) //these two shifts are my problem
}

func float16ToFloat32(half Float16) float32 {
	halfFloat := uint32(half)

	floatInt := ((halfFloat & 0x8000) << 16) | (((((halfFloat >> 10) & 0x1f) - 15 + 127) & 0xff) << 23) | ((halfFloat & 0x03FF) << 13)

	return math.Float32frombits(floatInt)
}

func Float32SliceToFloat16(dst []Float16, src []float32) error {
	dstLen := len(dst)
	srcLen := len(src)
	if dstLen != srcLen {
		return errors.New("len(dst) != len(src)")
	}

	for i := 0; i < dstLen; i++ {
		dst[i] = float32ToFloat16(src[i])
	}
	return nil
}

func Float16SliceToFloat32(dst []float32, src []Float16) error {
	dstLen := len(dst)
	srcLen := len(src)
	if dstLen != srcLen {
		return errors.New("len(dst) != len(src)")
	}

	for i := 0; i < dstLen; i++ {
		dst[i] = float16ToFloat32(src[i])
	}
	return nil
}
