// RUN: scalehls-opt %s | FileCheck %s

// CHECK: module {
func @test_mvt(%x1: memref<16xf32>, %x2: memref<16xf32>, %y1: memref<16xf32>, %y2: memref<16xf32>, %A: memref<16x16xf32>) {
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      %0 = affine.load %x1[%i] : memref<16xf32>
      %1 = affine.load %y1[%i] : memref<16xf32>
      %2 = affine.load %A[%i, %j] : memref<16x16xf32>
      %3 = mulf %2, %1 : f32
      %4 = addf %3, %0 : f32
      affine.store %4, %x1[%i] : memref<16xf32>
    }
  }
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      %5 = affine.load %x2[%i] : memref<16xf32>
      %6 = affine.load %y2[%i] : memref<16xf32>
      %7 = affine.load %A[%j, %i] : memref<16x16xf32>
      %8 = mulf %7, %6 : f32
      %9 = addf %8, %5 : f32
      affine.store %9, %x2[%i] : memref<16xf32>
    }
  }
  return
}