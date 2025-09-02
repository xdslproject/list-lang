// RUN: cat %s | python %lsl-parser | filecheck %s
// RUN: cat %s | python %lsl-parser --to=tensor | filecheck %s --check-prefix TENSOR

let x = 0..10;
let y = 10 + 1..15;
x

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 15 : i32
// CHECK-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-NEXT:    list.print %{{.*}} : !list.list<i32>
// CHECK-NEXT:  }

// TENSOR:       builtin.module {
// TENSOR-NEXT:    %{{.*}} = arith.constant 0 : i32
// TENSOR-NEXT:    %{{.*}} = arith.constant 10 : i32
// TENSOR-NEXT:    %{{.*}} = arith.constant 0 : index
// TENSOR-NEXT:    %{{.*}} = arith.constant 1 : index
// TENSOR-NEXT:    %{{.*}} = arith.subi %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:    %{{.*}} = arith.index_cast %{{.*}} : i32 to index
// TENSOR-NEXT:    %{{.*}} = tensor.empty(%{{.*}}) : tensor<?xi32>
// TENSOR-NEXT:    %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<?xi32>) {
// TENSOR-NEXT:      %{{.*}} = arith.index_cast %{{.*}} : index to i32
// TENSOR-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:      %{{.*}} = tensor.insert %{{.*}} into %{{.*}}[%{{.*}}] : tensor<?xi32>
// TENSOR-NEXT:      scf.yield %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:    }
// TENSOR-NEXT:    %{{.*}} = arith.constant 10 : i32
// TENSOR-NEXT:    %{{.*}} = arith.constant 1 : i32
// TENSOR-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:    %{{.*}} = arith.constant 15 : i32
// TENSOR-NEXT:    %{{.*}} = arith.constant 0 : index
// TENSOR-NEXT:    %{{.*}} = arith.constant 1 : index
// TENSOR-NEXT:    %{{.*}} = arith.subi %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:    %{{.*}} = arith.index_cast %{{.*}} : i32 to index
// TENSOR-NEXT:    %{{.*}} = tensor.empty(%{{.*}}) : tensor<?xi32>
// TENSOR-NEXT:    %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<?xi32>) {
// TENSOR-NEXT:      %{{.*}} = arith.index_cast %{{.*}} : index to i32
// TENSOR-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:      %{{.*}} = tensor.insert %{{.*}} into %{{.*}}[%{{.*}}] : tensor<?xi32>
// TENSOR-NEXT:      scf.yield %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:    }
// TENSOR-NEXT:    %{{.*}} = arith.constant 0 : index
// TENSOR-NEXT:    %{{.*}} = arith.constant 1 : index
// TENSOR-NEXT:    %{{.*}} = tensor.dim %{{.*}}, %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:    printf.print_format "["
// TENSOR-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// TENSOR-NEXT:      %{{.*}} = tensor.extract %{{.*}}[%{{.*}}] : tensor<?xi32>
// TENSOR-NEXT:      printf.print_format "{}", %{{.*}} : i32
// TENSOR-NEXT:      printf.print_format ","
// TENSOR-NEXT:    }
// TENSOR-NEXT:    printf.print_format "]"
// TENSOR-NEXT:  }
