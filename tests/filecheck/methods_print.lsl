// RUN: cat %s | python %lsl-parser | filecheck %s
// RUN: cat %s | python %lsl-parser --to=tensor | filecheck %s --check-prefix TENSOR
// RUN: cat %s | python %lsl-parser --to=interp | xdsl-run | filecheck %s --check-prefix INTERP

let a = 12;
let x = 0..10;

let x2 = x.map(|y| y * a + x.len());

let x3 = x2.map(|y| {
    let res = y * a;
    res + x.len()
}).map(|x| x < 800);

x3

// INTERP: [true,true,true,true,true,false,false,false,false,false,]

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 12 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-NEXT:    %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i32 {
// CHECK-NEXT:      %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      %{{.*}} = list.length %{{.*}} : !list.list<i32> -> i32
// CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      list.yield %{{.*}} : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i32 {
// CHECK-NEXT:      %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      %{{.*}} = list.length %{{.*}} : !list.list<i32> -> i32
// CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      list.yield %{{.*}} : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i1 {
// CHECK-NEXT:      %{{.*}} = arith.constant 800 : i32
// CHECK-NEXT:      %{{.*}} = arith.cmpi ult, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      list.yield %{{.*}} : i1
// CHECK-NEXT:    }
// CHECK-NEXT:    list.print %{{.*}} : !list.list<i1>
// CHECK-NEXT:  }

// TENSOR:       builtin.module {
// TENSOR-NEXT:    %{{.*}} = arith.constant 12 : i32
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
// TENSOR-NEXT:    %{{.*}} = arith.constant 0 : index
// TENSOR-NEXT:    %{{.*}} = arith.constant 1 : index
// TENSOR-NEXT:    %{{.*}} = tensor.dim %{{.*}}, %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:    %{{.*}} = tensor.empty(%{{.*}}) : tensor<?xi32>
// TENSOR-NEXT:    %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<?xi32>) {
// TENSOR-NEXT:      %{{.*}} = tensor.extract %{{.*}}[%{{.*}}] : tensor<?xi32>
// TENSOR-NEXT:      %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:      %{{.*}} = arith.constant 0 : index
// TENSOR-NEXT:      %{{.*}} = tensor.dim %{{.*}}, %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:      %{{.*}} = arith.index_cast %{{.*}} : index to i32
// TENSOR-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:      %{{.*}} = tensor.insert %{{.*}} into %{{.*}}[%{{.*}}] : tensor<?xi32>
// TENSOR-NEXT:      scf.yield %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:    }
// TENSOR-NEXT:    %{{.*}} = arith.constant 0 : index
// TENSOR-NEXT:    %{{.*}} = arith.constant 1 : index
// TENSOR-NEXT:    %{{.*}} = tensor.dim %{{.*}}, %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:    %{{.*}} = tensor.empty(%{{.*}}) : tensor<?xi32>
// TENSOR-NEXT:    %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<?xi32>) {
// TENSOR-NEXT:      %{{.*}} = tensor.extract %{{.*}}[%{{.*}}] : tensor<?xi32>
// TENSOR-NEXT:      %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:      %{{.*}} = arith.constant 0 : index
// TENSOR-NEXT:      %{{.*}} = tensor.dim %{{.*}}, %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:      %{{.*}} = arith.index_cast %{{.*}} : index to i32
// TENSOR-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:      %{{.*}} = tensor.insert %{{.*}} into %{{.*}}[%{{.*}}] : tensor<?xi32>
// TENSOR-NEXT:      scf.yield %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:    }
// TENSOR-NEXT:    %{{.*}} = arith.constant 0 : index
// TENSOR-NEXT:    %{{.*}} = arith.constant 1 : index
// TENSOR-NEXT:    %{{.*}} = tensor.dim %{{.*}}, %{{.*}} : tensor<?xi32>
// TENSOR-NEXT:    %{{.*}} = tensor.empty(%{{.*}}) : tensor<?xi1>
// TENSOR-NEXT:    %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<?xi1>) {
// TENSOR-NEXT:      %{{.*}} = tensor.extract %{{.*}}[%{{.*}}] : tensor<?xi32>
// TENSOR-NEXT:      %{{.*}} = arith.constant 800 : i32
// TENSOR-NEXT:      %{{.*}} = arith.cmpi ult, %{{.*}}, %{{.*}} : i32
// TENSOR-NEXT:      %{{.*}} = tensor.insert %{{.*}} into %{{.*}}[%{{.*}}] : tensor<?xi1>
// TENSOR-NEXT:      scf.yield %{{.*}} : tensor<?xi1>
// TENSOR-NEXT:    }
// TENSOR-NEXT:    %{{.*}} = arith.constant 0 : index
// TENSOR-NEXT:    %{{.*}} = arith.constant 1 : index
// TENSOR-NEXT:    %{{.*}} = tensor.dim %{{.*}}, %{{.*}} : tensor<?xi1>
// TENSOR-NEXT:    printf.print_format "["
// TENSOR-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// TENSOR-NEXT:      %{{.*}} = tensor.extract %{{.*}}[%{{.*}}] : tensor<?xi1>
// TENSOR-NEXT:      scf.if %{{.*}} {
// TENSOR-NEXT:        printf.print_format "true"
// TENSOR-NEXT:      } else {
// TENSOR-NEXT:        printf.print_format "false"
// TENSOR-NEXT:      }
// TENSOR-NEXT:      printf.print_format ","
// TENSOR-NEXT:    }
// TENSOR-NEXT:    printf.print_format "]"
// TENSOR-NEXT:  }
