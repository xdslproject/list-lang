// RUN: cat %s | python %lsl-parser | filecheck %s
// RUN: cat %s | python %lsl-parser --to=tensor | filecheck %s
// RUN: cat %s | python %lsl-parser --to=interp | xdsl-run | filecheck %s --check-prefix INTERP

let a = 10;
let c = 4 + 5;
let a = 1000;
let c = c + 2;
let d = {
    let x = a;
    let a = 5;
    a + x
};

a + c * d

// INTERP: 12055

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 4 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 5 : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 1000 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 5 : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    printf.print_format "{}", %{{.*}} : i32
// CHECK-NEXT:  }
