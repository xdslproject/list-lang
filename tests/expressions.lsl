// RUN: cat %s | python %lsl-parser | filecheck %s

let y = 1 == 2;
let x = 1 + 2 + 3;
let z = if y { 0 } else { 1 };
let w = x + z * (2 * z * 3);

1 + 2 + 3 + 4 * 5

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = scf.if %{{.*}} -> (i32) {
// CHECK-NEXT:      %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:      scf.yield %{{.*}} : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:      scf.yield %{{.*}} : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:    %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 4 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 5 : i32
// CHECK-NEXT:    %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:  }
