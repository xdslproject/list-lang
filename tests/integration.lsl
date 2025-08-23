// RUN: cat %s | python %lsl-parser | filecheck %s

let x = 4;
let y = 3;
let z = y + 3 * (x + y);
let w = if x * z == 0 {
    x + 1
} else {
    // this is the else branch
    let tmp = y + 1;
    if true { tmp * z } else { 0 }
} * 2;
w + 3

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 4 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = scf.if %{{.*}} -> (i32) {
// CHECK-NEXT:      %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      scf.yield %{{.*}} : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      %{{.*}} = arith.constant true
// CHECK-NEXT:      %{{.*}} = scf.if %{{.*}} -> (i32) {
// CHECK-NEXT:        %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:        scf.yield %{{.*}} : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:        scf.yield %{{.*}} : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %{{.*}} : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:  }
