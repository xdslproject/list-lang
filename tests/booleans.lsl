// RUN: cat %s | python %lsl-parser | filecheck %s

let x = 3;
let y0 = 1 == 2;
let y1 = x <= 2;
let z0 = y0 && y1;
let z1 = false || (z0 && 2 < 3);
let z2 = false || z0 && 2 < 3;
z1 || !z2 && x + 1 > 5

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.cmpi ule, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.andi %{{.*}}, %{{.*}} : i1
// CHECK-NEXT:    %{{.*}} = arith.constant false
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:    %{{.*}} = arith.cmpi ult, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.andi %{{.*}}, %{{.*}} : i1
// CHECK-NEXT:    %{{.*}} = arith.ori %{{.*}}, %{{.*}} : i1
// CHECK-NEXT:    %{{.*}} = arith.constant false
// CHECK-NEXT:    %{{.*}} = arith.ori %{{.*}}, %{{.*}} : i1
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:    %{{.*}} = arith.cmpi ult, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.andi %{{.*}}, %{{.*}} : i1
// CHECK-NEXT:    %{{.*}} = arith.constant true
// CHECK-NEXT:    %{{.*}} = arith.xori %{{.*}}, %{{.*}} : i1
// CHECK-NEXT:    %{{.*}} = arith.ori %{{.*}}, %{{.*}} : i1
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 5 : i32
// CHECK-NEXT:    %{{.*}} = arith.cmpi ugt, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:    %{{.*}} = arith.andi %{{.*}}, %{{.*}} : i1
// CHECK-NEXT:  }
