// RUN: cat %s | python %lsl-parser | filecheck %s

let x = 0..10;
let y = x.len();
let y2 = (10..20).len();
y == y2

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-NEXT:    %{{.*}} = list.length %{{.*}} : !list.list<i32> -> i32
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 20 : i32
// CHECK-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-NEXT:    %{{.*}} = list.length %{{.*}} : !list.list<i32> -> i32
// CHECK-NEXT:    %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:  }
