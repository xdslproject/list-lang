// RUN: cat %s | python %lsl-parser | filecheck %s

let a = 12;
let x = 0..10;

let x2 = x.map(|x| x < 2);

let x3 = x.map(|y| y * a + x.len());

let x4 = x3.map(|y| {
    let res = y * a;
    res + x.len()
});

let y = x.len();
let y2 = (10..20).len();

y == y2

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 12 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-NEXT:    %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i1 {
// CHECK-NEXT:      %{{.*}} = arith.constant 2 : i32
// CHECK-NEXT:      %{{.*}} = arith.cmpi ult, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      list.yield %{{.*}} : i1
// CHECK-NEXT:    }
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
// CHECK-NEXT:    %{{.*}} = list.length %{{.*}} : !list.list<i32> -> i32
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 20 : i32
// CHECK-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-NEXT:    %{{.*}} = list.length %{{.*}} : !list.list<i32> -> i32
// CHECK-NEXT:    %{{.*}} = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:  }
