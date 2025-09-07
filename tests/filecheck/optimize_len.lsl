// RUN: cat %s | python %lsl-parser | filecheck %s
// RUN: cat %s | python %lsl-parser --opt | filecheck %s --check-prefix CHECK-OPT
// RUN: cat %s | python %lsl-parser --to=interp | xdsl-run | filecheck %s --check-prefix INTERP
// RUN: cat %s | python %lsl-parser --opt --to=interp | xdsl-run | filecheck %s --check-prefix INTERP

(0..10).map(|x| x + 1).len()

// INTERP: 10

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-NEXT:    %[[MAP:.*]] = list.map %{{.*}} with (%{{.*}} : i32) -> i32 {
// CHECK-NEXT:      %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      list.yield %{{.*}} : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}} = list.length %[[MAP]] : !list.list<i32> -> i32
// CHECK-NEXT:    printf.print_format "{}", %{{.*}} : i32
// CHECK-NEXT:  }

// CHECK-OPT:       builtin.module {
// CHECK-OPT-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-OPT-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-OPT-NEXT:    %[[RANGE:.*]] = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-OPT-NEXT:    %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i32 {
// CHECK-OPT-NEXT:      %{{.*}} = arith.constant 1 : i32
// CHECK-OPT-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-OPT-NEXT:      list.yield %{{.*}} : i32
// CHECK-OPT-NEXT:    }
// CHECK-OPT-NEXT:    %{{.*}} = list.length %[[RANGE]] : !list.list<i32> -> i32
// CHECK-OPT-NEXT:    printf.print_format "{}", %{{.*}} : i32
// CHECK-OPT-NEXT:  }
