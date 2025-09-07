// RUN: cat %s | python %lsl-parser | filecheck %s
// RUN: cat %s | python %lsl-parser --opt | filecheck %s --check-prefix CHECK-OPT
// RUN: cat %s | python %lsl-parser --to=interp | xdsl-run | filecheck %s --check-prefix INTERP
// RUN: cat %s | python %lsl-parser --opt --to=interp | xdsl-run | filecheck %s --check-prefix INTERP

(0..10).map(|x| x + 1).map(|x| x < 3)

// INTERP: [true,true,false,false,false,false,false,false,false,false,]

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-NEXT:    %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i32 {
// CHECK-NEXT:      %{{.*}} = arith.constant 1 : i32
// CHECK-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      list.yield %{{.*}} : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i1 {
// CHECK-NEXT:      %{{.*}} = arith.constant 3 : i32
// CHECK-NEXT:      %{{.*}} = arith.cmpi ult, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:      list.yield %{{.*}} : i1
// CHECK-NEXT:    }
// CHECK-NEXT:    list.print %{{.*}} : !list.list<i1>
// CHECK-NEXT:  }

// CHECK-OPT:       builtin.module {
// CHECK-OPT-NEXT:    %{{.*}} = arith.constant 0 : i32
// CHECK-OPT-NEXT:    %{{.*}} = arith.constant 10 : i32
// CHECK-OPT-NEXT:    %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
// CHECK-OPT-NEXT:    %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i1 {
// CHECK-OPT-NEXT:      %{{.*}} = arith.constant 1 : i32
// CHECK-OPT-NEXT:      %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-OPT-NEXT:      %{{.*}} = arith.constant 3 : i32
// CHECK-OPT-NEXT:      %{{.*}} = arith.cmpi ult, %{{.*}}, %{{.*}} : i32
// CHECK-OPT-NEXT:      list.yield %{{.*}} : i1
// CHECK-OPT-NEXT:    }
// CHECK-OPT-NEXT:    list.print %{{.*}} : !list.list<i1>
// CHECK-OPT-NEXT:  }
