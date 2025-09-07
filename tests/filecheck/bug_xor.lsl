// RUN: cat %s | python %lsl-parser --to=interp | xdsl-run | filecheck %s --check-prefix INTERP

!(7 == 7)

// INTERP: false
