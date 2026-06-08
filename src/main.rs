// Welcome to Terry-2
//
// Terry is a cellular automata simulator written in Rust. It is designed to be super fast so we can
// simulate large worlds with many different types of cells.
// Anything that can be determined at build time is determined at build time, so we can optimize the
// code for the specific world we want to simulate. This means that we can have a very large number
// of different cell types without sacrificing performance.
//
// The ultimate goal is to create an automata that can solve https://docs.arcprize.org/
//
// However, step 1 is simply to choose the right action for the first move of the first level of the
// simplest game, "ls20".
// Since ARC3 is implemented in python, we can use the python implementation to determine the correct move for the first level of
fn main() {
    println!("Hello, World!");
}
