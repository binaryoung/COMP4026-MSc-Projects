#[path = "src/level.rs"] 
mod level;

use level::Collection;

fn main() {
    let collection =  Collection::build();
    collection.save("levels/collection.bin").unwrap();
}