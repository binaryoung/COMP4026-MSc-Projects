use fxhash::FxHashMap as HashMap;
use fxhash::FxHashSet as HashSet;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::PathBuf;
use std::io::{BufReader,Read};
use ndarray::prelude::*;
use pyo3::prelude::Python;
use numpy::{ToPyArray, PyArray};
struct Level {
    id: usize,   
    score: usize,
    trajectory: Vec<u8>,
    room: Array2<u8>,
}

fn file_to_levels(i:usize, path: PathBuf) -> Vec<Level> {
    let mut content = String::new();
    let levels:Vec<Level> = Vec::with_capacity(1000);

    let file = File::open(path).unwrap();
    let mut file = BufReader::new(file);
    file.read_to_string(&mut content).unwrap();

    content.split(';').skip(1).for_each(|level|{
        let (meta, room) = level.trim().split_once('\n').unwrap();
        let meta: Vec<&str> = meta.split(' ').collect();

        let id: usize = meta[0].parse::<usize>().unwrap() + i * 1000;
        let score: usize = meta[1].parse().unwrap();
        let trajectory: Vec<u8> = meta[2].split(',').map(|action| action.parse().unwrap()).collect();

        let room: Vec<Vec<u8>> = room.split('\n').map(|row|{
            row.chars().map(|x| {
                match x {
                    '#' => 0 ,
                    ' ' => 1 ,
                    '.' => 2 ,
                    '$' => 3 ,
                    '*' => 4 ,
                    '@' => 5 ,
                    '+' => 6 ,
                    _ => unreachable!(),
                }
            }).collect()
        }).collect();

        let gil = Python::acquire_gil();
        let py = gil.python();
        let room = room.into_pyarray(py);
        // let room = Array::from_shape_vec((10, 10), room).unwrap();
        dbg!(id,score,trajectory);
        println!("{}",room);
        panic!()
    });

    Vec::new()
}

fn main() {
    fs::read_dir("levels").unwrap().enumerate().for_each(|(i, file)| {
        let path = file.unwrap().path();
        file_to_levels(i,path);
    });
}
