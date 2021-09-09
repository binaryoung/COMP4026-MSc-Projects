use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use ndarray::prelude::*;
use rand::prelude::*;
use fxhash::FxHashMap as HashMap;
use serde::{Serialize, Deserialize};
use bincode::{serialize,deserialize};

type Room = Array2<u8>;
trait RoomExt {
     fn topology(&self) -> Self;
    fn to_ascii(&self) -> String;
}

impl RoomExt for Room {
    // Return room topology
    fn topology(&self) -> Self {
        self.mapv(|x| if x == 3 || x == 5 { 1 } else { x })
    }

    // Convert room to ASCII string
    fn to_ascii(&self) -> String {
        self.indexed_iter()
            .map(|(location, &x)| {
                let char = match x {
                    0 => '#',
                    1 => ' ',
                    2 => '.',
                    3 => '$',
                    4 => '*',
                    5 => '@',
                    6 => '+',
                    _ => unreachable!(),
                };

                if location.1 == 9 {
                    format!("{}\n", char)
                } else {
                    format!("{}", char)
                }
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Level {
    pub id: usize,   
    pub score: usize,
    pub trajectory: Vec<u8>,
    pub room: Room,
    pub topology: Room,
}

// Parse generated level file
fn parse_file(index:usize, path: PathBuf) -> Vec<Level> {
    let mut levels:Vec<Level> = Vec::with_capacity(1000);

    let content = fs::read_to_string(path).unwrap();

    content.split(';').skip(1).for_each(|level|{
        let (meta, room_str) = level.trim().split_once('\n').unwrap();
        let meta: Vec<&str> = meta.split(' ').collect();

        let id: usize = meta[0].parse::<usize>().unwrap() + index * 1000;
        let score: usize = meta[1].parse().unwrap();
        let trajectory: Vec<u8> = meta[2].split(',').map(|action| action.parse().unwrap()).collect();

        let room: Vec<Vec<u8>> = room_str.split('\n').map(|row|{
            row.chars().map(|x| {
                match x {
                    '#' => 0,
                    ' ' => 1,
                    '.' => 2,
                    '$' => 3,
                    '*' => 4,
                    '@' => 5,
                    '+' => 6,
                    _ => unreachable!(),
                }
            }).collect()
        }).collect();

        let room = room.into_iter().flatten().collect();
        let room:Room = Array::from_shape_vec((10, 10), room).unwrap();
        assert_eq!(room.to_ascii(), format!("{}\n",room_str));
        let topology:Room = room.topology();

        let level = Level{
            id,
            score,
            trajectory,
            room,
            topology,
        };

        levels.push(level);
    });

    levels
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Collection {
    map: HashMap<usize, Level>,
    tree: BTreeMap<usize, Vec<usize>>,
    len: usize,
}

impl Collection {
    //  Build level collection
    pub fn build() -> Collection {
        let mut map = HashMap::default();
        let mut tree: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut len = 0;

        fs::read_dir("levels").unwrap()
            .enumerate()
            .filter_map(|(index, file)| {
                let path = file.unwrap().path();

                if path.to_str().unwrap() != r"levels/collection.bin" {
                    Some(parse_file(index, path))
                } else {
                    None
                }
            })
            .flatten()
            .for_each(|level|{
                let id =level.id;
                let score = level.score;

                tree.entry(score).or_default().push(id);
                map.insert(id, level);
                len += 1;
            });

        Collection {
            map,
            tree,
            len,
        }
    }

    //  Save level collection to file
    pub fn save(&self, path: &str) -> Result<(), String>{
        let encoded = serialize(self).map_err(|_| "Can't serialize".to_string())?;
        fs::write(path, encoded).map_err(|_| "Can't save to file".to_string())?;
        Ok(())
    }

    //  Load level collection from file
    pub fn load(path: &str) -> Result<Collection, String>{
        let encoded =  fs::read(path).map_err(|_| "Can't read file".to_string())?;
        let levels: Collection = deserialize(&encoded[..]).map_err(|_| "Can't deserialize".to_string())?;
        Ok(levels)
    }

    //  Load level collection from bytes
    pub fn load_from_bytes(encoded: &[u8]) -> Result<Collection, String>{
        let levels: Collection = deserialize(encoded).map_err(|_| "Can't deserialize".to_string())?;
        Ok(levels)
    }

    // Return the level with the specified ID
    pub fn find(&self, id: usize) -> Result<&Level, String> {
        self.map.get(&id).ok_or(format!("Id {} not exist", id))
    }

    // Return a random level within the specified difficulty score range
    pub fn range(&self, min: usize, max: usize) -> Result<&Level, String> {
        let mut rng = rand::thread_rng();
        let (score, ids) = self.tree.range(min..=max).choose(&mut rng).ok_or(format!("Range {}-{} is empty", min, max))?;
        let &id = ids.choose(&mut rng).ok_or(format!("Score {} is empty", score))?;
        self.find(id)
    }

    // Return a random level
    pub fn random(&self) -> Result<&Level, String> {
        let id = thread_rng().gen_range(0..self.len);
        self.find(id)
    }
}