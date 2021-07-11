use fxhash::FxHashSet as HashSet;
use itertools::Itertools;
use ndarray::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::min;
use std::collections::{BTreeMap,VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::vec;


type Room = Array2<i32>;

trait RoomExt {
    fn new() -> Self;
    fn place_player_and_targets(&mut self) -> Result<(), &'static str>;
    fn topology(&self) -> Self;
    fn place_boxes(&self) -> Self;
    fn player_location(&self) -> (usize, usize);
    fn target_number(&self) -> usize;
    fn to_ascii(&self) -> String;
}

impl RoomExt for Room {
    fn new() -> Self {
        let mut rng: ThreadRng = thread_rng();

        let masks = [
            array![[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            array![[0, 1, 0], [0, 1, 0], [0, 1, 0]],
            array![[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            array![[0, 0, 0], [1, 1, 0], [1, 1, 0]],
            array![[0, 0, 0], [0, 1, 1], [0, 1, 0]],
        ];

        let directions = [(1, 0), (0, 1), (-1, 0), (0, -1)];
        let mut direction = directions.choose(&mut rng).unwrap();

        let mut position = (rng.gen_range(1..=9), rng.gen_range(1..=9));

        let mut room = Array::zeros((10, 10));

        for _ in 0..30 {
            if rng.gen::<f64>() < 0.35 {
                direction = directions.choose(&mut rng).unwrap();
            }

            position = (
                (position.0 + direction.0).clamp(1, 8),
                (position.1 + direction.1).clamp(1, 8),
            );

            let mask = masks.choose(&mut rng).unwrap();

            let mut level_slice = room.slice_mut(s![
                (position.0 - 1)..(position.0 + 2),
                (position.1 - 1)..(position.1 + 2)
            ]);
            level_slice += mask;
        }

        room.slice_mut(s![.., 0..10;9]).fill(0);
        room.slice_mut(s![0..10;9, ..]).fill(0);
        room.mapv_inplace(|x| min(x, 1));

        room
    }

    fn place_player_and_targets(&mut self) -> Result<(), &'static str> {
        let mut rng: ThreadRng = thread_rng();

        let mut empty_locations = Vec::with_capacity(50);
        self.indexed_iter().for_each(|(index, &x)| {
            if x == 1 {
                empty_locations.push(index)
            }
        });

        if empty_locations.len() <= (4 + 1) {
            return Err("Not enough room");
        }

        empty_locations.shuffle(&mut rng);

        let (x, y) = empty_locations[0];
        self[[x, y]] = 5;

        for i in 0..4 {
            let (x, y) = empty_locations[i + 1];
            self[[x, y]] = 2;
        }

        Ok(())
    }

    fn topology(&self) -> Self {
        self.mapv(|x| if x == 5 { 1 } else { x })
    }

    fn place_boxes(&self) -> Self {
        self.mapv(|x| if x == 2 { 4 } else { x })
    }

    fn player_location(&self) -> (usize, usize) {
        self.indexed_iter()
            .find(|(_, &x)| x == 5)
            .unwrap()
            .0
            .to_owned()
    }

    fn target_number(&self) -> usize {
        self.fold(0, |acc, &x| if x == 2 { acc + 1 } else { acc })
    }
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

thread_local! {
    static BEST_LEVEL: RefCell<Level> = RefCell::new(Level::default());
    static EXPLORED_STATES: RefCell<HashSet<Room>> = RefCell::new(HashSet::default());
}

#[derive(Hash, Eq, PartialEq, Clone, Default, Debug)]
struct Level {
    topology: Room,
    room: Room,
    player_location: (usize, usize),
    box_mapping: BTreeMap<(usize, usize), (usize, usize)>,
    box_swaps: usize,
    last_pull: (usize, usize),
    depth: usize,
    score: usize,
    trajectory: Vec<usize>,
}

impl Level {
    fn new() -> Result<Self, &'static str> {
        let mut room = Room::new();
        room.place_player_and_targets()?;
        let topology = room.topology();
        let room = room.place_boxes();

        let mut box_mapping = BTreeMap::new();
        topology.indexed_iter().for_each(|(location, &x)| {
            if x == 2 {
                box_mapping.insert(location, location);
            }
        });

        let trajectory = Vec::with_capacity(300);
        let player_location = room.player_location();

        let mut level = Level {
            topology,
            room,
            player_location,
            box_mapping,
            box_swaps: 0,
            last_pull: (0, 0),
            depth: 1,
            score: 0,
            trajectory,
        };

        level.calculate_score();

        Ok(level)
    }

    fn depth_first_search(&self) {
        let state_number = EXPLORED_STATES.with(|states| states.borrow().len());

        if self.depth > 300
            || state_number >= 1000000
            || EXPLORED_STATES.with(|states| states.borrow().contains(&self.room))
        {
            return;
        }

        EXPLORED_STATES.with(|states| states.borrow_mut().insert(self.room.clone()));

        BEST_LEVEL.with(|level| {
            let mut level = level.borrow_mut();
            if self.score > level.score {
                *level = self.clone();
            }
        });

        (0usize..=7).for_each(|action| {
            let level = self.reserve_move(action);
            level.depth_first_search();
        });
    }

    fn sublevels(&self) -> Vec<Level> {
        (0usize..=7).filter_map(|action| {
            let level = self.reserve_move(action);

            if level.validate() == true {
                Some(level)
            } else{ 
                None
            }
        }).collect()
    }

    fn breadth_first_search(&self) {
        let mut queue = VecDeque::with_capacity(100000);

        self.validate();
        queue.push_back(self.clone());

        while let Some(level) = queue.pop_front() {
            level.sublevels().into_iter().for_each(|level| queue.push_back(level));
        }
    }

    fn validate(&self) -> bool {
        let state_number = EXPLORED_STATES.with(|states| states.borrow().len());

        if self.depth > 300
            || state_number >= 1000000
            || EXPLORED_STATES.with(|states| states.borrow().contains(&self.room))
        {
            return false;
        }

        EXPLORED_STATES.with(|states| states.borrow_mut().insert(self.room.clone()));

        BEST_LEVEL.with(|level| {
            let mut level = level.borrow_mut();
            if self.score > level.score {
                *level = self.clone();
            }
        });

        true
    }

    fn reserve_move(&self, action: usize) -> Self {
        let mut level = self.clone();

        level.trajectory.push(action);
        level.depth += 1;

        let player_location = level.player_location;
        let moves = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        let change = moves[action % 4];
        let next_player_location = (
            (player_location.0 as isize + change.0) as usize,
            (player_location.1 as isize + change.1) as usize,
        );

        let next_player_location_value =
            level.room[[next_player_location.0, next_player_location.1]];

        if next_player_location_value == 1 || next_player_location_value == 2 {
            level.room[[next_player_location.0, next_player_location.1]] = 5;
            level.room[[player_location.0, player_location.1]] =
                level.topology[[player_location.0, player_location.1]];
            level.player_location = next_player_location;

            if action < 4 {
                let possible_box_location = (
                    (player_location.0 as isize + (change.0 * -1)) as usize,
                    (player_location.1 as isize + (change.1 * -1)) as usize,
                );
                let possible_box_location_value =
                    level.room[[possible_box_location.0, possible_box_location.1]];

                if possible_box_location_value == 3 || possible_box_location_value == 4 {
                    level.room[[player_location.0, player_location.1]] = 3;
                    level.room[[possible_box_location.0, possible_box_location.1]] =
                        level.topology[[possible_box_location.0, possible_box_location.1]];

                    let mut last_pull = level.last_pull;
                    level
                        .box_mapping
                        .iter_mut()
                        .for_each(|(&target, location)| {
                            if *location == possible_box_location {
                                *location = player_location;
                                last_pull = target;
                            }
                        });
                    if level.last_pull != last_pull {
                        level.last_pull = last_pull;
                        level.box_swaps += 1;
                    }
                }
            }
        }

        level.calculate_score();
        level
    }

    fn calculate_score(&mut self) {
        if self.room.target_number() != 4 {
            self.score = 0;
        } else {
            self.score = self.box_swaps * self.box_displacement_score()
        }
    }

    fn box_displacement_score(&self) -> usize {
        self.box_mapping
            .iter()
            .fold(0isize, |score, (&target, &location)| {
                score
                    + (target.0 as isize - location.0 as isize).abs()
                    + (target.1 as isize - location.1 as isize).abs()
            }) as usize
    }

    fn reverse_trajectory(&mut self) {
        self.trajectory = self
            .trajectory
            .iter()
            .rev()
            .map(|&action| match action {
                0 => 1,
                1 => 0,
                2 => 3,
                3 => 2,
                4 => 5,
                5 => 4,
                6 => 7,
                7 => 6,
                _ => unreachable!(),
            })
            .collect();
    }

    fn trajectory_to_string(&self) -> String {
        self.trajectory
            .iter()
            .map(|action| action.to_string())
            .intersperse(",".to_string())
            .collect()
    }

    fn metadata_to_string(&self, index: usize) -> String {
        format!("; {} {} {}", index, self.score, self.trajectory_to_string())
    }

    fn to_string(&self, index: usize) -> String {
        format!(
            "{}\n{}\n",
            self.metadata_to_string(index),
            self.room.to_ascii()
        )
    }
}

fn generate_level() -> Level {
    BEST_LEVEL.with(|level| {
        *level.borrow_mut() = Level::default();
    });
    EXPLORED_STATES.with(|states| states.borrow_mut().clear());

    loop {
        let level = Level::new();

        if level.is_err() {
            continue;
        }

        let level = level.unwrap();
        
        //  depth first search
        // level.depth_first_search();

        //  breadth first search
        level.breadth_first_search();

        let mut level = BEST_LEVEL.with(|level| level.borrow().clone());

        if level.score <= 0 {
            continue;
        }

        level.reverse_trajectory();
        break level;
    }
}

fn write_one_thousand_levels(index: usize) {
    let filename = format!("{:03}.txt", index);
    let path = format!("levels/{}", filename);

    let file = File::create(path).unwrap();
    let mut file = BufWriter::new(file);

    (0..1000).for_each(|i| {
        let level = generate_level();
        file.write_all(level.to_string(i).as_bytes()).unwrap();

        if i % 100 == 0 {
            println!("{}: {}", filename, i);
        }
        if i == 999 {
            println!("{}: done", filename);
        }
    });
}

fn main() {
    // (0..5).into_par_iter().for_each(|i| {
    //     write_one_thousand_levels(i);
    // });
    // dbg!(generate_level());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    #[test]
    fn it_works() {
        dbg!(generate_level());
    }

    #[test]
    fn performance() {
        let now = Instant::now();
        (0..100).for_each(|_| {
            generate_level();
        });
        dbg!(now.elapsed());
    }
}