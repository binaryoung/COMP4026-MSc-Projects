mod level;

use level::{Collection, Level};
use numpy::{IntoPyArray, PyArray2, ToPyArray};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

static COLLECTION_BYTES: &'static [u8] = include_bytes!(r"..\levels\collection.bin");
static COLLECTION: Lazy<Collection> = Lazy::new(|| Collection::load_from_bytes(COLLECTION_BYTES).unwrap());

type LevelPY<'py> = (usize, usize, Vec<u8>, &'py PyArray2<u8>, &'py PyArray2<u8>);

impl Level {
    fn to_py<'py>(&self, py: Python<'py>) -> LevelPY<'py> {
        let Level {
            id,
            score,
            trajectory,
            room,
            topology,
        } = self.clone();

        let room = room.into_pyarray(py);
        let topology = topology.into_pyarray(py);

        (id, score, trajectory, room, topology)
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn find<'py>(py: Python<'py>, id: usize) -> PyResult<LevelPY<'py>> {
    let level = COLLECTION.find(id).unwrap();
    Ok(level.to_py(py))
}

#[pyfunction]
fn range<'py>(py: Python<'py>, min: usize, max: usize) -> PyResult<LevelPY<'py>> {
    let level = COLLECTION.range(min, max).unwrap();
    Ok(level.to_py(py))
}

#[pyfunction]
fn random<'py>(py: Python<'py>) -> PyResult<LevelPY<'py>> {
    let level = COLLECTION.random().unwrap();
    Ok(level.to_py(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn boxoban(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find, m)?)?;
    m.add_function(wrap_pyfunction!(range, m)?)?;
    m.add_function(wrap_pyfunction!(random, m)?)?;

    Ok(())
}

#[test]
fn test_collection() {
    use std::time::Instant;

    let now = Instant::now();
    (0..10000).for_each(|_| {
        COLLECTION.range(16, 1593);
    });
    dbg!(now.elapsed());

    let now = Instant::now();
    (0..10000).for_each(|_| {
        COLLECTION.random();
    });
    dbg!(now.elapsed());

    dbg!(COLLECTION.range(13, 15));
    dbg!(COLLECTION.find(0));
    dbg!(COLLECTION.random());
}
