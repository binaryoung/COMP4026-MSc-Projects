mod level;

use level::{Collection, Level};
use numpy::{PyArray2, ToPyArray};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyException;

static COLLECTION_BYTES: &'static [u8] = include_bytes!(r"../levels/collection.bin");
static COLLECTION: Lazy<Collection> = Lazy::new(|| Collection::load_from_bytes(COLLECTION_BYTES).unwrap());

type LevelPy<'py> = (usize, usize, Vec<u8>, &'py PyArray2<u8>, &'py PyArray2<u8>);

impl Level {
    // Convert level to python data type
    fn to_py<'py>(&self, py: Python<'py>) -> LevelPy<'py> {
        let Level {
            id,
            score,
            trajectory,
            room,
            topology,
        } = self.clone();

        let room = room.to_pyarray(py);
        let topology = topology.to_pyarray(py);

        (id, score, trajectory, room, topology)
    }
}

// Return the level with the specified ID
#[pyfunction]
fn find<'py>(py: Python<'py>, id: usize) -> PyResult<LevelPy<'py>> {
    let level = COLLECTION.find(id).map_err(|e| PyException::new_err(e))?;
    Ok(level.to_py(py))
}

// Return a random level within the specified difficulty score range
#[pyfunction]
fn range<'py>(py: Python<'py>, min: usize, max: usize) -> PyResult<LevelPy<'py>> {
    let level = COLLECTION.range(min, max).map_err(|e| PyException::new_err(e))?;
    Ok(level.to_py(py))
}

// Return a random level
#[pyfunction]
fn random<'py>(py: Python<'py>) -> PyResult<LevelPy<'py>> {
    let level = COLLECTION.random().map_err(|e| PyException::new_err(e))?;
    Ok(level.to_py(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn boxoban_level_collection(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find, m)?)?;
    m.add_function(wrap_pyfunction!(range, m)?)?;
    m.add_function(wrap_pyfunction!(random, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
}

