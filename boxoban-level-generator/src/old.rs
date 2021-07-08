#[derive(Copy, Clone, Debug)]
enum State {
    Wall,
    Empty,
    Target,
    BoxOnTarget,
    Box,
    Player,
    PlayerOnTarget,
}

impl Default for State {
    fn default() -> Self {
        State::Wall
    }
}

impl Into<u8> for &State {
    fn into(self) -> u8 {
        match self {
            State::Wall => 0,
            State::Empty => 1,
            State::Target => 2,
            State::Box => 3,
            State::BoxOnTarget => 4,
            State::Player => 5,
            State::PlayerOnTarget => 6,
        }
    }
}

impl Into<char> for &State {
    fn into(self) -> char {
        match self {
            State::Wall => '0',
            State::Empty => '1',
            State::Target => '2',
            State::Box => '3',
            State::BoxOnTarget => '4',
            State::Player => '5',
            State::PlayerOnTarget => '6',
        }
    }
}

#[derive(Debug)]
pub struct Room<const W: usize, const H: usize> {
    array: [[State; W]; H],
}

impl<const W: usize, const H: usize> Default for Room<W, H>
{
    fn default() -> Self {
        Self {
            array: [[State::default(); W]; H],
        }
    }
}

impl<const W: usize, const H: usize> Display for Room<W, H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let matrix: String = self.array.iter().map(|row|{
            row.iter().map(|state| -> char {
                state.into()
            }).collect::<String>()
        }).intersperse("\n".into()).collect();

        write!(f, "{}", matrix)
    }
}
