pub const SIZE: usize = 15;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Cell {
    Empty,
    Black, // 先手
    White, // 後手
}

#[derive(Clone)]
pub struct Board {
    pub cells: [[Cell; SIZE]; SIZE],
}

impl Board {
    pub fn new() -> Self {
        Board {
            cells: [[Cell::Empty; SIZE]; SIZE],
        }
    }

    pub fn place(&mut self, row: usize, col: usize, cell: Cell) -> bool {
        if row >= SIZE || col >= SIZE {
            return false;
        }
        if self.cells[row][col] != Cell::Empty {
            return false;
        }
        self.cells[row][col] = cell;
        true
    }

    pub fn get(&self, row: usize, col: usize) -> Cell {
        self.cells[row][col]
    }

    pub fn is_full(&self) -> bool {
        self.cells
            .iter()
            .all(|row| row.iter().all(|&c| c != Cell::Empty))
    }

    /// 勝利判定: 指定セルが5つ連続しているか
    pub fn check_winner(&self, row: usize, col: usize, cell: Cell) -> bool {
        let directions: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for (dr, dc) in directions {
            let mut count = 1;
            let mut r = row as i32 + dr;
            let mut c = col as i32 + dc;
            while r >= 0
                && r < SIZE as i32
                && c >= 0
                && c < SIZE as i32
                && self.cells[r as usize][c as usize] == cell
            {
                count += 1;
                r += dr;
                c += dc;
            }
            r = row as i32 - dr;
            c = col as i32 - dc;
            while r >= 0
                && r < SIZE as i32
                && c >= 0
                && c < SIZE as i32
                && self.cells[r as usize][c as usize] == cell
            {
                count += 1;
                r -= dr;
                c -= dc;
            }
            if count >= 5 {
                return true;
            }
        }
        false
    }

    fn print_col_header() {
        // 十の位（0〜9は空白、10〜14は "1"）
        print!("    ");
        for c in 0..SIZE {
            if c >= 10 {
                print!("1 ");
            } else {
                print!("  ");
            }
        }
        println!();
        // 一の位
        print!("    ");
        for c in 0..SIZE {
            print!("{} ", c % 10);
        }
        println!();
    }

    pub fn print(&self) {
        Self::print_col_header();
        for row in 0..SIZE {
            print!("{:2}  ", row);
            for col in 0..SIZE {
                let symbol = match self.cells[row][col] {
                    Cell::Empty => "· ",
                    Cell::Black => "● ",
                    Cell::White => "○ ",
                };
                print!("{}", symbol);
            }
            println!();
        }
    }
}
