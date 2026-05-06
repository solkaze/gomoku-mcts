//! Monte Carlo Tree Search (AlphaZero方式) - 新設計版
//!
//! 設計:
//!   ・盤面は黒=Black, 白=White のまま
//!   ・各ノードは「次に打つ側 stone」を保持
//!   ・NN呼び出し時に network.predict(&board, stone) で視点変換
//!   ・backpropはpathにrootを含めて、reverseしながら符号反転＆加算

use crate::board::{Board, Cell, SIZE};
use crate::ai::network::Network;

const C_PUCT: f32 = 1.5;
pub const DEFAULT_SIMULATIONS: usize = 400;

const NUM_CELLS: usize = SIZE * SIZE;
const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];


fn check_winner_from(board: &Board, row: usize, col: usize, stone: Cell) -> bool {
    for (dr, dc) in DIRECTIONS {
        let mut count = 1;
        for sign in [1i32, -1] {
            let mut r = row as i32 + sign * dr;
            let mut c = col as i32 + sign * dc;
            while r >= 0 && r < SIZE as i32 && c >= 0 && c < SIZE as i32
                && board.cells[r as usize][c as usize] == stone
            {
                count += 1;
                r += sign * dr;
                c += sign * dc;
            }
        }
        if count >= 5 {
            return true;
        }
    }
    false
}

fn legal_moves(board: &Board) -> Vec<usize> {
    let mut moves = Vec::with_capacity(NUM_CELLS);
    for r in 0..SIZE {
        for c in 0..SIZE {
            if board.cells[r][c] == Cell::Empty {
                moves.push(r * SIZE + c);
            }
        }
    }
    moves
}

fn opposite(cell: Cell) -> Cell {
    match cell {
        Cell::Black => Cell::White,
        Cell::White => Cell::Black,
        Cell::Empty => unreachable!(),
    }
}


struct Node {
    board: Board,
    stone: Cell,
    last_move: Option<usize>,
    children: Vec<(usize, usize)>,
    prior: f32,
    visits: u32,
    value_sum: f32,
    expanded: bool,
}

impl Node {
    fn q(&self) -> f32 {
        if self.visits == 0 { 0.0 } else { self.value_sum / self.visits as f32 }
    }

    fn ucb(&self, parent_visits: u32) -> f32 {
        self.q() + C_PUCT * self.prior * (parent_visits as f32).sqrt() / (1.0 + self.visits as f32)
    }
}


pub struct Mcts {
    pub n_simulations: usize,
}

impl Mcts {
    pub fn new(n_simulations: usize) -> Self {
        Mcts { n_simulations }
    }

    pub fn best_move(
        &self,
        board: &Board,
        stone: Cell,
        net: &mut Network,
    ) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let mut nodes: Vec<Node> = Vec::with_capacity(self.n_simulations * 8);

        nodes.push(Node {
            board: board.clone(),
            stone,
            last_move: None,
            children: Vec::new(),
            prior: 0.0,
            visits: 0,
            value_sum: 0.0,
            expanded: false,
        });
        self.expand(&mut nodes, 0, net)?;

        for _ in 0..self.n_simulations {
            self.simulate(&mut nodes, net)?;
        }

        let root = &nodes[0];
        let (best_move, _) = root
            .children
            .iter()
            .max_by_key(|(_, idx)| nodes[*idx].visits)
            .ok_or("着手可能な手がありません")?;

        let row = best_move / SIZE;
        let col = best_move % SIZE;
        Ok((row, col))
    }

    fn simulate(&self, nodes: &mut Vec<Node>, net: &mut Network) -> Result<(), Box<dyn std::error::Error>> {
        // path にはルートも含める
        let mut path: Vec<usize> = vec![0];
        let mut current_idx = 0usize;

        // Selection
        while nodes[current_idx].expanded && !nodes[current_idx].children.is_empty() {
            let parent_visits = nodes[current_idx].visits.max(1);
            let mut best_idx = nodes[current_idx].children[0].1;
            let mut best_score = f32::NEG_INFINITY;
            for &(_, child_idx) in &nodes[current_idx].children {
                let s = nodes[child_idx].ucb(parent_visits);
                if s > best_score {
                    best_score = s;
                    best_idx = child_idx;
                }
            }
            current_idx = best_idx;
            path.push(current_idx);
        }

        // 終局判定
        let value: f32 = if let Some(last_move) = nodes[current_idx].last_move {
            let r = last_move / SIZE;
            let c = last_move % SIZE;
            let last_stone = opposite(nodes[current_idx].stone);
            if check_winner_from(&nodes[current_idx].board, r, c, last_stone) {
                -1.0
            } else if legal_moves(&nodes[current_idx].board).is_empty() {
                0.0
            } else {
                self.expand(nodes, current_idx, net)?
            }
        } else {
            // ルートはすでに展開済み（best_moveの最初で）なのでここには通常来ない
            self.expand(nodes, current_idx, net)?
        };

        // Backpropagation: pathを末端から辿り、視点を反転しながら加算
        let mut v = value;
        for &node_idx in path.iter().rev() {
            nodes[node_idx].visits += 1;
            nodes[node_idx].value_sum += v;
            v = -v;
        }

        Ok(())
    }

    fn expand(
        &self,
        nodes: &mut Vec<Node>,
        idx: usize,
        net: &mut Network,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let board_clone = nodes[idx].board.clone();
        let stone = nodes[idx].stone;

        let pred = net.predict(&board_clone, stone)?;
        let mut policy = pred.policy;
        let value = pred.value;

        let legal = legal_moves(&board_clone);
        if legal.is_empty() {
            nodes[idx].expanded = true;
            return Ok(value);
        }

        let mut sum = 0.0;
        for i in 0..NUM_CELLS {
            if !legal.contains(&i) {
                policy[i] = 0.0;
            } else {
                sum += policy[i];
            }
        }
        if sum > 0.0 {
            for v in policy.iter_mut() {
                *v /= sum;
            }
        } else {
            let p = 1.0 / legal.len() as f32;
            for &m in &legal {
                policy[m] = p;
            }
        }

        let next_stone = opposite(stone);
        let mut children = Vec::with_capacity(legal.len());
        for &m in &legal {
            let r = m / SIZE;
            let c = m % SIZE;
            let mut new_board = board_clone.clone();
            new_board.cells[r][c] = stone;

            let child_idx = nodes.len();
            nodes.push(Node {
                board: new_board,
                stone: next_stone,
                last_move: Some(m),
                children: Vec::new(),
                prior: policy[m],
                visits: 0,
                value_sum: 0.0,
                expanded: false,
            });
            children.push((m, child_idx));
        }
        nodes[idx].children = children;
        nodes[idx].expanded = true;

        Ok(value)
    }
}