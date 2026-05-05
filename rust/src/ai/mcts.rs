//! Monte Carlo Tree Search (AlphaZero方式)
//!
//! Python側のmcts.pyと同じアルゴリズムを Rust で実装。
//! PUCT 選択 → Expansion (NN推論) → Backpropagation を繰り返す。

use crate::ai::network::Network;
use crate::board::{Board, Cell, SIZE};

const C_PUCT: f32 = 1.5;
pub const DEFAULT_SIMULATIONS: usize = 400;

const NUM_CELLS: usize = SIZE * SIZE;
const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

// ── ヘルパ ─────────────────────────────────────────────────

fn check_winner_from(board: &Board, row: usize, col: usize, stone: Cell) -> bool {
    for (dr, dc) in DIRECTIONS {
        let mut count = 1;
        for sign in [1i32, -1] {
            let mut r = row as i32 + sign * dr;
            let mut c = col as i32 + sign * dc;
            while r >= 0
                && r < SIZE as i32
                && c >= 0
                && c < SIZE as i32
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

// ── ノード ────────────────────────────────────────────────

/// MCTSのノードはアリーナベースで管理し、子ノード参照はインデックスで持つ
struct Node {
    board: Board,
    stone: Cell, // このノードで打つ側
    parent: Option<usize>,
    last_move: Option<usize>, // 親→このノードに来た手（フラットインデックス）
    children: Vec<(usize, usize)>, // (move, child_node_idx)
    prior: f32,
    visits: u32,
    value_sum: f32,
    expanded: bool,
}

impl Node {
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    fn ucb(&self, parent_visits: u32) -> f32 {
        self.q() + C_PUCT * self.prior * (parent_visits as f32).sqrt() / (1.0 + self.visits as f32)
    }
}

// ── MCTS本体 ──────────────────────────────────────────────

pub struct Mcts {
    pub n_simulations: usize,
}

impl Mcts {
    pub fn new(n_simulations: usize) -> Self {
        Mcts { n_simulations }
    }

    /// 最善手を選ぶ（フラットインデックス→(row, col) で返す）
    pub fn best_move(
        &self,
        board: &Board,
        stone: Cell,
        net: &mut Network,
    ) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let mut nodes: Vec<Node> = Vec::with_capacity(self.n_simulations * 8);

        // ルート
        nodes.push(Node {
            board: board.clone(),
            stone,
            parent: None,
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

        // 訪問回数最大の子を選ぶ
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

    fn simulate(
        &self,
        nodes: &mut Vec<Node>,
        net: &mut Network,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // ── Selection ────────────────────────────────────
        let mut path: Vec<usize> = Vec::new();
        let mut current_idx = 0usize;

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

        // ── 終局判定 ─────────────────────────────────────
        // current_idx に到達した時点で last_move があれば、その手で勝敗が決まったかチェック
        let value: f32 = if let Some(last_move) = nodes[current_idx].last_move {
            let r = last_move / SIZE;
            let c = last_move % SIZE;
            // 直前に打った石は「親の手番」=「current.stone の逆」
            let last_stone = opposite(nodes[current_idx].stone);
            if check_winner_from(&nodes[current_idx].board, r, c, last_stone) {
                // 直前手で勝敗確定 → 「これから打つ側」から見ると -1（負け）
                -1.0
            } else if legal_moves(&nodes[current_idx].board).is_empty() {
                // 引き分け
                0.0
            } else {
                // 通常: NNで評価して展開
                self.expand(nodes, current_idx, net)?
            }
        } else {
            // ルートに来ることは expand 済みなのでここには来ないが念のため
            self.expand(nodes, current_idx, net)?
        };

        // ── Backpropagation ──────────────────────────────
        let mut v = value;
        for &node_idx in path.iter().rev() {
            nodes[node_idx].visits += 1;
            nodes[node_idx].value_sum += v;
            v = -v;
        }
        // ルートも更新
        nodes[0].visits += 1;
        nodes[0].value_sum += v;

        Ok(())
    }

    /// ノードを展開し、NN推論で価値を返す
    fn expand(
        &self,
        nodes: &mut Vec<Node>,
        idx: usize,
        net: &mut Network,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        // NN推論（borrow conflictを避けるためにフィールドだけ先に取り出す）
        let board_clone = nodes[idx].board.clone();
        let stone = nodes[idx].stone;

        let pred = net.predict(&board_clone, stone)?;
        let mut policy = pred.policy;
        let value = pred.value;

        // 合法手以外を0にしてから再正規化
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
            // すべて0なら一様分布
            let p = 1.0 / legal.len() as f32;
            for &m in &legal {
                policy[m] = p;
            }
        }

        // 子ノード作成
        let next_stone = opposite(stone);
        let parent_idx = idx;
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
                parent: Some(parent_idx),
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
