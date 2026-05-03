//! 禁じ手判定（先手=Black のみ適用）
//!
//! 五目並べの禁じ手ルール：
//!   - 長連禁：6つ以上連続は禁手
//!   - 四四禁：四（片側でも勝利確定できる連）が同時に2つ以上できる手は禁手
//!   - 三三禁：活三（両端が空いている三）が同時に2つ以上できる手は禁手
//!
//! ただし、禁じ手でも「五を作る手」は優先して勝利とする（五勝ちルール）

use crate::board::{Board, Cell, SIZE};

/// 指定座標が禁じ手かどうかを判定する
/// （盤面にはまだ石が置かれていない前提で呼ぶ）
pub fn is_forbidden(board: &Board, row: usize, col: usize) -> ForbiddenKind {
    // 仮置きして判定
    let mut b = board.clone();
    b.cells[row][col] = Cell::Black;

    // 五（ちょうど5連）または長連は最初に確認
    let five = count_in_a_row(&b, row, col, Cell::Black);

    // 五が作れるなら禁じ手より優先して勝利（長連でなければ）
    if five == 5 {
        return ForbiddenKind::None;
    }
    // 長連禁（6以上）
    if five >= 6 {
        return ForbiddenKind::Overline;
    }

    // 四四禁・三三禁のカウント
    let fours = count_fours(&b, row, col);
    if fours >= 2 {
        return ForbiddenKind::DoubleFour;
    }

    let threes = count_open_threes(&b, row, col);
    if threes >= 2 {
        return ForbiddenKind::DoubleThree;
    }

    ForbiddenKind::None
}

#[derive(Debug, PartialEq)]
pub enum ForbiddenKind {
    None,
    Overline,    // 長連
    DoubleFour,  // 四四
    DoubleThree, // 三三
}

impl std::fmt::Display for ForbiddenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForbiddenKind::None => write!(f, "なし"),
            ForbiddenKind::Overline => write!(f, "長連禁"),
            ForbiddenKind::DoubleFour => write!(f, "四四禁"),
            ForbiddenKind::DoubleThree => write!(f, "三三禁"),
        }
    }
}

// ── 内部ユーティリティ ────────────────────────────────────────

/// 盤面上の (row, col) を中心として、4方向それぞれの連続数の最大値を返す
fn count_in_a_row(board: &Board, row: usize, col: usize, cell: Cell) -> usize {
    let directions: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    let mut max_count = 0;
    for (dr, dc) in directions {
        let count = line_count(board, row, col, dr, dc, cell);
        if count > max_count {
            max_count = count;
        }
    }
    max_count
}

/// 1方向（正逆両方）の連続数を数える
fn line_count(board: &Board, row: usize, col: usize, dr: i32, dc: i32, cell: Cell) -> usize {
    let mut count = 1;
    for sign in [1i32, -1] {
        let mut r = row as i32 + sign * dr;
        let mut c = col as i32 + sign * dc;
        while in_bounds(r, c) && board.cells[r as usize][c as usize] == cell {
            count += 1;
            r += sign * dr;
            c += sign * dc;
        }
    }
    count
}

fn in_bounds(r: i32, c: i32) -> bool {
    r >= 0 && r < SIZE as i32 && c >= 0 && c < SIZE as i32
}

fn get(board: &Board, r: i32, c: i32) -> Option<Cell> {
    if in_bounds(r, c) {
        Some(board.cells[r as usize][c as usize])
    } else {
        None
    }
}

// ── 四（Four）の判定 ─────────────────────────────────────────
//
// 「四」= その方向に石を1つ追加すれば五になる状態。
// 四四禁は「片端が塞がれていても」カウントする。
// つまり、空きマスに仮置きして五になるかを確認する。
//
// 具体的には、(row,col) に Black を置いた後、
// 各方向の連の両端にある空きマスに仮置きして五になれば「四」とカウント。

/// (row,col) に Black を置いた後の四の数を返す（4方向を独立に検査）
fn count_fours(board: &Board, row: usize, col: usize) -> usize {
    let directions: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    let mut count = 0;
    for (dr, dc) in directions {
        if is_four_in_direction(board, row, col, dr, dc) {
            count += 1;
        }
    }
    count
}

/// 指定方向で「四」が成立しているか
/// 連のブロック全体を走査し、空き1マスを埋めれば五になるケースを探す
fn is_four_in_direction(board: &Board, row: usize, col: usize, dr: i32, dc: i32) -> bool {
    let r0 = row as i32;
    let c0 = col as i32;

    // この方向の連の端を求める（Black or Empty を辿る範囲）
    // 連 + 隣接空きを含めた「窓」を走査して四パターンを探す
    // ウィンドウ幅5で「Black が4つ・空きが1つ」のパターンを探す
    //
    // 探索範囲: (r0,c0) を中心に ±4 まで
    // ウィンドウ [i, i+4] を順にスライドさせる

    // まず基点から正逆方向に最大4マス分の座標列を取る
    let start_offset = -4i32;
    let mut found = false;

    'window: for offset in start_offset..=0 {
        // ウィンドウ: offset .. offset+4 の5マス
        let mut black_count = 0;
        let mut empty_pos: Option<(i32, i32)> = None;
        let mut valid = true;

        for i in 0..5i32 {
            let r = r0 + (offset + i) * dr;
            let c = c0 + (offset + i) * dc;
            match get(board, r, c) {
                Some(Cell::Black) => black_count += 1,
                Some(Cell::Empty) => {
                    if empty_pos.is_some() {
                        // 空きが2つ以上→このウィンドウは四ではない
                        valid = false;
                        break;
                    }
                    empty_pos = Some((r, c));
                }
                _ => {
                    // 盤外または相手石→このウィンドウは無効
                    valid = false;
                    break;
                }
            }
        }

        if !valid {
            continue 'window;
        }

        // Black4 + 空き1 のウィンドウが見つかった
        if black_count == 4 {
            if let Some((er, ec)) = empty_pos {
                // その空きに仮置きして五になるか確認（長連でないことも確認）
                let mut tmp = board.clone();
                tmp.cells[er as usize][ec as usize] = Cell::Black;
                let cnt = line_count(&tmp, er as usize, ec as usize, dr, dc, Cell::Black);
                if cnt == 5 {
                    found = true;
                    break;
                }
            }
        }
    }

    found
}

// ── 活三（Open Three）の判定 ──────────────────────────────────
//
// 「活三」= 両端が空いていて、あと1手で活四（両端空き四）になれる三。
// つまり「空きに1石追加→活四になる」パターンを探す。
//
// 活四 = 四であり、かつその四の連の両端が空いているもの。
// 三三禁は「両端が空いた三」が2つ以上同時にできる手を禁じる。

/// (row,col) に Black を置いた後の活三の数を返す
fn count_open_threes(board: &Board, row: usize, col: usize) -> usize {
    let directions: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
    let mut count = 0;
    for (dr, dc) in directions {
        if is_open_three_in_direction(board, row, col, dr, dc) {
            count += 1;
        }
    }
    count
}

/// 指定方向で「活三」が成立しているか
/// 「空きに1石追加すると活四になる」ケースを探す
fn is_open_three_in_direction(board: &Board, row: usize, col: usize, dr: i32, dc: i32) -> bool {
    let r0 = row as i32;
    let c0 = col as i32;

    // この方向の空きマス候補を列挙し、仮置きして「活四」になるか試す
    // 活四 = その方向で四が成立 かつ 連の両端が空き
    for offset in -4i32..=0 {
        // ウィンドウ5マス内の空きマスに仮置き
        for i in 0..5i32 {
            let r = r0 + (offset + i) * dr;
            let c = c0 + (offset + i) * dc;
            if get(board, r, c) != Some(Cell::Empty) {
                continue;
            }
            // 仮置き
            let mut tmp = board.clone();
            tmp.cells[r as usize][c as usize] = Cell::Black;

            // 仮置き後に (r,c) が「活四」になるか確認
            if is_open_four_at(&tmp, r as usize, c as usize, dr, dc) {
                return true;
            }
        }
    }
    false
}

/// (r,c) に Black を置いた後、その方向で「活四」かどうか
/// 活四 = 四 かつ 連の両端が空き
fn is_open_four_at(board: &Board, row: usize, col: usize, dr: i32, dc: i32) -> bool {
    let r0 = row as i32;
    let c0 = col as i32;

    // 連の端を求める
    let mut pos_end = (r0, c0);
    loop {
        let nr = pos_end.0 + dr;
        let nc = pos_end.1 + dc;
        if get(board, nr, nc) == Some(Cell::Black) {
            pos_end = (nr, nc);
        } else {
            break;
        }
    }
    let mut neg_end = (r0, c0);
    loop {
        let nr = neg_end.0 - dr;
        let nc = neg_end.1 - dc;
        if get(board, nr, nc) == Some(Cell::Black) {
            neg_end = (nr, nc);
        } else {
            break;
        }
    }

    let cnt = line_count(board, row, col, dr, dc, Cell::Black);
    if cnt != 4 {
        return false;
    }

    // 両端が空きであること
    let beyond_pos = (pos_end.0 + dr, pos_end.1 + dc);
    let beyond_neg = (neg_end.0 - dr, neg_end.1 - dc);
    get(board, beyond_pos.0, beyond_pos.1) == Some(Cell::Empty)
        && get(board, beyond_neg.0, beyond_neg.1) == Some(Cell::Empty)
}
