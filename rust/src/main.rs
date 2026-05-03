mod board;
mod forbidden;

use board::{Board, Cell};
use forbidden::{ForbiddenKind, is_forbidden};
use std::io::{self, BufRead, Write};

struct PlayerSide {
    player: Cell,
    ai: Cell,
}

fn parse_input(line: &str) -> Option<(usize, usize)> {
    let mut parts = line.split_whitespace();
    let x: usize = parts.next()?.parse().ok()?;
    let y: usize = parts.next()?.parse().ok()?;
    if x >= board::SIZE || y >= board::SIZE {
        return None;
    }
    Some((y, x))
}

fn flush_print(s: &str) {
    print!("{}", s);
    io::stdout().flush().unwrap();
}

fn choose_side(lines: &mut impl Iterator<Item = String>) -> PlayerSide {
    loop {
        flush_print("先手(●)か後手(○)か選んでください [1: 先手 / 2: 後手] > ");
        match lines.next().as_deref().map(str::trim) {
            Some("1") => {
                return PlayerSide {
                    player: Cell::Black,
                    ai: Cell::White,
                };
            }
            Some("2") => {
                return PlayerSide {
                    player: Cell::White,
                    ai: Cell::Black,
                };
            }
            _ => println!("1 か 2 を入力してください。"),
        }
    }
}

fn prompt(current: Cell, side: &PlayerSide) {
    let label = if current == side.player {
        if current == Cell::Black {
            "あなた (● 先手)"
        } else {
            "あなた (○ 後手)"
        }
    } else {
        if current == Cell::Black {
            "AI (● 先手)"
        } else {
            "AI (○ 後手)"
        }
    };
    flush_print(&format!("{} の番 (x y) > ", label));
}

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines().map(|l| l.unwrap());

    let side = choose_side(&mut lines);

    let mut board = Board::new();
    let mut current = Cell::Black;

    println!();
    board.print();
    prompt(current, &side);

    for line in lines {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        match parse_input(&line) {
            None => {
                println!("入力が無効です。\"x y\" の形式で 0〜14 の整数を入力してください。");
            }
            Some((row, col)) => {
                if !board.cells[row][col].eq(&Cell::Empty) {
                    println!("({}, {}) にはすでに石があります。", col, row);
                    prompt(current, &side);
                    continue;
                }

                // 先手(Black)のみ禁じ手チェック
                if current == Cell::Black {
                    match is_forbidden(&board, row, col) {
                        ForbiddenKind::None => {}
                        kind => {
                            println!("禁じ手です（{}）。先手の負けです。", kind);
                            return;
                        }
                    }
                }

                board.place(row, col, current);
                board.print();

                if board.check_winner(row, col, current) {
                    if current == side.player {
                        println!("あなたの勝利！");
                    } else {
                        println!("AIの勝利！");
                    }
                    return;
                }

                if board.is_full() {
                    println!("引き分け！");
                    return;
                }

                current = match current {
                    Cell::Black => Cell::White,
                    Cell::White => Cell::Black,
                    Cell::Empty => unreachable!(),
                };
            }
        }

        prompt(current, &side);
    }
}
