mod ai;
mod board;
mod forbidden;

use ai::{Mcts, Network};
use board::{Board, Cell};
use forbidden::{ForbiddenKind, is_forbidden};
use std::env;
use std::io::{self, BufRead, Write};

const DEFAULT_MODEL_PATH: &str = "../models/best_model.onnx";

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

fn prompt_player(current: Cell) {
    let label = if current == Cell::Black {
        "あなた (● 先手)"
    } else {
        "あなた (○ 後手)"
    };
    flush_print(&format!("{} の番 (x y) > ", label));
}

fn label_ai(current: Cell) -> &'static str {
    if current == Cell::Black {
        "AI (● 先手)"
    } else {
        "AI (○ 後手)"
    }
}

fn main() {
    // モデルパスは引数で上書き可能
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() >= 2 {
        args[1].clone()
    } else {
        DEFAULT_MODEL_PATH.to_string()
    };

    println!("ONNXモデルを読み込み中: {}", model_path);
    let mut net = match Network::load(&model_path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("モデル読み込み失敗: {}", e);
            eprintln!("ヒント: python/export_onnx.py で .bin から .onnx を生成してください。");
            std::process::exit(1);
        }
    };

    // MCTSのシミュレーション回数
    let n_sim: usize = env::var("MCTS_SIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(ai::mcts::DEFAULT_SIMULATIONS);
    println!("MCTSシミュレーション回数: {}\n", n_sim);
    let mcts = Mcts::new(n_sim);

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines().map(|l| l.unwrap());

    let side = choose_side(&mut lines);

    let mut board = Board::new();
    let mut current = Cell::Black;

    println!();
    board.print();

    loop {
        // 終局チェック用: 直前にAIが置いた手の勝敗判定はAIターン後に実施するため、ここでは入力受付のみ
        if current == side.ai {
            // ── AIのターン ────────────────────────────────
            flush_print(&format!("{} 思考中...", label_ai(current)));
            io::stdout().flush().unwrap();

            let (row, col) = match mcts.best_move(&board, current, &mut net) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("\nAI思考エラー: {}", e);
                    std::process::exit(1);
                }
            };
            println!(" → ({}, {})", col, row);

            // 先手のときだけ禁じ手チェック（AIも適用）
            if current == Cell::Black {
                match is_forbidden(&board, row, col) {
                    ForbiddenKind::None => {}
                    kind => {
                        println!("AIが禁じ手を打ちました（{}）。AIの負けです。", kind);
                        return;
                    }
                }
            }

            board.place(row, col, current);
            board.print();

            if board.check_winner(row, col, current) {
                println!("AIの勝利！");
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
            continue;
        }

        // ── プレイヤーのターン ──────────────────────────
        prompt_player(current);
        let line = match lines.next() {
            Some(l) => l,
            None => return,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        match parse_input(line) {
            None => {
                println!("入力が無効です。\"x y\" の形式で 0〜14 の整数を入力してください。");
                continue;
            }
            Some((row, col)) => {
                if board.cells[row][col] != Cell::Empty {
                    println!("({}, {}) にはすでに石があります。", col, row);
                    continue;
                }

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
                    println!("あなたの勝利！");
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
    }
}
