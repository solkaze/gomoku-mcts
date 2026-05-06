//! ONNX Runtime を使ってニューラルネットの推論を行う
//!
//! 入力: (1, 3, 15, 15) f32
//!   Ch.0 = to_play(=me)の石
//!   Ch.1 = 相手の石
//!   Ch.2 = 全1なら先手番、全0なら後手番
//!
//! 盤面そのものは黒=Black, 白=White のまま渡す。

use ndarray::Array4;
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use std::path::Path;

use crate::board::{Board, Cell, SIZE};

pub struct Network {
    session: Session,
}

#[derive(Debug)]
pub struct PredictionOutput {
    pub policy: Vec<f32>,
    pub value: f32,
}

impl Network {
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        Ok(Network { session })
    }

    /// 盤面と「次に打つ側 me」を渡して推論する。
    pub fn predict(
        &mut self,
        board: &Board,
        me: Cell,
    ) -> Result<PredictionOutput, Box<dyn std::error::Error>> {
        let mut input = Array4::<f32>::zeros((1, 3, SIZE, SIZE));

        let opp = match me {
            Cell::Black => Cell::White,
            Cell::White => Cell::Black,
            Cell::Empty => unreachable!("空のCellは渡せない"),
        };
        let is_first = me == Cell::Black;

        for r in 0..SIZE {
            for c in 0..SIZE {
                let cell = board.cells[r][c];
                if cell == me {
                    input[[0, 0, r, c]] = 1.0;
                } else if cell == opp {
                    input[[0, 1, r, c]] = 1.0;
                }
                // Ch.2: 先手番なら1、後手番なら0（学習時と一致）
                if is_first {
                    input[[0, 2, r, c]] = 1.0;
                }
            }
        }

        let input_tensor = Value::from_array(input)?;
        let outputs = self.session.run(ort::inputs!["board" => input_tensor])?;

        let policy_array = outputs["policy"].try_extract_array::<f32>()?;
        let policy: Vec<f32> = policy_array.iter().take(SIZE * SIZE).copied().collect();

        let value_array = outputs["value"].try_extract_array::<f32>()?;
        let value = *value_array.iter().next().ok_or("value tensor is empty")?;

        Ok(PredictionOutput { policy, value })
    }
}
