//! ONNX Runtime を使ってニューラルネットの推論を行う
//!
//! 入力: (1, 3, 15, 15) f32
//!   Ch.0 自分の石
//!   Ch.1 相手の石
//!   Ch.2 手番（自分が先手なら全1、後手なら全0）
//! 出力:
//!   policy: (1, 225) softmax済み
//!   value:  (1, 1)   tanh済み

use ndarray::{Array, Array4, Ix2};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use std::path::Path;

use crate::board::{Board, Cell, SIZE};

pub struct Network {
    session: Session,
}

#[derive(Debug)]
pub struct PredictionOutput {
    pub policy: Vec<f32>, // 長さ 225
    pub value: f32,
}

impl Network {
    /// ONNXモデルを読み込む
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        Ok(Network { session })
    }

    /// 盤面を入力して policy + value を取得する
    ///
    /// `me` は「自分の石」として扱う側（Black or White）。
    /// それ以外の石は相手として扱う。
    pub fn predict(
        &mut self,
        board: &Board,
        me: Cell,
    ) -> Result<PredictionOutput, Box<dyn std::error::Error>> {
        // (1, 3, 15, 15) のテンソルを構築
        let mut input = Array4::<f32>::zeros((1, 3, SIZE, SIZE));

        // Black が「先手」なので、自分が先手かどうかは me == Cell::Black
        let is_first = me == Cell::Black;
        let opp = match me {
            Cell::Black => Cell::White,
            Cell::White => Cell::Black,
            Cell::Empty => unreachable!("空のCellは渡せない"),
        };

        for r in 0..SIZE {
            for c in 0..SIZE {
                let cell = board.cells[r][c];
                if cell == me {
                    input[[0, 0, r, c]] = 1.0;
                } else if cell == opp {
                    input[[0, 1, r, c]] = 1.0;
                }
                if is_first {
                    input[[0, 2, r, c]] = 1.0;
                }
            }
        }

        // 推論実行
        let input_tensor = Value::from_array(input)?;
        let outputs = self.session.run(ort::inputs!["board" => input_tensor])?;

        // policy 取り出し: (1, 225) -> Vec<f32>
        let policy_view = outputs["policy"].try_extract_array::<f32>()?;
        let policy_2d = policy_view.into_dimensionality::<Ix2>()?;
        let policy: Vec<f32> = policy_2d.row(0).to_vec();

        // value 取り出し: (1, 1) -> f32
        let value_view = outputs["value"].try_extract_array::<f32>()?;
        let value_2d = value_view.into_dimensionality::<Ix2>()?;
        let value = value_2d[[0, 0]];

        Ok(PredictionOutput { policy, value })
    }
}
