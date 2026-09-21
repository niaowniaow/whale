use crate::board::state::BoardState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvaluationBackend {
    #[default]
    ClassicalNNUE,
    SFNN,
    HybridNNUE,
    WDIN,
    FutureModel,
}

impl EvaluationBackend {
    pub fn name(self) -> &'static str {
        match self {
            EvaluationBackend::ClassicalNNUE => "classical-nnue",
            EvaluationBackend::SFNN => "sfnn",
            EvaluationBackend::HybridNNUE => "hybrid-nnue",
            EvaluationBackend::WDIN => "wdin",
            EvaluationBackend::FutureModel => "future",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "classical" | "classical-nnue" | "nnue" => Some(EvaluationBackend::ClassicalNNUE),
            "sfnn" | "sfnn16" | "v16" => Some(EvaluationBackend::SFNN),
            "hybrid" | "hybrid-nnue" | "dcn" => Some(EvaluationBackend::HybridNNUE),
            "wdin" => Some(EvaluationBackend::WDIN),
            "future" => Some(EvaluationBackend::FutureModel),
            _ => None,
        }
    }
}

pub fn evaluate_with_backend(board: &mut BoardState, backend: EvaluationBackend) -> i16 {
    match backend {
        EvaluationBackend::ClassicalNNUE => crate::eval::evaluate_fast(board, 0),
        EvaluationBackend::SFNN => crate::eval::evaluate(board),
        EvaluationBackend::HybridNNUE => crate::eval::evaluate_with_depth(board, 0, 8),
        EvaluationBackend::WDIN | EvaluationBackend::FutureModel => crate::eval::evaluate(board),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn all_backends_return_bounded_scores() {
        for backend in [
            EvaluationBackend::ClassicalNNUE,
            EvaluationBackend::SFNN,
            EvaluationBackend::HybridNNUE,
            EvaluationBackend::WDIN,
            EvaluationBackend::FutureModel,
        ] {
            let mut board = BoardState::parse_fen(STARTING_FEN);
            let s = evaluate_with_backend(&mut board, backend);
            assert!(s.abs() <= 29000, "{backend:?} out of range");
        }
    }

    #[test]
    fn backend_names_roundtrip() {
        assert_eq!(
            EvaluationBackend::from_name("sfnn"),
            Some(EvaluationBackend::SFNN)
        );
        assert_eq!(
            EvaluationBackend::from_name("WDIN"),
            Some(EvaluationBackend::WDIN)
        );
        assert_eq!(EvaluationBackend::from_name("nope"), None);
        assert_eq!(EvaluationBackend::SFNN.name(), "sfnn");
    }
}
