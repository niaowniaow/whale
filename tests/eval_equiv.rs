use whale::board::node_threats::NodeThreats;
use whale::board::state::BoardState;
use whale::common::helpers::{ADVANCED_MOVE_FEN, ENDGAME_FEN, KIWI_PETE_FEN, STARTING_FEN};
use whale::common::move_list::MoveList;
use whale::eval::{evaluate_with_depth, evaluate_with_depth_cached};

#[test]
fn cached_eval_matches_fresh_everywhere() {
    let fens = [
        STARTING_FEN,
        KIWI_PETE_FEN,
        ENDGAME_FEN,
        ADVANCED_MOVE_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "7k/8/2b2b2/3pp3/8/8/8/K2RQ3 w - - 0 1",
        "4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
    ];
    for fen in fens {
        let mut board = BoardState::parse_fen(fen);

        let mut suite: Vec<BoardState> = vec![board.clone()];
        let mut ml = MoveList::new();
        board.generate_moves(&mut ml);
        for e in ml.iter().take(4) {
            board.make_move(e.mv);
            suite.push(board.clone());
            board.unmake_move(e.mv);
        }
        for mut b in suite {
            let nt = NodeThreats::compute(&b);
            for optimism in [-50, 0, 37] {
                for depth in [0u8, 1, 3, 5, 6, 8, 15, 16, 20, 40] {
                    let fresh = evaluate_with_depth(&mut b, optimism, depth);
                    let cached = evaluate_with_depth_cached(&mut b, optimism, depth, &nt);
                    assert_eq!(
                        fresh, cached,
                        "eval mismatch fen={fen} opt={optimism} depth={depth}"
                    );
                }
            }
        }
    }
}
