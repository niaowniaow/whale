use super::*;

#[inline(always)]
pub(super) fn is_cancelled(ctx: &SearchContext) -> bool {
    ctx.cancellation_token.load(Ordering::Relaxed)
}

pub struct SearchContext<'a> {
    pub allow_null_move: bool,
    pub on_pv_path: bool,
    pub previous_pv: &'a [Move],
    pub excluded_move: Option<Move>,

    pub cut_node: bool,
    pub(super) gtp_graph: gtp::GtpTreeGraph,
    pub(super) gtp_parent: Option<usize>,
    pub pv_table: &'a mut PvTable,
    pub cancellation_token: &'a AtomicBool,
    pub search_state: &'a mut SearchState,
}
