pub const MAX_GTP_NODES: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
pub struct GtpNode {
    pub depth: u8,
    pub eval_margin: i16,
    pub is_capture: bool,
    pub in_check: bool,
    pub history_score: i32,
    pub parent_idx: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct GtpTreeGraph {
    nodes: [GtpNode; MAX_GTP_NODES],
    adjacency: [u16; MAX_GTP_NODES],
    count: usize,
}

impl Default for GtpTreeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl GtpTreeGraph {
    pub const fn new() -> Self {
        Self {
            nodes: [GtpNode {
                depth: 0,
                eval_margin: 0,
                is_capture: false,
                in_check: false,
                history_score: 0,
                parent_idx: None,
            }; MAX_GTP_NODES],
            adjacency: [0; MAX_GTP_NODES],
            count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.adjacency.fill(0);
    }

    pub fn add_node(&mut self, node: GtpNode) -> usize {
        if self.count >= MAX_GTP_NODES || node.parent_idx.is_some_and(|parent| parent >= self.count)
        {
            return MAX_GTP_NODES;
        }

        let idx = self.count;
        self.nodes[idx] = node;
        self.count += 1;

        if let Some(parent) = node.parent_idx {
            self.adjacency[parent] |= 1 << idx;
            self.adjacency[idx] |= 1 << parent;
        }

        idx
    }
}

pub struct GtpPruner;

impl GtpPruner {
    pub fn node_score(graph: &GtpTreeGraph, idx: usize) -> i32 {
        let node = &graph.nodes[idx];
        let mut score = 50i32;
        score += (node.eval_margin as i32 / 64).clamp(-30, 30);
        score += (node.history_score / 512).clamp(-20, 20);
        if node.is_capture {
            score += 12;
        }
        if node.in_check {
            score -= 25;
        }
        let degree = (graph.adjacency[idx].count_ones() as i32).min(4);
        score += degree * 4;
        score += (node.depth.min(8) as i32) * 2;
        score.clamp(0, 100)
    }

    pub fn message_passing(graph: &GtpTreeGraph) -> [u8; MAX_GTP_NODES] {
        let mut importance = [50u8; MAX_GTP_NODES];
        if graph.count == 0 {
            return importance;
        }
        for (i, slot) in importance.iter_mut().enumerate().take(graph.count) {
            let own = Self::node_score(graph, i);
            let adj = graph.adjacency[i];
            let mut neighbor_sum = 0i32;
            let mut neighbor_count = 0i32;
            let mut n_idx = 0usize;
            while n_idx < graph.count {
                if (adj & (1 << n_idx)) != 0 {
                    neighbor_sum += Self::node_score(graph, n_idx);
                    neighbor_count += 1;
                }
                n_idx += 1;
            }
            let smoothed = if neighbor_count > 0 {
                (2 * own + neighbor_sum / neighbor_count) / 3
            } else {
                own
            };
            *slot = smoothed.clamp(0, 100) as u8;
        }
        importance
    }

    pub fn should_prune_subtree(graph: &GtpTreeGraph, node_idx: usize, threshold: u8) -> bool {
        if node_idx >= graph.count {
            return false;
        }
        let scores = Self::message_passing(graph);
        scores[node_idx] < threshold
    }
}

pub use GtpPruner as GtpModel;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gtp_tree_graph_creation() {
        let mut graph = GtpTreeGraph::new();
        let root = GtpNode {
            depth: 8,
            eval_margin: 50,
            is_capture: false,
            in_check: false,
            history_score: 100,
            parent_idx: None,
        };
        let r_idx = graph.add_node(root);
        assert_eq!(r_idx, 0);

        let child1 = GtpNode {
            depth: 7,
            eval_margin: -30,
            is_capture: true,
            in_check: false,
            history_score: 500,
            parent_idx: Some(0),
        };
        let c1_idx = graph.add_node(child1);
        assert_eq!(c1_idx, 1);
        assert_eq!(graph.adjacency[0] & (1 << 1), 1 << 1);
        assert_eq!(graph.adjacency[1] & (1 << 0), 1 << 0);
    }

    #[test]
    fn test_gtp_message_passing() {
        let mut graph = GtpTreeGraph::new();
        let n1 = graph.add_node(GtpNode {
            depth: 4,
            eval_margin: -200,
            is_capture: false,
            in_check: false,
            history_score: -1000,
            parent_idx: None,
        });
        let n2 = graph.add_node(GtpNode {
            depth: 3,
            eval_margin: -300,
            is_capture: false,
            in_check: false,
            history_score: -1500,
            parent_idx: Some(n1),
        });

        let importance = GtpModel::message_passing(&graph);
        assert!(importance[n1] <= 100);
        assert!(importance[n2] <= 100);
    }

    #[test]
    fn full_graph_does_not_alias_an_existing_candidate() {
        let mut graph = GtpTreeGraph::new();
        for _ in 0..MAX_GTP_NODES {
            graph.add_node(GtpNode::default());
        }
        let candidate = GtpNode {
            eval_margin: -30000,
            history_score: -30000,
            ..GtpNode::default()
        };
        let before = GtpModel::message_passing(&graph);
        let idx = graph.add_node(candidate);
        assert_eq!(idx, MAX_GTP_NODES);
        assert_eq!(graph.count, MAX_GTP_NODES);
        assert_eq!(graph.nodes[MAX_GTP_NODES - 1].eval_margin, 0);
        assert_eq!(GtpModel::message_passing(&graph), before);
        assert!(!GtpModel::should_prune_subtree(&graph, idx, 100));
        assert_eq!(
            graph.add_node(GtpNode {
                parent_idx: Some(idx),
                ..candidate
            }),
            MAX_GTP_NODES
        );
    }

    #[test]
    fn rejects_self_and_forward_parent_edges() {
        for parent in [0, 1, MAX_GTP_NODES - 1, usize::MAX] {
            let mut graph = GtpTreeGraph::new();
            graph.add_node(GtpNode {
                parent_idx: Some(parent),
                ..GtpNode::default()
            });
            assert_eq!(graph.count, 0);
            assert_eq!(graph.adjacency, [0; MAX_GTP_NODES]);
        }
    }

    #[test]
    fn empty_graph_uses_default_importance() {
        let graph = GtpTreeGraph::new();
        assert_eq!(GtpModel::message_passing(&graph), [50u8; MAX_GTP_NODES]);
        assert!(!GtpModel::should_prune_subtree(&graph, 0, 100));
        let _ = GtpTreeGraph::default();
    }

    #[test]
    fn reset_clears_nodes_and_edges() {
        let mut graph = GtpTreeGraph::new();
        let root = graph.add_node(GtpNode {
            depth: 4,
            ..GtpNode::default()
        });
        let _ = graph.add_node(GtpNode {
            parent_idx: Some(root),
            ..GtpNode::default()
        });
        assert_eq!(graph.count, 2);
        graph.reset();
        assert_eq!(graph.count, 0);
        assert_eq!(graph.adjacency, [0; MAX_GTP_NODES]);
        assert_eq!(GtpModel::message_passing(&graph), [50u8; MAX_GTP_NODES]);
    }

    #[test]
    fn prune_decision_follows_threshold() {
        let mut graph = GtpTreeGraph::new();
        let idx = graph.add_node(GtpNode {
            depth: 3,
            eval_margin: -400,
            history_score: -2000,
            ..GtpNode::default()
        });
        let score = GtpModel::message_passing(&graph)[idx];
        assert!(GtpModel::should_prune_subtree(
            &graph,
            idx,
            score.saturating_add(1)
        ));
        assert!(!GtpModel::should_prune_subtree(&graph, idx, score));
        assert!(!GtpModel::should_prune_subtree(&graph, idx, 0));
    }

    #[test]
    fn message_passing_covers_flags_and_clamps() {
        let mut graph = GtpTreeGraph::new();
        let a = graph.add_node(GtpNode {
            depth: 200,
            eval_margin: 30_000,
            is_capture: true,
            in_check: true,
            history_score: 100_000,
            parent_idx: None,
        });
        let b = graph.add_node(GtpNode {
            depth: 0,
            eval_margin: -30_000,
            is_capture: false,
            in_check: false,
            history_score: -100_000,
            parent_idx: Some(a),
        });
        let importance = GtpModel::message_passing(&graph);
        assert!(importance[a] <= 100);
        assert!(importance[b] <= 100);
        assert_eq!(importance[MAX_GTP_NODES - 1], 50);
    }

    #[test]
    fn connected_nodes_pull_toward_each_other() {
        let mut graph = GtpTreeGraph::new();
        let a = graph.add_node(GtpNode {
            depth: 4,
            eval_margin: 2000,
            history_score: 8000,
            ..GtpNode::default()
        });
        let _ = graph.add_node(GtpNode {
            depth: 4,
            eval_margin: -2000,
            history_score: -8000,
            parent_idx: Some(a),
            ..GtpNode::default()
        });
        let importance = GtpModel::message_passing(&graph);
        let solo_a = GtpPruner::node_score(&graph, a);
        assert!((importance[a] as i32 - solo_a).abs() <= 34);
    }
}
