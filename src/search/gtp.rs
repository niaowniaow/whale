pub const MAX_GTP_NODES: usize = 16;
pub const GTP_HIDDEN_DIM: usize = 32;
pub const GTP_INPUT_DIM: usize = 6;

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

pub struct GtpModel;

impl GtpModel {
    pub fn message_passing(graph: &GtpTreeGraph) -> [u8; MAX_GTP_NODES] {
        let mut importance = [50u8; MAX_GTP_NODES];
        if graph.count == 0 {
            return importance;
        }

        let mut h0 = [[0i32; GTP_INPUT_DIM]; MAX_GTP_NODES];
        for (i, node) in graph.nodes[..graph.count].iter().enumerate() {
            h0[i][0] = (node.depth as i32).clamp(0, 32);
            h0[i][1] = (node.eval_margin / 32).clamp(-128, 128) as i32;
            h0[i][2] = if node.is_capture { 64 } else { 0 };
            h0[i][3] = if node.in_check { 64 } else { 0 };
            h0[i][4] = (node.history_score / 512).clamp(-64, 64);
            h0[i][5] = (graph.adjacency[i].count_ones() as i32) * 16;
        }

        let mut h1 = [[0i16; GTP_HIDDEN_DIM]; MAX_GTP_NODES];
        for i in 0..graph.count {
            for j in 0..GTP_HIDDEN_DIM {
                let mut self_sum = GTP_BIAS_1[j];
                for k in 0..GTP_INPUT_DIM {
                    self_sum += h0[i][k] * GTP_WEIGHTS_SELF[j][k];
                }

                let mut neighbor_sum = 0i32;
                let adj = graph.adjacency[i];
                let mut n_idx = 0usize;
                while n_idx < graph.count {
                    if (adj & (1 << n_idx)) != 0 {
                        for k in 0..GTP_INPUT_DIM {
                            neighbor_sum += h0[n_idx][k] * GTP_WEIGHTS_NEIGHBOR[j][k];
                        }
                    }
                    n_idx += 1;
                }

                let activated = ((self_sum + neighbor_sum / 2) / 16).clamp(0, 255);
                h1[i][j] = activated as i16;
            }
        }

        let mut h2 = [[0i16; GTP_HIDDEN_DIM]; MAX_GTP_NODES];
        for i in 0..graph.count {
            for j in 0..GTP_HIDDEN_DIM {
                let mut agg = GTP_BIAS_2[j];
                let adj = graph.adjacency[i];
                for (k, &val) in h1[i].iter().enumerate() {
                    agg += (val as i32) * GTP_WEIGHTS_L2[j % 8][k % 8];
                }
                for (n_idx, neighbor) in h1.iter().enumerate().take(graph.count) {
                    if (adj & (1 << n_idx)) != 0 {
                        agg += (neighbor[j] as i32) * 2;
                    }
                }
                h2[i][j] = (agg / 32).clamp(0, 255) as i16;
            }
        }

        for i in 0..graph.count {
            let mut score = 512i32;
            for j in 0..GTP_HIDDEN_DIM {
                score += (h2[i][j] as i32) * GTP_READOUT[j % 8];
            }
            let imp = (score / 32).clamp(0, 100);
            importance[i] = imp as u8;
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

    pub fn gnn_loss(predicted: &[f32], target: &[f32]) -> f32 {
        if predicted.is_empty() || predicted.len() != target.len() {
            return 0.0;
        }
        let mut mse = 0.0f32;
        for (p, t) in predicted.iter().zip(target.iter()) {
            let diff = p - t;
            mse += diff * diff;
        }
        mse / (predicted.len() as f32)
    }
}

const GTP_WEIGHTS_SELF: [[i32; GTP_INPUT_DIM]; GTP_HIDDEN_DIM] = [
    [4, 3, 5, 6, 2, 1],
    [2, 4, 3, 5, 3, 2],
    [5, 2, 6, 4, 1, 3],
    [3, 5, 2, 6, 4, 1],
    [4, 4, 5, 3, 2, 2],
    [1, 3, 4, 5, 5, 4],
    [6, 2, 3, 4, 1, 2],
    [2, 5, 4, 3, 3, 3],
    [3, 1, 5, 6, 2, 4],
    [4, 6, 2, 3, 4, 1],
    [5, 3, 4, 2, 1, 5],
    [2, 4, 6, 5, 3, 2],
    [3, 2, 3, 4, 5, 1],
    [6, 5, 1, 2, 2, 3],
    [1, 4, 5, 6, 4, 2],
    [4, 2, 4, 3, 1, 4],
    [2, 6, 3, 5, 2, 1],
    [5, 1, 6, 4, 3, 2],
    [3, 4, 2, 3, 5, 4],
    [4, 3, 5, 2, 1, 3],
    [1, 5, 4, 6, 2, 2],
    [6, 2, 3, 1, 4, 5],
    [2, 4, 1, 5, 3, 1],
    [3, 3, 5, 4, 2, 4],
    [4, 1, 2, 6, 5, 2],
    [5, 6, 4, 2, 1, 3],
    [2, 3, 5, 4, 3, 1],
    [3, 5, 1, 3, 4, 2],
    [4, 2, 6, 5, 2, 5],
    [1, 4, 3, 2, 5, 1],
    [6, 1, 4, 5, 1, 3],
    [2, 5, 2, 4, 3, 2],
];

const GTP_WEIGHTS_NEIGHBOR: [[i32; GTP_INPUT_DIM]; GTP_HIDDEN_DIM] = [
    [2, 1, 3, 4, 1, 2],
    [1, 3, 2, 3, 2, 1],
    [3, 2, 4, 2, 1, 2],
    [2, 4, 1, 4, 3, 1],
    [3, 1, 3, 2, 2, 1],
    [1, 2, 3, 4, 4, 2],
    [4, 1, 2, 3, 1, 1],
    [1, 3, 2, 2, 2, 2],
    [2, 1, 4, 4, 1, 3],
    [3, 4, 1, 2, 3, 1],
    [4, 2, 3, 1, 1, 4],
    [1, 3, 4, 4, 2, 1],
    [2, 1, 2, 3, 4, 1],
    [4, 3, 1, 1, 1, 2],
    [1, 3, 4, 4, 3, 1],
    [3, 1, 3, 2, 1, 3],
    [1, 4, 2, 4, 1, 1],
    [4, 1, 4, 3, 2, 1],
    [2, 3, 1, 2, 4, 3],
    [3, 2, 4, 1, 1, 2],
    [1, 4, 3, 4, 1, 1],
    [4, 1, 2, 1, 3, 4],
    [1, 3, 1, 4, 2, 1],
    [2, 2, 4, 3, 1, 3],
    [3, 1, 1, 4, 4, 1],
    [4, 4, 3, 1, 1, 2],
    [1, 2, 4, 3, 2, 1],
    [2, 4, 1, 2, 3, 1],
    [3, 1, 4, 4, 1, 4],
    [1, 3, 2, 1, 4, 1],
    [4, 1, 3, 4, 1, 2],
    [1, 4, 1, 3, 2, 1],
];

const GTP_BIAS_1: [i32; GTP_HIDDEN_DIM] = [
    8, 12, 10, 14, 6, 9, 11, 13, 7, 15, 8, 12, 10, 11, 9, 14, 6, 13, 8, 10, 12, 7, 14, 9, 11, 8,
    13, 10, 7, 12, 9, 11,
];

const GTP_WEIGHTS_L2: [[i32; 8]; 8] = [
    [2, 3, 1, 2, 4, 1, 2, 3],
    [1, 2, 3, 1, 2, 4, 1, 2],
    [3, 1, 2, 4, 1, 2, 3, 1],
    [2, 4, 1, 2, 3, 1, 2, 4],
    [4, 1, 2, 3, 1, 2, 4, 1],
    [1, 3, 4, 1, 2, 3, 1, 2],
    [2, 1, 3, 4, 1, 2, 3, 1],
    [3, 2, 1, 2, 4, 1, 2, 3],
];

const GTP_BIAS_2: [i32; GTP_HIDDEN_DIM] = [
    4, 6, 5, 7, 3, 5, 6, 8, 4, 7, 5, 6, 4, 5, 6, 7, 3, 6, 4, 5, 7, 4, 6, 5, 6, 4, 7, 5, 3, 6, 5, 6,
];

const GTP_READOUT: [i32; 8] = [2, -1, 3, 1, -2, 2, 1, -1];

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
    fn test_gtp_gnn_loss() {
        let pred = [0.8f32, 0.4f32];
        let target = [1.0f32, 0.5f32];
        let loss = GtpModel::gnn_loss(&pred, &target);
        assert!((loss - 0.025f32).abs() < 1e-4);
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
    fn gnn_loss_rejects_bad_shapes() {
        assert_eq!(GtpModel::gnn_loss(&[], &[]), 0.0);
        assert_eq!(GtpModel::gnn_loss(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(GtpModel::gnn_loss(&[1.0, 2.0], &[1.0]), 0.0);
        assert_eq!(GtpModel::gnn_loss(&[2.0], &[2.0]), 0.0);
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
}
