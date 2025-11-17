// Shared types for gradient analysis

// Helper structures for gradient analysis
#[derive(Debug)]
pub(super) struct GradientStatistics {
    pub(super) total_norm: Option<f64>,
    pub(super) max_gradient: Option<f64>,
    pub(super) variance: Option<f64>,
    pub(super) sparsity: Option<f64>, // Fraction of near-zero gradients
    pub(super) outlier_count: Option<usize>,
}

#[derive(Debug)]
pub(super) struct GradientFlowInfo {
    pub(super) vanishing_layers: usize,
    pub(super) exploding_layers: usize,
    pub(super) flow_balance: Option<f64>,
}

// Enhanced gradient statistics computation using lawkit incremental patterns
pub(super) struct EnhancedGradientStats {
    pub(super) total_norm: Option<f64>,
    pub(super) max_gradient: Option<f64>,
    pub(super) variance: Option<f64>,
}
