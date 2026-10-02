export type Horizon = "1d" | "5d" | "20d";
export type ExperimentModel = "ridge" | "momentum" | "mean_reversion";

export interface Health {
  status: string;
  version: string;
  model_count: number;
  cache_backend: string;
  data_mode: string;
  live_intelligence_configured: boolean;
  reddit_configured: boolean;
  environment?: "development" | "test" | "staging" | "production";
}

export interface ModelRegistry {
  artifacts: string[];
  tickers: string[];
  horizons: Horizon[];
  data_provider: string;
  live_intelligence_configured?: boolean;
  reddit_configured?: boolean;
}

export interface Prediction {
  ticker: string;
  data_timestamp: string;
  horizon: Horizon;
  current_price: number;
  predicted_price: number;
  predicted_return_pct: number;
  lower_bound: number;
  upper_bound: number;
  signal: string;
  signal_strength: number;
  signal_score: number;
  bullish_score: number;
  neutral_score: number;
  bearish_score: number;
  predicted_volatility_pct: number;
  risk_score: number;
  var_95: number;
  explanation: string;
  feature_importance: Record<string, number>;
  model_metadata: { model_type: string; train_start: string; train_end: string; n_samples: number };
  data_mode: string;
  latency_ms: number;
  cache_hit: boolean;
}

export interface History {
  ticker: string;
  rows: Array<{date:string; open:number; high:number; low:number; close:number; volume:number}>;
}

export interface Overview {
  fear_greed_index: number;
  vix_proxy: number;
  top_movers: Array<{ticker:string; return_pct:number}>;
  sector_performance: Record<string, number>;
  data_mode: string;
}

export interface Backtest {
  ticker: string;
  horizon: Horizon;
  stats: Record<string, number>;
  equity_curve: Array<{date:string; equity:number}>;
  warnings: string[];
}

export interface ExperimentComparison {
  model: ExperimentModel;
  kind: "machine_learning" | "baseline";
  stats: Record<string, number>;
  trades: number;
  equity_curve: Array<{date:string; equity:number}>;
}

export interface ExperimentResult {
  ticker: string;
  horizon: Horizon;
  costs: {commission_bps:number; slippage_bps:number};
  methodology: string;
  comparisons: ExperimentComparison[];
  warning: string | null;
}

export interface SystemOverview {
  service: string;
  version: string;
  uptime_seconds: number;
  requests_total: number;
  average_http_latency_ms: number;
  prediction_requests: number;
  prediction_cache_hit_rate: number;
  cache_backend: string;
  models_loaded: number;
  data_provider: string;
  live_intelligence_configured: boolean;
  reddit_configured: boolean;
  architecture: string[];
  capabilities: string[];
}

export interface StockCandidate {
  symbol: string;
  name: string;
  asset_type: string;
  region: string;
  currency: string;
  match_score: number;
}

export interface LiveSearchResult {
  query: string;
  matches: StockCandidate[];
  count: number;
}

export interface SourceSummary {
  score: number;
  volume: number;
  engagement: number;
  positive: number;
  negative: number;
  neutral: number;
}

export interface EvidenceItem {
  source: string;
  kind: "news" | "reddit_post" | "reddit_comment";
  title: string;
  snippet: string;
  published_at: string;
  url: string;
  sentiment: number;
  relevance: number;
  engagement: number;
  subreddit: string;
  market_session?: string | null;
  session_assignment_reason?: string | null;
  model_usage?: string;
}

export interface LiveIntelligence {
  query: string;
  ticker: string;
  company: StockCandidate;
  timestamp: string;
  horizon: Horizon;
  latency_ms: number;
  cache_hit: boolean;
  market: {
    current_price: number;
    returns_pct: Record<string, number>;
    annualised_volatility_pct: number;
    high_52w: number;
    low_52w: number;
    distance_to_20d_ma_pct: number;
    distance_to_50d_ma_pct: number;
    trend_regime: string;
    history_rows: number;
    history_start: string;
    history_end: string;
  };
  sentiment: {
    news: SourceSummary;
    reddit: SourceSummary;
    combined_score: number;
    reddit_configured: boolean;
    reddit_scope: string;
    source_errors: Record<string,string>;
  };
  prediction: {
    signal: string;
    signal_strength: number;
    signal_score: number;
    predicted_return_pct: number;
    predicted_price: number;
    lower_bound: number;
    upper_bound: number;
    bullish_score: number;
    neutral_score: number;
    bearish_score: number;
    holdout_residual_volatility_pct: number;
    source_coverage: number;
    contribution_mix: Record<"technical"|"news"|"reddit", number>;
    top_drivers: Array<{feature:string; effect:number; direction:"up"|"down"}>;
  };
  model: {
    type: string;
    feature_names: string[];
    technical_features: string[];
    alternative_data_features: string[];
    training_rows: number;
    fit_rows_before_holdout: number;
    final_refit_rows: number;
    train_start: string;
    train_end: string;
    validation_start: string;
    validation_end: string;
    ridge_alpha: number;
    ridge_alpha_selection: {
      method: string;
      candidate_alphas: number[];
      selected_gcv: number;
    };
    reddit_overlay?: {
      type: string;
      max_residual_sigma_fraction: number;
      selection: string;
      current_effect: number;
    };
    validation: {
      mae_pct_points: number;
      directional_accuracy: number;
      correlation: number;
      rows: number;
      purge_rows: number;
      effective_non_overlapping_observations: number;
      reliability: "LOW"|"MEDIUM"|"HIGH";
      residual_std_pct_points: number;
    };
    methodology: string;
  };
  evidence: {
    news_count: number;
    reddit_post_count: number;
    reddit_comment_count: number;
    items: EvidenceItem[];
  };
  history: Array<{date:string; close:number}>;
  providers: Record<string,string>;
  disclaimer: string;
}
