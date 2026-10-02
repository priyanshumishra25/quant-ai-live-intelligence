import { api, AuthError, clearCredentials, setCredentials, storedUsername } from "./api.js";
import { lineChart } from "./chart.js";
const $ = (id) => document.getElementById(id);
const money = (n) => `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const pct = (n, d = 2) => `${n >= 0 ? "+" : ""}${n.toFixed(d)}%`;
const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[ch] || ch));
const safeUrl = (value) => { try {
    const u = new URL(value);
    return ["http:", "https:"].includes(u.protocol) ? u.href : "#";
}
catch {
    return "#";
} };
const sentimentClass = (v) => v > 0.12 ? "positive" : v < -0.12 ? "negative" : "neutral-text";
const signalClass = (value) => value.includes("BULLISH") ? "positive" : value.includes("BEARISH") ? "negative" : "";
const humanSignal = (value) => value.replaceAll("_", " ").toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
let currentTicker = "AAPL";
let currentHorizon = "5d";
let liveHorizon = "5d";
let latestPrediction = null;
let liveConfigured = false;
let redditConfigured = false;
let appDataReady = false;
let currentHealth = null;
let pendingView = "liveView";
function status(kind, text) {
    $("status").className = `status ${kind}`;
    $("statusText").textContent = text;
}
function metric(label, value, sub = "") {
    return `<dl class="metric-row"><dt>${esc(label)}</dt><dd>${esc(value)}</dd>${sub ? `<small>${esc(sub)}</small>` : ""}</dl>`;
}
function modelLabel(id) {
    return { ridge: "Ridge ML", momentum: "Momentum", mean_reversion: "Mean reversion" }[id] || id;
}
function setProofFromPrediction(p) {
    const proofPrice = document.getElementById("proofPrice");
    if (!proofPrice)
        return;
    proofPrice.textContent = money(p.current_price);
    $("proofReturn").textContent = pct(p.predicted_return_pct);
    $("proofSignal").textContent = humanSignal(p.signal);
    $("proofTarget").textContent = money(p.predicted_price);
    $("proofInterval").textContent = `${money(p.lower_bound)} to ${money(p.upper_bound)}`;
    $("proofRows").textContent = String(p.model_metadata.n_samples);
}
function scoreRows(entries) {
    return entries.map(([name, value, kind]) => `
    <div class="score-row">
      <span>${esc(name)}</span>
      <div class="score-bar" aria-hidden="true"><span class="${kind}" style="width:${Math.max(0, Math.min(100, value * 100))}%"></span></div>
      <strong>${(value * 100).toFixed(0)}</strong>
    </div>`).join("");
}
function featureRows(entries, percentageScale = 100) {
    const max = Math.max(...entries.map(([, v]) => Math.abs(v)), 0.000001);
    return entries.map(([name, value]) => `
    <div class="feature-row">
      <span>${esc(name.replaceAll("_", " "))}</span>
      <div class="feature-bar" aria-hidden="true"><span style="width:${Math.max(2, Math.min(100, Math.abs(value) / max * 100))}%"></span></div>
      <strong>${(value * percentageScale).toFixed(1)}${percentageScale === 100 ? "%" : ""}</strong>
    </div>`).join("");
}
function renderPrediction(p) {
    latestPrediction = p;
    setProofFromPrediction(p);
    $("heroTicker").textContent = p.ticker;
    $("heroPrice").textContent = money(p.current_price);
    $("heroReturn").textContent = pct(p.predicted_return_pct);
    $("heroReturn").className = p.predicted_return_pct >= 0 ? "positive" : "negative";
    $("heroSignal").textContent = humanSignal(p.signal);
    $("heroSignal").className = `signal-label ${signalClass(p.signal)}`;
    $("heroMeta").textContent = `${p.horizon} horizon · ${p.model_metadata.model_type} · ${p.latency_ms.toFixed(2)} ms${p.cache_hit ? " · cached" : ""}`;
    $("predictionMetrics").innerHTML = [
        metric("Forecast price", money(p.predicted_price)),
        metric("95% interval", `${money(p.lower_bound)} to ${money(p.upper_bound)}`),
        metric("Signal strength", `${(p.signal_strength * 100).toFixed(1)}`, "heuristic score out of 100"),
        metric("Volatility", `${p.predicted_volatility_pct.toFixed(1)}%`),
        metric("Risk score", `${p.risk_score.toFixed(1)} / 10`),
        metric("Training rows", String(p.model_metadata.n_samples)),
    ].join("");
    const features = Object.entries(p.feature_importance).sort((a, b) => b[1] - a[1]).slice(0, 6);
    $("featureList").innerHTML = featureRows(features);
    $("explanation").textContent = p.explanation;
    $("probabilities").innerHTML = scoreRows([
        ["Bullish", p.bullish_score, "positive"], ["Neutral", p.neutral_score, "neutral"], ["Bearish", p.bearish_score, "negative"],
    ]);
}
async function refreshExplorer() {
    status("busy", "Running deterministic fixture inference");
    const [pred, hist, overview] = await Promise.all([api.prediction(currentTicker, currentHorizon), api.history(currentTicker), api.overview()]);
    renderPrediction(pred);
    lineChart($("priceChart"), [{ values: hist.rows.map(r => r.close), className: "primary" }]);
    $("marketMetrics").innerHTML = [
        metric("Fear / greed", overview.fear_greed_index.toFixed(1)),
        metric("VIX proxy", `${overview.vix_proxy.toFixed(1)}%`),
        ...Object.entries(overview.sector_performance).map(([k, v]) => metric(k, pct(v))),
    ].join("");
    $("movers").innerHTML = overview.top_movers.map(m => `<div class="driver-row"><strong>${esc(m.ticker)}</strong><span class="${m.return_pct >= 0 ? "positive" : "negative"}">${pct(m.return_pct)}</span></div>`).join("");
    status("ok", liveConfigured ? "Live intelligence ready" : "Offline research mode");
}
function sourceSummaryRow(name, score, volume, engagement, detail) {
    const width = Math.min(100, Math.abs(score) * 100);
    return `<div class="source-row">
    <strong class="${sentimentClass(score)}">${esc(name)} ${score >= 0 ? "+" : ""}${score.toFixed(3)}</strong>
    <div class="source-meter" aria-hidden="true"><span class="${score >= 0 ? "positive-bg" : "negative-bg"}" style="width:${width}%"></span></div>
    <p>${esc(detail)}</p>
    <small>${volume} items · ${engagement.toLocaleString()} engagement</small>
  </div>`;
}
function renderLive(result) {
    const p = result.prediction, m = result.market, s = result.sentiment;
    $("liveCompany").textContent = `${result.company.name} · ${result.ticker}`;
    $("livePrice").textContent = money(m.current_price);
    $("liveSignal").textContent = humanSignal(p.signal);
    $("liveSignal").className = `signal-label ${signalClass(p.signal)}`;
    $("livePredReturn").textContent = pct(p.predicted_return_pct);
    $("livePredReturn").className = p.predicted_return_pct >= 0 ? "positive" : "negative";
    $("liveMeta").textContent = `${result.horizon} horizon · ${m.trend_regime} · generated in ${result.latency_ms.toFixed(0)} ms${result.cache_hit ? " · cached" : ""}`;
    $("livePredictionMetrics").classList.remove("empty-metrics");
    $("livePredictionMetrics").innerHTML = [
        metric("Forecast price", money(p.predicted_price)),
        metric("95% interval", `${money(p.lower_bound)} to ${money(p.upper_bound)}`),
        metric("Direction", humanSignal(p.signal), `${(p.signal_strength * 100).toFixed(1)} signal score`),
        metric("Source coverage", `${(p.source_coverage * 100).toFixed(0)}%`, "evidence-volume diagnostic"),
        metric("Holdout residual sigma", `${p.holdout_residual_volatility_pct.toFixed(2)} pp`, "interval basis"),
        metric("Validation reliability", result.model.validation.reliability, `≈${result.model.validation.effective_non_overlapping_observations} non-overlapping observations`),
    ].join("");
    lineChart($("livePriceChart"), [{ values: result.history.map(x => x.close), className: "primary" }]);
    $("liveTrendMetrics").innerHTML = [
        metric("1 day", pct(m.returns_pct["1d"] ?? 0)),
        metric("5 days", pct(m.returns_pct["5d"] ?? 0)),
        metric("20 days", pct(m.returns_pct["20d"] ?? 0)),
        metric("60 days", pct(m.returns_pct["60d"] ?? 0)),
        metric("Annualized volatility", `${m.annualised_volatility_pct.toFixed(1)}%`),
        metric("52-week range", `${money(m.low_52w)} to ${money(m.high_52w)}`),
    ].join("");
    $("liveSourceSummary").innerHTML =
        sourceSummaryRow("News", s.news.score, s.news.volume, s.news.engagement, "Market news and ticker-level provider sentiment") +
            sourceSummaryRow("Reddit", s.reddit.score, s.reddit.volume, s.reddit.engagement, s.reddit_configured ? s.reddit_scope : "Not configured; no Reddit contribution applied");
    $("liveProbabilities").innerHTML = scoreRows([
        ["Bullish", p.bullish_score, "positive"], ["Neutral", p.neutral_score, "neutral"], ["Bearish", p.bearish_score, "negative"],
    ]);
    $("contributionMix").innerHTML = featureRows([
        ["technical", p.contribution_mix.technical || 0],
        ["news", p.contribution_mix.news || 0],
        ["reddit", p.contribution_mix.reddit || 0],
    ]);
    $("liveDrivers").innerHTML = p.top_drivers.map(d => `<div class="driver-row"><strong>${esc(d.feature.replaceAll("_", " "))}</strong><span class="${d.direction === 'up' ? "positive" : "negative"}">${d.direction === 'up' ? "Up" : "Down"} ${Math.abs(d.effect).toFixed(4)}</span></div>`).join("");
    $("liveModelMetrics").innerHTML = [
        metric("Training rows", String(result.model.fit_rows_before_holdout), `${result.model.train_start} to training cutoff`),
        metric("Purged rows", String(result.model.validation.purge_rows), "horizon-length separation"),
        metric("Holdout rows", String(result.model.validation.rows), `${result.model.validation_start} to ${result.model.validation_end}`),
        metric("Effective observations", `≈${result.model.validation.effective_non_overlapping_observations}`, "approximate non-overlapping targets"),
        metric("Holdout MAE", `${result.model.validation.mae_pct_points.toFixed(2)} pp`),
        metric("Directional accuracy", `${(result.model.validation.directional_accuracy * 100).toFixed(1)}%`, "diagnostic only"),
        metric("Holdout correlation", result.model.validation.correlation.toFixed(3)),
        metric("Selected Ridge alpha", result.model.ridge_alpha.toFixed(3), "training-only generalized cross-validation"),
        metric("Reliability", result.model.validation.reliability, "sample-size diagnostic"),
    ].join("");
    $("liveMethodology").textContent = result.model.methodology;
    const sourceErrors = Object.entries(s.source_errors);
    $("sourceErrors").innerHTML = sourceErrors.length
        ? `<strong>Provider degradation:</strong> ${sourceErrors.map(([k, v]) => `${esc(k)}: ${esc(v)}`).join(" · ")}`
        : "All configured providers completed successfully.";
    const overlay = result.model.reddit_overlay;
    if (overlay && s.reddit_configured) {
        const effectPct = overlay.current_effect * 100;
        $("redditOverlayDisclosure").hidden = false;
        $("redditOverlayDisclosure").textContent = `Reddit inference adjustment: ${effectPct >= 0 ? "+" : ""}${effectPct.toFixed(3)} percentage points. Maximum bound: ${overlay.max_residual_sigma_fraction.toFixed(2)} residual sigma. This is a fixed design guardrail, not an empirically optimized coefficient.`;
    }
    else {
        $("redditOverlayDisclosure").hidden = true;
    }
    $("evidenceList").innerHTML = result.evidence.items.length ? result.evidence.items.map(item => {
        const origin = item.kind === "news" ? item.source : `r/${item.subreddit || "reddit"}`;
        const when = new Date(item.published_at).toLocaleString();
        const assigned = item.market_session ? item.market_session : "Not exposed";
        const usage = item.model_usage || (item.kind === "news" ? "Historical feature input" : "Inference only");
        return `<article class="evidence-row">
      <div class="evidence-meta"><strong>${esc(origin)}</strong><span>${esc(item.kind.replaceAll('_', ' '))}</span><span>${esc(when)}</span></div>
      <div class="evidence-content"><a href="${esc(safeUrl(item.url))}" target="_blank" rel="noopener noreferrer">${esc(item.title)}</a><p>${esc(item.snippet)}</p></div>
      <div class="evidence-stats"><span>Sentiment ${item.sentiment >= 0 ? "+" : ""}${item.sentiment.toFixed(3)}</span><span>Relevance ${item.relevance.toFixed(2)}</span><span>${item.engagement ? `${item.engagement} engagement` : "No engagement value"}</span><span>Market session ${esc(assigned)}</span><span>${esc(usage)}</span></div>
    </article>`;
    }).join("") : `<p class="empty-copy">No source items returned. The technical model can still run, but alternative-data contribution is zero.</p>`;
    $("liveDisclaimer").textContent = result.disclaimer;
}
function showLogin(message = "Authentication is required to use the application API.") {
    const dialog = $("loginDialog");
    $("loginError").hidden = true;
    $("loginHint").textContent = currentHealth?.environment === "development"
        ? "Development mode detected. Use AUTH_USERNAME and AUTH_PASSWORD from your local .env configuration."
        : message;
    const user = $("loginUser");
    if (!user.value)
        user.value = storedUsername();
    if (!dialog.open)
        dialog.showModal();
}
async function ensureAppData(promptForAuth = true) {
    if (appDataReady)
        return true;
    try {
        const models = await api.models();
        const ticker = $("ticker");
        ticker.innerHTML = models.tickers.map(t => `<option>${esc(t)}</option>`).join("");
        currentTicker = ticker.value || "AAPL";
        appDataReady = true;
        await refreshExplorer();
        return true;
    }
    catch (e) {
        if (e instanceof AuthError) {
            if (promptForAuth) {
                showLogin();
                return false;
            }
            throw e;
        }
        status("error", e.message);
        return false;
    }
}
async function runLiveAnalysis() {
    const query = $("liveQuery").value.trim();
    if (!query)
        return;
    const btn = $("liveAnalyzeButton");
    btn.disabled = true;
    btn.textContent = "Analyzing";
    $("liveError").hidden = true;
    $("analysisProgress").hidden = false;
    status("busy", `Running live analysis for ${query}`);
    try {
        const result = await api.liveAnalyze(query, liveHorizon, $("includeComments").checked);
        renderLive(result);
        status("ok", `${result.ticker} live intelligence · ${result.evidence.news_count + result.evidence.reddit_post_count + result.evidence.reddit_comment_count} evidence items`);
    }
    catch (e) {
        if (e instanceof AuthError) {
            showLogin("Sign in before running live analysis.");
        }
        else {
            const message = e.message;
            $("liveError").innerHTML = `<strong>Live analysis could not complete.</strong><p>${esc(message)}</p><p>Cached analyses and the Offline Lab may remain available.</p>`;
            $("liveError").hidden = false;
            status("error", message);
        }
    }
    finally {
        $("analysisProgress").hidden = true;
        btn.disabled = false;
        btn.textContent = "Analyze stock";
    }
}
async function runBacktest() {
    const btn = $("backtestButton");
    btn.disabled = true;
    btn.textContent = "Running";
    try {
        const r = await api.backtest(currentTicker, currentHorizon);
        const keys = [["total_return_pct", "Return"], ["cagr_pct", "CAGR"], ["sharpe_ratio", "Sharpe"], ["max_drawdown_pct", "Maximum drawdown"], ["win_rate_pct", "Win rate"]];
        $("backtestMetrics").innerHTML = keys.map(([k, label]) => metric(label, k.includes("pct") ? `${Number(r.stats[k] || 0).toFixed(2)}%` : Number(r.stats[k] || 0).toFixed(3))).join("");
        lineChart($("equityChart"), [{ values: r.equity_curve.map(x => x.equity), className: "primary" }]);
        $("backtestWarning").textContent = [...r.warnings, "Historical simulation does not imply future performance."].filter(Boolean).join(" ");
        $("exportButton").disabled = false;
    }
    catch (e) {
        if (e instanceof AuthError)
            showLogin();
        else
            status("error", e.message);
    }
    finally {
        btn.disabled = false;
        btn.textContent = "Run walk-forward";
    }
}
async function runExperiment() {
    const selected = Array.from(document.querySelectorAll('input[name="experimentModel"]:checked')).map(x => x.value);
    if (!selected.length)
        return;
    const commission = Number($("commission").value);
    const slippage = Number($("slippage").value);
    const btn = $("experimentButton");
    btn.disabled = true;
    btn.textContent = "Evaluating";
    try {
        const r = await api.experiment(currentTicker, currentHorizon, selected, commission, slippage);
        $("experimentTable").innerHTML = `<table><thead><tr><th>Model</th><th>Type</th><th>Return</th><th>Sharpe</th><th>Maximum drawdown</th><th>Win rate</th><th>Trades</th></tr></thead><tbody>${r.comparisons.map(c => `<tr><td><strong>${esc(modelLabel(c.model))}</strong></td><td><span class="badge">${esc(c.kind.replaceAll('_', ' '))}</span></td><td>${Number(c.stats.total_return_pct || 0).toFixed(2)}%</td><td>${Number(c.stats.sharpe_ratio || 0).toFixed(3)}</td><td>${Number(c.stats.max_drawdown_pct || 0).toFixed(2)}%</td><td>${Number(c.stats.win_rate_pct || 0).toFixed(1)}%</td><td>${c.trades}</td></tr>`).join("")}</tbody></table>`;
        lineChart($("experimentChart"), r.comparisons.map((c, i) => ({ values: c.equity_curve.map(x => x.equity), className: `series-${i}` })));
        $("experimentLegend").innerHTML = r.comparisons.map((c, i) => `<span><i class="series-${i}"></i>${esc(modelLabel(c.model))}</span>`).join("");
        $("experimentNote").textContent = `${r.methodology}. Commission ${r.costs.commission_bps} bps plus slippage ${r.costs.slippage_bps} bps. ${r.warning || ""}`;
    }
    catch (e) {
        if (e instanceof AuthError)
            showLogin();
        else
            status("error", e.message);
    }
    finally {
        btn.disabled = false;
        btn.textContent = "Run comparison";
    }
}
async function refreshSystem() {
    try {
        const s = await api.system();
        $("systemMetrics").innerHTML = [
            metric("API version", s.version),
            metric("Uptime", `${Math.round(s.uptime_seconds)} s`),
            metric("Requests", String(s.requests_total)),
            metric("Average latency", `${s.average_http_latency_ms.toFixed(2)} ms`),
            metric("Prediction cache hit", `${(s.prediction_cache_hit_rate * 100).toFixed(1)}%`),
            metric("Models loaded", String(s.models_loaded)),
            metric("Alpha Vantage", s.live_intelligence_configured ? "Ready" : "Not configured"),
            metric("Reddit", s.reddit_configured ? "Ready" : "Not configured"),
        ].join("");
        $("architecture").innerHTML = s.architecture.map((x, i) => `<li><span>${String(i + 1).padStart(2, "0")}</span><strong>${esc(x)}</strong></li>`).join("");
        $("capabilities").innerHTML = s.capabilities.map((x, i) => `<div class="capability-row"><span>${String(i + 1).padStart(2, "0")}</span><strong>${esc(x)}</strong></div>`).join("");
    }
    catch (e) {
        if (e instanceof AuthError)
            showLogin();
        else
            status("error", e.message);
    }
}
async function exportCsv() {
    try {
        const { url, token } = await api.equityCsvUrl(currentTicker, currentHorizon);
        const res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
        const blob = await res.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${currentTicker}_${currentHorizon}_equity.csv`;
        a.click();
        URL.revokeObjectURL(a.href);
    }
    catch (e) {
        if (e instanceof AuthError)
            showLogin();
        else
            status("error", e.message);
    }
}
function switchAppView(viewId) {
    document.querySelectorAll(".view").forEach(v => v.hidden = true);
    const target = document.getElementById(viewId);
    if (target)
        target.hidden = false;
    document.querySelectorAll("[data-view]").forEach(x => x.classList.toggle("active", x.dataset.view === viewId));
    pendingView = viewId;
    if (viewId === "systemView")
        void refreshSystem();
}
function showPublicPage(pageId) {
    $("appShell").hidden = true;
    $("publicShell").hidden = false;
    $("siteHeader").hidden = false;
    $("siteFooter").hidden = false;
    document.querySelectorAll(".public-view").forEach(page => page.hidden = page.id !== pageId);
    document.querySelectorAll("[data-public-view]").forEach(btn => btn.classList.toggle("active", btn.dataset.publicView === pageId));
    window.scrollTo({ top: 0, behavior: "auto" });
}
async function openApp(viewId = "liveView") {
    $("publicShell").hidden = true;
    $("siteHeader").hidden = true;
    $("siteFooter").hidden = true;
    $("appShell").hidden = false;
    switchAppView(viewId);
    window.scrollTo({ top: 0, behavior: "auto" });
    await ensureAppData();
}
function bindNavigation() {
    document.querySelectorAll("[data-public-view]").forEach(btn => btn.addEventListener("click", () => showPublicPage(String(btn.dataset.publicView))));
    document.querySelectorAll("[data-open-app]").forEach(el => el.addEventListener("click", () => void openApp(el.dataset.appView || "liveView")));
    document.querySelectorAll("[data-close-app]").forEach(btn => btn.addEventListener("click", () => showPublicPage("productPage")));
    document.querySelectorAll("[data-view]").forEach(btn => btn.addEventListener("click", () => switchAppView(String(btn.dataset.view))));
    document.querySelectorAll("[data-reference]").forEach(btn => btn.addEventListener("click", () => showPublicPage(String(btn.dataset.reference))));
}
function bindControls() {
    $("ticker").addEventListener("change", async (e) => { currentTicker = e.target.value; await refreshExplorer(); });
    $("horizon").addEventListener("change", async (e) => { currentHorizon = e.target.value; await refreshExplorer(); });
    $("liveHorizon").addEventListener("change", e => { liveHorizon = e.target.value; });
    $("refreshButton").addEventListener("click", () => void refreshExplorer());
    $("backtestButton").addEventListener("click", () => void runBacktest());
    $("experimentButton").addEventListener("click", () => void runExperiment());
    $("exportButton").addEventListener("click", () => void exportCsv());
    $("liveForm").addEventListener("submit", async (e) => { e.preventDefault(); await runLiveAnalysis(); });
    $("signOutButton").addEventListener("click", () => { clearCredentials(); appDataReady = false; status("ok", "Signed out"); showPublicPage("productPage"); });
    $("loginCancel").addEventListener("click", () => $("loginDialog").close());
    $("loginForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        const user = $("loginUser").value;
        const pass = $("loginPass").value;
        setCredentials(user, pass);
        try {
            appDataReady = false;
            const ok = await ensureAppData(false);
            if (ok) {
                $("loginError").hidden = true;
                $("loginPass").value = "";
                $("loginDialog").close();
                switchAppView(pendingView);
            }
        }
        catch (err) {
            $("loginError").textContent = err.message;
            $("loginError").hidden = false;
        }
    });
}
async function boot() {
    bindNavigation();
    bindControls();
    try {
        currentHealth = await api.health();
        liveConfigured = currentHealth.live_intelligence_configured;
        redditConfigured = currentHealth.reddit_configured;
        $("releaseMeta").textContent = `v${currentHealth.version} · ${currentHealth.model_count} model artifacts · ${currentHealth.cache_backend} cache`;
        $("sidebarVersion").textContent = `v${currentHealth.version}`;
        $("providerState").textContent = liveConfigured ? `Live provider configured${redditConfigured ? " · Reddit configured" : " · Reddit optional"}` : "Live provider not configured · Offline Lab available";
        $("alphaState").textContent = liveConfigured ? "Ready" : "Not configured";
        $("redditState").textContent = redditConfigured ? "Ready" : "Not configured";
        status("ok", liveConfigured ? "Live intelligence ready. No provider request has been made." : "Offline research mode. Live provider not configured.");
    }
    catch (e) {
        status("error", `Runtime unavailable: ${e.message}`);
        $("alphaState").textContent = "Unavailable";
        $("redditState").textContent = "Unavailable";
    }
}
void boot();
