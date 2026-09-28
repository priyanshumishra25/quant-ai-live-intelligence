import { api, AuthError, setCredentials } from "./api.js";
import { lineChart } from "./chart.js";
import type { ExperimentModel, ExperimentResult, Horizon, LiveIntelligence, Prediction } from "./types.js";

const $ = <T extends HTMLElement = HTMLElement>(id:string) => document.getElementById(id) as T;
const money = (n:number) => `$${n.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`;
const pct = (n:number,d=2) => `${n>=0?"+":""}${n.toFixed(d)}%`;
const esc = (value:unknown) => String(value ?? "").replace(/[&<>'"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[ch] || ch));
const safeUrl = (value:string) => { try { const u=new URL(value); return ["http:","https:"].includes(u.protocol)?u.href:"#"; } catch { return "#"; } };
const sentimentClass = (v:number) => v > 0.12 ? "positive" : v < -0.12 ? "negative" : "neutral-text";

let currentTicker = "AAPL";
let currentHorizon: Horizon = "5d";
let liveHorizon: Horizon = "5d";
let latestPrediction: Prediction | null = null;
let liveConfigured = false;
let redditConfigured = false;

function status(kind:string,text:string){ $("status").className=`status ${kind}`; $("statusText").textContent=text; }
function metric(label:string,value:string,sub=""){ return `<article class="metric"><small>${esc(label)}</small><b>${esc(value)}</b>${sub?`<span>${esc(sub)}</span>`:""}</article>`; }
function modelLabel(id:string){ return ({ridge:"Ridge ML",momentum:"Momentum",mean_reversion:"Mean reversion"} as Record<string,string>)[id] || id; }

function renderPrediction(p:Prediction) {
  latestPrediction = p;
  $("heroTicker").textContent = p.ticker;
  $("heroPrice").textContent = money(p.current_price);
  $("heroReturn").textContent = pct(p.predicted_return_pct);
  $("heroReturn").className = p.predicted_return_pct >= 0 ? "positive" : "negative";
  $("heroSignal").textContent = p.signal.replaceAll("_"," ");
  $("heroSignal").className = `signal ${p.signal.includes("BULLISH")?"positive":p.signal.includes("BEARISH")?"negative":""}`;
  $("heroMeta").textContent = `${p.horizon} horizon · ${p.model_metadata.model_type} · ${p.latency_ms.toFixed(2)} ms ${p.cache_hit?"· cache hit":""}`;
  $("predictionMetrics").innerHTML = [
    metric("Target",money(p.predicted_price)), metric("95% low",money(p.lower_bound)), metric("95% high",money(p.upper_bound)),
    metric("Signal strength",`${(p.signal_strength*100).toFixed(1)}%`,`heuristic score · not probability`), metric("Volatility",`${p.predicted_volatility_pct.toFixed(1)}%`), metric("Risk",`${p.risk_score.toFixed(1)}/10`),
  ].join("");
  const features = Object.entries(p.feature_importance).sort((a,b)=>b[1]-a[1]).slice(0,6);
  $("featureList").innerHTML = features.map(([name,val]) => `<div class="feature"><span>${esc(name)}</span><i><em style="width:${Math.min(100,val*260)}%"></em></i><b>${(val*100).toFixed(1)}%</b></div>`).join("");
  $("explanation").textContent = p.explanation;
  $("probabilities").innerHTML = [
    ["BULLISH",p.bullish_score,"positive"],["NEUTRAL",p.neutral_score,"neutral"],["BEARISH",p.bearish_score,"negative"],
  ].map(([name,v,cls])=>`<div><span>${name}</span><i><em class="${cls}" style="width:${Number(v)*100}%"></em></i><b>${(Number(v)*100).toFixed(0)}</b></div>`).join("");
}

async function refreshExplorer() {
  status("busy","Running fixture inference…");
  const [pred, hist, overview] = await Promise.all([api.prediction(currentTicker,currentHorizon),api.history(currentTicker),api.overview()]);
  renderPrediction(pred);
  lineChart($("priceChart") as unknown as SVGSVGElement,[{values:hist.rows.map(r=>r.close),className:"primary"}]);
  $("marketMetrics").innerHTML = metric("Fear / greed",overview.fear_greed_index.toFixed(1))+metric("VIX proxy",`${overview.vix_proxy.toFixed(1)}%`)+
    Object.entries(overview.sector_performance).map(([k,v])=>metric(k,pct(v))).join("");
  $("movers").innerHTML = overview.top_movers.map(m=>`<div class="row"><strong>${esc(m.ticker)}</strong><span class="${m.return_pct>=0?"positive":"negative"}">${pct(m.return_pct)}</span></div>`).join("");
  status("ok",liveConfigured?"live intelligence ready":"offline research mode");
}

function sourceSummaryCard(name:string, score:number, volume:number, engagement:number, detail:string) {
  const width = Math.min(100, Math.abs(score)*100);
  return `<article class="source-summary"><div><small>${esc(name)}</small><b class="${sentimentClass(score)}">${score>=0?"+":""}${score.toFixed(3)}</b></div><div class="sentiment-axis"><i class="${score>=0?"positive-bg":"negative-bg"}" style="width:${width}%;margin-left:${score>=0?"50%":`${50-width}%`}"></i></div><p>${volume} items · ${engagement.toLocaleString()} engagement</p><span>${esc(detail)}</span></article>`;
}

function renderLive(result:LiveIntelligence) {
  const p=result.prediction, m=result.market, s=result.sentiment;
  $("liveCompany").textContent = `${result.company.name} · ${result.ticker}`;
  $("livePrice").textContent = money(m.current_price);
  $("liveSignal").textContent = p.signal.replaceAll("_"," ");
  $("liveSignal").className=`signal ${p.signal.includes("BULLISH")?"positive":p.signal.includes("BEARISH")?"negative":""}`;
  $("livePredReturn").textContent = pct(p.predicted_return_pct);
  $("livePredReturn").className = p.predicted_return_pct>=0?"positive":"negative";
  $("liveMeta").textContent = `${result.horizon} horizon · ${m.trend_regime} · ${result.latency_ms.toFixed(0)} ms ${result.cache_hit?"· cached":""}`;
  $("livePredictionMetrics").innerHTML = [
    metric("Target",money(p.predicted_price)), metric("95% low",money(p.lower_bound)), metric("95% high",money(p.upper_bound)),
    metric("Signal strength",`${(p.signal_strength*100).toFixed(1)}%`,`heuristic score · not probability`),
    metric("Source coverage",`${(p.source_coverage*100).toFixed(0)}%`,`evidence-volume diagnostic`),
    metric("Holdout σ",`${p.holdout_residual_volatility_pct.toFixed(2)}%`,`purged holdout residuals`),
  ].join("");
  lineChart($("livePriceChart") as unknown as SVGSVGElement,[{values:result.history.map(x=>x.close),className:"primary"}]);

  $("liveTrendMetrics").innerHTML = [
    metric("1D",pct(m.returns_pct["1d"] ?? 0)), metric("5D",pct(m.returns_pct["5d"] ?? 0)), metric("20D",pct(m.returns_pct["20d"] ?? 0)),
    metric("60D",pct(m.returns_pct["60d"] ?? 0)), metric("Ann. vol",`${m.annualised_volatility_pct.toFixed(1)}%`), metric("52W range",`${money(m.low_52w)} – ${money(m.high_52w)}`),
  ].join("");
  $("liveSourceSummary").innerHTML =
    sourceSummaryCard("News",s.news.score,s.news.volume,s.news.engagement,"Alpha Vantage news + ticker sentiment")+
    sourceSummaryCard("Reddit",s.reddit.score,s.reddit.volume,s.reddit.engagement,s.reddit_configured?"matched posts + top-thread comments":"not configured")+
    sourceSummaryCard("Combined",s.combined_score,s.news.volume+s.reddit.volume,s.news.engagement+s.reddit.engagement,"65% news / 35% Reddit diagnostic");

  $("liveProbabilities").innerHTML = [
    ["BULLISH",p.bullish_score,"positive"],["NEUTRAL",p.neutral_score,"neutral"],["BEARISH",p.bearish_score,"negative"],
  ].map(([name,v,cls])=>`<div><span>${name}</span><i><em class="${cls}" style="width:${Number(v)*100}%"></em></i><b>${(Number(v)*100).toFixed(0)}</b></div>`).join("");

  $("contributionMix").innerHTML = (["technical","news","reddit"] as const).map(name=>{
    const value=p.contribution_mix[name] || 0;
    return `<div class="feature"><span>${esc(name)}</span><i><em style="width:${Math.min(100,value*100)}%"></em></i><b>${(value*100).toFixed(1)}%</b></div>`;
  }).join("");
  $("liveDrivers").innerHTML = p.top_drivers.map(d=>`<div class="row"><strong>${esc(d.feature)}</strong><span class="${d.direction==='up'?"positive":"negative"}">${d.direction==='up'?"↑":"↓"} ${Math.abs(d.effect).toFixed(4)}</span></div>`).join("");

  $("liveModelMetrics").innerHTML = [
    metric("Fit rows",String(result.model.fit_rows_before_holdout),`${result.model.train_start} → purged holdout`),
    metric("Holdout rows",String(result.model.validation.rows),`purge gap ${result.model.validation.purge_rows} rows`),
    metric("Effective obs",`≈${result.model.validation.effective_non_overlapping_observations}`,`${result.model.validation.reliability} validation reliability`),
    metric("Directional acc.",`${(result.model.validation.directional_accuracy*100).toFixed(1)}%`,`diagnostic only`),
    metric("Holdout MAE",`${result.model.validation.mae_pct_points.toFixed(2)} pp`),
    metric("Ridge α",result.model.ridge_alpha.toFixed(2),`training-only GCV`),
  ].join("");
  $("liveMethodology").textContent = result.model.methodology;

  const sourceErrors = Object.entries(s.source_errors);
  $("sourceErrors").innerHTML = sourceErrors.length ? `<strong>Provider degradation:</strong> ${sourceErrors.map(([k,v])=>`${esc(k)} — ${esc(v)}`).join(" · ")}` : "All configured providers completed successfully.";

  $("evidenceList").innerHTML = result.evidence.items.length ? result.evidence.items.map(item=>{
    const origin=item.kind==="news"?item.source:`r/${item.subreddit || "reddit"}`;
    const when=new Date(item.published_at).toLocaleString();
    return `<article class="evidence-card"><div class="evidence-top"><span class="badge ${item.kind==='news'?'ai':''}">${esc(item.kind.replaceAll('_',' '))}</span><span class="${sentimentClass(item.sentiment)}">${item.sentiment>=0?"+":""}${item.sentiment.toFixed(3)}</span></div><a href="${esc(safeUrl(item.url))}" target="_blank" rel="noopener noreferrer">${esc(item.title)}</a><p>${esc(item.snippet)}</p><footer><span>${esc(origin)}</span><span>${esc(when)}</span><span>${item.engagement?`${item.engagement} engagement`:""}</span></footer></article>`;
  }).join("") : `<div class="empty">No source items returned. The technical model can still run, but alternative-data contribution is zero.</div>`;
  $("liveDisclaimer").textContent=result.disclaimer;
}

async function runLiveAnalysis() {
  const query=($("liveQuery") as HTMLInputElement).value.trim();
  if(!query) return;
  const btn=$("liveAnalyzeButton") as HTMLButtonElement;
  btn.disabled=true; btn.textContent="Collecting evidence…";
  $("liveError").hidden=true;
  status("busy",`Resolving ${query} + collecting live evidence…`);
  try {
    const result=await api.liveAnalyze(query,liveHorizon,($("includeComments") as HTMLInputElement).checked);
    renderLive(result);
    status("ok",`${result.ticker} live intelligence · ${result.evidence.news_count+result.evidence.reddit_post_count+result.evidence.reddit_comment_count} evidence items`);
  } catch(e) {
    const message=(e as Error).message;
    $("liveError").textContent=message;
    $("liveError").hidden=false;
    status("error",message);
  } finally { btn.disabled=false; btn.textContent="Analyze stock"; }
}

async function runBacktest() {
  const btn = $("backtestButton") as HTMLButtonElement; btn.disabled=true; btn.textContent="Running…";
  try {
    const r = await api.backtest(currentTicker,currentHorizon);
    const keys:[string,string][]=[["total_return_pct","Return"],["cagr_pct","CAGR"],["sharpe_ratio","Sharpe"],["max_drawdown_pct","Max drawdown"],["win_rate_pct","Win rate"]];
    $("backtestMetrics").innerHTML=keys.map(([k,label])=>metric(label,k.includes("pct")?`${Number(r.stats[k]||0).toFixed(2)}%`:Number(r.stats[k]||0).toFixed(3))).join("");
    lineChart($("equityChart") as unknown as SVGSVGElement,[{values:r.equity_curve.map(x=>x.equity),className:"primary"}]);
    $("backtestWarning").textContent=r.warnings.join(" ");
    ($("exportButton") as HTMLButtonElement).disabled=false;
  } finally { btn.disabled=false; btn.textContent="Run walk-forward"; }
}

async function runExperiment() {
  const selected = Array.from(document.querySelectorAll<HTMLInputElement>('input[name="experimentModel"]:checked')).map(x=>x.value as ExperimentModel);
  if (!selected.length) return;
  const commission = Number(($("commission") as HTMLInputElement).value);
  const slippage = Number(($("slippage") as HTMLInputElement).value);
  const btn=$("experimentButton") as HTMLButtonElement; btn.disabled=true; btn.textContent="Evaluating…";
  try {
    const r:ExperimentResult=await api.experiment(currentTicker,currentHorizon,selected,commission,slippage);
    $("experimentTable").innerHTML = `<table><thead><tr><th>Model</th><th>Type</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Win rate</th><th>Trades</th></tr></thead><tbody>${r.comparisons.map(c=>`<tr><td><strong>${esc(modelLabel(c.model))}</strong></td><td><span class="badge ${c.kind==='machine_learning'?'ai':''}">${esc(c.kind.replaceAll('_',' '))}</span></td><td>${Number(c.stats.total_return_pct||0).toFixed(2)}%</td><td>${Number(c.stats.sharpe_ratio||0).toFixed(3)}</td><td>${Number(c.stats.max_drawdown_pct||0).toFixed(2)}%</td><td>${Number(c.stats.win_rate_pct||0).toFixed(1)}%</td><td>${c.trades}</td></tr>`).join("")}</tbody></table>`;
    lineChart($("experimentChart") as unknown as SVGSVGElement,r.comparisons.map((c,i)=>({values:c.equity_curve.map(x=>x.equity),className:`series-${i}`})));
    $("experimentLegend").innerHTML=r.comparisons.map((c,i)=>`<span><i class="series-${i}"></i>${esc(modelLabel(c.model))}</span>`).join("");
    $("experimentNote").textContent=`${r.methodology}. Commission ${r.costs.commission_bps} bps + slippage ${r.costs.slippage_bps} bps. ${r.warning||""}`;
  } finally { btn.disabled=false; btn.textContent="Run comparison"; }
}

async function refreshSystem() {
  const s=await api.system();
  $("systemMetrics").innerHTML=metric("API version",s.version)+metric("Uptime",`${Math.round(s.uptime_seconds)}s`)+metric("Requests",String(s.requests_total))+metric("Avg latency",`${s.average_http_latency_ms.toFixed(2)} ms`)+metric("Cache hit",`${(s.prediction_cache_hit_rate*100).toFixed(1)}%`)+metric("Models",String(s.models_loaded))+metric("Live intel",s.live_intelligence_configured?"READY":"OFF")+metric("Reddit",s.reddit_configured?"READY":"OFF");
  $("architecture").innerHTML=s.architecture.map((x,i)=>`<div class="arch-node"><small>${String(i+1).padStart(2,"0")}</small><strong>${esc(x)}</strong></div>`).join("<span class=\"arrow\">→</span>");
  $("capabilities").innerHTML=s.capabilities.map(x=>`<li>${esc(x)}</li>`).join("");
}

function bindNavigation() {
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach(btn=>btn.addEventListener("click",async()=>{
    document.querySelectorAll("[data-view]").forEach(x=>x.classList.remove("active")); btn.classList.add("active");
    document.querySelectorAll<HTMLElement>(".view").forEach(v=>v.hidden=true); $(String(btn.dataset.view)).hidden=false;
    if(btn.dataset.view==="systemView") await refreshSystem();
  }));
}

async function exportCsv(){ const {url,token}=await api.equityCsvUrl(currentTicker,currentHorizon); const res=await fetch(url,{headers:{Authorization:`Bearer ${token}`}}); const blob=await res.blob(); const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=`${currentTicker}_${currentHorizon}_equity.csv`; a.click(); URL.revokeObjectURL(a.href); }

async function boot() {
  bindNavigation();
  try {
    const [health,models]=await Promise.all([api.health(),api.models()]);
    liveConfigured=health.live_intelligence_configured; redditConfigured=health.reddit_configured;
    const ticker=$("ticker") as HTMLSelectElement; ticker.innerHTML=models.tickers.map(t=>`<option>${esc(t)}</option>`).join(""); currentTicker=ticker.value;
    $("releaseMeta").textContent=`v${health.version} · ${health.model_count} persisted artefacts · ${health.cache_backend} cache`;
    $("providerState").textContent=liveConfigured?`LIVE READY · click Analyze · free-tier caching enabled${redditConfigured?" · Reddit ready":" · Reddit optional"}`:"LIVE KEYS NOT CONFIGURED · offline lab remains available";
    $("providerState").className=liveConfigured?"provider-state ready":"provider-state";
    status("ok",liveConfigured?"live intelligence ready · no provider calls made yet":"offline research mode");
    // Deliberately do NOT run a live analysis during boot. Free-tier provider
    // quotas should only be consumed after the user explicitly clicks Analyze.
    await refreshExplorer();
  } catch(e) {
    if(e instanceof AuthError) ($("loginDialog") as HTMLDialogElement).showModal(); else status("error",(e as Error).message);
  }
}

$("ticker").addEventListener("change",async e=>{currentTicker=(e.target as HTMLSelectElement).value; await refreshExplorer();});
$("horizon").addEventListener("change",async e=>{currentHorizon=(e.target as HTMLSelectElement).value as Horizon; await refreshExplorer();});
$("liveHorizon").addEventListener("change",e=>{liveHorizon=(e.target as HTMLSelectElement).value as Horizon;});
$("refreshButton").addEventListener("click",refreshExplorer); $("backtestButton").addEventListener("click",runBacktest); $("experimentButton").addEventListener("click",runExperiment); $("exportButton").addEventListener("click",exportCsv);
$("liveForm").addEventListener("submit",async e=>{e.preventDefault();await runLiveAnalysis();});
$("loginForm").addEventListener("submit",async e=>{e.preventDefault();setCredentials(($('loginUser') as HTMLInputElement).value,($('loginPass') as HTMLInputElement).value);($('loginDialog') as HTMLDialogElement).close();await boot();});
boot();
