import type {
  Backtest, ExperimentModel, ExperimentResult, Health, History, Horizon,
  LiveIntelligence, LiveSearchResult, ModelRegistry, Overview, Prediction, SystemOverview,
} from "./types.js";

const dockerOrigin = location.port === "3000" || location.port === "80" || location.port === "";
const API = (window as Window & { QUANT_API_BASE?: string }).QUANT_API_BASE || (dockerOrigin ? "" : "http://localhost:8000");
let token = sessionStorage.getItem("quant_token") || "";
let tokenExp = Number(sessionStorage.getItem("quant_token_exp") || 0);
let username = sessionStorage.getItem("quant_user") || "";
let password = "";

export class AuthError extends Error {}

export function setCredentials(user:string, pass:string) {
  username = user.trim();
  password = pass;
  token = "";
  tokenExp = 0;
  sessionStorage.setItem("quant_user", username);
  sessionStorage.removeItem("quant_token");
  sessionStorage.removeItem("quant_token_exp");
}

export function clearCredentials() {
  token = "";
  tokenExp = 0;
  username = "";
  password = "";
  sessionStorage.removeItem("quant_token");
  sessionStorage.removeItem("quant_token_exp");
  sessionStorage.removeItem("quant_user");
  sessionStorage.removeItem("quant_pass");
}

export function storedUsername() { return username; }

async function ensureToken() {
  if (token && Date.now() < tokenExp - 10_000) return token;
  if (!username || !password) throw new AuthError("Authentication required");
  const res = await fetch(`${API}/api/v1/auth/token`, {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({username,password}),
  });
  if (!res.ok) throw new AuthError("Authentication failed");
  const body = await res.json() as {access_token:string;expires_in:number};
  token = body.access_token;
  tokenExp = Date.now() + body.expires_in * 1000;
  sessionStorage.setItem("quant_token", token);
  sessionStorage.setItem("quant_token_exp", String(tokenExp));
  return token;
}

async function request<T>(path:string, init:RequestInit={}) {
  await ensureToken();
  const headers = new Headers(init.headers || {});
  headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type","application/json");
  let res = await fetch(`${API}${path}`, {...init, headers});
  if (res.status === 401) {
    token = "";
    sessionStorage.removeItem("quant_token");
    sessionStorage.removeItem("quant_token_exp");
    await ensureToken();
    headers.set("Authorization", `Bearer ${token}`);
    res = await fetch(`${API}${path}`, {...init, headers});
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = String((await res.json()).detail || detail); } catch {}
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => fetch(`${API}/api/health`).then(r => r.json() as Promise<Health>),
  models: () => request<ModelRegistry>("/api/v1/models"),
  prediction: (ticker:string, horizon:Horizon) => request<Prediction>(`/api/v1/predict/${encodeURIComponent(ticker)}?horizon=${horizon}`),
  history: (ticker:string, limit=180) => request<History>(`/api/v1/market/history/${encodeURIComponent(ticker)}?limit=${limit}`),
  overview: () => request<Overview>("/api/v1/market/overview"),
  backtest: (ticker:string, horizon:Horizon) => request<Backtest>("/api/v1/backtest", {method:"POST", body:JSON.stringify({ticker,horizon})}),
  experiment: (ticker:string, horizon:Horizon, models:ExperimentModel[], commission_bps:number, slippage_bps:number) =>
    request<ExperimentResult>("/api/v1/experiments/compare", {method:"POST", body:JSON.stringify({ticker,horizon,models,commission_bps,slippage_bps})}),
  system: () => request<SystemOverview>("/api/v1/system"),
  liveSearch: (query:string, limit=8) => request<LiveSearchResult>(`/api/v1/intelligence/search?q=${encodeURIComponent(query)}&limit=${limit}`),
  liveAnalyze: (query:string, horizon:Horizon, include_reddit_comments=true) => request<LiveIntelligence>("/api/v1/intelligence/analyze", {
    method:"POST", body:JSON.stringify({query,horizon,include_reddit_comments}),
  }),
  equityCsvUrl: async (ticker:string, horizon:Horizon) => {
    await ensureToken();
    return {url:`${API}/api/v1/backtest/${encodeURIComponent(ticker)}/equity.csv?horizon=${horizon}`, token};
  },
};
