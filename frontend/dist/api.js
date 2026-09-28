const dockerOrigin = location.port === "3000" || location.port === "80" || location.port === "";
const API = window.QUANT_API_BASE || (dockerOrigin ? "" : "http://localhost:8000");
let token = sessionStorage.getItem("quant_token") || "";
let tokenExp = Number(sessionStorage.getItem("quant_token_exp") || 0);
let username = sessionStorage.getItem("quant_user") || "demo";
let password = sessionStorage.getItem("quant_pass") || "quantai-demo";
export class AuthError extends Error {
}
export function setCredentials(user, pass) {
    username = user;
    password = pass;
    token = "";
    tokenExp = 0;
    sessionStorage.setItem("quant_user", user);
    sessionStorage.setItem("quant_pass", pass);
}
async function ensureToken() {
    if (token && Date.now() < tokenExp - 10_000)
        return token;
    const res = await fetch(`${API}/api/v1/auth/token`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }),
    });
    if (!res.ok)
        throw new AuthError("Authentication failed");
    const body = await res.json();
    token = body.access_token;
    tokenExp = Date.now() + body.expires_in * 1000;
    sessionStorage.setItem("quant_token", token);
    sessionStorage.setItem("quant_token_exp", String(tokenExp));
    return token;
}
async function request(path, init = {}) {
    await ensureToken();
    const headers = new Headers(init.headers || {});
    headers.set("Authorization", `Bearer ${token}`);
    if (init.body && !headers.has("Content-Type"))
        headers.set("Content-Type", "application/json");
    let res = await fetch(`${API}${path}`, { ...init, headers });
    if (res.status === 401) {
        token = "";
        await ensureToken();
        headers.set("Authorization", `Bearer ${token}`);
        res = await fetch(`${API}${path}`, { ...init, headers });
    }
    if (!res.ok) {
        let detail = `HTTP ${res.status}`;
        try {
            detail = String((await res.json()).detail || detail);
        }
        catch { }
        throw new Error(detail);
    }
    return res.json();
}
export const api = {
    health: () => fetch(`${API}/api/health`).then(r => r.json()),
    models: () => request("/api/v1/models"),
    prediction: (ticker, horizon) => request(`/api/v1/predict/${encodeURIComponent(ticker)}?horizon=${horizon}`),
    history: (ticker, limit = 180) => request(`/api/v1/market/history/${encodeURIComponent(ticker)}?limit=${limit}`),
    overview: () => request("/api/v1/market/overview"),
    backtest: (ticker, horizon) => request("/api/v1/backtest", { method: "POST", body: JSON.stringify({ ticker, horizon }) }),
    experiment: (ticker, horizon, models, commission_bps, slippage_bps) => request("/api/v1/experiments/compare", { method: "POST", body: JSON.stringify({ ticker, horizon, models, commission_bps, slippage_bps }) }),
    system: () => request("/api/v1/system"),
    liveSearch: (query, limit = 8) => request(`/api/v1/intelligence/search?q=${encodeURIComponent(query)}&limit=${limit}`),
    liveAnalyze: (query, horizon, include_reddit_comments = true) => request("/api/v1/intelligence/analyze", {
        method: "POST", body: JSON.stringify({ query, horizon, include_reddit_comments }),
    }),
    equityCsvUrl: async (ticker, horizon) => {
        await ensureToken();
        return { url: `${API}/api/v1/backtest/${encodeURIComponent(ticker)}/equity.csv?horizon=${horizon}`, token };
    },
};
