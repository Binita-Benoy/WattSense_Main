/**
 * Tariff, cost and recommendation API layer (US-20 to US-24).
 * Token-scoped like dashboardApi.js: the backend works out whose data to use from the token.
 * Kept in its own file so dashboardApi.js is left untouched.
 */

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function authedJson(path, token, options = {}) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...options,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(options.body ? { "Content-Type": "application/json" } : {}),
      },
    });
  } catch {
    throw new Error("Can't reach the server. Is the backend running?");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`;
    const error = new Error(detail);
    error.status = res.status;
    throw error;
  }
  return res.json();
}

export const billingApi = {
  // US-20
  getTariff: (token) => authedJson("/api/my-household/tariff", token),
  saveTariff: (token, peakRate, offPeakRate) =>
    authedJson("/api/my-household/tariff", token, {
      method: "PUT",
      body: JSON.stringify({ peak_rate: peakRate, off_peak_rate: offPeakRate }),
    }),
  // US-21
  getCurrentCost: (token) => authedJson("/api/my-household/cost", token),
  // US-22: horizon is "24h" | "7day" | "1month", the same values the forecast uses
  getPredictedCost: (token, horizon) =>
    authedJson(`/api/my-household/cost/predicted?horizon=${horizon}`, token),
  // US-23 / US-24
  getAdvice: (token, horizon) => authedJson(`/api/my-household/advice?horizon=${horizon}`, token),
};