/**
 * Dashboard API layer. Talks to the token-scoped endpoints:
 *   /api/profile/*              new vs existing user choice
 *   /api/forecast/new-user      upload a CSV, build + fetch the forecast (new users)
 *   /api/my-household/*         dataset household data (existing users)
 * Kept separate from the original api.js, which is left untouched.
 */

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function messageFrom(body, status) {
  const detail = body && body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    const text = detail.map((d) => d.msg).filter(Boolean).join("; ");
    if (text) return text;
  }
  return `Request failed (${status})`;
}

async function authedFetch(path, token, options = {}) {
  const isForm = typeof FormData !== "undefined" && options.body instanceof FormData;
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...options,
      headers: {
        Authorization: `Bearer ${token}`,
        // For FormData the browser must set Content-Type itself (it adds the multipart boundary).
        ...(options.body && !isForm ? { "Content-Type": "application/json" } : {}),
      },
    });
  } catch {
    throw new Error("Can't reach the server. Is the backend running?");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(messageFrom(body, res.status));
  }
  return res;
}

async function authedJson(path, token, options) {
  const res = await authedFetch(path, token, options);
  return res.json();
}

export const dashboardApi = {
  // --- onboarding choice ---------------------------------------------------
  getProfile: (token) => authedJson("/api/profile/me", token),
  createProfile: (token, userType) =>
    authedJson("/api/profile", token, {
      method: "POST",
      body: JSON.stringify({ user_type: userType }),
    }),
  resetProfile: (token) => authedJson("/api/profile", token, { method: "DELETE" }),

  // --- new users: upload + forecast ----------------------------------------
  getNewUserForecast: (token) => authedJson("/api/forecast/new-user", token),
  startNewUserForecast: (token, { file, location, countryCode, state }) => {
    const form = new FormData();
    form.append("file", file);
    form.append("location", location);
    form.append("country_code", countryCode);
    form.append("state", state);
    return authedJson("/api/forecast/new-user", token, { method: "POST", body: form });
  },
  deleteNewUserForecast: (token) =>
    authedJson("/api/forecast/new-user", token, { method: "DELETE" }),

  // --- existing users: dataset household -----------------------------------
  getMySummary: (token) => authedJson("/api/my-household/summary", token),

  getMyReadings: (token, { start, end, limit = 2000 } = {}) => {
    const params = new URLSearchParams();
    if (start) params.append("start", start);
    if (end) params.append("end", end);
    params.append("limit", limit);
    return authedJson(`/api/my-household/readings?${params.toString()}`, token);
  },

  getMyAnomalies: (token) => authedJson("/api/my-household/anomalies", token),

  // asOf: the forecast continues from the last reading at or before this time
  getMyForecast: (token, horizon = "24h", asOf) =>
    authedJson(
      `/api/my-household/forecast?horizon=${horizon}${asOf ? `&as_of=${encodeURIComponent(asOf)}` : ""}`,
      token
    ),

  // --- reports (both kinds of user) ----------------------------------------
  // source: "household" (dataset household, optional start/end) or "upload" (the saved upload)
  downloadReport: async (token, format, { source = "household", start, end, ratePerKwh } = {}) => {
    const base = source === "upload" ? "/api/forecast/new-user" : "/api/my-household";
    const params = new URLSearchParams();
    if (source !== "upload") {
      if (start) params.append("start", start);
      if (end) params.append("end", end);
    }
    if (ratePerKwh) params.append("rate_per_kwh", ratePerKwh);
    const query = params.toString() ? `?${params.toString()}` : "";

    const res = await authedFetch(`${base}/report.${format}${query}`, token);
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `energy_report.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  },
};