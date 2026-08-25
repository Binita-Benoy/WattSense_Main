const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path) {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  getHouseholds: () => request("/api/households"),
  getReadings: (householdId, limit = 500) =>
    request(`/api/households/${householdId}/readings?limit=${limit}`),
  getSummary: (householdId) => request(`/api/households/${householdId}/summary`),
};