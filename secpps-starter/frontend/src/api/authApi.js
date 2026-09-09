/**
 * Auth API layer — talks to the real backend (US-1).
 * Base URL assumes the backend is reachable at localhost:8000 in dev.
 * Move this to an env variable (VITE_API_URL) if/when deploying.
 */

const API_BASE = "http://localhost:8000";

export async function registerUser({ email, password }) {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(data?.detail || "Registration failed. Please try again.");
  }

  return data; // { id, email }
}

export async function loginUser({ email, password }) {
  // Backend expects OAuth2 form data, not JSON — field name is "username"
  const body = new URLSearchParams();
  body.append("username", email);
  body.append("password", password);

  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });

  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(data?.detail || "Incorrect email or password.");
  }

  return data; // { access_token, token_type }
}

export async function fetchCurrentUser(token) {
  const response = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    throw new Error("Session expired. Please log in again.");
  }

  return response.json(); // { id, email }
}
