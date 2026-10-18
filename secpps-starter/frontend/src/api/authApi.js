
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

  return data; // 
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
