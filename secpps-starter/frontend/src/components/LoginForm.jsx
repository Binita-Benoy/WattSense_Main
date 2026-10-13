import { useState } from "react";
import { loginUser } from "../api/authApi";
import { useAuth } from "../context/AuthContext";

export default function LoginForm({ onSwitchToRegister }) {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const { access_token } = await loginUser({ email, password });
      login(access_token);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="auth-form" onSubmit={handleSubmit}>
      <h2>Log in</h2>
      <p className="auth-subtitle">Access your household energy dashboard</p>

      <label>
        Email
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          autoComplete="email"
        />
      </label>

      <label>
        Password
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          autoComplete="current-password"
        />
      </label>

      {error && <div className="auth-error" role="alert">{error}</div>}

      <button type="submit" disabled={loading}>
        {loading ? "Logging in..." : "Log in"}
      </button>

      <p className="auth-switch">
        Don't have an account?{" "}
        <button type="button" className="link-button" onClick={onSwitchToRegister}>
          Register
        </button>
      </p>
    </form>
  );
}
