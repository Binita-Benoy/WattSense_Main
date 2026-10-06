import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { dashboardApi } from "../api/dashboardApi";
import "./AuthPage.css";

/** Shown once after login: new user (upload your own data) or existing user (data already on file). */
export default function OnboardingPage({ onDone }) {
  const { token, logout } = useAuth();
  const [userType, setUserType] = useState(null); // "new" | "existing"
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userType) {
      setError("Please choose one of the two options above.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      await dashboardApi.createProfile(token, userType);
      const profile = await dashboardApi.getProfile(token);
      onDone(profile);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">
          <div className="auth-logo">⚡</div>
          <span>SECPPS</span>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <h2>Welcome to SECPPS</h2>
          <p className="auth-subtitle">Which of these describes you?</p>

          <div className="user-type-options">
            <button
              type="button"
              className={`user-type-option ${userType === "new" ? "selected" : ""}`}
              onClick={() => setUserType("new")}
            >
              <strong>I'm a new user</strong>
              <span>
                I'll upload my past electricity usage and my location, and get a forecast of my
                future usage.
              </span>
            </button>
            <button
              type="button"
              className={`user-type-option ${userType === "existing" ? "selected" : ""}`}
              onClick={() => setUserType("existing")}
            >
              <strong>I'm an existing user</strong>
              <span>
                My household's smart-meter history is already on file. Show it with a forecast.
              </span>
            </button>
          </div>

          {error && <div className="auth-error" role="alert">{error}</div>}

          <button type="submit" disabled={loading || !userType}>
            {loading ? "Saving..." : "Continue to dashboard"}
          </button>

          <p className="auth-switch">
            <button type="button" className="link-button" onClick={logout}>
              Log out
            </button>
          </p>
        </form>
      </div>
    </div>
  );
}