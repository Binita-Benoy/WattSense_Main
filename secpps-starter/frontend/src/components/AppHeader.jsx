import { useAuth } from "../context/AuthContext";

/** Shared top bar for the dashboards. `actions` = extra buttons: [{ label, onClick, disabled }]. */
export default function AppHeader({ subtitle, actions = [] }) {
  const { user, logout } = useAuth();
  return (
    <div className="header">
      <div>
        {/* >>> CHANGED (UI polish): small brand mark in front of the title (same bolt as the login card) */}
        <h1>
          {/* CHANGED (branding): was "SECPPS Energy Dashboard" with the bolt in front */}
          WattSense Energy Dashboard
          <span className="brand-mark" aria-hidden="true">⚡</span>
        </h1>
        {/* <<< END CHANGED */}
        <p>{subtitle}</p>
      </div>
      <div className="header-right">
        <span>{user.email}</span>
        {actions.map((a) => (
          <button key={a.label} onClick={a.onClick} disabled={a.disabled}>
            {a.label}
          </button>
        ))}
        <button onClick={logout}>Log out</button>
      </div>
    </div>
  );
}