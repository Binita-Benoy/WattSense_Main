import { useAuth } from "../context/AuthContext";

/** Shared top bar for the dashboards. `actions` = extra buttons: [{ label, onClick, disabled }]. */
export default function AppHeader({ subtitle, actions = [] }) {
  const { user, logout } = useAuth();
  return (
    <div className="header">
      <div>
        <h1>SECPPS Energy Dashboard</h1>
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