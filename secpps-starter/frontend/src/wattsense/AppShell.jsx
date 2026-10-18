import { useAuth } from "../context/AuthContext";
import { Link, usePath } from "./lib";
import EnergyPage from "./EnergyPage";
import "./theme.css";

const TABS = [
  { path: "/", label: "Dashboard" },
  { path: "/energy", label: "Energy" },
];

/**
 * Everything a logged-in user sees: one navigation bar, then either
 *   Dashboard  - the team's dashboard (forecast + anomaly detection), passed in as children
 *   Energy     - tariff, current cost, predicted cost and recommendations (US-20 to US-24)
 */
export default function AppShell({ children }) {
  const { user, logout } = useAuth();
  const path = usePath();
  const onEnergy = path === "/energy";

  return (
    <div className="ws-theme">
      <header className="ws-nav">
        <div className="ws-nav-inner">
          <Link to="/" className="ws-brand" aria-label="WattSense, go to the dashboard">
            <svg viewBox="0 0 32 32" aria-hidden="true" className="ws-brand-mark">
              <circle cx="16" cy="16" r="13" fill="none" stroke="currentColor" strokeOpacity="0.35" strokeWidth="3" />
              <path d="M16 3 A13 13 0 0 1 27.3 22.5" fill="none" stroke="var(--ws-peak)" strokeWidth="3" strokeLinecap="round" />
              <path d="M17.5 8 11 17.5h4.5L14 24l7-10h-4.6z" fill="currentColor" />
            </svg>
            <span>WattSense</span>
          </Link>

          <nav className="ws-tabs" aria-label="Pages">
            {TABS.map((t) => {
              const active = t.path === "/energy" ? onEnergy : !onEnergy;
              return (
                <Link key={t.path} to={t.path} className={`ws-tab ${active ? "is-active" : ""}`} aria-current={active ? "page" : undefined}>
                  {t.label}
                </Link>
              );
            })}
          </nav>

          <div className="ws-user">
            <span className="ws-user-email">{user?.email}</span>
            <button className="ws-btn ws-btn-quiet" onClick={logout}>
              Log out
            </button>
          </div>
        </div>
      </header>

      {onEnergy ? (
        <main className="ws-main">
          <EnergyPage />
        </main>
      ) : (
        <div className="ws-dashboard">{children}</div>
      )}
    </div>
  );
}