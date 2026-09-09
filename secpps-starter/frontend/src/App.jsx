import { useAuth } from "./context/AuthContext";
import AuthPage from "./components/AuthPage";

function App() {
  const { user, token, checkingSession, logout } = useAuth();

  if (checkingSession) {
    return <div className="app-loading">Loading...</div>;
  }

  if (!token || !user) {
    return <AuthPage />;
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <span>SECPPS</span>
        <div className="app-header-right">
          <span>{user.email}</span>
          <button onClick={logout}>Log out</button>
        </div>
      </header>

      <main className="app-main">
        <p>You're logged in. Household consumption view goes here (US-16).</p>
      </main>
    </div>
  );
}

export default App;
