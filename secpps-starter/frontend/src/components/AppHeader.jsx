import { useAuth } from "../context/AuthContext";


export default function AppHeader({ subtitle, actions = [] }) {
  const { user, logout } = useAuth();
  return (
    <div className="header">
      <div>
     
        <h1>
         
          WattSense Energy Dashboard
          <span className="brand-mark" aria-hidden="true">⚡</span>
        </h1>
       
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