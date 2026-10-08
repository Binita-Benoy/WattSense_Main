import { useState } from "react";
import LoginForm from "./LoginForm";
import RegisterForm from "./RegisterForm";
import "./AuthPage.css";

export default function AuthPage() {
  const [mode, setMode] = useState("login"); // "login" | "register"

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">
          {/* >>> CHANGED (branding): now reads "WattSense ⚡". Original lines were:
              <div className="auth-logo">⚡</div>
              <span>SECPPS</span> */}
          <span>WattSense</span>
          <div className="auth-logo">⚡</div>
          {/* <<< END CHANGED */}
        </div>

        {mode === "login" ? (
          <LoginForm onSwitchToRegister={() => setMode("register")} />
        ) : (
          <RegisterForm onSwitchToLogin={() => setMode("login")} />
        )}
      </div>
    </div>
  );
}