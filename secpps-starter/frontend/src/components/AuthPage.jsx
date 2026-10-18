import { useState } from "react";
import LoginForm from "./LoginForm";
import RegisterForm from "./RegisterForm";
import "./AuthPage.css";

export default function AuthPage() {
  const [mode, setMode] = useState("login"); 

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">
          
          <span>WattSense</span>
          <div className="auth-logo">⚡</div>
          
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