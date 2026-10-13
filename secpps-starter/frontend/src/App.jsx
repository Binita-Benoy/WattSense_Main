// Root component. Nobody is asked what kind of user they are; the account decides:
//   not logged in                         -> login / register
//   account registered as a NEW user      -> NewUserFlow: upload the CSV, then the dashboard
//   any other account (logging in)        -> ExistingDashboard: the dataset household
//
// "new" is recorded by the backend when an account is registered (api/auth.py). An account without
// that mark, such as one created before this existed, counts as an existing user.
//
// HouseholdSelector.jsx, api.js and OnboardingPage.jsx are no longer used; the files are left untouched.

import { useEffect, useState } from "react";
import { useAuth } from "./context/AuthContext";
import { dashboardApi } from "./api/dashboardApi";

import AuthPage from "./components/AuthPage";
import NewUserFlow from "./components/NewUserFlow";
import ExistingDashboard from "./components/ExistingDashboard";

export default function App() {
  const { user, token, checkingSession } = useAuth();
  const [profile, setProfile] = useState(null); // null = not loaded yet
  const [profileError, setProfileError] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);

  const loggedIn = Boolean(token && user);

  useEffect(() => {
    if (!loggedIn) {
      setProfile(null);
      return;
    }
    let cancelled = false;
    setProfile(null);
    setProfileError(null);
    dashboardApi
      .getProfile(token)
      .then((p) => {
        if (!cancelled) setProfile(p);
      })
      .catch((err) => {
        if (!cancelled) setProfileError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [token, loggedIn, reloadKey]);

  if (checkingSession) return <div className="status">Loading...</div>;
  if (!token || !user) return <AuthPage />;

  if (profileError) {
    return (
      <div className="status error">
        <p>{profileError}</p>
        <button onClick={() => setReloadKey((k) => k + 1)}>Try again</button>
      </div>
    );
  }
  if (profile === null) return <div className="status">Loading...</div>;

  if (profile.has_profile && profile.user_type === "new") return <NewUserFlow />;
  return <ExistingDashboard />;
}