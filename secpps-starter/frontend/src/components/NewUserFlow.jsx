import { useCallback, useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { dashboardApi } from "../api/dashboardApi";
import AppHeader from "./AppHeader";
import UploadForm from "./UploadForm";
import ForecastProgress from "./ForecastProgress";
import NewUserDashboard from "./NewUserDashboard";
import Warnings from "./Warnings";

/**
 * Everything a "new user" sees, driven by the saved job's status:
 *   none / failed  -> upload form (failed also shows why)
 *   queued/running -> progress screen (polls until it finishes)
 *   done           -> dashboard
 */
export default function NewUserFlow({ pollMs = 3000 }) {
  const { token } = useAuth();
  const [job, setJob] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [actionError, setActionError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setJob(await dashboardApi.getNewUserForecast(token));
      setLoadError(null);
    } catch (err) {
      setLoadError(err.message);
    }
  }, [token]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const status = job ? job.status : null;
  useEffect(() => {
    if (status !== "queued" && status !== "running") return undefined;
    const id = setInterval(refresh, pollMs);
    return () => clearInterval(id);
  }, [status, refresh, pollMs]);

  const handleUploadAgain = async () => {
    setActionError(null);
    try {
      await dashboardApi.deleteNewUserForecast(token);
      setJob({ status: "none" });
    } catch (err) {
      setActionError(err.message);
    }
  };

  const actions = [];
  if (status === "done") actions.push({ label: "Upload a different file", onClick: handleUploadAgain });

  let body;
  if (!job && loadError) {
    body = (
      <div className="status error">
        <p>{loadError}</p>
        <button onClick={refresh}>Try again</button>
      </div>
    );
  } else if (!job) {
    body = <div className="status">Loading…</div>;
  } else if (status === "queued" || status === "running") {
    body = (
      <>
        <ForecastProgress job={job} />
        <Warnings items={job.warnings} />
      </>
    );
  } else if (status === "done") {
    body = <NewUserDashboard data={job} token={token} />;
  } else {
    body = (
      <>
        {status === "failed" && (
          <div className="status error" role="alert">
            <strong>Your last forecast couldn't be built.</strong>
            <p>{job.error}</p>
          </div>
        )}
        <UploadForm onStarted={refresh} />
      </>
    );
  }

  return (
    <div className="app">
      <AppHeader subtitle="Your uploaded usage and forecast" actions={actions} />
      {job && loadError && <div className="status error">Connection problem, retrying… ({loadError})</div>}
      {actionError && <div className="status error">{actionError}</div>}
      {body}
    </div>
  );
}