// Shared helpers for the WattSense shell and Energy page: a tiny hash router,
// an API-loading hook that uses the login token, and number formatting.
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { useAuth } from "../context/AuthContext";

// ---- Hash router (#/ and #/energy). No extra npm package needed. ----------
function subscribe(callback) {
  window.addEventListener("hashchange", callback);
  return () => window.removeEventListener("hashchange", callback);
}
const currentPath = () => window.location.hash.replace(/^#/, "") || "/";

export function usePath() {
  return useSyncExternalStore(subscribe, currentPath);
}

export function Link({ to, children, ...rest }) {
  return (
    <a href={`#${to}`} {...rest}>
      {children}
    </a>
  );
}

// ---- useApi(fetcher, deps): runs fetcher(token); { data, error, loading, reload } ----
// Old figures stay on screen while new ones load, and a slow earlier response
// can never overwrite a newer one.
export function useApi(fetcher, deps = [], enabled = true) {
  const { token } = useAuth();
  const [state, setState] = useState({ data: null, error: null, loading: enabled });
  const latest = useRef(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const run = useCallback(() => {
    if (!token || !enabled) return;
    const id = ++latest.current;
    setState((s) => ({ ...s, error: null, loading: true }));
    fetcherRef
      .current(token)
      .then((data) => id === latest.current && setState({ data, error: null, loading: false }))
      .catch((error) => id === latest.current && setState({ data: null, error, loading: false }));
  }, [token, enabled]);

  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, ...deps]);

  return { ...state, reload: run };
}

// ---- Formatting ----------------------------------------------------------------
export const money = (n, currency = "$") =>
  n == null || Number.isNaN(n)
    ? "—"
    : `${currency}${Number(n).toLocaleString("en-AU", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const kwh = (n) =>
  n == null || Number.isNaN(n) ? "—" : `${Number(n).toLocaleString("en-AU", { maximumFractionDigits: 1 })} kWh`;

// "2014-03-03T09:00:00" -> "3 Mar 2014"
export const day = (iso) =>
  iso ? new Date(iso).toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" }) : "—";

// "2014-03-03T09:00:00" -> "3 Mar"
export const shortDay = (iso) =>
  iso ? new Date(iso).toLocaleDateString("en-AU", { day: "numeric", month: "short" }) : "—";

// "3:00 PM" -> 15 ; "9:30 AM" -> 9.5
export function clockToHour(label) {
  const m = /^(\d{1,2}):(\d{2})\s*(AM|PM)$/i.exec(label || "");
  if (!m) return null;
  let hour = Number(m[1]) % 12;
  if (m[3].toUpperCase() === "PM") hour += 12;
  return hour + Number(m[2]) / 60;
}

// 15 -> "3pm", 21 -> "9pm", 0 -> "12am"
export function hourLabel(h) {
  const hour = ((Math.floor(h) % 24) + 24) % 24;
  const suffix = hour < 12 ? "am" : "pm";
  return `${hour % 12 === 0 ? 12 : hour % 12}${suffix}`;
}