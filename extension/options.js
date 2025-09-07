const DEFAULT_API_BASE = "http://127.0.0.1:8000";

const elApiBase = document.getElementById("apiBase");
const elSave    = document.getElementById("save");
const elTest    = document.getElementById("test");
const elReset   = document.getElementById("reset");
const elStatus  = document.getElementById("status");

function normalizeBase(input) {
  if (!input) return "";
  let s = String(input).trim();
  // If user typed just host:port, assume http
  if (!/^https?:\/\//i.test(s)) s = "http://" + s;
  // strip trailing slashes
  return s.replace(/\/+$/, "");
}

function setStatus(text, cls = "muted") {
  elStatus.className = `status ${cls}`;
  elStatus.textContent = text;
}

async function load() {
  chrome.storage.local.get(["apiBase"], ({ apiBase }) => {
    elApiBase.value = apiBase || DEFAULT_API_BASE;
    setStatus("Loaded.", "muted");
  });
}

async function save() {
  const base = normalizeBase(elApiBase.value);
  if (!base) return setStatus("Please enter a valid URL.", "err");

  await chrome.storage.local.set({ apiBase: base });
  setStatus(`Saved ✓  (${base})`, "ok");

  // optional: ping after save
  try {
    const res = await fetch(`${base}/`, { method: "GET" });
    if (!res.ok) return setStatus(`Saved, but health check: ${res.status}`, "warn");
    const j = await res.json().catch(() => ({}));
    const m = j?.ok ? "online" : "reachable (no ok flag)";
    setStatus(`Saved ✓  • API ${m} • service=${j?.service || "?"} • llm=${j?.llm_mode ?? "?"}`, j?.ok ? "ok" : "warn");
  } catch (e) {
    setStatus(`Saved, but offline: ${e?.message || e}`, "err");
  }
}

async function test() {
  const base = normalizeBase(elApiBase.value || DEFAULT_API_BASE);
  setStatus("Testing…", "muted");
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 6000);
    const res = await fetch(`${base}/`, { method: "GET", signal: ctl.signal });
    clearTimeout(t);
    if (!res.ok) return setStatus(`Health check failed: ${res.status}`, "err");
    const j = await res.json().catch(() => ({}));
    const m = j?.ok ? "online" : "reachable (no ok flag)";
    setStatus(`API ${m} • version=${j?.version ?? "?"} • llm=${j?.llm_mode ?? "?"}`, j?.ok ? "ok" : "warn");
  } catch (e) {
    setStatus(`Offline: ${e?.name === "AbortError" ? "timeout" : (e?.message || e)}`, "err");
  }
}

function resetToDefault() {
  elApiBase.value = DEFAULT_API_BASE;
  setStatus(`Reset to ${DEFAULT_API_BASE}`, "muted");
}

elSave.addEventListener("click", save);
elTest.addEventListener("click", test);
elReset.addEventListener("click", resetToDefault);
document.addEventListener("DOMContentLoaded", load);
