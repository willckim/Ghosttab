// ======== CONFIG ========
const CLOUD_RUN_BASE = "https://ghosttab-api-611064069137.us-central1.run.app";
const DEFAULT_API_BASE = CLOUD_RUN_BASE;

// Heuristic: treat builds with "(Dev)" or "Dev" in name as local-dev
const IS_DEV_BUILD = /\bdev\b/i.test(chrome.runtime.getManifest?.().name || "");

// Detect localhost-like bases
const LOCAL_RE   = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?/i;
const RUN_APP_RE = /^https?:\/\/[^/]*\.run\.app(?::\d+)?$/i;

// ======== DOM ========
const elApiBase = document.getElementById("apiBase");
const elSave    = document.getElementById("save");
const elTest    = document.getElementById("test");
const elReset   = document.getElementById("reset");
const elStatus  = document.getElementById("status");
const elDevTip  = document.getElementById("devTip");

// ======== Safe helpers ========
const safe = (cond, fn) => { if (cond) { try { fn(); } catch {} } };

function getLocal(keys) {
  return new Promise((resolve) => chrome.storage.local.get(keys, resolve));
}
function setLocal(obj) {
  return new Promise((resolve) => chrome.storage.local.set(obj, resolve));
}
function removeLocal(keys) {
  return new Promise((resolve) => chrome.storage.local.remove(keys, resolve));
}

// ======== URL helpers ========
function normalizeBase(input) {
  if (!input) return "";
  let s = String(input).trim();

  // If missing protocol, assume http
  if (!/^https?:\/\//i.test(s)) s = "http://" + s;

  try {
    const u = new URL(s);
    // If it's a *.run.app, force https and collapse to origin
    if (RUN_APP_RE.test(u.origin)) u.protocol = "https:";
    return u.origin; // strip paths & trailing slash
  } catch {
    // Fallback: just strip trailing slashes
    return s.replace(/\/+$/, "");
  }
}

function validateUrlStrict(maybeUrl) {
  try {
    const u = new URL(maybeUrl);
    if (!/^https?:$/.test(u.protocol)) {
      return { ok: false, reason: "Only http/https are allowed." };
    }
    return { ok: true, url: u.origin };
  } catch {
    return { ok: false, reason: "Invalid URL." };
  }
}

// ======== UI helpers ========
function setStatus(text, cls = "muted") {
  if (!elStatus) return;
  elStatus.className = `status ${cls}`;
  elStatus.textContent = text;
}

function disableButtons(disabled) {
  [elSave, elTest, elReset].forEach(b => { if (b) b.disabled = disabled; });
}

// ======== KV helpers ========
async function getApiKey() {
  const { apiKey } = await getLocal(["apiKey"]);
  return (typeof apiKey === "string" && apiKey.trim()) ? apiKey.trim() : null;
}

// NEW: optional provider preference ("azure" | "openai")
async function getLlmPref() {
  const { llmProvider } = await getLocal(["llmProvider"]);
  return (llmProvider === "azure" || llmProvider === "openai") ? llmProvider : null;
}

// ======== Ping (CSP-safe) ========
// Tries GET /ping → /health → /, then falls back to POST /summarize
// Marks auth-gated servers as { ok:false, auth:true } (degraded)
async function pingBase(base, timeoutMs = 6000) {
  if (!IS_DEV_BUILD && LOCAL_RE.test(base)) {
    return { ok: false, json: { ok: false, detail: "CSP blocks localhost in store build. Use the Cloud Run URL." } };
  }

  const ctl = new AbortController();
  const to  = setTimeout(() => ctl.abort(), timeoutMs);

  const tryGet = async (path) => {
    try {
      const start = performance.now();
      const res = await fetch(`${base}${path}?t=${Date.now()}`, {
        method: "GET",
        cache: "no-store",
        signal: ctl.signal,
      });
      const ms = Math.max(0, Math.round(performance.now() - start));
      let j = {};
      try { j = await res.json(); } catch {}
      return { ok: res.ok, status: res.status, json: j, ms, path };
    } catch (e) {
      return { ok: false, status: 0, json: { error: e?.message || "network" } };
    }
  };

  try {
    // 1) /ping → 2) /health → 3) /
    let r = await tryGet("/ping");
    if (!r.ok) r = await tryGet("/health");
    if (!r.ok) r = await tryGet("/");

    // If still not OK, it might be auth-gated. Do a tiny POST to /summarize
    if (!r.ok) {
      const headers = { "content-type": "application/json" };
      const key = await getApiKey();
      const pref = await getLlmPref();         // ← include provider override if set
      if (key) headers["x-api-key"] = key;
      if (pref) headers["x-llm"] = pref;

      try {
        const start = performance.now();
        const resp = await fetch(`${base}/summarize`, {
          method: "POST",
          headers,
          body: JSON.stringify({ text: "ping" }),
          signal: ctl.signal,
        });
        const ms = Math.max(0, Math.round(performance.now() - start));
        clearTimeout(to);

        if (resp.status === 200) {
          const j = await resp.json().catch(() => ({}));
          return { ok: true, json: { ok: true, via: "post", ...j }, pathTried: "POST /summarize", ms };
        }
        if (resp.status === 401 || resp.status === 403) {
          return { ok: false, json: { ok: false, auth: true, detail: "auth required" }, pathTried: "POST /summarize", ms };
        }
        return { ok: false, json: { ok: false, detail: `status ${resp.status}` }, pathTried: "POST /summarize", ms };
      } catch (e) {
        clearTimeout(to);
        return { ok: false, json: { ok: false, detail: e?.message || "network" } };
      }
    }

    clearTimeout(to);
    // Normalize output
    const j = r.json || {};
    const pathTried = r.path || "GET";
    return { ok: r.ok, json: j, pathTried, ms: r.ms };
  } catch (e) {
    clearTimeout(to);
    return { ok: false, json: { ok: false, detail: e?.message || (e?.name === "AbortError" ? "timeout" : "error") } };
  }
}

// ======== Load / Save ========
async function load() {
  // Toggle Dev tip without inline script
  safe(elDevTip, () => {
    if (IS_DEV_BUILD) elDevTip.classList.remove("hidden");
  });

  try {
    let { apiBase } = await getLocal(["apiBase"]);

    // Auto-migrate to Cloud Run if a localhost URL is saved in a production build
    if (!IS_DEV_BUILD && LOCAL_RE.test(apiBase || "")) {
      await setLocal({ apiBase: CLOUD_RUN_BASE });
      apiBase = CLOUD_RUN_BASE;
      setStatus("Detected localhost in production. Switched to Cloud Run.", "warn");
    } else {
      setStatus("Loaded.", "muted");
    }

    if (elApiBase) elApiBase.value = apiBase || DEFAULT_API_BASE;
  } catch {
    setStatus("Failed to load settings.", "err");
  }
}

async function save() {
  if (!elApiBase) return;

  let base = normalizeBase(elApiBase.value);
  const v  = validateUrlStrict(base);
  if (!v.ok) return setStatus(v.reason, "err");
  base = v.url;

  // In production builds, don’t allow localhost (prevents CSP errors)
  if (!IS_DEV_BUILD && LOCAL_RE.test(base)) {
    base = CLOUD_RUN_BASE;
    elApiBase.value = base;
    setStatus("Localhost is blocked in Web Store build. Using Cloud Run instead.", "warn");
  }

  try {
    disableButtons(true);
    await setLocal({ apiBase: base });
    setStatus(`Saved ✓ (${base})`, "ok");

    // Optional health check after save (auth-aware)
    setStatus("Saved. Checking health…", "muted");
    const result = await pingBase(base, 6000);

    if (result.ok) {
      const j = result.json || {};
      const info = [
        "API online",
        `endpoint=/ping|/health|/ (or POST)`,
        `latency=${result.ms ?? "?"}ms`,
        `version=${j?.version ?? "?"}`,
        `llm=${j?.llm_mode ?? "?"}`
      ].join(" • ");
      setStatus(`Saved ✓ • ${info}`, "ok");
    } else if (result.json?.auth) {
      setStatus("Saved ✓ • reachable, but API key required (set it in Options).", "warn");
    } else {
      setStatus(`Saved, but offline: ${result.json?.detail || "No response"}`, "err");
    }
  } catch (e) {
    setStatus(`Failed to save: ${e?.message || e}`, "err");
  } finally {
    disableButtons(false);
  }
}

async function test() {
  const current = elApiBase ? elApiBase.value : DEFAULT_API_BASE;
  const base    = normalizeBase(current || DEFAULT_API_BASE);
  const v       = validateUrlStrict(base);
  if (!v.ok) return setStatus(v.reason, "err");

  setStatus("Testing…", "muted");
  try {
    disableButtons(true);
    const result = await pingBase(v.url, 6000);

    if (result.ok) {
      const j = result.json || {};
      const info = [
        "API online",
        `endpoint=/ping|/health|/ (or POST)`,
        `latency=${result.ms ?? "?"}ms`,
        `version=${j?.version ?? "?"}`,
        `llm=${j?.llm_mode ?? "?"}`
      ].join(" • ");
      setStatus(info, "ok");
    } else if (result.json?.auth) {
      setStatus("Reachable, but API key required (set it in Options).", "warn");
    } else {
      setStatus(`Health check failed: ${result.json?.detail || "No response"}`, "err");
    }
  } catch (e) {
    setStatus(`Offline: ${e?.name === "AbortError" ? "timeout" : (e?.message || e)}`, "err");
  } finally {
    disableButtons(false);
  }
}

function resetToDefault() {
  const base = IS_DEV_BUILD ? "http://127.0.0.1:8000" : CLOUD_RUN_BASE;
  if (elApiBase) elApiBase.value = base;
  setStatus(`Reset to ${base}`, "muted");
}

// ======== UX bindings ========
function onKeydown(e) {
  // Ctrl/Cmd+S to save
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    save();
  }
  // "/" to test (matches tooltip)
  if (!e.ctrlKey && !e.metaKey && e.key === "/") {
    e.preventDefault();
    test();
  }
}

function bindUX() {
  safe(elSave,  () => elSave.addEventListener("click", save));
  safe(elTest,  () => elTest.addEventListener("click", test));
  safe(elReset, () => elReset.addEventListener("click", resetToDefault));

  // Enter to save, blur to auto-save
  safe(elApiBase, () => {
    elApiBase.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        save();
      }
    });
    elApiBase.addEventListener("blur", async () => {
      let base = normalizeBase(elApiBase.value);
      if (!base) return;

      const v = validateUrlStrict(base);
      if (!v.ok) {
        setStatus(v.reason, "err");
        return;
      }

      let saveBase = v.url;
      if (!IS_DEV_BUILD && LOCAL_RE.test(saveBase)) {
        saveBase = CLOUD_RUN_BASE;
        elApiBase.value = saveBase;
        setStatus("Localhost is blocked in Web Store build. Using Cloud Run instead.", "warn");
      }
      await setLocal({ apiBase: saveBase });
      setStatus(`Saved ✓ (${saveBase})`, "ok");
    });
  });

  document.addEventListener("keydown", onKeydown);
}

// ======== Init ========
document.addEventListener("DOMContentLoaded", () => {
  load();
  bindUX();
});
