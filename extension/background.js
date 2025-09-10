// ─────────────────────────── Focus / Tab Limit Logic ───────────────────────────
let tabLimitListener = null;
let lastNotificationTime = 0; // Prevent notification spam
let focusModeEnabled = false;

function showRateLimitedNotification(options) {
  const now = Date.now();
  if (now - lastNotificationTime > 3000) {
    try { chrome.notifications.create(options); } catch {}
    lastNotificationTime = now;
  }
}

// Helper to parse and clamp a number from storage
function parseLimit(n, fallback = 3) {
  const v = Number(n);
  return Number.isFinite(v) && v > 0 ? Math.floor(v) : fallback;
}

async function updateBadge(tabsCount, maxTabs, enforce) {
  try {
    const text = focusModeEnabled ? `${Math.min(tabsCount, 99)}` : "";
    await chrome.action.setBadgeText({ text });
    await chrome.action.setBadgeBackgroundColor({ color: enforce ? "#D7263D" : "#2D7DD2" });
    await chrome.action.setTitle({
      title: focusModeEnabled
        ? `GhostTab Focus: ${tabsCount}/${maxTabs} ${enforce ? "(strict)" : ""}`
        : "GhostTab",
    });
  } catch {
    // action API might not be available in some contexts
  }
}

function enableFocusMode() {
  if (tabLimitListener) return; // avoid duplicate listeners
  focusModeEnabled = true;

  tabLimitListener = function (tab) {
    chrome.storage.local.get(["enforceTabs", "customTabLimit"], (data) => {
      const enforce = Boolean(data?.enforceTabs);
      const MAX_TABS = parseLimit(data?.customTabLimit, 3);

      chrome.tabs.query({}, async (tabs) => {
        const count = tabs.length;
        await updateBadge(count, MAX_TABS, enforce);

        if (count > MAX_TABS) {
          if (enforce && tab?.id) {
            chrome.tabs.remove(tab.id, () => void chrome.runtime.lastError);
          }

          showRateLimitedNotification({
            type: "basic",
            iconUrl: chrome.runtime.getURL("icon.png"),
            title: "GhostTab Alert",
            message: enforce
              ? `🚫 Too many tabs! Extra tab closed. (${count}/${MAX_TABS})`
              : `⚠️ Tab limit exceeded. You have ${count} tabs open.`
          });
        }
      });
    });
  };

  chrome.tabs.onCreated.addListener(tabLimitListener);

  // Also react when tabs change/activate quickly
  const boundUpdated = async () => {
    chrome.storage.local.get(["enforceTabs", "customTabLimit"], (data) => {
      const enforce = Boolean(data?.enforceTabs);
      const MAX_TABS = parseLimit(data?.customTabLimit, 3);
      chrome.tabs.query({}, (tabs) => void updateBadge(tabs.length, MAX_TABS, enforce));
    });
  };
  chrome.tabs.onRemoved.addListener(boundUpdated);
  chrome.tabs.onUpdated.addListener(boundUpdated);
  chrome.tabs.onActivated?.addListener(boundUpdated);
  chrome.windows.onFocusChanged.addListener(boundUpdated);

  enableFocusMode._boundUpdated = boundUpdated;
}

function disableFocusMode() {
  if (tabLimitListener) {
    chrome.tabs.onCreated.removeListener(tabLimitListener);
    tabLimitListener = null;
  }
  if (enableFocusMode._boundUpdated) {
    chrome.tabs.onRemoved.removeListener(enableFocusMode._boundUpdated);
    chrome.tabs.onUpdated.removeListener(enableFocusMode._boundUpdated);
    chrome.tabs.onActivated?.removeListener(enableFocusMode._boundUpdated);
    chrome.windows.onFocusChanged.removeListener(enableFocusMode._boundUpdated);
    enableFocusMode._boundUpdated = null;
  }
  focusModeEnabled = false;
  try { chrome.action.setBadgeText({ text: "" }); } catch {}
}

// Enable focus mode on startup if previously enabled
chrome.storage.local.get("focusMode", ({ focusMode }) => {
  if (focusMode) enableFocusMode();
});

// React to focus mode changes
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.focusMode) {
    if (changes.focusMode.newValue === true) {
      enableFocusMode();
    } else {
      disableFocusMode();
    }
  }
});

// ────────────────────────────── AI Bridge (Prod-safe) ─────────────────────────────
const CLOUD_RUN_BASE = "https://ghosttab-api-611064069137.us-central1.run.app";
// Treat builds with “Dev” in the name as local-dev (optional convenience)
const IS_DEV_BUILD = /\bdev\b/i.test(chrome.runtime.getManifest?.().name || "");
const LOCAL_RE = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?/i;

// Default base: Cloud in prod, localhost in dev
const DEFAULT_API_BASE = IS_DEV_BUILD ? "http://127.0.0.1:8000" : CLOUD_RUN_BASE;

async function getApiBase() {
  const { apiBase } = await chrome.storage.local.get(["apiBase"]);
  const saved = (apiBase || DEFAULT_API_BASE).replace(/\/+$/, "");
  // In production, never return a localhost base (prevents CSP violations)
  if (!IS_DEV_BUILD && LOCAL_RE.test(saved)) return CLOUD_RUN_BASE;
  return saved;
}

async function getApiKey() {
  const { apiKey } = await chrome.storage.local.get(["apiKey"]);
  return (typeof apiKey === "string" && apiKey.trim()) ? apiKey.trim() : null;
}

// NEW: optional provider preference ("azure" | "openai")
async function getLlmPref() {
  const { llmProvider } = await chrome.storage.local.get(["llmProvider"]);
  return (llmProvider === "azure" || llmProvider === "openai") ? llmProvider : null;
}

function isRestrictedUrl(url = "") {
  return /^chrome:\/\//i.test(url) || /^https?:\/\/chrome\.google\.com\/webstore/i.test(url);
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

// Prefer content.js (fast). Fallback to executeScript if content is missing.
async function getPageData() {
  const tab = await getActiveTab();
  if (!tab?.id || isRestrictedUrl(tab.url)) {
    return { ok: false, error: "Restricted page; cannot access content." };
  }

  // Try content.js
  try {
    const res = await chrome.tabs.sendMessage(tab.id, { type: "GHOSTTAB_GET_PAGE_TEXT" });
    if (res && typeof res === "object" && ("ok" in res)) return res;
  } catch {
    // no content script injected or messaging failure
  }

  // Fallback to executeScript (MV3)
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        try {
          const selection = window.getSelection?.().toString() || "";
          const text = document.body?.innerText || "";
          return { ok: true, text, selection };
        } catch (e) {
          return { ok: false, error: String(e) };
        }
      }
    });
    return result || { ok: false, error: "Unknown executeScript failure" };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
}

/** postJSON with timeout & retries + x-api-key header (+ x-llm override if set) */
async function postJSON(path, body, opts = {}) {
  const { timeoutMs = 15000, retries = 2 } = opts;
  const apiBase = await getApiBase();
  const url = `${apiBase}${path}`;
  const apiKey = await getApiKey();
  const llmPref = await getLlmPref();

  let attempt = 0;
  const backoff = (n) => new Promise((r) => setTimeout(r, 400 * Math.pow(2, n)));

  const manifest = chrome.runtime.getManifest?.();
  const clientVersion = manifest?.version || "0.0.0";

  while (true) {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const headers = {
        "Content-Type": "application/json",
        "X-GhostTab-Client": `ext/${clientVersion}`
      };
      if (apiKey) headers["x-api-key"] = apiKey;
      if (llmPref) headers["x-llm"] = llmPref; // ← provider override

      const res = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(body || {}),
        signal: controller.signal
      });

      clearTimeout(t);

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        if (res.status === 401 || res.status === 403) {
          throw new Error(`Unauthorized. Check API key/config. (${res.status}) ${text}`);
        }
        if (res.status === 404) {
          throw new Error(`Endpoint not found: ${path}. Is the server exposing this route?`);
        }
        throw new Error(`API ${path} failed: ${res.status} ${text}`);
      }

      return await res.json();
    } catch (err) {
      clearTimeout(t);
      // Improve timeout message for the UI
      if (err?.name === "AbortError") {
        if (attempt >= retries) throw new Error("Request timed out");
      }
      if (attempt < retries) {
        await backoff(attempt++);
        continue;
      }
      throw err;
    }
  }
}

function requireString(field, value) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`Missing/invalid "${field}"`);
  }
}

// Helper: resolve text from message or fall back to active page
async function resolveTextFromMessageOrPage(msg, opts = {}) {
  const { preferSelection = true, minLen = 1, allowEmpty = false } = opts;
  const raw = msg?.payload?.text;
  if (typeof raw === "string" && (allowEmpty || raw.trim().length >= minLen)) {
    return raw;
  }
  const { ok, selection, text, error } = await getPageData();
  if (!ok) throw new Error(error || "Failed to read page text");
  const picked = preferSelection && selection?.trim()?.length >= minLen ? selection : text;
  if (!allowEmpty && (!picked || !picked.trim())) {
    throw new Error("No readable text found on this page.");
  }
  return picked;
}

// ─────────────────────────── Message Routing ───────────────────────────
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    try {
      switch (msg?.type) {
        case "GET_PAGE_TEXT": {
          const data = await getPageData();
          return sendResponse(data);
        }

        case "SUMMARIZE": {
          const text = await resolveTextFromMessageOrPage(msg, { preferSelection: false });
          const data = await postJSON("/summarize", { text });
          return sendResponse(data);
        }
        case "REWRITE": {
          const text = await resolveTextFromMessageOrPage(msg, { preferSelection: true });
          const tone = typeof msg?.payload?.tone === "string" ? msg.payload.tone : undefined;
          const data = await postJSON("/rewrite", { text, tone });
          return sendResponse(data);
        }
        case "TRANSLATE": {
          requireString("to", msg?.payload?.to);
          const text = await resolveTextFromMessageOrPage(msg, { preferSelection: true });
          const data = await postJSON("/translate", { text, to: msg.payload.to });
          return sendResponse(data);
        }
        case "SENTIMENT": {
          const text = await resolveTextFromMessageOrPage(msg, { preferSelection: true });
          const raw = await postJSON("/sentiment", { text });
          const s = (raw && typeof raw === "object" && raw.sentiment && typeof raw.sentiment === "object")
            ? raw.sentiment
            : raw;
          const label = (s && (s.sentiment ?? s.label)) || null;
          const confidence = (s && (typeof s.confidence === "number" ? s.confidence :
                                     typeof s.score === "number" ? s.score : null)) ?? null;
          return sendResponse({ sentiment: { sentiment: label, confidence } });
        }
        case "ANALYZE": {
          const text = await resolveTextFromMessageOrPage(msg, { preferSelection: true });
          const data = await postJSON("/analyze", { text });
          return sendResponse(data);
        }

        // Page/selection flows
        case "ASK_PAGE": {
          requireString("text", msg?.payload?.text);
          requireString("question", msg?.payload?.question);
          const data = await postJSON("/ask_page", {
            text: msg.payload.text,
            question: msg.payload.question,
            top_k: 5
          });
          return sendResponse(data);
        }
        case "ASK_SELECTION": {
          const q = (msg?.payload?.question || "").trim();
          requireString("question", q);
          const { ok, selection, text, error } = await getPageData();
          if (!ok) throw new Error(error || "Failed to read page");
          const used = (selection && selection.trim().length >= 40) ? selection : text;
          const data = await postJSON("/ask_page", { text: used, question: q });
          return sendResponse(data);
        }
        case "SUMMARIZE_SELECTION": {
          const { ok, selection, error } = await getPageData();
          if (!ok) throw new Error(error || "Failed to read selection");
          const sel = (selection || "").trim();
          if (!sel) throw new Error("No text selected.");
          const data = await postJSON("/summarize", { text: sel });
          return sendResponse(data);
        }
        case "REWRITE_SELECTION": {
          const tone = typeof msg?.payload?.tone === "string" ? msg.payload.tone : undefined;
          const { ok, selection, error } = await getPageData();
          if (!ok) throw new Error(error || "Failed to read selection");
          const sel = (selection || "").trim();
          if (!sel) throw new Error("No text selected.");
          const data = await postJSON("/rewrite", { text: sel, tone });
          return sendResponse(data);
        }
        case "TRANSLATE_SELECTION": {
          const to = (msg?.payload?.to || "").trim();
          requireString("to", to);
          const { ok, selection, error } = await getPageData();
          if (!ok) throw new Error(error || "Failed to read selection");
          const sel = (selection || "").trim();
          if (!sel) throw new Error("No text selected.");
          const data = await postJSON("/translate", { text: sel, to });
          return sendResponse(data);
        }

        // ─────────────────────────── HEALTH (Upgraded) ───────────────────────────
        case "HEALTH": {
          const apiBase = await getApiBase();
          const controller = new AbortController();
          const t = setTimeout(() => controller.abort(), 6000);

          const tryGet = async (path) => {
            try {
              const res = await fetch(`${apiBase}${path}?t=${Date.now()}`, {
                method: "GET",
                cache: "no-store",
                signal: controller.signal,
              });
              let json = {};
              try { json = await res.json(); } catch {}
              return { ok: res.ok, status: res.status, json };
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
              const apiKey = await getApiKey();
              const llmPref = await getLlmPref();
              if (apiKey) headers["x-api-key"] = apiKey;
              if (llmPref) headers["x-llm"] = llmPref;
              try {
                const resp = await fetch(`${apiBase}/summarize`, {
                  method: "POST",
                  headers,
                  body: JSON.stringify({ text: "ping" }),
                  signal: controller.signal,
                });
                // 200 → online, 401/403 → reachable but needs auth (degraded)
                if (resp.status === 200) {
                  clearTimeout(t);
                  const j = await resp.json().catch(() => ({}));
                  return sendResponse({ ok: true, summary: j?.summary ?? "", mode: "post-ok" });
                }
                if (resp.status === 401 || resp.status === 403) {
                  clearTimeout(t);
                  return sendResponse({ ok: false, auth: true, error: "auth required" });
                }
              } catch (e) {
                // fall through
              }
            }

            clearTimeout(t);
            if (r.ok) return sendResponse({ ok: true, ...r.json });
            if (r.status === 401 || r.status === 403) {
              // reachable but needs auth
              return sendResponse({ ok: false, auth: true, ...r.json });
            }
            return sendResponse({ ok: false, ...r.json });
          } catch (e) {
            clearTimeout(t);
            return sendResponse({ ok: false, error: e?.message || "unreachable" });
          }
        }

        default:
          return sendResponse({ error: "Unknown message type" });
      }
    } catch (e) {
      console.error("[GhostTab AI Bridge] Error:", e);
      return sendResponse({ error: e?.message || "Network error" });
    }
  })();

  return true; // keep channel open for async response
});

// Optional: clean up on unload (useful in dev hot-reloads)
self.addEventListener?.("unload", () => {
  disableFocusMode();
});

// ─────────────────────────── OPTIONAL EXTRAS ───────────────────────────
chrome.runtime.onInstalled?.addListener(async () => {
  try {
    // Default apiBase on first install; migrate away from localhost in prod
    const { apiBase } = await chrome.storage.local.get(["apiBase"]);
    let base = apiBase || DEFAULT_API_BASE;
    if (!IS_DEV_BUILD && LOCAL_RE.test(base)) base = CLOUD_RUN_BASE;
    await chrome.storage.local.set({ apiBase: base });
  } catch {}

  try {
    chrome.contextMenus?.create({
      id: "ghosttab-analyze",
      title: "GhostTab: Analyze Selection",
      contexts: ["selection"]
    });
  } catch {}
});

chrome.contextMenus?.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "ghosttab-analyze" || !info.selectionText) return;
  try {
    const data = await postJSON("/analyze", { text: info.selectionText });
    const conf = data?.sentiment?.confidence;
    try {
      chrome.notifications.create({
        type: "basic",
        iconUrl: chrome.runtime.getURL("icon.png"),
        title: "GhostTab Analyze",
        message: `Sentiment: ${data?.sentiment?.sentiment} ${
          typeof conf === "number" ? `(${(conf * 100).toFixed(1)}%)` : ""
        }`
      });
    } catch {}
  } catch (e) {
    try {
      chrome.notifications.create({
        type: "basic",
        iconUrl: chrome.runtime.getURL("icon.png"),
        title: "GhostTab Error",
        message: e?.message || "Network error"
      });
    } catch {}
  }
});

chrome.commands?.onCommand.addListener(async (command) => {
  if (!["summarize", "rewrite"].includes(command)) return;
  const tab = await getActiveTab();
  if (!tab?.id || isRestrictedUrl(tab.url)) {
    try {
      chrome.notifications.create({
        type: "basic",
        iconUrl: chrome.runtime.getURL("icon.png"),
        title: "GhostTab",
        message: "This page blocks content access. Try a normal website."
      });
    } catch {}
    return;
  }

  let text = "";
  try {
    const res = await getPageData();
    if (res.ok) text = (res.selection || res.text || "").slice(0, 8000);
  } catch {}
  if (!text) return;

  try {
    if (command === "summarize") {
      await postJSON("/summarize", { text });
      try {
        chrome.notifications.create({
          type: "basic", iconUrl: chrome.runtime.getURL("icon.png"),
          title: "GhostTab", message: "Summary requested."
        });
      } catch {}
    } else if (command === "rewrite") {
      await postJSON("/rewrite", { text });
      try {
        chrome.notifications.create({
          type: "basic", iconUrl: chrome.runtime.getURL("icon.png"),
          title: "GhostTab", message: "Rewrite requested."
        });
      } catch {}
    }
  } catch (e) {
    try {
      chrome.notifications.create({
        type: "basic", iconUrl: chrome.runtime.getURL("icon.png"),
        title: "GhostTab Error", message: e?.message || "Network error"
      });
    } catch {}
  }
});

// Mark file as an ES module (helps some bundlers/linting setups)
export {};
