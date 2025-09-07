// ─────────────────────────── Focus / Tab Limit Logic ───────────────────────────
let tabLimitListener = null;
let lastNotificationTime = 0; // Prevent notification spam
let focusModeEnabled = false;

function showRateLimitedNotification(options) {
  const now = Date.now();
  if (now - lastNotificationTime > 3000) {
    chrome.notifications.create(options);
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
    await chrome.action.setBadgeBackgroundColor({
      color: enforce ? "#D7263D" : "#2D7DD2",
    });
    await chrome.action.setTitle({
      title: focusModeEnabled
        ? `GhostTab Focus: ${tabsCount}/${maxTabs} ${enforce ? "(strict)" : ""}`
        : "GhostTab",
    });
  } catch { /* badge might not be available on all builds */ }
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
          console.log(`[GhostTab] Too many tabs: ${count}/${MAX_TABS}`);

          if (enforce && tab?.id) {
            // Best-effort close; ignore errors if tab already gone
            chrome.tabs.remove(tab.id, () => void chrome.runtime.lastError);
          }

          showRateLimitedNotification({
            type: "basic",
            iconUrl: chrome.runtime.getURL("icon.png"),
            title: "👻 GhostTab Alert",
            message: enforce
              ? `🚫 Too many tabs! Extra tab closed. (${count}/${MAX_TABS})`
              : `⚠️ Tab limit exceeded. You have ${count} tabs open.`,
          });
        } else {
          console.log(
            `[GhostTab] Tab created: ${count}/${MAX_TABS} (enforced: ${enforce})`
          );
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
      chrome.tabs.query({}, (tabs) => updateBadge(tabs.length, MAX_TABS, enforce));
    });
  };
  chrome.tabs.onRemoved.addListener(boundUpdated);
  chrome.tabs.onUpdated.addListener(boundUpdated);
  chrome.tabs.onActivated?.addListener(boundUpdated); // NEW: keep badge fresh on tab switch
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
  chrome.action.setBadgeText({ text: "" }).catch?.(() => {});
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

// ────────────────────────────── AI Bridge (Phase 1) ─────────────────────────────
// Storage-configurable API base. Set in Options page as { apiBase: "https://your-api" }.
// Falls back to localhost during dev.
const DEFAULT_API_BASE = "http://localhost:8000";

function getApiBase() {
  return new Promise((resolve) => {
    chrome.storage.local.get(["apiBase"], (data) => {
      const base = (data?.apiBase || DEFAULT_API_BASE).replace(/\/+$/, "");
      resolve(base);
    });
  });
}

/**
 * postJSON with timeout & retries
 * @param {string} path - e.g. "/summarize"
 * @param {any} body  - JSON payload
 * @param {object} opts - { timeoutMs?: number, retries?: number }
 */
async function postJSON(path, body, opts = {}) {
  const { timeoutMs = 15000, retries = 2 } = opts;
  const apiBase = await getApiBase();
  const url = `${apiBase}${path}`;

  let attempt = 0;
  const backoff = (n) => new Promise((r) => setTimeout(r, 400 * Math.pow(2, n)));

  const manifest = chrome.runtime.getManifest?.();
  const clientVersion = manifest?.version || "0.0.0";

  while (true) {
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-GhostTab-Client": `ext/${clientVersion}`,
        },
        body: JSON.stringify(body || {}),
        signal: controller.signal,
      });

      clearTimeout(t);

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        if (res.status === 401 || res.status === 403) {
          throw new Error(
            `Unauthorized. Check backend key/config. (${res.status}) ${text}`
          );
        }
        if (res.status === 404) {
          throw new Error(
            `Endpoint not found: ${path}. Is the server running the correct route?`
          );
        }
        throw new Error(`API ${path} failed: ${res.status} ${text}`);
      }

      return await res.json();
    } catch (err) {
      clearTimeout(t);
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

// ─────────────────────────── Message Routing ───────────────────────────
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    try {
      switch (msg?.type) {
        case "SUMMARIZE": {
          requireString("text", msg?.payload?.text);
          const data = await postJSON("/summarize", { text: msg.payload.text });
          return sendResponse(data); // { summary: "..." }
        }
        case "REWRITE": {
          requireString("text", msg?.payload?.text);
          const tone = typeof msg?.payload?.tone === "string" ? msg.payload.tone : undefined;
          const data = await postJSON("/rewrite", { text: msg.payload.text, tone });
          return sendResponse(data); // { rewrite: "..." }
        }
        case "TRANSLATE": {
          requireString("text", msg?.payload?.text);
          requireString("to", msg?.payload?.to);
          const data = await postJSON("/translate", {
            text: msg.payload.text,
            to: msg.payload.to,
          });
          return sendResponse(data); // { translated: "..." }
        }
        case "SENTIMENT": {
          requireString("text", msg?.payload?.text);
          const data = await postJSON("/sentiment", { text: msg.payload.text });
          return sendResponse(data); // { sentiment, confidence, ... }
        }
        case "ANALYZE": {
          // one-shot pipeline: ONNX sentiment + GPT summary (if you added /analyze)
          requireString("text", msg?.payload?.text);
          const data = await postJSON("/analyze", { text: msg.payload.text });
          return sendResponse(data); // { sentiment: {...}, summary: "..." }
        }
        case "HEALTH": {
          // Robust GET with timeout
          const apiBase = await getApiBase();
          const controller = new AbortController();
          const t = setTimeout(() => controller.abort(), 5000);
          try {
            const res = await fetch(`${apiBase}/`, { method: "GET", signal: controller.signal });
            clearTimeout(t);
            const json = await res.json().catch(() => ({}));
            json.ok = Boolean(json?.ok);
            return sendResponse(json); // { ok, service, sentiment_model, ... }
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
// Add these only if you update your manifest.json accordingly.

// 1) Right-click “Analyze Selection” (requires "contextMenus" permission)
chrome.runtime.onInstalled?.addListener(() => {
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
    chrome.notifications.create({
      type: "basic",
      iconUrl: chrome.runtime.getURL("icon.png"),
      title: "GhostTab Analyze",
      message: `Sentiment: ${data?.sentiment?.sentiment} ${
        typeof conf === "number" ? `(${(conf * 100).toFixed(1)}%)` : ""
      }`
    });
  } catch (e) {
    chrome.notifications.create({
      type: "basic",
      iconUrl: chrome.runtime.getURL("icon.png"),
      title: "GhostTab Error",
      message: e?.message || "Network error"
    });
  }
});

// 2) Keyboard shortcuts (requires "commands" in manifest)
chrome.commands?.onCommand.addListener(async (command) => {
  if (!["summarize", "rewrite"].includes(command)) return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;

  // Avoid restricted pages
  if (/^chrome:\/\//i.test(tab.url) || /^https?:\/\/chrome\.google\.com\/webstore/i.test(tab.url)) {
    chrome.notifications.create({
      type: "basic",
      iconUrl: chrome.runtime.getURL("icon.png"),
      title: "GhostTab",
      message: "This page blocks content access. Try a normal website."
    });
    return;
  }

  // Grab selection or page text
  let text = "";
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const sel = window.getSelection?.().toString() || "";
        const body = document.body?.innerText || "";
        return (sel || body).slice(0, 8000);
      }
    });
    text = result || "";
  } catch {
    text = "";
  }
  if (!text) return;

  try {
    if (command === "summarize") {
      await postJSON("/summarize", { text });
      chrome.notifications.create({
        type: "basic", iconUrl: chrome.runtime.getURL("icon.png"),
        title: "GhostTab", message: "Summary requested."
      });
    } else if (command === "rewrite") {
      await postJSON("/rewrite", { text });
      chrome.notifications.create({
        type: "basic", iconUrl: chrome.runtime.getURL("icon.png"),
        title: "GhostTab", message: "Rewrite requested."
      });
    }
  } catch (e) {
    chrome.notifications.create({
      type: "basic", iconUrl: chrome.runtime.getURL("icon.png"),
      title: "GhostTab Error", message: e?.message || "Network error"
    });
  }
});
