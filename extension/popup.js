// ================== STORAGE HELPERS & MIGRATION ==================
async function getLocal(keys) { return chrome.storage.local.get(keys); }
async function setLocal(obj)  { return chrome.storage.local.set(obj); }
async function removeLocal(keys) { return chrome.storage.local.remove(keys); }

// One-time migration of apiKey from sync -> local (privacy, reliability)
(async () => {
  try {
    const syncData  = await chrome.storage.sync.get(['apiKey']);
    const localData = await getLocal(['apiKey']);
    if (syncData.apiKey && !localData.apiKey) {
      await setLocal({ apiKey: syncData.apiKey });
      await chrome.storage.sync.remove(['apiKey']);
    }
  } catch {}
})();

// ================== ELEMENT HOOKS ==================
const startBtn = document.getElementById("startFocus");
const stopBtn = document.getElementById("stopFocus");
const statusText = document.getElementById("status");
const themeSelect = document.getElementById("theme");
const enforceToggle = document.getElementById("enforceToggle");
// Old dropdowns kept as optional (hidden in UI)
const workDropdown = document.getElementById("workDuration");
const breakDropdown = document.getElementById("breakDuration");

const pomoProgress = document.getElementById("pomoProgress");
const ghostMeter = document.getElementById("ghostMeter");

const resetGoalBtn = document.getElementById("resetGoal");
const addSessionBtn = document.getElementById("addSession");

const tabLimitSection = document.getElementById("customTabLimitSection");
const customTabLimitInput = document.getElementById("customTabLimit");

const customGoalSection = document.getElementById("customGoalSection");
const customGoalInput = document.getElementById("customGoal");

const toggleHistoryBtn = document.getElementById("toggleHistoryBtn");
const historyDisplay = document.getElementById("historyDisplay");
const historySection = document.getElementById("historySection");

const timerDropdownSection = document.getElementById("defaultTimerSection");
const customTimerInputSection = document.getElementById("customTimerSection");
const customWorkInput = document.getElementById("customWork");
const customBreakInput = document.getElementById("customBreak");

const timerDisplay = document.getElementById("timerDisplay");
const timerMode = document.getElementById("timerMode");
const startTimer = document.getElementById("startTimer");
const pauseTimer = document.getElementById("pauseTimer");
const resetTimer = document.getElementById("resetTimer");

// AI
const summarizeBtn = document.getElementById("summarizePage");
const rewriteBtn   = document.getElementById("rewriteText");
const sentimentBtn = document.getElementById("sentimentText");
const analyzeBtn   = document.getElementById("analyzeText");
const healthBtn    = document.getElementById("checkHealth");
const aiResult     = document.getElementById("aiResult");
const askBtn       = document.getElementById("askPageBtn");
const askInput     = document.getElementById("askQuestion");
const translateBtn = document.getElementById("translateBtn");
const translateTo  = document.getElementById("translateTo");

// Optional legacy selection tools (no-op if missing)
const btnAskSelection  = document.getElementById("btnAskSelection");
const askOut           = document.getElementById("askOut");
const btnSummarizeSel  = document.getElementById("btnSummarizeSel");
const sumOut           = document.getElementById("sumOut");
const btnRewriteSel    = document.getElementById("btnRewriteSel");
const rewriteTone      = document.getElementById("rewriteTone");
const rewriteOut       = document.getElementById("rewriteOut");

// API status (dot + label)
const apiStatusCard  = document.getElementById("apiStatus");
const apiStatusDot   = document.getElementById("apiStatusDot");
const apiStatusLabel = document.getElementById("apiStatusLabel");

// ================== CONSTANTS / STATE ==================
let MAX_TABS = 3;
let dailyGoal = 6;

// ================== UTILITIES ==================
function setStatusDot(state /* ok|warn|err */, label) {
  if (!apiStatusCard || !apiStatusDot || !apiStatusLabel) return;
  apiStatusCard.classList.remove("status-ok", "status-warn", "status-err");
  if (state === "ok")   apiStatusCard.classList.add("status-ok");
  if (state === "warn") apiStatusCard.classList.add("status-warn");
  if (state === "err")  apiStatusCard.classList.add("status-err");
  apiStatusLabel.textContent = label || (state === "ok" ? "online" : state === "warn" ? "degraded" : "offline");
}

function isRestrictedUrl(url = "") {
  return /^chrome:\/\//i.test(url) || /^https?:\/\/chrome\.google\.com\/webstore/i.test(url);
}

function setAIBusy(busy) {
  aiResult?.toggleAttribute("disabled", !!busy);
  if (summarizeBtn) summarizeBtn.disabled = busy;
  if (rewriteBtn)   rewriteBtn.disabled   = busy;
  if (sentimentBtn) sentimentBtn.disabled = busy;
  if (analyzeBtn)   analyzeBtn.disabled   = busy;
  if (askBtn)       askBtn.disabled       = busy;
  if (translateBtn) translateBtn.disabled = busy;
}

function showResult(text) { if (aiResult) aiResult.value = text; }

function sendBg(message) {
  // Safe wrapper around background messaging
  return new Promise((resolve) => {
    try { chrome.runtime.sendMessage(message, (res) => resolve(res || null)); }
    catch { resolve(null); }
  });
}

// ================== API HEALTH ON LOAD ==================
(async () => {
  if (!apiStatusDot || !apiStatusLabel) return;
  setStatusDot("warn", "checking…");
  const res = await sendBg({ type: "HEALTH" });
  if (res?.ok) setStatusDot("ok", "online");
  else if (res) setStatusDot("warn", "degraded");
  else setStatusDot("err", "offline");
})();

// ================== INIT STORAGE/UI ==================
chrome.storage.local.get(
  ["focusMode", "ghostTheme", "enforceTabs", "customTabLimit", "customDailyGoal"],
  (data) => {
    if (statusText) statusText.innerText = data.focusMode ? "🎯 Focus Mode: ON" : "🎯 Focus Mode: OFF";

    if (data.ghostTheme) {
      document.body.classList.remove("pastel", "midnight", "mint", "cyber");
      document.body.classList.add(data.ghostTheme);
      if (themeSelect) themeSelect.value = data.ghostTheme;
    }

    if (enforceToggle) enforceToggle.checked = data.enforceTabs || false;

    if (data.customTabLimit && customTabLimitInput) {
      customTabLimitInput.value = data.customTabLimit;
      MAX_TABS = parseInt(data.customTabLimit, 10);
    }
    if (data.customDailyGoal && customGoalInput) {
      customGoalInput.value = data.customDailyGoal;
      dailyGoal = parseInt(data.customDailyGoal, 10);
    }

    updateProUI();
    updatePomoProgressDisplay();
  }
);

function updateProUI() {
  if (tabLimitSection) tabLimitSection.style.display = "block";
  if (customGoalSection) customGoalSection.style.display = "block";
  if (historySection) historySection.style.display = "block";
  if (historyDisplay) historyDisplay.style.display = "none";
  if (toggleHistoryBtn) toggleHistoryBtn.innerText = "📊 Show History";
  if (timerDropdownSection) timerDropdownSection.style.display = "none";
  if (customTimerInputSection) customTimerInputSection.style.display = "block";
}

// Theme
themeSelect?.addEventListener("change", (e) => {
  const selected = e.target.value;
  document.body.classList.remove("pastel", "midnight", "mint", "cyber");
  document.body.classList.add(selected);
  chrome.storage.local.set({ ghostTheme: selected });
});

enforceToggle?.addEventListener("change", (e) => {
  chrome.storage.local.set({ enforceTabs: e.target.checked });
});

// ================== FOCUS MODE ==================
startBtn?.addEventListener("click", () => {
  chrome.tabs.query({}, (tabs) => {
    chrome.storage.local.get("enforceTabs", ({ enforceTabs }) => {
      if (enforceTabs && tabs.length > MAX_TABS) {
        try {
          chrome.notifications.create({
            type: "basic",
            iconUrl: chrome.runtime.getURL("icon.png"),
            title: "GhostTab",
            message: `❌ You have ${tabs.length} tabs open. Close some before starting focus.`
          });
        } catch {}
      } else {
        chrome.storage.local.set({ focusMode: true });
        if (statusText) statusText.innerText = "🎯 Focus Mode: ON";
      }
    });
  });
});

stopBtn?.addEventListener("click", () => {
  chrome.storage.local.set({ focusMode: false });
  if (statusText) statusText.innerText = "🎯 Focus Mode: OFF";
});

// Custom inputs
customTabLimitInput?.addEventListener("input", (e) => {
  const val = parseInt(e.target.value, 10);
  if (!isNaN(val)) {
    MAX_TABS = val;
    chrome.storage.local.set({ customTabLimit: val });
  }
});

customGoalInput?.addEventListener("input", (e) => {
  const goal = parseInt(e.target.value, 10);
  if (!isNaN(goal)) {
    dailyGoal = goal;
    chrome.storage.local.set({ customDailyGoal: goal });
    updatePomoProgressDisplay();
  }
});

function updatePomoProgressDisplay() {
  const today = new Date().toISOString().split("T")[0];
  chrome.storage.local.get(["pomoLog"], ({ pomoLog }) => {
    const count = pomoLog?.[today] || 0;
    if (pomoProgress) pomoProgress.innerText = `🍅 Pomodoros Today: ${count} / ${dailyGoal}`;
    let ghost = "😴";
    if (count >= 1 && count <= 2) ghost = "🙀";
    else if (count >= 3 && count <= 4) ghost = "😊";
    else if (count >= 5) ghost = "😈";
    if (ghostMeter) ghostMeter.innerText = ghost;
  });
}

resetGoalBtn?.addEventListener("click", () => {
  const today = new Date().toISOString().split("T")[0];
  chrome.storage.local.get(["pomoLog"], ({ pomoLog }) => {
    pomoLog = pomoLog || {};
    pomoLog[today] = 0;
    chrome.storage.local.set({ pomoLog }, updatePomoProgressDisplay);
  });
});

addSessionBtn?.addEventListener("click", () => {
  const today = new Date().toISOString().split("T")[0];
  chrome.storage.local.get(["pomoLog"], ({ pomoLog }) => {
    pomoLog = pomoLog || {};
    pomoLog[today] = (pomoLog[today] || 0) + 1;
    chrome.storage.local.set({ pomoLog }, updatePomoProgressDisplay);
  });
});

toggleHistoryBtn?.addEventListener("click", () => {
  const showing = historyDisplay?.style.display === "block";
  if (!historyDisplay || !toggleHistoryBtn) return;
  if (showing) {
    historyDisplay.style.display = "none";
    toggleHistoryBtn.innerText = "📊 Show History";
    toggleHistoryBtn.setAttribute("aria-expanded", "false");
  } else {
    loadPomodoroHistory();
    historyDisplay.style.display = "block";
    toggleHistoryBtn.innerText = "🙈 Hide History";
    toggleHistoryBtn.setAttribute("aria-expanded", "true");
  }
});

function loadPomodoroHistory() {
  chrome.storage.local.get("pomoLog", ({ pomoLog }) => {
    if (!historyDisplay) return;
    if (!pomoLog || Object.keys(pomoLog).length === 0) {
      historyDisplay.innerHTML = "<p>No Pomodoro history yet. 🍅</p>";
      return;
    }
    const sorted = Object.entries(pomoLog)
      .sort((a, b) => new Date(b[0]) - new Date(a[0]))
      .slice(0, 7)
      .map(([date, count]) => `📅 <strong>${date}</strong>: ${"🍅".repeat(Math.min(count, 20))} (${count})`);
    historyDisplay.innerHTML = `<div>${sorted.join("<br>")}</div>`;
  });
}

// ================== TIMER ==================
let timerInterval;
let timeRemaining = 30 * 60;
let isBreak = false;
let isRunning = false;

function updateDisplay() {
  if (!timerDisplay || !timerMode) return;
  const minutes = Math.floor(timeRemaining / 60).toString().padStart(2, "0");
  const seconds = (timeRemaining % 60).toString().padStart(2, "0");
  timerDisplay.innerText = `⏳ ${minutes}:${seconds}`;
  timerMode.innerText = isBreak ? "☕ Break" : "💼 Work";
}

function getDurationMinutes() {
  const work = parseInt(customWorkInput?.value || "30", 10);
  const brk  = parseInt(customBreakInput?.value || "5", 10);
  return isBreak ? brk : work;
}

function startPomodoro() {
  if (isRunning) return;
  isRunning = true;

  const now = Date.now();
  const duration = Math.max(1, getDurationMinutes()) * 60;
  timeRemaining = duration;

  chrome.storage.local.set({
    pomodoroStart: now,
    pomodoroDuration: duration,
    pomodoroIsBreak: isBreak
  });

  updateDisplay();
  timerInterval = setInterval(updateTimerFromStorage, 1000);
}

function updateTimerFromStorage() {
  chrome.storage.local.get(["pomodoroStart", "pomodoroDuration", "pomodoroIsBreak"], (data) => {
    const { pomodoroStart, pomodoroDuration, pomodoroIsBreak } = data || {};
    if (!pomodoroStart || !pomodoroDuration) {
      clearInterval(timerInterval);
      return;
    }
    const now = Date.now();
    const elapsed = Math.floor((now - pomodoroStart) / 1000);
    const remaining = pomodoroDuration - elapsed;

    isBreak = !!pomodoroIsBreak;
    timeRemaining = remaining;

    if (remaining <= 0) {
      clearInterval(timerInterval);
      isRunning = false;

      if (!isBreak) {
        const todayKey = new Date().toISOString().split("T")[0];
        chrome.storage.local.get(["pomoLog"], ({ pomoLog }) => {
          pomoLog = pomoLog || {};
          pomoLog[todayKey] = (pomoLog[todayKey] || 0) + 1;
          chrome.storage.local.set({ pomoLog }, updatePomoProgressDisplay);
        });
      }

      isBreak = !isBreak;
      startPomodoro();

      try {
        chrome.notifications.create({
          type: "basic",
          iconUrl: chrome.runtime.getURL("icon.png"),
          title: "GhostTab Timer",
          message: isBreak ? "☕ Time for a break!" : "🔥 Back to work!"
        });
      } catch {}

      return;
    }

    updateDisplay();
  });
}

function pausePomodoro() {
  clearInterval(timerInterval);
  isRunning = false;
}

function resetPomodoro() {
  clearInterval(timerInterval);
  isRunning = false;
  isBreak = false;
  timeRemaining = Math.max(1, getDurationMinutes()) * 60;
  chrome.storage.local.remove(["pomodoroStart", "pomodoroDuration", "pomodoroIsBreak"]);
  updateDisplay();
}

startTimer?.addEventListener("click", startPomodoro);
pauseTimer?.addEventListener("click", pausePomodoro);
resetTimer?.addEventListener("click", resetPomodoro);

// On load, resume if mid-session
chrome.storage.local.get(["pomodoroStart", "pomodoroDuration", "pomodoroIsBreak"], (data) => {
  if (data?.pomodoroStart && data?.pomodoroDuration) {
    isBreak = !!data.pomodoroIsBreak;
    isRunning = true;
    timerInterval = setInterval(updateTimerFromStorage, 1000);
  } else {
    updateDisplay();
  }
});

// ================== PAGE TEXT/SELECTION HELPERS ==================
async function getActiveTabTextAndSelection() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id || isRestrictedUrl(tab.url)) {
    return { ok: false, error: "Restricted page; cannot access content.", text: "", selection: "" };
  }
  // Prefer content script
  try {
    const res = await chrome.tabs.sendMessage(tab.id, { type: "GHOSTTAB_GET_PAGE_TEXT" });
    if (res && typeof res === "object" && ("ok" in res)) {
      if (res.ok && (res.text?.trim()?.length || res.selection?.trim()?.length)) return res;
    }
  } catch {}
  // Fallback to background
  try {
    const res = await sendBg({ type: "GET_PAGE_TEXT" });
    if (res && res.ok && (res.text?.trim()?.length || res.selection?.trim()?.length)) return res;
    return res || { ok: false, error: "Background fallback failed.", text: "", selection: "" };
  } catch (e) {
    return { ok: false, error: String(e || "Fallback failed"), text: "", selection: "" };
  }
}
function warnIfRestricted(info) {
  if (!info?.ok) { showResult(info?.error || "Cannot access this page."); return true; }
  if (!info.text || !info.text.trim()) { showResult("No readable text found on this page."); return true; }
  return false;
}

// ================== AI ACTIONS ==================
summarizeBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Summarizing…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return;
    const res = await sendBg({ type: "ASK_PAGE", payload: { text: info.text, question: "Summarize this page." } });
    if (res?.answer) showResult(res.answer);
    else if (res?.summary) showResult(res.summary);
    else showResult(res?.error || "No response");
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
  } finally { setAIBusy(false); }
});

rewriteBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Rewriting selection…");
    const res = await sendBg({ type: "REWRITE_SELECTION", payload: { tone: null } });
    showResult(res?.rewrite || res?.error || "No response");
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
  } finally { setAIBusy(false); }
});

sentimentBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Detecting sentiment…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return;
    const payload = (info.selection && info.selection.trim().length >= 1) ? info.selection : info.text;
    const res = await sendBg({ type: "SENTIMENT", payload: { text: payload } });
    if (res?.error) return showResult(`❌ ${res.error}`);
    const s = res?.sentiment || {};
    const label = s.sentiment ?? s.label ?? "(unknown)";
    const conf  = typeof s.confidence === "number" ? (s.confidence * 100).toFixed(2) + "%" : "";
    showResult(`Sentiment: ${label}${conf ? ` (${conf})` : ""}`);
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
  } finally { setAIBusy(false); }
});

analyzeBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Analyzing…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return;
    const payload = (info.selection && info.selection.trim().length >= 1) ? info.selection : info.text;
    const res = await sendBg({ type: "ANALYZE", payload: { text: payload } });
    if (res?.error) return showResult(`❌ ${res.error}`);
    const s = res?.sentiment || {};
    const label = s.sentiment ?? s.label ?? "(unknown)";
    const conf  = typeof s.confidence === "number" ? (s.confidence * 100).toFixed(2) + "%" : "";
    const summary = res?.summary || "(no summary)";
    showResult(`Sentiment: ${label}${conf ? ` (${conf})` : ""}\n\nSummary:\n${summary}`);
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
  } finally { setAIBusy(false); }
});

healthBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Checking API…");
    const res = await sendBg({ type: "HEALTH" });
    if (!res) {
      setStatusDot("err", "offline");
      return showResult("Background not available. Make sure background.js registers listeners.");
    }
    if (res.ok) setStatusDot("ok", "online");
    else setStatusDot("warn", "degraded");
    showResult(JSON.stringify(res, null, 2));
  } catch (e) {
    setStatusDot("err", "offline");
    showResult(`Error: ${e?.message || e}`);
  } finally { setAIBusy(false); }
});

askBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    const q = (askInput?.value || "").trim();
    if (!q) { showResult("Please type a question for Ask Page."); return; }
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return;
    const res = await sendBg({ type: "ASK_PAGE", payload: { text: info.text, question: q } });
    if (res?.error) return showResult(`❌ ${res.error}`);
    const src = (res?.sources || [])
      .map(s => `[#${s.rank} idx=${s.chunk_idx} ${s.start}-${s.end}] ${s.preview || ""}`)
      .join("\n");
    showResult(`Answer:\n${res?.answer || "(no answer)"}\n\nSources:\n${src}`);
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
  } finally { setAIBusy(false); }
});

translateBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    const lang = (translateTo?.value || "").trim();
    if (!lang) return showResult("Enter a target language (e.g., es, fr, ko).");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return;
    const payload = info.selection?.trim() ? info.selection : info.text;
    const res = await sendBg({ type: "TRANSLATE", payload: { text: payload, to: lang } });
    showResult(res?.translated || res?.error || "No response");
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
  } finally { setAIBusy(false); }
});

// ================== LEGACY SELECTION TOOLS (if present) ==================
btnAskSelection?.addEventListener("click", async () => {
  const q = (askInput?.value || "").trim();
  if (!q) return (askOut.textContent = "Enter a question.");
  askOut.textContent = "Asking (selection)…";
  const res = await sendBg({ type: "ASK_SELECTION", payload: { question: q } });
  if (res?.error) return (askOut.textContent = `❌ ${res.error}`);
  const lines = (res?.sources || []).map((s) => `[${s.rank}] chunk ${s.chunk_idx} — ${s.preview}`);
  askOut.textContent = `${res?.answer || "(no answer)"}\n\nSources:\n${lines.join("\n")}`;
});

btnSummarizeSel?.addEventListener("click", async () => {
  sumOut.textContent = "Summarizing selection…";
  const res = await sendBg({ type: "SUMMARIZE_SELECTION" });
  sumOut.textContent = res?.summary || res?.error || "(no summary)";
});

btnRewriteSel?.addEventListener("click", async () => {
  rewriteOut.textContent = "Rewriting selection…";
  const tone = (rewriteTone?.value || "").trim() || null;
  const res = await sendBg({ type: "REWRITE_SELECTION", payload: { tone } });
  rewriteOut.textContent = res?.rewrite || res?.error || "(no rewrite)";
});
