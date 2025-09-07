// ========== EXISTING ELEMENT HOOKS ==========
const startBtn = document.getElementById("startFocus");
const stopBtn = document.getElementById("stopFocus");
const statusText = document.getElementById("status");
const themeSelect = document.getElementById("theme");
const enforceToggle = document.getElementById("enforceToggle");
const workDropdown = document.getElementById("workDuration");
const breakDropdown = document.getElementById("breakDuration");
const pomoProgress = document.getElementById("pomoProgress");
const ghostMeter = document.getElementById("ghostMeter");
const resetGoalBtn = document.getElementById("resetGoal");
const addSessionBtn = document.getElementById("addSession");
const tabLimitSection = document.getElementById("customTabLimitSection");
const customTabLimitInput = document.getElementById("customTabLimit");
const customGoalInput = document.getElementById("customGoal");
const customGoalSection = document.getElementById("customGoalSection");
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

// ========== AI ELEMENTS (present in your HTML) ==========
const summarizeBtn = document.getElementById("summarizePage");
const rewriteBtn   = document.getElementById("rewriteText");
const sentimentBtn = document.getElementById("sentimentText");
const analyzeBtn   = document.getElementById("analyzeText");
const healthBtn    = document.getElementById("checkHealth");
const aiResult     = document.getElementById("aiResult");

// NEW: Ask + Translate UI (updated IDs per your request)
const askBtn       = document.getElementById("askPageBtn");
const askInput     = document.getElementById("askQuestion");
const translateBtn = document.getElementById("translateBtn");
const translateTo  = document.getElementById("translateTo");

// (Optional legacy selection tools – keep if they exist in your HTML)
const btnAskSelection  = document.getElementById("btnAskSelection");
const askOut           = document.getElementById("askOut");
const btnSummarizeSel  = document.getElementById("btnSummarizeSel");
const sumOut           = document.getElementById("sumOut");
const btnRewriteSel    = document.getElementById("btnRewriteSel");
const rewriteTone      = document.getElementById("rewriteTone");
const rewriteOut       = document.getElementById("rewriteOut");

// API status (if present)
const apiStatusDot   = document.getElementById("apiStatusDot");
const apiStatusLabel = document.getElementById("apiStatusLabel");

// ========== ALWAYS-ON PRO UI ==========
let isProUser = true;
let MAX_TABS = 3;
let dailyGoal = 6;

// Ping API root once to color the status dot
(async () => {
  if (!apiStatusDot || !apiStatusLabel) return;
  try {
    const res = await new Promise(r => chrome.runtime.sendMessage({ type: "HEALTH" }, r));
    if (res?.ok) {
      apiStatusDot.style.background = "#16a34a";
      apiStatusLabel.textContent = "online";
      apiStatusLabel.style.color = "#16a34a";
    } else {
      apiStatusDot.style.background = "#f59e0b";
      apiStatusLabel.textContent = "degraded";
      apiStatusLabel.style.color = "#f59e0b";
    }
  } catch {
    apiStatusDot.style.background = "#ef4444";
    apiStatusLabel.textContent = "offline";
    apiStatusLabel.style.color = "#ef4444";
  }
})();

// ========== INIT STORAGE/UI ==========
chrome.storage.local.get(
  ["focusMode", "ghostTheme", "enforceTabs", "customTabLimit", "customDailyGoal"],
  (data) => {
    statusText.innerText = data.focusMode ? "🎯 Focus Mode: ON" : "🎯 Focus Mode: OFF";

    if (data.ghostTheme) {
      document.body.classList.add(data.ghostTheme);
      themeSelect.value = data.ghostTheme;
    }

    enforceToggle.checked = data.enforceTabs || false;

    if (data.customTabLimit) {
      customTabLimitInput.value = data.customTabLimit;
      MAX_TABS = parseInt(data.customTabLimit);
    }
    if (data.customDailyGoal) {
      customGoalInput.value = data.customDailyGoal;
      dailyGoal = parseInt(data.customDailyGoal);
    }

    updateProUI();
  }
);

function updateProUI() {
  tabLimitSection.style.display = "block";
  customGoalSection.style.display = "block";
  historySection.style.display = "block";
  if (historyDisplay) historyDisplay.style.display = "none";
  if (toggleHistoryBtn) toggleHistoryBtn.innerText = "📊 Show History";
  if (timerDropdownSection) timerDropdownSection.style.display = "none";
  if (customTimerInputSection) customTimerInputSection.style.display = "block";
}

// Theme
themeSelect.addEventListener("change", (e) => {
  const selected = e.target.value;
  document.body.classList.remove("pastel", "midnight", "mint", "cyber");
  document.body.classList.add(selected);
  chrome.storage.local.set({ ghostTheme: selected });
});

enforceToggle.addEventListener("change", (e) => {
  chrome.storage.local.set({ enforceTabs: e.target.checked });
});

// ========== FOCUS MODE ==========
startBtn.addEventListener("click", () => {
  chrome.tabs.query({}, (tabs) => {
    chrome.storage.local.get("enforceTabs", ({ enforceTabs }) => {
      if (enforceTabs && tabs.length > MAX_TABS) {
        alert(`❌ You have ${tabs.length} tabs open. Please close some before focusing.`);
      } else {
        chrome.storage.local.set({ focusMode: true });
        statusText.innerText = "🎯 Focus Mode: ON";
      }
    });
  });
});

stopBtn.addEventListener("click", () => {
  chrome.storage.local.set({ focusMode: false });
  statusText.innerText = "🎯 Focus Mode: OFF";
});

// Custom inputs
customTabLimitInput.addEventListener("input", (e) => {
  const val = parseInt(e.target.value);
  if (!isNaN(val)) {
    MAX_TABS = val;
    chrome.storage.local.set({ customTabLimit: val });
  }
});

customGoalInput.addEventListener("input", (e) => {
  const goal = parseInt(e.target.value);
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
    pomoProgress.innerText = `🍅 Pomodoros Today: ${count} / ${dailyGoal}`;
    let ghost = "😴";
    if (count >= 1 && count <= 2) ghost = "🙀";
    else if (count >= 3 && count <= 4) ghost = "😊";
    else if (count >= 5) ghost = "😈";
    ghostMeter.innerText = ghost;
  });
}

resetGoalBtn.addEventListener("click", () => {
  const today = new Date().toISOString().split("T")[0];
  chrome.storage.local.get(["pomoLog"], ({ pomoLog }) => {
    pomoLog = pomoLog || {};
    pomoLog[today] = 0;
    chrome.storage.local.set({ pomoLog }, updatePomoProgressDisplay);
  });
});

addSessionBtn.addEventListener("click", () => {
  const today = new Date().toISOString().split("T")[0];
  chrome.storage.local.get(["pomoLog"], ({ pomoLog }) => {
    pomoLog = pomoLog || {};
    pomoLog[today] = (pomoLog[today] || 0) + 1;
    chrome.storage.local.set({ pomoLog }, updatePomoProgressDisplay);
  });
});

toggleHistoryBtn.addEventListener("click", () => {
  const showing = historyDisplay.style.display === "block";
  if (showing) {
    historyDisplay.style.display = "none";
    toggleHistoryBtn.innerText = "📊 Show History";
  } else {
    loadPomodoroHistory();
    historyDisplay.style.display = "block";
    toggleHistoryBtn.innerText = "🙈 Hide History";
  }
});

function loadPomodoroHistory() {
  chrome.storage.local.get("pomoLog", ({ pomoLog }) => {
    if (!pomoLog || Object.keys(pomoLog).length === 0) {
      historyDisplay.innerHTML = "<p>No Pomodoro history yet. 🍅</p>";
      return;
    }
    const sorted = Object.entries(pomoLog)
      .sort((a, b) => new Date(b[0]) - new Date(a[0]))
      .slice(0, 7)
      .map(([date, count]) => `📅 <strong>${date}</strong>: ${"🍅".repeat(count)} (${count})`);
    historyDisplay.innerHTML = `<div>${sorted.join("<br>")}</div>`;
  });
}

// ========== TIMER ==========
let timerInterval;
let timeRemaining = 30 * 60;
let isBreak = false;
let isRunning = false;

function updateDisplay() {
  const minutes = Math.floor(timeRemaining / 60).toString().padStart(2, "0");
  const seconds = (timeRemaining % 60).toString().padStart(2, "0");
  timerDisplay.innerText = `⏳ ${minutes}:${seconds}`;
  timerMode.innerText = isBreak ? "☕ Break" : "💼 Work";
}

function getDurationMinutes() {
  return isBreak ? parseInt(customBreakInput.value) : parseInt(customWorkInput.value);
}

function startPomodoro() {
  if (isRunning) return;
  isRunning = true;

  const now = Date.now();
  const duration = getDurationMinutes() * 60;
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
    const { pomodoroStart, pomodoroDuration, pomodoroIsBreak } = data;
    if (!pomodoroStart || !pomodoroDuration) {
      clearInterval(timerInterval);
      return;
    }
    const now = Date.now();
    const elapsed = Math.floor((now - pomodoroStart) / 1000);
    const remaining = pomodoroDuration - elapsed;

    isBreak = pomodoroIsBreak;
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

      chrome.notifications.create({
        type: "basic",
        iconUrl: chrome.runtime.getURL("icon.png"),
        title: "GhostTab Timer",
        message: isBreak ? "☕ Time for a break!" : "🔥 Back to work!"
      });

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
  timeRemaining = getDurationMinutes() * 60;
  chrome.storage.local.remove(["pomodoroStart", "pomodoroDuration", "pomodoroIsBreak"]);
  updateDisplay();
}

startTimer.addEventListener("click", startPomodoro);
pauseTimer.addEventListener("click", pausePomodoro);
resetTimer.addEventListener("click", resetPomodoro);

// On load, resume if mid-session
chrome.storage.local.get(["pomodoroStart", "pomodoroDuration", "pomodoroIsBreak"], (data) => {
  if (data.pomodoroStart && data.pomodoroDuration) {
    isBreak = data.pomodoroIsBreak;
    isRunning = true;
    timerInterval = setInterval(updateTimerFromStorage, 1000);
  } else {
    updateDisplay();
  }
});
updatePomoProgressDisplay();

// ========== AI HELPERS ==========
function setAIBusy(busy) {
  if (!aiResult) return;
  aiResult.toggleAttribute("disabled", !!busy);
  if (summarizeBtn) summarizeBtn.disabled = busy;
  if (rewriteBtn)   rewriteBtn.disabled   = busy;
  if (sentimentBtn) sentimentBtn.disabled = busy;
  if (analyzeBtn)   analyzeBtn.disabled   = busy;
  if (askBtn)       askBtn.disabled       = busy;
  if (translateBtn) translateBtn.disabled = busy;
}
function showResult(text) {
  if (aiResult) aiResult.value = text;
}

// Helpers for fetching page text/selection from the active tab
function isRestrictedUrl(url = "") {
  return /^chrome:\/\//i.test(url) || /^https?:\/\/chrome\.google\.com\/webstore/i.test(url);
}

async function getActiveTabTextAndSelection() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id || isRestrictedUrl(tab.url)) {
    return { ok: false, error: "Restricted page; cannot access content.", text: "", selection: "" };
  }

  // 1) Try content script first
  try {
    const res = await chrome.tabs.sendMessage(tab.id, { type: "GHOSTTAB_GET_PAGE_TEXT" });
    if (res && typeof res === "object" && ("ok" in res)) {
      if (res.ok && (res.text?.trim()?.length || res.selection?.trim()?.length)) return res;
    }
  } catch {
    // ignore; we'll fall back to background
  }

  // 2) Fallback: ask background to read the page via executeScript
  try {
    const res = await new Promise(r => chrome.runtime.sendMessage({ type: "GET_PAGE_TEXT" }, r));
    if (res && res.ok && (res.text?.trim()?.length || res.selection?.trim()?.length)) return res;
    return res || { ok: false, error: "Background fallback failed.", text: "", selection: "" };
  } catch (e) {
    return { ok: false, error: String(e || "Fallback failed"), text: "", selection: "" };
  }
}

function warnIfRestricted(info) {
  if (!info?.ok) {
    showResult(info?.error || "Cannot access this page.");
    return true;
  }
  if (!info.text || !info.text.trim()) {
    showResult("No readable text found on this page.");
    return true;
  }
  return false;
}

// ========== AI: BASIC BUTTONS (updated to pass text properly) ==========
summarizeBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Summarizing…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return setAIBusy(false);
    chrome.runtime.sendMessage(
      { type: "ASK_PAGE", payload: { text: info.text, question: "Summarize this page." } },
      (res) => {
        if (res?.answer) showResult(res.answer);
        else if (res?.summary) showResult(res.summary);
        else showResult(res?.error || "No response");
        setAIBusy(false);
      }
    );
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
    setAIBusy(false);
  }
});

rewriteBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Rewriting selection…");
    // Keep using background's selection path so it prefers selected text
    chrome.runtime.sendMessage({ type: "REWRITE_SELECTION", payload: { tone: null } }, (res) => {
      showResult(res?.rewrite || res?.error || "No response");
      setAIBusy(false);
    });
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
    setAIBusy(false);
  }
});

sentimentBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Detecting sentiment…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return setAIBusy(false);
    const payload = (info.selection && info.selection.trim().length >= 1) ? info.selection : info.text;
    chrome.runtime.sendMessage({ type: "SENTIMENT", payload: { text: payload } }, (res) => {
      if (res?.error) return showResult(`❌ ${res.error}`);
      const s = res?.sentiment || {};
      const label = s.sentiment ?? s.label ?? "(unknown)";
      const conf  = typeof s.confidence === "number" ? (s.confidence * 100).toFixed(2) + "%" : "";
      showResult(`Sentiment: ${label}${conf ? ` (${conf})` : ""}`);
      setAIBusy(false);
    });
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
    setAIBusy(false);
  }
});

analyzeBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Analyzing…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return setAIBusy(false);
    const payload = (info.selection && info.selection.trim().length >= 1) ? info.selection : info.text;
    chrome.runtime.sendMessage({ type: "ANALYZE", payload: { text: payload } }, (res) => {
      if (res?.error) return showResult(`❌ ${res.error}`);
      const s = res?.sentiment || {};
      const label = s.sentiment ?? s.label ?? "(unknown)";
      const conf  = typeof s.confidence === "number" ? (s.confidence * 100).toFixed(2) + "%" : "";
      const summary = res?.summary || "(no summary)";
      showResult(`Sentiment: ${label}${conf ? ` (${conf})` : ""}\n\nSummary:\n${summary}`);
      setAIBusy(false);
    });
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
    setAIBusy(false);
  }
});

healthBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Checking API…");
    chrome.runtime.sendMessage({ type: "HEALTH" }, (res) => {
      showResult(JSON.stringify(res, null, 2));
      setAIBusy(false);
    });
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
    setAIBusy(false);
  }
});

// ========== NEW: Ask Page (+ top_k happens in background) ==========
askBtn?.addEventListener("click", async () => {
  setAIBusy(true);
  const q = (askInput?.value || "").trim();
  if (!q) { showResult("Please type a question for Ask Page."); return setAIBusy(false); }
  const info = await getActiveTabTextAndSelection();
  if (warnIfRestricted(info)) return setAIBusy(false);
  chrome.runtime.sendMessage(
    { type: "ASK_PAGE", payload: { text: info.text, question: q } },
    (res) => {
      if (res?.error) return showResult(`❌ ${res.error}`);
      const src = (res?.sources || [])
        .map(s => `[#${s.rank} idx=${s.chunk_idx} ${s.start}-${s.end}] ${s.preview || ""}`)
        .join("\n");
      showResult(`Answer:\n${res?.answer || "(no answer)"}\n\nSources:\n${src}`);
      setAIBusy(false);
    }
  );
});

// ========== NEW: Translate (selection preferred, fallback to page) ==========
translateBtn?.addEventListener("click", async () => {
  setAIBusy(true);
  const lang = (translateTo?.value || "").trim();
  if (!lang) { showResult("Enter a target language (e.g., es, fr, ko)."); return setAIBusy(false); }
  const info = await getActiveTabTextAndSelection();
  if (warnIfRestricted(info)) return setAIBusy(false);
  const payload = info.selection?.trim() ? info.selection : info.text;
  chrome.runtime.sendMessage({ type: "TRANSLATE", payload: { text: payload, to: lang } }, (res) => {
    showResult(res?.translated || res?.error || "No response");
    setAIBusy(false);
  });
});

// ========== OPTIONAL: Selection tools (unchanged) ==========
btnAskSelection?.addEventListener("click", async () => {
  const q = (askInput?.value || "").trim();
  if (!q) return (askOut.textContent = "Enter a question.");
  askOut.textContent = "Asking (selection)…";
  chrome.runtime.sendMessage({ type: "ASK_SELECTION", payload: { question: q } }, (res) => {
    if (res?.error) return (askOut.textContent = `❌ ${res.error}`);
    const lines = (res?.sources || []).map((s) => `[${s.rank}] chunk ${s.chunk_idx} — ${s.preview}`);
    askOut.textContent = `${res?.answer || "(no answer)"}\n\nSources:\n${lines.join("\n")}`;
  });
});

btnSummarizeSel?.addEventListener("click", async () => {
  sumOut.textContent = "Summarizing selection…";
  chrome.runtime.sendMessage({ type: "SUMMARIZE_SELECTION" }, (res) => {
    sumOut.textContent = res?.summary || res?.error || "(no summary)";
  });
});

btnRewriteSel?.addEventListener("click", async () => {
  rewriteOut.textContent = "Rewriting selection…";
  const tone = (rewriteTone?.value || "").trim() || null;
  chrome.runtime.sendMessage({ type: "REWRITE_SELECTION", payload: { tone } }, (res) => {
    rewriteOut.textContent = res?.rewrite || res?.error || "(no rewrite)";
  });
});
