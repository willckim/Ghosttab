// Elements
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

// --- AI Feature Elements ---
const summarizeBtn = document.getElementById("summarizePage");
const rewriteBtn = document.getElementById("rewriteText");
const aiResult = document.getElementById("aiResult");

// NEW: optional extra buttons (add these to popup.html if you want)
const sentimentBtn = document.getElementById("sentimentText");
const analyzeBtn  = document.getElementById("analyzeText");
const healthBtn   = document.getElementById("checkHealth");

// NEW: API status dot/label (exists if you added them in HTML)
const apiStatusDot   = document.getElementById("apiStatusDot");
const apiStatusLabel = document.getElementById("apiStatusLabel");

// Always-on "Pro"
let isProUser = true;
let MAX_TABS = 3;
let dailyGoal = 6;

// NEW: ping background HEALTH once on load to set dot
(async () => {
  if (!apiStatusDot || !apiStatusLabel) return;
  try {
    const res = await new Promise(r => chrome.runtime.sendMessage({ type: "HEALTH" }, r));
    if (res?.ok) {
      apiStatusDot.style.background = "#16a34a"; // green
      apiStatusLabel.textContent = "online";
      apiStatusLabel.style.color = "#16a34a";
    } else {
      apiStatusDot.style.background = "#f59e0b"; // amber
      apiStatusLabel.textContent = "degraded";
      apiStatusLabel.style.color = "#f59e0b";
    }
  } catch {
    apiStatusDot.style.background = "#ef4444"; // red
    apiStatusLabel.textContent = "offline";
    apiStatusLabel.style.color = "#ef4444";
  }
})();

// Initialize state
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

// Pro UI is always visible
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

// Focus Mode
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

// Custom Inputs
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

// Timer
const timerDisplay = document.getElementById("timerDisplay");
const timerMode = document.getElementById("timerMode");
const startTimer = document.getElementById("startTimer");
const pauseTimer = document.getElementById("pauseTimer");
const resetTimer = document.getElementById("resetTimer");

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

// On load
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


// ─────────────────────────────── AI FUNCTIONS ───────────────────────────────

// Get page selection if any; fallback to body text (first 8k chars).
async function getActiveTabTextAndSelection() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // NEW: guard for restricted pages (chrome://, web store, etc.)
  const restricted = /^chrome:\/\//i.test(tab.url) || /^https?:\/\/chrome\.google\.com\/webstore/i.test(tab.url);
  if (restricted) {
    return { selection: "", text: "" , _restricted: true };
  }

  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const selection = window.getSelection && window.getSelection().toString();
        const bodyText = document.body ? document.body.innerText : "";
        return {
          selection: (selection || "").slice(0, 4000),
          text: (bodyText || "").slice(0, 8000)
        };
      }
    });
    return result || { selection: "", text: "" };
  } catch (e) {
    // If injection fails for any reason, treat as restricted
    return { selection: "", text: "", _restricted: true };
  }
}

function setAIBusy(busy) {
  if (!aiResult) return;
  if (busy) aiResult.setAttribute("disabled", "true");
  else aiResult.removeAttribute("disabled");
  if (summarizeBtn) summarizeBtn.disabled = busy;
  if (rewriteBtn) rewriteBtn.disabled = busy;
  if (sentimentBtn) sentimentBtn.disabled = busy;  // NEW
  if (analyzeBtn) analyzeBtn.disabled = busy;      // NEW
}

function showResult(text) {
  if (!aiResult) return;
  aiResult.value = text;
}

function warnIfRestricted(info) {
  if (info?._restricted) {
    showResult("⚠️ This page blocks extensions from reading content. Try another site (e.g., Wikipedia).");
    return true;
  }
  return false;
}

// Summarize page
summarizeBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Summarizing…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return setAIBusy(false);
    chrome.runtime.sendMessage({ type: "SUMMARIZE", payload: { text: info.text } }, (res) => {
      showResult(res?.summary || res?.error || "No response");
      setAIBusy(false);
    });
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
    setAIBusy(false);
  }
});

// Rewrite selection (or page)
rewriteBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Rewriting selection…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return setAIBusy(false);
    const payload = info.selection && info.selection.trim().length > 0 ? info.selection : info.text;
    chrome.runtime.sendMessage({ type: "REWRITE", payload: { text: payload } }, (res) => {
      showResult(res?.rewrite || res?.error || "No response");
      setAIBusy(false);
    });
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
    setAIBusy(false);
  }
});

// Sentiment (ONNX)
sentimentBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Detecting sentiment…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return setAIBusy(false);
    const payload = info.selection && info.selection.trim().length > 0 ? info.selection : info.text;
    chrome.runtime.sendMessage({ type: "SENTIMENT", payload: { text: payload } }, (res) => {
      if (res?.error) return showResult(`❌ ${res.error}`);
      const label = res?.sentiment ?? res?.label ?? "(unknown)";
      const conf  = typeof res?.confidence === "number" ? (res.confidence * 100).toFixed(2) + "%" : "";
      showResult(`Sentiment: ${label}${conf ? ` (${conf})` : ""}`);
      setAIBusy(false);
    });
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
    setAIBusy(false);
  }
});

// Analyze (Sentiment + Summary; requires /analyze)
analyzeBtn?.addEventListener("click", async () => {
  try {
    setAIBusy(true);
    showResult("Analyzing…");
    const info = await getActiveTabTextAndSelection();
    if (warnIfRestricted(info)) return setAIBusy(false);
    const payload = info.selection && info.selection.trim().length > 0 ? info.selection : info.text;
    chrome.runtime.sendMessage({ type: "ANALYZE", payload: { text: payload } }, (res) => {
      if (res?.error) return showResult(`❌ ${res.error}`);
      const s = res?.sentiment;
      const label = s?.sentiment ?? s?.label ?? "(unknown)";
      const conf  = typeof s?.confidence === "number" ? (s.confidence * 100).toFixed(2) + "%" : "";
      const summary = res?.summary || "(no summary)";
      showResult(`Sentiment: ${label}${conf ? ` (${conf})` : ""}\n\nSummary:\n${summary}`);
      setAIBusy(false);
    });
  } catch (e) {
    showResult(`Error: ${e?.message || e}`);
    setAIBusy(false);
  }
});

// Health ping (optional)
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
