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
const simulateProToggle = document.getElementById("simulatePro");
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

let isProUser = false;
let MAX_TABS = 3;
let dailyGoal = 6;

// Initialize state
chrome.storage.local.get([
  "focusMode", "ghostTheme", "enforceTabs", "ghostPro",
  "customTabLimit", "customDailyGoal"
], (data) => {
  statusText.innerText = data.focusMode ? "🎯 Focus Mode: ON" : "🎯 Focus Mode: OFF";

  if (data.ghostTheme) {
    document.body.classList.add(data.ghostTheme);
    themeSelect.value = data.ghostTheme;
  }

  enforceToggle.checked = data.enforceTabs || false;
  isProUser = !!data.ghostPro;
  simulateProToggle.checked = isProUser;
  updateProUI(isProUser);

  if (isProUser) {
    if (data.customTabLimit) {
      customTabLimitInput.value = data.customTabLimit;
      MAX_TABS = parseInt(data.customTabLimit);
    }
    if (data.customDailyGoal) {
      customGoalInput.value = data.customDailyGoal;
      dailyGoal = parseInt(data.customDailyGoal);
    }
  }
});

// UI switching based on Pro
function updateProUI(enabled) {
  tabLimitSection.style.display = enabled ? "block" : "none";
  customGoalSection.style.display = enabled ? "block" : "none";
  historySection.style.display = enabled ? "block" : "none";
  historyDisplay.style.display = "none";
  toggleHistoryBtn.innerText = "📊 Show History";
  timerDropdownSection.style.display = enabled ? "none" : "block";
  customTimerInputSection.style.display = enabled ? "block" : "none";
}

simulateProToggle.addEventListener("change", (e) => {
  const isPro = e.target.checked;
  chrome.storage.local.set({ ghostPro: isPro }, () => {
    isProUser = isPro;
    updateProUI(isPro);
  });
});

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

// Custom Pro Inputs
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
  if (isProUser) {
    return isBreak ? parseInt(customBreakInput.value) : parseInt(customWorkInput.value);
  } else {
    return isBreak ? parseInt(breakDropdown.value) : parseInt(workDropdown.value);
  }
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
