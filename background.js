let tabLimitListener = null;
let lastNotificationTime = 0; // Prevent notification spam

function showRateLimitedNotification(options) {
  const now = Date.now();
  if (now - lastNotificationTime > 3000) { // 3-second cooldown
    chrome.notifications.create(options);
    lastNotificationTime = now;
  }
}

function enableFocusMode() {
  tabLimitListener = function (tab) {
    chrome.storage.local.get(["enforceTabs", "ghostPro", "customTabLimit"], (data) => {
      const enforce = data.enforceTabs;
      const isPro = data.ghostPro;
      const customLimit = parseInt(data.customTabLimit);
      const MAX_TABS = isPro && !isNaN(customLimit) ? customLimit : 3;

      chrome.tabs.query({}, (tabs) => {
        if (tabs.length > MAX_TABS) {
          console.log(`[GhostTab] Too many tabs: ${tabs.length}/${MAX_TABS}`);

          if (enforce) {
            chrome.tabs.remove(tab.id); // Strict mode: auto-close
          }

          showRateLimitedNotification({
            type: "basic",
            iconUrl: chrome.runtime.getURL("icon.png"),
            title: "👻 GhostTab Alert",
            message: enforce
              ? `🚫 Too many tabs! Extra tab closed. (${tabs.length}/${MAX_TABS})`
              : `⚠️ Tab limit exceeded. You have ${tabs.length} tabs open.`,
          });
        } else {
          console.log(`[GhostTab] Tab created: ${tabs.length}/${MAX_TABS} (enforced: ${enforce})`);
        }
      });
    });
  };

  chrome.tabs.onCreated.addListener(tabLimitListener);
}

function disableFocusMode() {
  if (tabLimitListener) {
    chrome.tabs.onCreated.removeListener(tabLimitListener);
    tabLimitListener = null;
  }
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
