// content.js
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === "GHOSTTAB_GET_PAGE_TEXT") {
    try {
      const selection = window.getSelection?.().toString() || "";
      const text = document.body?.innerText || "";
      sendResponse({ ok: true, text, selection });
    } catch (e) {
      sendResponse({ ok: false, error: String(e) });
    }
    return true;
  }
});
