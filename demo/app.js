const form = document.querySelector("#generationForm");
const openingLine = document.querySelector("#openingLine");
const temperature = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperatureValue");
const seed = document.querySelector("#seed");
const generateButton = document.querySelector("#generateButton");
const statusText = document.querySelector("#statusText");
const modelStatus = document.querySelector(".model-status");
const resultState = document.querySelector("#resultState");
const sonnetLines = document.querySelector("#sonnetLines");
const lineCount = document.querySelector("#lineCount");
const elapsedTime = document.querySelector("#elapsedTime");

temperature.addEventListener("input", () => {
  temperatureValue.value = Number(temperature.value).toFixed(1);
});

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Unavailable");
    }
    const payload = await response.json();
    statusText.textContent = payload.status === "ready" ? "Model ready" : "Unavailable";
    modelStatus.classList.toggle("ready", payload.status === "ready");
    modelStatus.classList.toggle("error", payload.status !== "ready");
  } catch (_error) {
    statusText.textContent = "Disconnected";
    modelStatus.classList.add("error");
  }
}

function showState(message, isError = false) {
  resultState.hidden = false;
  resultState.classList.toggle("error", isError);
  resultState.replaceChildren(Object.assign(document.createElement("p"), {
    textContent: message,
  }));
  sonnetLines.hidden = true;
}

function showSonnet(text) {
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  sonnetLines.replaceChildren(...lines.map((line) => {
    const item = document.createElement("li");
    item.textContent = line;
    return item;
  }));
  resultState.hidden = true;
  sonnetLines.hidden = false;
  return lines.length;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  generateButton.disabled = true;
  generateButton.textContent = "Generating";
  lineCount.textContent = "--";
  elapsedTime.textContent = "--";
  showState("Generating sonnet...");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        opening_line: openingLine.value,
        temperature: Number(temperature.value),
        seed: Number(seed.value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Generation failed");
    }
    lineCount.textContent = String(showSonnet(payload.text));
    elapsedTime.textContent = `${payload.elapsed_seconds.toFixed(1)}s`;
  } catch (error) {
    showState(error.message || "Generation failed", true);
  } finally {
    generateButton.disabled = false;
    generateButton.textContent = "Generate sonnet";
  }
});

refreshStatus();
