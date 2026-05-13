"use strict";

const runsTableBody = document.querySelector("#runs-table tbody");
const streamLog = document.querySelector("#stream-log");

let activeEventSource = null;

async function loadRuns() {
  const res = await fetch("/api/runs");
  const runs = await res.json();
  runsTableBody.innerHTML = "";
  for (const run of runs) {
    const tr = document.createElement("tr");
    const state = run.state || "pending";
    tr.innerHTML = `
      <td><code>${run.run_id}</code></td>
      <td>${run.kind}</td>
      <td>${escapeHtml(run.title || "")}</td>
      <td class="status-${state}">${state}</td>
      <td>
        <button data-action="watch" data-id="${run.run_id}">stream</button>
        <button data-action="report" data-id="${run.run_id}" data-name="summary">summary</button>
        <button data-action="report" data-id="${run.run_id}" data-name="analysis">analysis</button>
        <button data-action="report" data-id="${run.run_id}" data-name="brief">brief</button>
        ${run.kind === "compare"
          ? `<button data-action="report" data-id="${run.run_id}" data-name="comparison">comparison</button>`
          : ""}
      </td>`;
    runsTableBody.appendChild(tr);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function watchRun(runId) {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  streamLog.textContent = `[stream] connected to ${runId}\n`;
  const es = new EventSource(`/api/runs/${runId}/stream`);
  activeEventSource = es;
  es.onmessage = (e) => {
    streamLog.textContent += e.data + "\n";
    streamLog.scrollTop = streamLog.scrollHeight;
  };
  es.addEventListener("status", (e) => {
    streamLog.textContent += `\n[stream] run finished: ${e.data}\n`;
    es.close();
    loadRuns();
  });
  es.onerror = () => {
    streamLog.textContent += "[stream] disconnected\n";
    es.close();
  };
}

document.addEventListener("click", (evt) => {
  const btn = evt.target.closest("button[data-action]");
  if (!btn) return;
  const action = btn.dataset.action;
  const runId = btn.dataset.id;
  if (action === "watch") {
    watchRun(runId);
  } else if (action === "report") {
    window.open(
      `/api/runs/${runId}/report/${btn.dataset.name}`,
      "_blank",
      "noopener"
    );
  }
});

document.querySelector("#run-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const fd = new FormData(form);
  const res = await fetch("/api/run", { method: "POST", body: fd });
  if (!res.ok) {
    alert(`Run failed to submit: ${res.status} ${res.statusText}`);
    return;
  }
  const run = await res.json();
  form.reset();
  await loadRuns();
  watchRun(run.run_id);
});

document.querySelector("#compare-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.currentTarget;
  const fd = new FormData(form);
  const res = await fetch("/api/compare", { method: "POST", body: fd });
  if (!res.ok) {
    alert(`Compare failed to submit: ${res.status} ${res.statusText}`);
    return;
  }
  const run = await res.json();
  form.reset();
  await loadRuns();
  watchRun(run.run_id);
});

loadRuns();
setInterval(loadRuns, 5000);
