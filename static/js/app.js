const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const SOURCE_LABEL = { private_kb: "Company KB", web: "Web search", fallback: "No verified source", direct: "Direct" };

function addMessage(role, html) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="bubble">${html}</div>`;
  $("messages").appendChild(el);
  $("messages").scrollTop = $("messages").scrollHeight;
  return el;
}

function renderPath(trace) {
  document.querySelectorAll("#path li").forEach((li) => { li.classList.remove("hit"); li.removeAttribute("data-order"); });
  trace.forEach((s, i) => {
    const li = document.querySelector(`#path li[data-node="${s.node}"]`);
    if (li) { li.classList.add("hit"); li.dataset.order = li.dataset.order ? `${li.dataset.order},${i + 1}` : i + 1; }
  });
  $("trace").innerHTML = trace.map((s, i) => `<div class="step">${i + 1}. <b>${esc(s.node)}</b><br>${esc(s.detail)}</div>`).join("");
}

function renderCitations(citations) {
  if (!citations.length) { $("citations").innerHTML = '<p class="muted">No sources for this answer.</p>'; return; }
  $("citations").innerHTML = citations.map((c) => {
    const title = c.url ? `<a href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.title)}</a>` : esc(c.title);
    const extra = [c.detail, c.score != null ? `score ${c.score}` : ""].filter(Boolean).join(" · ");
    return `<div class="cite">[${c.n}] ${title}${extra ? `<br><span class="muted">${esc(extra)}</span>` : ""}</div>`;
  }).join("");
}

async function ask(question) {
  addMessage("user", esc(question));
  const pending = addMessage("bot", '<span class="muted">Thinking: routing, retrieving, grading…</span>');
  $("send").disabled = true;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);

    const badges = [
      `<span class="badge ${data.source_used}">${SOURCE_LABEL[data.source_used] || data.source_used}</span>`,
      data.confidence !== "n/a" ? `<span class="badge plain">confidence: ${esc(data.confidence)}</span>` : "",
      data.retries ? `<span class="badge plain">${data.retries} retry</span>` : "",
      `<span class="badge plain">${(data.latency_ms / 1000).toFixed(1)}s</span>`,
    ].join("");
    pending.querySelector(".bubble").innerHTML = `${esc(data.answer)}<div class="meta">${badges}</div>`;
    renderPath(data.trace);
    renderCitations(data.citations);
  } catch (err) {
    pending.querySelector(".bubble").innerHTML = `<span style="color:#c92a2a">Error: ${esc(err.message)}</span>`;
  } finally {
    $("send").disabled = false;
  }
}

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("question").value.trim();
  if (q.length < 2) return;
  $("question").value = "";
  ask(q);
});

$("suggestions").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") ask(e.target.textContent);
});

$("upload").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = new FormData();
  form.append("file", $("file").files[0]);
  $("uploadResult").textContent = "Indexing…";
  try {
    const res = await fetch("/api/ingest", { method: "POST", headers: { "X-Admin-Key": $("adminKey").value }, body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    $("uploadResult").textContent = `Indexed ${data.file}: ${data.chunks} chunks. Ask about it now.`;
  } catch (err) {
    $("uploadResult").textContent = `Failed: ${err.message}`;
  }
});

fetch("/api/health").then((r) => r.json()).then((h) => {
  const missing = Object.entries(h.keys_present).filter(([, ok]) => !ok).map(([k]) => k);
  $("status").textContent = missing.length ? `missing keys: ${missing.join(", ")}` : `${h.llm} · ${h.embeddings}`;
}).catch(() => { $("status").textContent = "API unreachable"; });
