const FALLBACK_ERROR = "The assistant couldn't answer just now. Try again in a moment.";
const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");

let thread;
let threadEmpty;
let askForm;
let askInput;
let askButton;
let starters = [];
let note;
let scoreboard;
let failsList;

let conversationId = crypto.randomUUID();
let messages = [];
let turnCount = 0;
let busy = false;
let lastOpenFilm = null;

function cloneTemplate(id) {
  const template = document.getElementById(id);
  return template.content.firstElementChild.cloneNode(true);
}

function formatMs(ms) {
  const value = Math.round((ms || 0) * 10) / 10;
  const text = value.toLocaleString("en-US", { maximumFractionDigits: 1 });
  return text.endsWith(".0") ? text.slice(0, -2) : text;
}

function formatMoney(price) {
  return price.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatRate(rate) {
  return Math.round(rate * 100) + "%";
}

function formatCitation(title) {
  const base = String(title).replace(/\.pdf$/i, "").replace(/_/g, " ");
  return base.charAt(0).toUpperCase() + base.slice(1);
}

function capitalize(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

function formatDate(iso) {
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const date = new Date(iso);
  return date.getUTCDate() + " " + months[date.getUTCMonth()] + " " + date.getUTCFullYear();
}

function scrollThread() {
  thread.scrollTo({
    top: thread.scrollHeight,
    behavior: REDUCED_MOTION.matches ? "auto" : "smooth",
  });
}

function setBusyUI(isBusy) {
  askButton.disabled = isBusy;
  for (const starter of starters) starter.disabled = isBusy;
}

function appendPending(text) {
  const userMsg = cloneTemplate("tpl-user");
  userMsg.querySelector(".bubble").textContent = text;
  const asstMsg = cloneTemplate("tpl-assistant");
  asstMsg.classList.add("msg-pending");
  asstMsg.querySelector(".bubble").textContent = "Looking that up…";
  const film = asstMsg.querySelector(".film");
  film.hidden = true;
  thread.append(userMsg, asstMsg);
  scrollThread();
  return asstMsg;
}

async function parseError(response) {
  try {
    const data = await response.json();
    if (data && typeof data.error === "string" && data.error) return data.error;
  } catch {}
  return response.statusText || "Request failed.";
}

async function postChat() {
  let response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: conversationId,
        messages: messages.slice(-10),
      }),
    });
  } catch {
    throw new Error();
  }
  if (!response.ok) {
    const error = new Error(await parseError(response));
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function renderCard(item) {
  const card = cloneTemplate("tpl-card");
  card.querySelector(".card-name").textContent = item.name;
  card.querySelector(".card-meta").textContent = item.category;
  card.querySelector(".card-price").textContent = "EGP " + formatMoney(item.price_egp);
  const stock = card.querySelector(".stock");
  if (item.stock > 0) {
    stock.textContent = "In stock (" + item.stock + ")";
    stock.classList.add("stock-in");
  } else {
    stock.textContent = "Out of stock";
    stock.classList.add("stock-out");
  }
  return card;
}

function renderCitation(item) {
  const citation = cloneTemplate("tpl-citation");
  citation.querySelector(".citation-source").textContent =
    formatCitation(item.title) + ", part " + (item.chunk_index + 1);
  citation.querySelector(".citation-snippet").textContent = item.snippet;
  return citation;
}

function renderFilm(film, trace) {
  film.hidden = false;
  const steps = trace.steps || [];
  const totalMs = trace.total_ms || 0;
  turnCount += 1;
  film.querySelector(".marker").textContent = "T" + turnCount;
  film.querySelector(".total").textContent = formatMs(totalMs) + " ms";

  const exposure = film.querySelector(".exposure");
  const readings = film.querySelector(".readings");
  exposure.textContent = "";
  readings.textContent = "";

  let hotIndex = -1;
  let hotMs = -Infinity;
  steps.forEach((step, index) => {
    if (step.ms > hotMs) {
      hotMs = step.ms;
      hotIndex = index;
    }
  });

  steps.forEach((step, index) => {
    const isHot = index === hotIndex;
    const seg = document.createElement("div");
    seg.className = "seg seg-" + step.kind;
    if (isHot) seg.classList.add("seg-hot");
    seg.style.flexBasis = (totalMs > 0 ? (step.ms / totalMs) * 100 : 0) + "%";
    exposure.append(seg);

    const reading = document.createElement("div");
    reading.className = "reading";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    if (isHot) {
      reading.classList.add("hot");
      swatch.style.background = "var(--hot)";
    } else if (step.kind === "llm") {
      swatch.style.background = "var(--bone)";
    } else {
      swatch.style.background = "var(--glow)";
      if (step.kind === "search") swatch.style.opacity = "0.55";
    }
    reading.append(swatch);
    reading.append(document.createTextNode(step.label + " "));
    const ms = document.createElement("b");
    ms.className = "num";
    ms.textContent = formatMs(step.ms) + " ms";
    reading.append(ms);
    if (step.kind === "embed" && step.cache === "hit") {
      const badge = document.createElement("span");
      badge.className = "cache-hit";
      badge.textContent = "cache hit";
      reading.append(badge);
    } else if (step.kind === "embed" && step.cache === "miss") {
      const badge = document.createElement("span");
      badge.className = "cache-miss";
      badge.textContent = "cache miss";
      reading.append(badge);
    }
    readings.append(reading);
  });

  if (lastOpenFilm) lastOpenFilm.open = false;
  film.open = true;
  lastOpenFilm = film;
}

function renderReply(asstMsg, data) {
  asstMsg.classList.remove("msg-pending");

  const bubble = asstMsg.querySelector(".bubble");
  bubble.textContent = "";
  for (const paragraph of data.reply.split(/\n\s*\n/)) {
    const p = document.createElement("p");
    p.textContent = paragraph;
    bubble.append(p);
  }

  const cards = asstMsg.querySelector(".cards");
  if (data.cards && data.cards.length) {
    cards.hidden = false;
    for (const item of data.cards) cards.append(renderCard(item));
  }

  const citations = asstMsg.querySelector(".citations");
  if (data.citations && data.citations.length) {
    citations.hidden = false;
    for (const item of data.citations) citations.append(renderCitation(item));
  }

  renderFilm(asstMsg.querySelector(".film"), data.trace);

  messages.push({ role: "assistant", content: data.reply });

  if (data.turns_left <= 5) {
    note.textContent = data.turns_left + " questions left in this conversation.";
    note.hidden = false;
  }
  scrollThread();
}

function showNewConversationButton() {
  note.textContent = "";
  note.hidden = false;
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn btn-quiet";
  button.textContent = "Start a new conversation";
  button.addEventListener("click", resetConversation);
  note.append(button);
}

function resetConversation() {
  conversationId = crypto.randomUUID();
  messages = [];
  turnCount = 0;
  lastOpenFilm = null;
  thread.textContent = "";
  thread.append(threadEmpty);
  threadEmpty.hidden = false;
  note.textContent = "";
  note.hidden = true;
}

function handleError(asstMsg, error) {
  asstMsg.classList.remove("msg-pending");
  asstMsg.classList.add("msg-error");
  const bubble = asstMsg.querySelector(".bubble");
  if (error.status === 401) {
    bubble.textContent = "Enter the access code first.";
    window.location.assign("/login.html");
    return;
  }
  if (error.status === 429) {
    bubble.textContent = error.message || "Too many questions from this network. Try again in an hour.";
    messages.pop();
    showNewConversationButton();
    return;
  }
  bubble.textContent = error.message || FALLBACK_ERROR;
  messages.pop();
  scrollThread();
}

async function send(question) {
  const text = String(question == null ? "" : question).trim();
  if (!text || busy) return;
  busy = true;
  setBusyUI(true);
  threadEmpty.hidden = true;
  const pending = appendPending(text);
  messages.push({ role: "user", content: text });

  try {
    const data = await postChat();
    renderReply(pending, data);
    askInput.value = "";
  } catch (error) {
    handleError(pending, error);
  } finally {
    busy = false;
    setBusyUI(false);
    askInput.focus();
  }
}

function headerCell(text) {
  const th = document.createElement("th");
  th.textContent = text;
  return th;
}

function renderScoreboardSection(title, section, columns) {
  const frag = document.createDocumentFragment();
  const heading = document.createElement("h3");
  heading.textContent = title;
  frag.append(heading);

  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headerRow.append(headerCell("Search setup"));
  for (const column of columns) headerRow.append(headerCell(column.header));
  thead.append(headerRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  const configs = section.configs || [];
  const bests = columns.map((column) => {
    const values = configs.map((cfg) => cfg.metrics[column.key]);
    return column.direction === "low" ? Math.min(...values) : Math.max(...values);
  });
  for (const cfg of configs) {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = cfg.name + (cfg.key === section.production_config ? " (live)" : "");
    row.append(name);
    columns.forEach((column, index) => {
      const cell = document.createElement("td");
      cell.className = "metric";
      const value = cfg.metrics[column.key];
      cell.textContent = column.format(value, section.k);
      if (value === bests[index]) cell.classList.add("best");
      row.append(cell);
    });
    tbody.append(row);
  }
  table.append(tbody);
  wrap.append(table);
  frag.append(wrap);
  return frag;
}

function renderScoreboard(data) {
  scoreboard.textContent = "";

  const productColumns = [
    { key: "hit_at_1", header: "Right answer first", format: formatRate, direction: "high" },
    { key: "hit_at_5", header: "Right answer in top 5", format: formatRate, direction: "high" },
    { key: "mrr", header: "Rank score (MRR)", format: formatMRR, direction: "high" },
    { key: "diversity_at_5", header: "Variety in top 5", format: formatVariety, direction: "high" },
    { key: "p50_search_ms", header: "Search time", format: formatTime, direction: "low" },
  ];
  const documentColumns = [
    { key: "hit_at_1", header: "Right answer first", format: formatRate, direction: "high" },
    { key: "hit_at_3", header: "Right answer in top 3", format: formatRate, direction: "high" },
    { key: "mrr", header: "Rank score (MRR)", format: formatMRR, direction: "high" },
    { key: "p50_search_ms", header: "Search time", format: formatTime, direction: "low" },
  ];

  scoreboard.append(renderScoreboardSection("Products", data.products, productColumns));
  scoreboard.append(renderScoreboardSection("Store documents", data.documents, documentColumns));

  const lede = document.createElement("p");
  lede.className = "lede";
  lede.textContent = "Measured " + formatDate(data.generated_at) + " on a " + data.environment + ".";
  scoreboard.append(lede);
}

function formatMRR(value) {
  return value.toFixed(2);
}

function formatVariety(value, k) {
  return value.toFixed(1) + " of " + k;
}

function formatTime(value) {
  return formatMs(value) + " ms";
}

function failItem(failure, emptyText) {
  const li = document.createElement("li");
  li.className = "fail";
  if (!failure) {
    const q = document.createElement("div");
    q.className = "q";
    q.textContent = emptyText;
    li.append(q);
    return li;
  }
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = capitalize(failure.tag);
  li.append(tag);
  const q = document.createElement("div");
  q.className = "q";
  q.textContent = "\u201C" + failure.query + "\u201D";
  li.append(q);
  const why = document.createElement("div");
  why.className = "why";
  why.textContent = "Top results: " + (failure.top || []).join(", ");
  li.append(why);
  return li;
}

function renderFails(data) {
  failsList.textContent = "";
  const failures = [];
  for (const section of [data.products, data.documents]) {
    const live = (section.configs || []).find((cfg) => cfg.key === section.production_config);
    if (live) failures.push(...(live.failures || []));
  }
  if (!failures.length) {
    failsList.append(failItem(null, "No failing questions."));
    return;
  }
  for (const failure of failures) failsList.append(failItem(failure));
}

async function loadScoreboard() {
  try {
    const response = await fetch("/api/eval");
    if (!response.ok) throw new Error();
    const data = await response.json();
    renderScoreboard(data);
    renderFails(data);
  } catch {
    scoreboard.textContent = "";
    const placeholder = document.createElement("p");
    placeholder.className = "lede";
    placeholder.textContent = "Scores will appear here after the first evaluation run.";
    scoreboard.append(placeholder);
  }
}

function init() {
  thread = document.getElementById("thread");
  threadEmpty = document.getElementById("thread-empty");
  askForm = document.getElementById("ask-form");
  askInput = document.getElementById("ask-input");
  askButton = document.getElementById("ask-button");
  starters = Array.from(document.querySelectorAll("#starters .starter"));
  note = document.getElementById("limit-note");
  scoreboard = document.getElementById("scoreboard");
  failsList = document.getElementById("scoreboard-fails");

  askForm.addEventListener("submit", (event) => {
    event.preventDefault();
    send(askInput.value);
  });
  for (const starter of starters) {
    starter.addEventListener("click", () => send(starter.dataset.q));
  }

  fetch("/api/health").catch(() => {});
  loadScoreboard();
}

document.addEventListener("DOMContentLoaded", init);