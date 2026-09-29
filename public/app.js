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
let scoreboardLede;
let plugCount;
let nextMedian;
let nextNegation;
let nextDiversity;
let nextHit1;

let conversationId = crypto.randomUUID();
let messages = [];
let turnCount = 0;
let busy = false;
const turnsById = new Map();
let yourTurns, turnCountEl, turnsEmpty, turnList;

let myAnswers = new Map();
let monLatestMine = null;
let monInspectedId = null;
let monCount, monTotal, monCache, monChart, monFramesEl, monEmpty;
let monInspectTitle, monInspectReadings, monBars, monVerdict, monRoll;
let monInFlight = false;
let monReloadQueued = false;
let monPolling = null;
let monVisible = false;
let monFirstRender = true;
let monPrevIds = new Set();
const MONITOR_POLL_MS = 15000;

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

const KIND_LABELS = { semantic: "Meaning", negation: "Negation", brand: "Brand", direct: "Direct" };
const PRODUCT_COLUMNS = [
  { key: "hit_at_1", header: "Right answer first", format: formatRate, direction: "high" },
  { key: "hit_at_5", header: "Right answer in top 5", format: formatRate, direction: "high" },
  { key: "mrr", header: "Rank score (MRR)", format: formatMRR, direction: "high" },
  { key: "diversity_at_5", header: "Variety in top 5", format: formatVariety, direction: "high" },
  { key: "p50_search_ms", header: "Search time", format: formatTime, direction: "low" },
];
const DOCUMENT_COLUMNS = [
  { key: "hit_at_1", header: "Right answer first", format: formatRate, direction: "high" },
  { key: "hit_at_3", header: "Right answer in top 3", format: formatRate, direction: "high" },
  { key: "mrr", header: "Rank score (MRR)", format: formatMRR, direction: "high" },
  { key: "p50_search_ms", header: "Search time", format: formatTime, direction: "low" },
];

function productionConfig(section) {
  return (section.configs || []).find((cfg) => cfg.key === section.production_config) || null;
}

function configQuestionCount(config) {
  if (!config || !config.by_tag) return 0;
  return Object.values(config.by_tag).reduce((sum, bucket) => sum + (bucket.n || 0), 0);
}

function displayConfigName(cfg) {
  return String(cfg.name)
    .replace("chunks", "passages")
    .replace("Name, category and description", "Name, category, description");
}

function typeOf(name) {
  const parts = String(name).split(" ");
  return parts.length > 1 ? parts.slice(1).join(" ") : String(name);
}

function countWord(count) {
  const words = { 2: "two", 3: "three", 4: "four", 5: "five" };
  return words[count] || String(count);
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

function renderFilm(film, trace, number) {
  const steps = trace.steps || [];
  const totalMs = trace.total_ms || 0;
  film.querySelector(".marker").textContent = "T" + number;
  const firstSound = firstSoundLabel(trace);
  film.querySelector(".total").textContent = formatDuration(totalMs) +
    (firstSound ? " · " + firstSound : "");

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
    if (isHot) reading.classList.add("hot");
    const swatch = document.createElement("span");
    swatch.className = "swatch sw-" + (isHot ? "hot" : step.kind);
    reading.append(swatch);
    reading.append(document.createTextNode(step.label + " "));
    const ms = document.createElement("b");
    ms.className = "num";
    ms.textContent = formatDuration(step.ms);
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

}

function replyLanguage(text) {
  const letters = String(text || "").match(/\p{L}/gu) || [];
  if (!letters.length) return "Mixed";
  const arabic = letters.filter((letter) => /\p{Script=Arabic}/u.test(letter)).length;
  if (arabic / letters.length >= 0.3) return "Arabic";
  return arabic === 0 ? "English" : "Mixed";
}

function languageLabel(language, messageNode) {
  const labels = { ar: "Arabic", en: "English", mixed: "Mixed" };
  return labels[String(language || "").toLowerCase()] ||
    replyLanguage(messageNode.querySelector(".bubble").textContent);
}

function activateTurn(id, { scroll = false } = {}) {
  document.querySelectorAll(".turn-chip.is-active, .turn-entry.is-active, .mon-frame.is-active")
    .forEach((element) => element.classList.remove("is-active"));
  const turn = turnsById.get(id);
  if (!turn) return;
  turn.chip.classList.add("is-active");
  turn.logNode.classList.add("is-active");
  document.querySelectorAll(".mon-frame[data-turn]").forEach((frame) => {
    frame.classList.toggle("is-active", frame.dataset.turn === id);
  });
  yourTurns.open = true;
  if (scroll) {
    turn.logNode.scrollIntoView({ block: "center", behavior: REDUCED_MOTION.matches ? "auto" : "smooth" });
  }
}

function addTurn({ kind, trace, messageNode, monitorId, language }) {
  turnCount += 1;
  const number = turnCount;
  const id = "T" + number;
  const logNode = cloneTemplate("tpl-log-entry");
  logNode.id = id;
  renderFilm(logNode, trace, number);

  const label = languageLabel(language || trace.language, messageNode);
  const hasSearch = (trace.steps || []).some((step) => step.kind === "search");
  const hasCards = !messageNode.querySelector(".cards").hidden;
  const meta = [kind, label];
  if (!hasSearch && !hasCards) meta.push("asked to clarify");
  logNode.querySelector(".meta").textContent = meta.join(" · ");
  turnList.prepend(logNode);
  turnCountEl.textContent = String(turnCount);
  turnsEmpty.hidden = true;

  messageNode.id = "msg-" + id;
  const chip = messageNode.querySelector(".turn-chip");
  chip.querySelector(".chip-t").textContent = id;
  const chipParts = [kind];
  if (kind === "spoken" && label !== "English") chipParts.push(label);
  if (!hasSearch && !hasCards) chipParts.push("asked to clarify");
  chipParts.push(formatDuration(trace.total_ms || 0));
  const firstSound = firstSoundLabel(trace);
  if (firstSound) chipParts.push(firstSound);
  chip.querySelector(".chip-text").textContent = chipParts.join(" · ");
  chip.hidden = false;
  chip.addEventListener("click", () => activateTurn(id, { scroll: true }));
  logNode.querySelector(".back").addEventListener("click", (event) => {
    event.preventDefault();
    messageNode.scrollIntoView({ block: "center", behavior: REDUCED_MOTION.matches ? "auto" : "smooth" });
  });

  turnsById.set(id, {
    id, number, kind, language: label, totalMs: trace.total_ms || 0, trace,
    messageNode, logNode, chip, monitorId,
  });
  noteMyAnswer(monitorId, number);
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

  addTurn({ kind: "typed", trace: data.trace, messageNode: asstMsg, monitorId: data.monitor_id });
  loadMonitor();

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
  turnsById.clear();
  turnList.textContent = "";
  turnsEmpty.hidden = false;
  turnCountEl.textContent = "0";
  myAnswers.clear();
  monLatestMine = null;
  monInspectedId = null;
  thread.textContent = "";
  thread.append(threadEmpty);
  threadEmpty.hidden = false;
  note.textContent = "";
  note.hidden = true;
  loadMonitor();
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
  const products = data.products || {};
  const documents = data.documents || {};

  const boards = document.createElement("div");
  boards.className = "ev-boards";
  boards.append(buildBoard("Products", products, [
    { name: "Right answer first", key: "hit_at_1" },
    { name: "Right answer in the top 5", key: "hit_at_5" },
    { name: "Different products in the top 5", key: "diversity_at_5", variety: true },
  ]));
  boards.append(buildDocumentBoard(documents));
  scoreboard.append(boards);

  const foot = document.createElement("div");
  foot.className = "ev-board-foot";
  const measured = document.createElement("span");
  measured.className = "ev-small";
  measured.textContent = "Measured " + formatDate(data.generated_at) + " on a " + data.environment + ".";
  foot.append(measured);

  const details = document.createElement("details");
  details.className = "ev-full";
  const summary = document.createElement("summary");
  summary.textContent = "Full scores";
  details.append(summary);
  details.append(renderScoreboardSection("Products", products, PRODUCT_COLUMNS));
  details.append(renderScoreboardSection("Store documents", documents, DOCUMENT_COLUMNS));
  foot.append(details);
  scoreboard.append(foot);

  populateScoreboardNumbers(data);
  populateNextNumbers(data);
}

function buildBoard(title, section, metrics) {
  const board = document.createElement("div");
  board.className = "ev-board";

  const head = document.createElement("div");
  head.className = "ev-board-head";
  const heading = document.createElement("h3");
  heading.textContent = title;
  head.append(heading);
  const count = document.createElement("span");
  count.className = "num";
  count.textContent = configQuestionCount(productionConfig(section)) + " questions";
  head.append(count);
  board.append(head);

  const production = productionConfig(section);
  const ordered = production
    ? [production, ...section.configs.filter((cfg) => cfg !== production)]
    : section.configs || [];

  for (const metric of metrics) {
    const wrap = document.createElement("div");
    wrap.className = "ev-metric";
    const name = document.createElement("div");
    name.className = "ev-metric-name";
    name.textContent = metric.name;
    wrap.append(name);
    for (const cfg of ordered) wrap.append(buildPair(cfg, section, metric));
    board.append(wrap);
  }
  return board;
}

function buildDocumentBoard(section) {
  const board = buildBoard("Store documents", section, [
    { name: "Right passage first", key: "hit_at_1" },
  ]);
  const decided = document.createElement("p");
  decided.className = "ev-decided";
  const lead = document.createElement("b");
  lead.textContent = "This score picked the passage size.";
  decided.append(lead);
  decided.append(
    document.createTextNode(
      " Only three documents, so read it as a direction; the product numbers carry the weight."
    )
  );
  board.append(decided);
  return board;
}

function buildPair(cfg, section, metric) {
  const pair = document.createElement("div");
  pair.className = "ev-pair";
  const live = cfg.key === section.production_config;
  if (live) pair.classList.add("ev-live");

  const who = document.createElement("span");
  who.className = "ev-who";
  if (live) {
    const strong = document.createElement("b");
    strong.textContent = displayConfigName(cfg);
    who.append(strong);
    const chip = document.createElement("span");
    chip.className = "ev-live-chip";
    chip.textContent = "live";
    who.append(chip);
  } else {
    who.textContent = displayConfigName(cfg);
  }
  pair.append(who);

  const bar = document.createElement("div");
  bar.className = "ev-bar";
  const fill = document.createElement("i");
  fill.style.width = metricWidth(cfg, section, metric) + "%";
  bar.append(fill);
  pair.append(bar);

  const value = document.createElement("span");
  value.className = "ev-v num";
  value.textContent = metric.variety
    ? formatVariety(cfg.metrics[metric.key], section.k)
    : formatRate(cfg.metrics[metric.key]);
  pair.append(value);
  return pair;
}

function metricWidth(cfg, section, metric) {
  const value = cfg.metrics[metric.key] || 0;
  const ratio = metric.variety ? value / section.k : value;
  return Math.round(ratio * 100 * 10) / 10;
}

function populateScoreboardNumbers(data) {
  const products = (data && data.products) || {};
  const documents = (data && data.documents) || {};
  const total =
    configQuestionCount(productionConfig(products)) + configQuestionCount(productionConfig(documents));
  scoreboardLede.textContent =
    total + " shopper questions with a known right answer, run against every version of the search.";
  plugCount.textContent = total + " questions per release";
}

function populateNextNumbers(data) {
  const products = (data && data.products) || {};
  const live = productionConfig(products);
  if (!live) return;
  const metrics = live.metrics || {};
  const byTag = live.by_tag || {};
  const negation = byTag.negation;
  if (negation && negation.n) {
    nextNegation.textContent = Math.round(negation.hit_at_5 * negation.n) + " of " + negation.n;
    nextNegation.classList.toggle("bad", negation.hit_at_5 < 0.5);
  }
  if (metrics.diversity_at_5 != null && products.k) {
    nextDiversity.textContent = formatVariety(metrics.diversity_at_5, products.k);
  }
  if (metrics.hit_at_1 != null) {
    nextHit1.textContent = formatRate(metrics.hit_at_1);
  }
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

function boldEl(text) {
  const b = document.createElement("b");
  b.textContent = text;
  return b;
}

function buildGotLine(failure) {
  const got = document.createElement("span");
  got.className = "ev-got";
  const names = failure.top || [];
  if (!names.length) {
    got.textContent = "Got nothing back";
    return got;
  }
  const types = names.map(typeOf);
  const word = countWord(names.length);
  got.append(document.createTextNode("Got "));
  if (names.every((name) => name === names[0])) {
    got.append(boldEl(types[0]), document.createTextNode(" " + word + " times"));
  } else if (types.every((type) => type === types[0])) {
    got.append(boldEl(types[0]), document.createTextNode(" from " + word + " brands"));
  } else {
    types.forEach((type, index) => {
      if (index > 0) got.append(document.createTextNode(", "));
      got.append(boldEl(type));
    });
  }
  return got;
}

function failItem(failure) {
  const li = document.createElement("li");
  li.className = "ev-miss";

  const kind = document.createElement("span");
  kind.className = "ev-kind";
  kind.textContent = KIND_LABELS[failure.tag] || capitalize(failure.tag);
  li.append(kind);

  const asked = document.createElement("span");
  asked.className = "ev-asked";
  asked.textContent = "\u201C" + failure.query + "\u201D";
  li.append(asked);

  li.append(buildGotLine(failure));
  return li;
}

function renderFails(data) {
  failsList.textContent = "";
  const failures = [];
  for (const section of [data.products, data.documents]) {
    const live = productionConfig(section);
    if (live) failures.push(...(live.failures || []));
  }
  if (!failures.length) {
    const li = document.createElement("li");
    li.className = "ev-miss";
    const note = document.createElement("span");
    note.className = "ev-got";
    note.textContent = "No failing questions.";
    li.append(note);
    failsList.append(li);
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

/* ---------- voice ---------- */
const VOICE_SDK_URL =
  "https://cdn.jsdelivr.net/npm/livekit-client@2.22.3/dist/livekit-client.umd.js";
const VOICE_SESSION_SECONDS = 180;

let talkButton;
let voiceHint;
let voiceBar;
let voiceStatus;
let voiceTimer;
let voiceEnd;
let voiceRoom = null;
let voiceSdkPromise = null;
let voiceTimerId = null;
let voiceSecondsLeft = 0;
let voiceAudioEls = [];
let voiceAssistantBubbles = new Map();
let voiceTurnNodes = new Map();
let voiceLatestAssistant = null;
let voiceAwaitingReply = null;

function formatVoiceTime(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes + ":" + String(seconds).padStart(2, "0");
}

function showVoiceThreadContent() {
  threadEmpty.hidden = true;
  scrollThread();
}

function voiceAttrs(stream) {
  return stream.info.attributes || {};
}

function segmentKey(stream) {
  return voiceAttrs(stream)["lk.segment_id"] || stream.info.id;
}

function voiceBubbleForSegment(segmentId) {
  let node = voiceAssistantBubbles.get(segmentId);
  if (!node) {
    // The agent sends product cards while it searches, before it speaks; the
    // spoken answer that follows fills the bubble the cards are already under.
    if (voiceAwaitingReply) {
      node = voiceAwaitingReply;
      voiceAwaitingReply = null;
    } else {
      node = cloneTemplate("tpl-assistant");
      thread.append(node);
      showVoiceThreadContent();
    }
    voiceAssistantBubbles.set(segmentId, node);
  }
  voiceLatestAssistant = node;
  return node;
}

function setVoiceBubbleText(node, text) {
  const bubble = node.querySelector(".bubble");
  if (bubble.textContent !== text) {
    bubble.textContent = text;
    scrollThread();
  }
}

function renderVoiceError(message) {
  const node = cloneTemplate("tpl-assistant");
  node.classList.add("msg-error");
  node.querySelector(".bubble").textContent = message;
  thread.append(node);
  showVoiceThreadContent();
}

function restoreTalkButton() {
  talkButton.disabled = false;
}

function resetVoiceUI() {
  stopVoiceTimer();
  voiceBar.hidden = true;
  restoreTalkButton();
}

function cleanupVoiceNotes() {
  for (const el of voiceAudioEls) el.remove();
  voiceAudioEls = [];
  voiceAssistantBubbles = new Map();
  voiceTurnNodes = new Map();
  voiceLatestAssistant = null;
  voiceAwaitingReply = null;
}

async function handleVoiceTranscription(stream, isUserSpeech) {
  let text = "";
  for await (const chunk of stream) {
    text += chunk;
    if (!isUserSpeech && text) {
      setVoiceBubbleText(voiceBubbleForSegment(segmentKey(stream)), text);
    }
  }
  const final = voiceAttrs(stream)["lk.transcription_final"] === "true";
  const turnId = voiceAttrs(stream).turn_id;
  if (isUserSpeech) {
    if (final && text) {
      const node = cloneTemplate("tpl-user");
      const bubble = node.querySelector(".bubble");
      const mic = document.createElement("span");
      mic.className = "mic";
      mic.setAttribute("aria-label", "spoken");
      mic.textContent = "🎙";
      bubble.append(mic, document.createTextNode(text));
      thread.append(node);
      showVoiceThreadContent();
    }
  } else if (text) {
    const node = voiceBubbleForSegment(segmentKey(stream));
    if (turnId && !voiceTurnNodes.has(turnId)) voiceTurnNodes.set(turnId, node);
    setVoiceBubbleText(node, text);
  }
}

async function handleVoiceCards(stream) {
  let text;
  try {
    text = await stream.readAll();
  } catch {
    return;
  }
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    return;
  }
  const cards = data.cards;
  if (!Array.isArray(cards) || !cards.length) return;
  const node = cloneTemplate("tpl-assistant");
  thread.append(node);
  voiceAwaitingReply = node;
  if (data.turn_id && !voiceTurnNodes.has(data.turn_id)) voiceTurnNodes.set(data.turn_id, node);
  const wrap = node.querySelector(".cards");
  for (const item of cards) wrap.append(renderCard(item));
  wrap.hidden = false;
  scrollThread();
}

async function handleVoiceTrace(stream) {
  let trace;
  try {
    trace = JSON.parse(await stream.readAll());
  } catch {
    return;
  }
  const node = (trace.turn_id && voiceTurnNodes.get(trace.turn_id)) || voiceLatestAssistant;
  if (!node || !Array.isArray(trace.steps)) return;
  if (trace.turn_id && !voiceTurnNodes.has(trace.turn_id)) voiceTurnNodes.set(trace.turn_id, node);
  addTurn({
    kind: "spoken", trace, messageNode: node,
    monitorId: trace.monitor_id, language: trace.language,
  });
  loadMonitor();
  scrollThread();
}

function startVoiceTimer() {
  stopVoiceTimer();
  voiceSecondsLeft = VOICE_SESSION_SECONDS;
  voiceTimer.textContent = formatVoiceTime(voiceSecondsLeft);
  voiceTimerId = setInterval(() => {
    voiceSecondsLeft -= 1;
    voiceTimer.textContent = formatVoiceTime(Math.max(voiceSecondsLeft, 0));
    if (voiceSecondsLeft <= 0) {
      stopVoiceTimer();
      if (voiceRoom) voiceRoom.disconnect();
    }
  }, 1000);
}

function stopVoiceTimer() {
  if (voiceTimerId !== null) {
    clearInterval(voiceTimerId);
    voiceTimerId = null;
  }
}

function cleanupVoiceCall() {
  voiceRoom = null;
  resetVoiceUI();
  cleanupVoiceNotes();
  const note = cloneTemplate("tpl-assistant");
  setVoiceBubbleText(note, "Voice session ended.");
  thread.append(note);
  showVoiceThreadContent();
}

function loadVoiceSdk() {
  if (!voiceSdkPromise) {
    voiceSdkPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = VOICE_SDK_URL;
      script.onload = () => resolve(window.LivekitClient);
      script.onerror = () => reject(new Error("Couldn't load the voice SDK."));
      document.head.append(script);
    });
  }
  return voiceSdkPromise;
}

async function checkVoiceStatus() {
  let response;
  try {
    response = await fetch("/api/voice/status");
  } catch {
    return;
  }
  if (!response.ok) return;
  let data;
  try {
    data = await response.json();
  } catch {
    return;
  }
  if (data.enabled) {
    talkButton.hidden = false;
    voiceHint.hidden = false;
    document.getElementById("legend-voice").hidden = false;
  }
}

async function startVoice() {
  talkButton.disabled = true;
  voiceBar.hidden = false;
  voiceStatus.textContent = "Connecting…";

  let response;
  try {
    response = await fetch("/api/voice/token", { method: "POST" });
  } catch {
    renderVoiceError("Couldn't reach the voice service. Try again in a moment.");
    resetVoiceUI();
    return;
  }
  if (!response.ok) {
    renderVoiceError(await parseError(response));
    resetVoiceUI();
    return;
  }
  let data;
  try {
    data = await response.json();
  } catch {
    renderVoiceError("The voice service sent an unreadable reply. Try again in a moment.");
    resetVoiceUI();
    return;
  }

  let LivekitClient;
  try {
    LivekitClient = await loadVoiceSdk();
  } catch (error) {
    renderVoiceError(error && error.message ? error.message : "Couldn't load the voice SDK.");
    resetVoiceUI();
    return;
  }

  const room = new LivekitClient.Room();
  voiceRoom = room;

  room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === "audio") {
      const attached = track.attach();
      const elements = Array.isArray(attached) ? attached : [attached];
      for (const el of elements) {
        voiceAudioEls.push(el);
        document.body.append(el);
      }
    }
  });

  room.registerTextStreamHandler("lk.transcription", (stream, participantInfo) => {
    const isUserSpeech = participantInfo.identity === room.localParticipant.identity;
    handleVoiceTranscription(stream, isUserSpeech);
  });
  room.registerTextStreamHandler("ravo.cards", handleVoiceCards);
  room.registerTextStreamHandler("ravo.trace", handleVoiceTrace);
  room.on(LivekitClient.RoomEvent.ParticipantConnected, markAssistantPresent);
  room.on(LivekitClient.RoomEvent.Disconnected, cleanupVoiceCall);

  try {
    await room.connect(data.url, data.token);
  } catch (error) {
    voiceRoom = null;
    renderVoiceError(error && error.message ? error.message : "Voice couldn't connect.");
    resetVoiceUI();
    return;
  }

  // Browsers with strict autoplay rules (Safari) only play the agent's voice
  // after an explicit start inside the user's Talk gesture.
  try {
    await room.startAudio();
  } catch {}

  try {
    await room.localParticipant.setMicrophoneEnabled(true);
  } catch {
    renderVoiceError(
      "Microphone access is blocked. Allow it in your browser to talk, or type your question instead."
    );
    room.disconnect();
    return;
  }

  // An idle assistant is asleep and takes a few seconds to join; the call's
  // three minutes start when it arrives, not while it wakes.
  if (room.remoteParticipants.size > 0) {
    markAssistantPresent();
  } else {
    voiceStatus.textContent = "Starting the assistant… up to 20 seconds";
  }
}

function markAssistantPresent() {
  voiceStatus.textContent = "Listening";
  if (voiceTimerId === null) startVoiceTimer();
}

/* ---------- live monitor ---------- */
function formatDuration(ms) {
  if (ms == null) return "–";
  return ms >= 1000 ? (ms / 1000).toFixed(1) + " s" : Math.round(ms) + " ms";
}

function firstSoundLabel(trace) {
  const firstSoundMs = trace.first_sound_ms;
  return Number.isFinite(firstSoundMs) && firstSoundMs < trace.total_ms
    ? "first sound " + (firstSoundMs / 1000).toFixed(1) + " s"
    : "";
}

function noteMyAnswer(id, number) {
  if (!id) return;
  myAnswers.set(id, number);
  monLatestMine = id;
  monInspectedId = id;
}

function monShownTurns(data) {
  const turns = data.turns || [];
  const cap = window.matchMedia("(max-width: 520px)").matches ? 24 : turns.length;
  return turns.slice(-cap);
}

function monScaleTop(turns) {
  const totals = turns.map((turn) => turn.total_ms).sort((a, b) => a - b);
  const p95 = totals[Math.floor(0.95 * (totals.length - 1))] || 3000;
  return Math.max(1000, Math.ceil(p95 / 1000) * 1000);
}

function monDescribe(turn, index) {
  const how = turn.spoken ? "Spoken answer" : "Typed answer";
  const cache = turn.cache === "hit" ? ", question served from the cache" : "";
  return how + " " + (index + 1) + ", " + formatDuration(turn.total_ms) + cache;
}

function renderMonitorGrid(turns) {
  monChart.querySelectorAll(".mon-gridline").forEach((gridline) => gridline.remove());
  if (!turns.length) return;
  const top = monScaleTop(turns);
  for (let s = 1000; s <= top; s += 1000) {
    const gridline = document.createElement("div");
    gridline.className = "mon-gridline";
    gridline.style.bottom = (s / top) * 100 + "%";
    const label = document.createElement("span");
    label.className = "num";
    label.textContent = s / 1000 + " s";
    gridline.append(label);
    monChart.prepend(gridline);
  }
}

function renderMonitorFrames(turns) {
  const framesEl = monFramesEl;
  framesEl.textContent = "";
  const entering = !monFirstRender && !REDUCED_MOTION.matches;
  monFirstRender = false;
  if (!turns.length) {
    monPrevIds = new Set();
    return;
  }

  const prevIds = monPrevIds;
  const newIds = new Set();
  const fresh = [];
  const top = monScaleTop(turns);

  turns.forEach((turn, index) => {
    const frame = document.createElement("button");
    frame.type = "button";
    frame.className = "mon-frame";
    frame.setAttribute("role", "listitem");
    frame.dataset.id = turn.id;
    frame.setAttribute("aria-label", monDescribe(turn, index));
    if (turn.spoken) frame.classList.add("spoken");
    const mine = myAnswers.get(turn.id);
    if (mine) {
      frame.classList.add("mine");
      frame.dataset.turn = "T" + mine;
    }
    newIds.add(turn.id);

    const stack = document.createElement("div");
    stack.className = "mon-stack";
    stack.style.height = Math.min(100, (turn.total_ms / top) * 100) + "%";
    const parts = [
      ["listen", turn.listen_ms],
      ["search", turn.search_ms],
      ["embed", turn.embed_ms],
      ["llm", turn.llm_ms],
      ["speak", turn.speak_ms],
    ];
    const sum = parts.reduce((acc, [, ms]) => acc + ms, 0) || 1;
    for (const [kind, ms] of parts) {
      if (!ms) continue;
      const band = document.createElement("div");
      band.className = "mon-band mon-b-" + kind;
      band.style.flex = "0 0 " + (ms / sum) * 100 + "%";
      stack.append(band);
    }
    const isOwnLatest = mine && turn.id === monLatestMine;
    if (isOwnLatest) {
      const marker = document.createElement("span");
      marker.className = "marker mon-marker";
      marker.textContent = "T" + mine;
      stack.append(marker);
    }
    if (entering && !prevIds.has(turn.id)) {
      frame.classList.add("entering");
      fresh.push(frame);
    }
    frame.append(stack);

    frame.addEventListener("mouseenter", () => inspectMonitorFrame(frame, turns));
    frame.addEventListener("focus", () => inspectMonitorFrame(frame, turns));
    frame.addEventListener("click", () => {
      inspectMonitorFrame(frame, turns);
      if (frame.dataset.turn) activateTurn(frame.dataset.turn, { scroll: false });
    });
    framesEl.append(frame);
  });

  monPrevIds = newIds;

  if (fresh.length) {
    requestAnimationFrame(() => requestAnimationFrame(() => {
      for (const frame of fresh) frame.classList.remove("entering");
    }));
  }
}

function monReading(swatchClass, label, ms, tag) {
  const reading = document.createElement("div");
  reading.className = "reading";
  const swatch = document.createElement("span");
  swatch.className = "swatch " + swatchClass;
  reading.append(swatch);
  reading.append(document.createTextNode(label));
  if (tag) {
    const badge = document.createElement("span");
    badge.className = tag[0];
    badge.textContent = tag[1];
    reading.append(badge);
  }
  const value = document.createElement("b");
  value.className = "num";
  value.textContent = formatDuration(ms);
  reading.append(value);
  return reading;
}

function inspectMonitorFrame(frame, turns) {
  monInspectedId = frame.dataset.id;
  for (const other of monFramesEl.children) {
    other.classList.toggle("inspected", other === frame);
  }
  const turn = turns.find((t) => t.id === frame.dataset.id);
  if (!turn) return;

  const mine = myAnswers.get(turn.id);
  let who;
  if (mine) {
    who = "Your answer, T" + mine;
  } else if (turn === turns[turns.length - 1]) {
    who = "Latest answer";
  } else {
    who = "Answer " + (turns.indexOf(turn) + 1) + " of " + turns.length;
  }

  monInspectTitle.textContent = "";
  const lead = document.createElement("b");
  lead.textContent = who;
  monInspectTitle.append(lead);
  monInspectTitle.append(
    document.createTextNode(", " + (turn.spoken ? "spoken" : "typed") + ", " + formatDuration(turn.total_ms) + " in all")
  );

  monInspectReadings.textContent = "";
  if (turn.spoken) monInspectReadings.append(monReading("sw-listen", "Hearing you finish", turn.listen_ms));
  monInspectReadings.append(monReading("sw-llm", "Thinking and writing", turn.llm_ms));
  const embedTag =
    turn.cache === "hit" ? ["cache-hit", "cache hit"]
    : turn.cache === "miss" ? ["cache-miss", "new question"] : null;
  monInspectReadings.append(monReading("sw-embed", "Understanding the question", turn.embed_ms, embedTag));
  monInspectReadings.append(monReading("sw-search", "Searching", turn.search_ms));
  if (turn.spoken) monInspectReadings.append(monReading("sw-listen", "Starting to speak", turn.speak_ms));
}

function resolveInspected(turns) {
  const ids = new Set(turns.map((turn) => turn.id));
  if (monInspectedId && ids.has(monInspectedId)) return monInspectedId;
  return turns.length ? turns[turns.length - 1].id : null;
}

function inspectMonitorId(id, turns) {
  const frame = Array.from(monFramesEl.children).find((f) => f.dataset.id === id);
  if (!frame) {
    monInspectTitle.textContent = "";
    monInspectReadings.textContent = "";
    return;
  }
  inspectMonitorFrame(frame, turns);
}

function renderMonitorBars(summary) {
  const rows = [
    ["Thinking and writing", summary.median_llm_ms, "llm"],
    ["Understanding a new question", summary.median_embed_miss_ms, "embed"],
    ["Understanding a repeat, from cache", summary.median_embed_hit_ms, "embed"],
    ["Searching", summary.median_search_ms, "search"],
  ];
  const max = Math.max(...rows.map((row) => (row[1] == null ? 0 : row[1])));
  monBars.textContent = "";
  for (const [label, ms, kind] of rows) {
    const bar = document.createElement("div");
    bar.className = "mon-bar" + (ms === max && max > 0 ? " hot" : "");
    const labels = document.createElement("div");
    labels.className = "mon-bar-label";
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("b");
    value.className = "num";
    value.textContent = formatDuration(ms);
    labels.append(name, value);

    const track = document.createElement("div");
    track.className = "mon-track";
    const fill = document.createElement("div");
    fill.className = "mon-fill mon-fill-" + kind;
    fill.style.width = ms == null ? "0" : (ms / max) * 100 + "%";
    track.append(fill);

    bar.append(labels, track);
    monBars.append(bar);
  }
}

function renderMonitorVerdict(summary) {
  monVerdict.textContent = "";
  const llm = summary.median_llm_ms;
  const search = summary.median_search_ms;
  if (llm > 0 && search > 0) {
    const times = document.createElement("b");
    times.className = "num";
    times.textContent = Math.round(llm / search) + "×";
    monVerdict.append(
      document.createTextNode("The model takes "),
      times,
      document.createTextNode(" longer than the database. Speed comes from the model, not the search.")
    );
    monVerdict.hidden = false;
  } else {
    monVerdict.hidden = true;
  }
}

function renderMonitor(data) {
  const summary = data.summary || {};
  const turns = monShownTurns(data);
  const empty = turns.length === 0;
  nextMedian.textContent = formatDuration(summary.median_total_ms);

  monEmpty.hidden = !empty;
  if (empty) {
    monCount.textContent = "–";
    monTotal.textContent = "–";
    monCache.textContent = "–";
  } else {
    const spoken = turns.filter((turn) => turn.spoken).length;
    const label = "Last " + turns.length + (turns.length === 1 ? " answer" : " answers");
    monCount.textContent = spoken > 0 ? label + ", " + spoken + " spoken" : label;
    monTotal.textContent = formatDuration(summary.median_total_ms);
    monCache.textContent = summary.cache_hit_rate == null ? "–" : Math.round(summary.cache_hit_rate * 100) + "%";
  }

  renderMonitorGrid(turns);
  renderMonitorFrames(turns);
  renderMonitorBars(summary);
  renderMonitorVerdict(summary);
  inspectMonitorId(resolveInspected(turns), turns);
}

async function loadMonitor() {
  // A refresh asked for mid-flight (say, right after the shopper's own answer) runs
  // once the current one lands, so their new column is never missed.
  if (monInFlight) {
    monReloadQueued = true;
    return;
  }
  monInFlight = true;
  try {
    const response = await fetch("/api/monitor");
    if (!response.ok) return;
    const data = await response.json();
    renderMonitor(data);
  } catch {
    // keep whatever is on screen
  } finally {
    monInFlight = false;
    if (monReloadQueued) {
      monReloadQueued = false;
      loadMonitor();
    }
  }
}

function updateMonitorPolling() {
  const active = monVisible && document.visibilityState === "visible";
  if (active && monPolling === null) {
    monPolling = setInterval(loadMonitor, MONITOR_POLL_MS);
  } else if (!active && monPolling !== null) {
    clearInterval(monPolling);
    monPolling = null;
  }
}

function setupMonitorPolling() {
  if (typeof IntersectionObserver === "undefined") {
    monVisible = true;
  } else {
    const observer = new IntersectionObserver((entries) => {
      monVisible = entries.some((entry) => entry.isIntersecting);
      updateMonitorPolling();
    });
    observer.observe(monRoll);
  }
  document.addEventListener("visibilitychange", () => {
    updateMonitorPolling();
    if (document.visibilityState === "visible" && monVisible) loadMonitor();
  });
  updateMonitorPolling();
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
  scoreboardLede = document.getElementById("scoreboard-lede");
  plugCount = document.getElementById("plug-count");
  nextMedian = document.getElementById("next-median");
  nextNegation = document.getElementById("next-negation");
  nextDiversity = document.getElementById("next-diversity");
  nextHit1 = document.getElementById("next-hit1");
  yourTurns = document.getElementById("your-turns");
  turnCountEl = document.getElementById("turn-count");
  turnsEmpty = document.getElementById("turns-empty");
  turnList = document.getElementById("turn-list");
  if (window.matchMedia("(max-width: 640px)").matches) yourTurns.open = false;

  monCount = document.getElementById("mon-count");
  monTotal = document.getElementById("mon-total");
  monCache = document.getElementById("mon-cache");
  monChart = document.getElementById("mon-chart");
  monFramesEl = document.getElementById("mon-frames");
  monEmpty = document.getElementById("mon-empty");
  monInspectTitle = document.getElementById("mon-inspect-title");
  monInspectReadings = document.getElementById("mon-inspect-readings");
  monBars = document.getElementById("mon-bars");
  monVerdict = document.getElementById("mon-verdict");
  monRoll = document.querySelector(".mon-roll");
  setupMonitorPolling();

  askForm.addEventListener("submit", (event) => {
    event.preventDefault();
    send(askInput.value);
  });
  for (const starter of starters) {
    starter.addEventListener("click", () => send(starter.dataset.q));
  }

  talkButton = document.getElementById("talk-button");
  voiceHint = document.getElementById("voice-hint");
  voiceBar = document.getElementById("voice-bar");
  voiceStatus = document.getElementById("voice-status");
  voiceTimer = document.getElementById("voice-timer");
  voiceEnd = document.getElementById("voice-end");
  talkButton.addEventListener("click", startVoice);
  voiceEnd.addEventListener("click", () => {
    if (voiceRoom) voiceRoom.disconnect();
  });

  checkVoiceStatus();
  fetch("/api/health").catch(() => {});
  loadScoreboard();
  loadMonitor();
}

document.addEventListener("DOMContentLoaded", init);
