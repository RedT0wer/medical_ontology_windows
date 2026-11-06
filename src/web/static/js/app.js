/**
 * Medical Ontology & Questionnaires - Client Application
 */

// Application State
const state = {
  activeTab: "questionnaires",
  questionnaires: [],
  selectedDocId: null,
  currentDocData: null,
  docViewMode: "both", // "both", "json", "raw"
  graphData: null,
  network: null,
  selectedNodeId: null,
  currentNodeData: null,
  currentTableFilter: "all", // "all", "filled", "empty"
  graphLayout: "hierarchical-ud", // "hierarchical-ud", "hierarchical-lr", "physics"
};

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSearch();
  initGraphControls();
  loadQuestionnaires();

  const editBtn = document.getElementById("btnEditNodeName");
  if (editBtn) {
    editBtn.addEventListener("click", (e) => {
      e.preventDefault();
      startRenamingNode();
    });
  }

  // Handle URL hash routing if present
  window.addEventListener("hashchange", handleHashRouting);
  handleHashRouting();
});

function handleHashRouting() {
  const hash = window.location.hash;
  if (!hash) return;

  if (hash.startsWith("#doc-")) {
    const docId = hash.replace("#doc-", "");
    switchTab("questionnaires");
    selectQuestionnaire(docId);
  } else if (hash.startsWith("#node-")) {
    const nodeId = hash.replace("#node-", "");
    switchTab("graph");
    if (state.network) {
      openNodeInspector(nodeId);
    } else {
      loadGraph().then(() => openNodeInspector(nodeId));
    }
  } else if (hash === "#graph") {
    switchTab("graph");
  } else if (hash === "#questionnaires") {
    switchTab("questionnaires");
  }
}

// ============================================================================
// Tab Navigation
// ============================================================================
function initTabs() {
  document.querySelectorAll(".nav-item[data-tab]").forEach((item) => {
    item.addEventListener("click", () => {
      const targetTab = item.getAttribute("data-tab");
      switchTab(targetTab);
    });
  });
}

function switchTab(tabName) {
  if (state.activeTab === tabName) return;
  state.activeTab = tabName;

  // Update sidebar active state
  document.querySelectorAll(".nav-item[data-tab]").forEach((item) => {
    item.classList.toggle("active", item.getAttribute("data-tab") === tabName);
  });

  // Update panels active state
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `view-${tabName}`);
  });

  // If switched to graph, ensure graph is loaded and canvas resized
  if (tabName === "graph") {
    if (!state.graphData) {
      loadGraph();
    } else if (state.network) {
      setTimeout(() => state.network.fit(), 100);
    }
  }
}

// ============================================================================
// Tab 1: Questionnaires Management
// ============================================================================
async function loadQuestionnaires() {
  const listEl = document.getElementById("qListContainer");
  const countBadge = document.getElementById("qTotalCount");

  listEl.innerHTML = `<div class="empty-placeholder" style="padding: 20px;"><span style="font-size: 20px;">⏳</span> Загрузка анкет...</div>`;

  try {
    const res = await fetch("/api/documents");
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    state.questionnaires = data;

    if (countBadge) countBadge.textContent = data.length;

    renderQuestionnairesList(data);

    // Auto-select first questionnaire if none selected
    if (data.length > 0 && !state.selectedDocId) {
      selectQuestionnaire(data[0].id);
    }
  } catch (err) {
    console.error("Ошибка загрузки анкет:", err);
    listEl.innerHTML = `<div class="empty-placeholder" style="color: #ef4444; padding: 20px;">⚠️ Ошибка загрузки списка: ${err.message}</div>`;
  }
}

function renderQuestionnairesList(items) {
  const listEl = document.getElementById("qListContainer");
  if (!items || items.length === 0) {
    listEl.innerHTML = `<div class="empty-placeholder" style="padding: 20px;">Анкеты не найдены</div>`;
    return;
  }

  listEl.innerHTML = items
    .map((item) => {
      const isSelected = String(item.id) === String(state.selectedDocId);
      const dateFormatted = item.consultation_date
        ? new Date(item.consultation_date).toLocaleDateString("ru-RU", {
            day: "2-digit",
            month: "2-digit",
            year: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })
        : "Дата не указана";

      return `
      <div class="q-card ${isSelected ? "selected" : ""}" data-doc-id="${item.id}" onclick="selectQuestionnaire('${item.id}')">
        <div class="q-card-top">
          <span class="q-id-pill">ID #${item.id}</span>
          <span class="q-date">${dateFormatted}</span>
        </div>
        <div class="q-patient-name">${escapeHtml(item.patient_name || `Пациент #${item.patient_id}`)}</div>
        <div class="q-card-footer">
          <span>${item.doctor ? escapeHtml(item.doctor) : "Врач не указан"}</span>
          <span class="q-feat-count">📊 ${item.values_count || 0} призн.</span>
        </div>
      </div>
    `;
    })
    .join("");
}

async function selectQuestionnaire(docId) {
  state.selectedDocId = docId;

  // Highlight selected card
  document.querySelectorAll(".q-card").forEach((card) => {
    card.classList.toggle("selected", card.getAttribute("data-doc-id") === String(docId));
  });

  const detailPane = document.getElementById("qDetailContainer");
  detailPane.innerHTML = `<div class="empty-placeholder"><span class="empty-icon">⏳</span><span>Загрузка данных анкеты #${docId}...</span></div>`;

  try {
    const res = await fetch(`/api/documents/${docId}`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const doc = await res.json();
    state.currentDocData = doc;
    renderQuestionnaireDetail(doc);
    window.location.hash = `doc-${docId}`;
  } catch (err) {
    console.error(`Ошибка загрузки анкеты ${docId}:`, err);
    detailPane.innerHTML = `<div class="empty-placeholder" style="color: #ef4444;"><span class="empty-icon">⚠️</span><span>Не удалось загрузить анкету #${docId}: ${err.message}</span></div>`;
  }
}

function setDocViewMode(mode) {
  state.docViewMode = mode;
  if (state.currentDocData) {
    renderQuestionnaireDetail(state.currentDocData);
  }
}

function copyRawText() {
  if (!state.currentDocData || !state.currentDocData.raw_text) return;
  navigator.clipboard
    .writeText(state.currentDocData.raw_text)
    .then(() => showToast("✅ Исходный текст скопирован в буфер обмена!"))
    .catch(() => showToast("❌ Не удалось скопировать"));
}

function renderQuestionnaireDetail(doc) {
  const detailPane = document.getElementById("qDetailContainer");
  const jsonContent = doc.json || {};
  const admin = doc.administrative || {};
  const rawText = doc.raw_text || "";

  const dateStr = doc.consultation_date
    ? new Date(doc.consultation_date).toLocaleString("ru-RU")
    : "Не указана";

  const rawJsonString = JSON.stringify(jsonContent, null, 2);
  const viewMode = state.docViewMode || "both";
  const isSingle = viewMode !== "both";

  detailPane.innerHTML = `
    <!-- Top Header -->
    <div class="q-detail-header">
      <div class="q-title-group">
        <h2>
          <span>${escapeHtml(doc.patient_name || `Пациент #${doc.patient_id}`)}</span>
          <span class="q-id-pill" style="font-size: 13px;">ID #${doc.id}</span>
        </h2>
        <div class="q-meta-chips">
          <span class="q-meta-chip">📅 <b>Осмотр:</b> ${dateStr}</span>
          ${admin.doctor ? `<span class="q-meta-chip">👨‍⚕️ <b>Врач:</b> ${escapeHtml(admin.doctor)}</span>` : ""}
          ${admin.institution ? `<span class="q-meta-chip" title="${escapeHtml(admin.institution)}">🏥 ${escapeHtml(admin.institution.slice(0, 45))}...</span>` : ""}
          <span class="q-meta-chip">📊 <b>Признаков:</b> ${doc.values_count || 0}</span>
        </div>
      </div>
      <div class="header-actions">
        <!-- View mode switcher tabs -->
        <div class="view-mode-tabs">
          <button class="view-mode-btn ${viewMode === 'both' ? 'active' : ''}" onclick="setDocViewMode('both')">📑 Оба блока</button>
          <button class="view-mode-btn ${viewMode === 'json' ? 'active' : ''}" onclick="setDocViewMode('json')">{ } Только JSON</button>
          <button class="view-mode-btn ${viewMode === 'raw' ? 'active' : ''}" onclick="setDocViewMode('raw')">📄 Исходный текст</button>
        </div>

        <a class="btn btn-default" href="/api/documents/${doc.id}/raw-json" download="questionnaire_${doc.id}.json" target="_blank" title="Скачать чистый файл .json">
          ⬇️ Скачать .json
        </a>
      </div>
    </div>

    <!-- Viewer Body -->
    <div class="q-json-container">
      ${
        viewMode === "both" || viewMode === "json"
          ? `
        <div class="json-card ${isSingle ? 'single-view' : ''}">
          <div class="json-toolbar">
            <span class="json-toolbar-title">{ } Скомпилированный нормализованный JSON</span>
            <div class="json-toolbar-actions">
              <span style="font-family: monospace; font-size: 11px;">${rawJsonString.length} симв.</span>
              <button class="btn btn-default btn-sm" onclick="copyCurrentJson()">
                📋 Копировать JSON
              </button>
            </div>
          </div>
          <pre class="json-view-pre" id="jsonPreCode">${syntaxHighlightJson(jsonContent)}</pre>
        </div>
      `
          : ""
      }

      ${
        (viewMode === "both" || viewMode === "raw") && rawText
          ? `
        <div class="json-card ${isSingle ? 'single-view' : ''}">
          <div class="json-toolbar">
            <span class="json-toolbar-title">📄 Исходный клинический протокол (из базы данных)</span>
            <div class="json-toolbar-actions">
              <span style="font-family: monospace; font-size: 11px;">${rawText.length} симв.</span>
              <button class="btn btn-default btn-sm" onclick="copyRawText()">
                📋 Копировать текст
              </button>
            </div>
          </div>
          <pre class="raw-text-pre">${escapeHtml(rawText)}</pre>
        </div>
      `
          : ""
      }
    </div>
  `;
}

function copyCurrentJson() {
  if (!state.currentDocData || !state.currentDocData.json) return;
  const jsonText = JSON.stringify(state.currentDocData.json, null, 2);
  navigator.clipboard
    .writeText(jsonText)
    .then(() => showToast("✅ JSON скопирован в буфер обмена!"))
    .catch(() => showToast("❌ Не удалось скопировать"));
}

function initSearch() {
  const searchInput = document.getElementById("qSearchInput");
  if (!searchInput) return;

  searchInput.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase().trim();
    if (!q) {
      renderQuestionnairesList(state.questionnaires);
      return;
    }
    const filtered = state.questionnaires.filter((item) => {
      const name = (item.patient_name || "").toLowerCase();
      const id = String(item.id);
      const pid = String(item.patient_id || "").toLowerCase();
      const doc = (item.doctor || "").toLowerCase();
      const inst = (item.institution || "").toLowerCase();
      return name.includes(q) || id.includes(q) || pid.includes(q) || doc.includes(q) || inst.includes(q);
    });
    renderQuestionnairesList(filtered);
  });
}

// ============================================================================
// Tab 2: Feature Ontology Graph (Vis.js)
// ============================================================================
async function loadGraph() {
  const container = document.getElementById("graphCanvas");
  container.innerHTML = `<div class="empty-placeholder"><span class="empty-icon">⏳</span><span>Загрузка графа онтологии признаков...</span></div>`;

  try {
    const res = await fetch("/api/ontology/graph");
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    state.graphData = data;

    // Update stats bar
    document.getElementById("statTotalNodes").textContent = data.stats.total_nodes;
    document.getElementById("statLeafNodes").textContent = data.stats.leaf_nodes;
    document.getElementById("statDocsCount").textContent = data.stats.total_documents;

    renderVisNetwork(data);
  } catch (err) {
    console.error("Ошибка загрузки графа:", err);
    container.innerHTML = `<div class="empty-placeholder" style="color: #ef4444;"><span class="empty-icon">⚠️</span><span>Ошибка построения графа: ${err.message}</span></div>`;
  }
}

function getVisNetworkOptions(layoutType) {
  const isHierarchical = layoutType.startsWith("hierarchical");
  const direction = layoutType === "hierarchical-lr" ? "LR" : "UD";

  return {
    autoResize: true,
    nodes: {
      shape: "dot",
      size: 20,
      font: {
        face: "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
        size: 12,
        color: "#1e293b",
      },
      borderWidth: 2,
      shadow: true,
    },
    edges: {
      width: 1.5,
      arrows: { to: { enabled: true, scaleFactor: 0.8 } },
      smooth: isHierarchical
        ? { enabled: true, type: "cubicBezier", forceDirection: direction === "UD" ? "vertical" : "horizontal", roundness: 0.5 }
        : { enabled: true, type: "continuous" },
      color: { color: "#94a3b8", highlight: "#2563eb", hover: "#60a5fa" },
    },
    layout: isHierarchical
      ? {
          hierarchical: {
            enabled: true,
            direction: direction,
            sortMethod: "directed",
            levelSeparation: direction === "UD" ? 140 : 200,
            nodeSpacing: direction === "UD" ? 180 : 120,
            treeSpacing: 220,
          },
        }
      : {
          hierarchical: { enabled: false },
        },
    // Для иерархического графа полностью отключаем физику, чтобы исключить любые скачки масштаба и смещения узлов
    physics: isHierarchical
      ? false
      : {
          forceAtlas2Based: {
            gravitationalConstant: -70,
            centralGravity: 0.015,
            springLength: 100,
            springConstant: 0.08,
            damping: 0.4,
          },
          solver: "forceAtlas2Based",
          stabilization: { enabled: true, iterations: 150, fit: false },
        },
    interaction: {
      hover: true,
      tooltipDelay: 80,
      hideEdgesOnDrag: false,
      zoomView: true,
      dragView: true,
    },
  };
}

function renderVisNetwork(graphData) {
  const container = document.getElementById("graphCanvas");
  container.innerHTML = "";

  // Преобразуем подсказки каждого узла в DOM-элементы без сырого HTML-текста
  graphData.nodes.forEach((n) => {
    const tipEl = document.createElement("div");
    tipEl.style.fontFamily = "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif";
    tipEl.style.fontSize = "12px";
    tipEl.style.lineHeight = "1.4";
    tipEl.style.maxWidth = "280px";

    const nameEl = document.createElement("div");
    nameEl.style.fontWeight = "bold";
    nameEl.style.fontSize = "13px";
    nameEl.style.color = "#0f172a";
    nameEl.style.marginBottom = "3px";
    nameEl.textContent = n.display_name || n.key_name || n.label;
    tipEl.appendChild(nameEl);

    const pathEl = document.createElement("div");
    pathEl.style.color = "#475569";
    pathEl.style.fontSize = "11px";
    pathEl.style.marginBottom = "2px";
    pathEl.textContent = `Путь: ${n.full_path || ""}`;
    tipEl.appendChild(pathEl);

    const typeEl = document.createElement("div");
    typeEl.style.color = "#475569";
    typeEl.style.fontSize = "11px";
    typeEl.style.marginBottom = "2px";
    typeEl.textContent = `Тип: ${n.data_type || "—"}${n.unit ? " (" + n.unit + ")" : ""}`;
    tipEl.appendChild(typeEl);

    if (n.records_count !== undefined) {
      const recsEl = document.createElement("div");
      recsEl.style.color = "#2563eb";
      recsEl.style.fontSize = "11px";
      recsEl.textContent = `Заполнено: ${n.records_count} из ${n.total_documents || 0} анкет`;
      tipEl.appendChild(recsEl);
    }

    n.title = tipEl;
  });

  state.visNodes = new vis.DataSet(graphData.nodes);
  state.visEdges = new vis.DataSet(graphData.edges);

  const data = { nodes: state.visNodes, edges: state.visEdges };
  const options = getVisNetworkOptions(state.graphLayout);

  state.network = new vis.Network(container, data, options);

  // Однократно центрируем граф при первоначальной загрузке
  state.network.once("afterDrawing", () => {
    state.network.fit({ animation: false });
  });

  // Клик по узлу: открывает инспектор БЕЗ сброса масштаба и смещения холста
  state.network.on("click", (params) => {
    if (params.nodes && params.nodes.length > 0) {
      const selectedId = params.nodes[0];
      if (selectedId === "root_0") {
        showToast("Корневой узел схемы онтологии");
        return;
      }
      openNodeInspector(selectedId);
    }
  });

  // Двойной клик: центрирование узла с сохранением текущего масштаба
  state.network.on("doubleClick", (params) => {
    if (params.nodes && params.nodes.length > 0) {
      const currentScale = state.network.getScale();
      state.network.focus(params.nodes[0], {
        scale: currentScale,
        animation: { duration: 300, easingFunction: "easeInOutQuad" },
      });
    }
  });
}

function initGraphControls() {
  // Layout switch
  const layoutSelect = document.getElementById("graphLayoutSelect");
  if (layoutSelect) {
    layoutSelect.addEventListener("change", (e) => {
      state.graphLayout = e.target.value;
      if (state.network) {
        state.network.setOptions(getVisNetworkOptions(state.graphLayout));
        setTimeout(() => state.network.fit(), 200);
      }
    });
  }

  // Graph Search (плавный переход к узлу без скачка масштаба)
  const graphSearch = document.getElementById("graphSearchInput");
  if (graphSearch) {
    graphSearch.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!q || !state.graphData || !state.network) return;

      const match = state.graphData.nodes.find(
        (n) =>
          (n.display_name && n.display_name.toLowerCase().includes(q)) ||
          (n.full_path && n.full_path.toLowerCase().includes(q)) ||
          (n.key_name && n.key_name.toLowerCase().includes(q))
      );

      if (match) {
        state.network.selectNodes([match.id]);
        const currentScale = state.network.getScale();
        state.network.focus(match.id, {
          scale: currentScale,
          animation: { duration: 400, easingFunction: "easeInOutQuad" },
        });
      }
    });
  }

  // Клавиатурные события для поля переименования узла
  const renameInput = document.getElementById("renameNodeInput");
  if (renameInput) {
    renameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        saveNodeRename();
      } else if (e.key === "Escape") {
        e.preventDefault();
        cancelNodeRename();
      }
    });
  }

  // Reset View / Fit button
  const fitBtn = document.getElementById("btnFitGraph");
  if (fitBtn) {
    fitBtn.addEventListener("click", () => {
      if (state.network) state.network.fit({ animation: true });
    });
  }
}

// ============================================================================
// Node Inspector & Mini Table of Feature Across All Questionnaires
// ============================================================================
async function openNodeInspector(nodeId) {
  state.selectedNodeId = nodeId;
  cancelNodeRename();
  const drawer = document.getElementById("nodeInspectorDrawer");
  drawer.classList.add("open");

  const titleEl = document.getElementById("drawerTitle");
  const pathEl = document.getElementById("drawerPath");
  const bodyEl = document.getElementById("drawerBody");

  titleEl.textContent = "Загрузка данных признака...";
  pathEl.textContent = "...";
  bodyEl.innerHTML = `<div class="empty-placeholder" style="padding: 40px;"><span class="empty-icon">⏳</span><span>Загрузка значений по всем анкетам...</span></div>`;

  try {
    const res = await fetch(`/api/ontology/nodes/${nodeId}/values`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    state.currentNodeData = data;
    renderNodeInspector(data);

    // Обновляем URL без срабатывания hashchange для предотвращения скачков масштаба
    if (window.location.hash !== `#node-${nodeId}`) {
      history.replaceState(null, "", `#node-${nodeId}`);
    }
  } catch (err) {
    console.error("Ошибка загрузки значений признака:", err);
    bodyEl.innerHTML = `<div class="empty-placeholder" style="color: #ef4444;"><span class="empty-icon">⚠️</span><span>Ошибка: ${err.message}</span></div>`;
  }
}

function closeNodeInspector() {
  const drawer = document.getElementById("nodeInspectorDrawer");
  drawer.classList.remove("open");
  state.selectedNodeId = null;
  state.currentNodeData = null;
  cancelNodeRename();
  if (window.location.hash.startsWith("#node-")) {
    history.replaceState(null, "", "#graph");
  }
}

function startRenamingNode() {
  const form = document.getElementById("renameNodeForm");
  const input = document.getElementById("renameNodeInput");
  if (!form || !input) return;

  // Извлекаем текущее имя из доступных источников (не блокируя показ формы)
  let currentName = "";
  if (state.currentNodeData && state.currentNodeData.node) {
    currentName = state.currentNodeData.node.display_name || state.currentNodeData.node.key_name || "";
  } else if (state.graphData && state.graphData.nodes && state.selectedNodeId) {
    const gNode = state.graphData.nodes.find((n) => n.id == state.selectedNodeId);
    if (gNode) {
      currentName = gNode.display_name || gNode.key_name || gNode.label || "";
    }
  }
  if (!currentName) {
    const titleEl = document.getElementById("drawerTitle");
    if (titleEl) {
      const text = titleEl.textContent.trim();
      if (text !== "Загрузка данных признака..." && text !== "Параметр онтологии") {
        currentName = text;
      }
    }
  }

  input.value = currentName;
  form.style.display = "block";
  setTimeout(() => {
    input.focus();
    input.select();
  }, 30);
}
window.startRenamingNode = startRenamingNode;

function cancelNodeRename() {
  const form = document.getElementById("renameNodeForm");
  if (form) form.style.display = "none";
}
window.cancelNodeRename = cancelNodeRename;

async function saveNodeRename() {
  const nodeId = state.selectedNodeId;
  if (!nodeId) {
    showToast("Сначала выберите узел для переименования");
    return;
  }

  const input = document.getElementById("renameNodeInput");
  if (!input) return;
  const newName = input.value.trim();
  if (!newName) {
    showToast("Название не может быть пустым");
    return;
  }

  try {
    const res = await fetch(`/api/ontology/nodes/${nodeId}/rename`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: newName }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    // 1. Обновляем локальное состояние и заголовок инспектора
    if (state.currentNodeData && state.currentNodeData.node) {
      state.currentNodeData.node.display_name = newName;
    }
    const titleEl = document.getElementById("drawerTitle");
    if (titleEl) titleEl.textContent = newName;

    cancelNodeRename();

    // 2. Обновляем узел в датасете Vis.js
    if (state.visNodes) {
      const existing = state.visNodes.get(nodeId);
      if (existing) {
        state.visNodes.update({
          id: nodeId,
          label: newName,
          display_name: newName,
        });
      }
    }

    // 3. Обновляем в кэше графа
    if (state.graphData && state.graphData.nodes) {
      const gNode = state.graphData.nodes.find((n) => n.id == nodeId);
      if (gNode) {
        gNode.display_name = newName;
        gNode.label = newName;
      }
    }

    showToast(`Узел переименован: "${newName}"`);
  } catch (err) {
    console.error("Ошибка переименования узла:", err);
    showToast(`Ошибка: ${err.message}`);
  }
}
window.saveNodeRename = saveNodeRename;

function renderNodeInspector(data) {
  const node = data.node;
  const stats = data.stats;
  const docs = data.documents;

  const titleEl = document.getElementById("drawerTitle");
  const pathEl = document.getElementById("drawerPath");
  const bodyEl = document.getElementById("drawerBody");

  titleEl.textContent = node.display_name || node.key_name;
  pathEl.textContent = node.full_path;

  // Render Inspector Body
  bodyEl.innerHTML = `
    <!-- 1. Metadata card of the feature -->
    <div class="feat-meta-card">
      <div>
        <div class="meta-field-label">Тип данных (Data Type):</div>
        <div class="meta-field-val"><code>${node.data_type}</code>${node.unit ? ` (${node.unit})` : ""}</div>
      </div>
      <div>
        <div class="meta-field-label">Иерархический уровень:</div>
        <div class="meta-field-val">${node.is_leaf ? "🍃 Листовой признак" : "📁 Ветка / Категория"}</div>
      </div>
      <div>
        <div class="meta-field-label">ID узла в БД:</div>
        <div class="meta-field-val">#${node.id}</div>
      </div>
      <div>
        <div class="meta-field-label">Статус узла:</div>
        <div class="meta-field-val"><span style="color: #10b981;">✓ ${node.status || "approved"}</span></div>
      </div>
      <div style="grid-column: span 2;">
        <div class="meta-field-label">Отображаемое имя узла:</div>
        <div class="meta-field-val" style="display: flex; align-items: center; gap: 8px;">
          <strong style="color: #0f172a;">${escapeHtml(node.display_name || node.key_name)}</strong>
          <button type="button" class="btn-rename-trigger" onclick="startRenamingNode()" title="Переименовать узел">✎ Переименовать</button>
        </div>
      </div>
      ${
        node.aliases && node.aliases.length > 0
          ? `
        <div style="grid-column: span 2;">
          <div class="meta-field-label">Синонимы (Aliases):</div>
          <div class="meta-tags-cell" style="margin-top: 4px;">
            ${node.aliases.map((a) => `<span class="meta-pill">${escapeHtml(a)}</span>`).join("")}
          </div>
        </div>
      `
          : ""
      }
      ${
        data.children && data.children.length > 0
          ? `
        <div style="grid-column: span 2;">
          <div class="meta-field-label">Дочерние признаки (${data.children.length}):</div>
          <div class="meta-tags-cell" style="margin-top: 4px;">
            ${data.children
              .map(
                (c) =>
                  `<span class="meta-pill meta-pill-accent" style="cursor: pointer;" onclick="openNodeInspector(${c.id})" title="Открыть узел ${escapeHtml(c.full_path)}">🔹 ${escapeHtml(c.display_name || c.key_name)}</span>`
              )
              .join("")}
          </div>
        </div>
      `
          : ""
      }
    </div>

    <!-- 2. Summary stats across questionnaires -->
    <div class="drawer-stats-row">
      <div class="drawer-stat-item">
        <span>📋 Заполнено:</span>
        <strong style="color: #10b981;">${stats.filled_documents} из ${stats.total_documents}</strong>
        <span style="color: #64748b;">(${stats.fill_percentage}%)</span>
      </div>
      ${
        stats.numeric_avg !== null
          ? `
        <div class="drawer-stat-item" style="margin-left: auto;">
          <span>Ср. значение:</span>
          <strong>${stats.numeric_avg}${node.unit ? ` ${node.unit}` : ""}</strong>
          <span style="color: #64748b;">(min: ${stats.numeric_min}, max: ${stats.numeric_max})</span>
        </div>
      `
          : ""
      }
    </div>

    <!-- 3. Mini Table Section -->
    <div class="mini-table-section">
      <div class="section-head-row">
        <h4>Таблица значений у всех анкет</h4>
        <div class="filter-btn-group">
          <button class="filter-chip ${state.currentTableFilter === "all" ? "active" : ""}" onclick="filterMiniTable('all')">Все (${docs.length})</button>
          <button class="filter-chip ${state.currentTableFilter === "filled" ? "active" : ""}" onclick="filterMiniTable('filled')">С данными (${stats.filled_documents})</button>
          <button class="filter-chip ${state.currentTableFilter === "empty" ? "active" : ""}" onclick="filterMiniTable('empty')">Пустые (${stats.missing_documents})</button>
        </div>
      </div>

      <div class="mini-table-wrapper">
        <table class="mini-table">
          <thead>
            <tr>
              <th style="width: 75px;">Анкета</th>
              <th>Пациент</th>
              <th>Значение узла</th>
              <th>Метаданные (пара признак-документ)</th>
            </tr>
          </thead>
          <tbody id="miniTableBody">
            ${renderMiniTableRows(docs, state.currentTableFilter)}
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function filterMiniTable(filterType) {
  state.currentTableFilter = filterType;
  if (!state.currentNodeData) return;

  document.querySelectorAll(".filter-chip").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent.toLowerCase().includes(filterType === "all" ? "все" : filterType === "filled" ? "данными" : "пустые"));
  });

  const tbody = document.getElementById("miniTableBody");
  if (tbody) {
    tbody.innerHTML = renderMiniTableRows(state.currentNodeData.documents, filterType);
  }
}

function renderMiniTableRows(docs, filterType) {
  let filtered = docs;
  if (filterType === "filled") {
    filtered = docs.filter((d) => d.is_present);
  } else if (filterType === "empty") {
    filtered = docs.filter((d) => !d.is_present);
  }

  if (filtered.length === 0) {
    return `<tr><td colspan="4" style="text-align: center; color: #94a3b8; padding: 20px;">Записей не найдено</td></tr>`;
  }

  return filtered
    .map((d) => {
      const meta = d.metadata || {};
      const valBadge = d.is_present
        ? `<span class="val-badge filled">${escapeHtml(String(d.value))}</span>`
        : `<span class="val-badge empty">NaN (отсутствует)</span>`;

      // Build metadata badges
      const metaBadges = [];
      metaBadges.push(`<span class="meta-pill meta-pill-accent">document_id: ${d.document_id}</span>`);
      if (meta.action) {
        metaBadges.push(`<span class="meta-pill">action: ${escapeHtml(meta.action)}</span>`);
      }
      if (meta.similarity !== undefined && meta.similarity !== null) {
        metaBadges.push(`<span class="meta-pill">similarity: ${meta.similarity}</span>`);
      }
      if (meta.confidence !== undefined && meta.confidence !== null) {
        metaBadges.push(`<span class="meta-pill">conf: ${Math.round(meta.confidence * 100)}%</span>`);
      }
      if (meta.raw_key) {
        metaBadges.push(`<span class="meta-pill">raw_key: ${escapeHtml(meta.raw_key)}</span>`);
      }

      return `
      <tr class="clickable-row" onclick="jumpToQuestionnaire('${d.document_id}')" title="Кликните, чтобы открыть анкету #${d.document_id}">
        <td>
          <span class="q-id-pill" style="cursor: pointer;">#${d.document_id}</span>
        </td>
        <td>
          <div style="font-weight: 600; color: #1e293b;">${escapeHtml(d.patient_name)}</div>
          <div style="font-size: 11px; color: #64748b;">Пациент ID: ${escapeHtml(d.patient_id)}</div>
        </td>
        <td>${valBadge}</td>
        <td>
          <div class="meta-tags-cell">
            ${metaBadges.join("")}
          </div>
          ${
            d.raw_context
              ? `<div style="margin-top: 4px; font-size: 11px; color: #64748b; font-style: italic; max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${escapeHtml(d.raw_context)}">«${escapeHtml(d.raw_context)}»</div>`
              : ""
          }
        </td>
      </tr>
    `;
    })
    .join("");
}

function jumpToQuestionnaire(docId) {
  closeNodeInspector();
  switchTab("questionnaires");
  selectQuestionnaire(docId);
}

// ============================================================================
// Utilities
// ============================================================================
function escapeHtml(text) {
  if (text === null || text === undefined) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function syntaxHighlightJson(json) {
  if (typeof json !== "string") {
    json = JSON.stringify(json, null, 2);
  }
  json = escapeHtml(json);
  return json.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
    (match) => {
      let cls = "json-number";
      if (/^"/.test(match)) {
        if (/:$/.test(match)) {
          cls = "json-key";
        } else {
          cls = "json-string";
        }
      } else if (/true|false/.test(match)) {
        cls = "json-boolean";
      } else if (/null/.test(match)) {
        cls = "json-null";
      }
      return `<span class="${cls}">${match}</span>`;
    }
  );
}

function showToast(msg) {
  const toast = document.getElementById("toastMsg");
  if (!toast) return;
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 2500);
}
