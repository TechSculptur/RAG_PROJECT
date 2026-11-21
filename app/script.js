/**
 * Document Assistant — Frontend Application Script
 * With Live RAG Evaluation & Analytics Dashboard
 */

(function () {
    "use strict";

    // ================= CONFIGURATION & SESSION =================
    const CLOUD_FALLBACK = "https://rag-document-assistant-sfs0.onrender.com";
    const API_BASE = window.location.protocol.startsWith("http") ? "" : CLOUD_FALLBACK;

    // Persistent browser session ID
    let sessionId = sessionStorage.getItem("rag_session_id");
    if (!sessionId) {
        sessionId = "sess_" + Math.random().toString(36).substring(2, 10);
        sessionStorage.setItem("rag_session_id", sessionId);
    }

    // ================= STATE =================
    let chatHistory = [];
    let selectedFile = null;
    let isProcessing = false;
    let currentView = "chat";
    let activeDocumentName = null;
    let analyticsPollingTimer = null;
    let activePolls = new Set();

    // ================= DOM ELEMENTS =================
    const sidebar = document.getElementById("sidebar");
    const sidebarOverlay = document.getElementById("sidebar-overlay");
    const mobileToggleBtn = document.getElementById("mobile-toggle-btn");
    const mobileCloseBtn = document.getElementById("mobile-close-btn");

    // Views & Nav
    const viewChat = document.getElementById("view-chat");
    const viewAnalytics = document.getElementById("view-analytics");
    const navBtnChat = document.getElementById("nav-btn-chat");
    const navBtnAnalytics = document.getElementById("nav-btn-analytics");
    const viewTabChat = document.getElementById("view-tab-chat");
    const viewTabAnalytics = document.getElementById("view-tab-analytics");
    const navEvalBadge = document.getElementById("nav-eval-badge");
    const analyticsPulseDot = document.getElementById("analytics-pulse-dot");

    // Upload Elements
    const dropZone = document.getElementById("drop-zone");
    const pdfInput = document.getElementById("pdf-input");
    const fileCard = document.getElementById("file-card");
    const fileCardName = document.getElementById("file-card-name");
    const fileCardSize = document.getElementById("file-card-size");
    const fileCardRemove = document.getElementById("file-card-remove");
    const uploadBtn = document.getElementById("upload-btn");
    const uploadSpinner = document.getElementById("upload-spinner");
    const uploadStatus = document.getElementById("upload-status");
    const docList = document.getElementById("doc-list");
    const docListEmpty = document.getElementById("doc-list-empty");
    const indexedCountBadge = document.getElementById("indexed-count-badge");
    const topbarStatus = document.getElementById("topbar-status");
    const attachBtn = document.getElementById("attach-btn");

    // Chat Elements
    const chatFeed = document.getElementById("chat-feed");
    const emptyState = document.getElementById("empty-state");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const clearChatBtn = document.getElementById("clear-chat-btn");

    // Indicator & Toast
    const indicatorDot = document.getElementById("indicator-dot");
    const indicatorText = document.getElementById("indicator-text");
    const toastShelf = document.getElementById("toast-shelf");

    // Analytics Dashboard Elements
    const analyticsDocFilter = document.getElementById("analytics-doc-filter");
    const analyticsRefreshBtn = document.getElementById("analytics-refresh-btn");
    const analyticsClearBtn = document.getElementById("analytics-clear-btn");

    // Summary Stats
    const statTotalQuestions = document.getElementById("stat-total-questions");
    const statEvaluatedQuestions = document.getElementById("stat-evaluated-questions");
    const statPendingQuestions = document.getElementById("stat-pending-questions");
    const statFailedQuestions = document.getElementById("stat-failed-questions");
    const statAvgLatency = document.getElementById("stat-avg-latency");

    // Overall Composite Score Card
    const metricOverallScoreValue = document.getElementById("metric-overall-score-value");
    const metricOverallScoreBar = document.getElementById("metric-overall-score-bar");
    const metricOverallScoreCount = document.getElementById("metric-overall-score-count");

    // Metric Cards
    const metricFaithfulnessValue = document.getElementById("metric-faithfulness-value");
    const metricFaithfulnessBar = document.getElementById("metric-faithfulness-bar");
    const metricFaithfulnessCount = document.getElementById("metric-faithfulness-count");

    const metricContextRelevanceValue = document.getElementById("metric-context-relevance-value");
    const metricContextRelevanceBar = document.getElementById("metric-context-relevance-bar");
    const metricContextRelevanceCount = document.getElementById("metric-context-relevance-count");

    const metricAnswerRelevanceValue = document.getElementById("metric-answer-relevance-value");
    const metricAnswerRelevanceBar = document.getElementById("metric-answer-relevance-bar");
    const metricAnswerRelevanceCount = document.getElementById("metric-answer-relevance-count");

    // Latency Elements
    const statRetrievalLatency = document.getElementById("stat-retrieval-latency");
    const statGenerationLatency = document.getElementById("stat-generation-latency");
    const statTotalLatency = document.getElementById("stat-total-latency");

    // History Elements
    const evalHistoryList = document.getElementById("eval-history-list");
    const evalHistoryEmpty = document.getElementById("eval-history-empty");

    // ================= UTILITIES =================
    function formatBytes(bytes) {
        if (!bytes) return "0 B";
        const k = 1024;
        const units = ["B", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + units[i];
    }

    function formatTime(isoString) {
        const d = isoString ? new Date(isoString) : new Date();
        return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function formatDate(isoString) {
        if (!isoString) return "";
        const d = new Date(isoString);
        return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function formatPercentage(score) {
        if (score === null || score === undefined) return "No evaluations yet";
        return (score * 100).toFixed(1) + "%";
    }

    function getScoreClass(score) {
        if (score === null || score === undefined) return "unavailable";
        if (score >= 0.8) return "high";
        if (score >= 0.6) return "medium";
        return "low";
    }

    function escapeHtml(text) {
        if (!text) return "";
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function showToast(message, isError = false) {
        if (!toastShelf) return;
        const toast = document.createElement("div");
        toast.className = `toast-msg${isError ? " error" : ""}`;
        toast.textContent = message;
        toastShelf.appendChild(toast);

        setTimeout(() => {
            toast.style.transition = "opacity 0.25s, transform 0.25s";
            toast.style.opacity = "0";
            toast.style.transform = "translateY(-6px)";
            setTimeout(() => toast.remove(), 260);
        }, 3200);
    }

    // ================= VIEW SWITCHING =================
    function switchView(viewName) {
        currentView = viewName;

        if (viewName === "chat") {
            viewChat.classList.add("active");
            viewChat.style.display = "flex";
            viewAnalytics.classList.remove("active");
            viewAnalytics.style.display = "none";

            navBtnChat.classList.add("active");
            navBtnAnalytics.classList.remove("active");
            viewTabChat.classList.add("active");
            viewTabAnalytics.classList.remove("active");
            analyticsPulseDot.style.display = "none";
        } else {
            viewChat.classList.remove("active");
            viewChat.style.display = "none";
            viewAnalytics.classList.add("active");
            viewAnalytics.style.display = "flex";

            navBtnChat.classList.remove("active");
            navBtnAnalytics.classList.add("active");
            viewTabChat.classList.remove("active");
            viewTabAnalytics.classList.add("active");
            analyticsPulseDot.style.display = "none";

            loadAnalyticsDashboard();
        }

        if (window.innerWidth < 860) {
            sidebar.classList.remove("open");
            sidebarOverlay.classList.remove("active");
        }
    }

    navBtnChat.addEventListener("click", () => switchView("chat"));
    navBtnAnalytics.addEventListener("click", () => switchView("analytics"));
    viewTabChat.addEventListener("click", () => switchView("chat"));
    viewTabAnalytics.addEventListener("click", () => switchView("analytics"));

    // ================= BACKEND STATUS CHECK =================
    async function checkConnection() {
        indicatorDot.className = "indicator-dot";
        indicatorText.textContent = "Connecting...";

        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 6000);

            const res = await fetch(`${API_BASE}/`, {
                method: "GET",
                signal: controller.signal
            }).catch(() => null);

            clearTimeout(timeoutId);

            if (res && (res.ok || res.status === 404 || res.status === 200)) {
                indicatorDot.className = "indicator-dot online";
                indicatorText.textContent = "Connected";
            } else {
                throw new Error("Offline");
            }
        } catch {
            indicatorDot.className = "indicator-dot offline";
            indicatorText.textContent = "Backend Offline";
        }
    }

    // ================= DOCUMENT LIST MANAGEMENT =================
    function getStoredDocs() {
        try {
            return JSON.parse(localStorage.getItem("doc_assistant_files") || "[]");
        } catch {
            return [];
        }
    }

    function saveDocRecord(doc) {
        const docs = getStoredDocs().filter(d => d.name !== doc.name);
        docs.unshift(doc);
        localStorage.setItem("doc_assistant_files", JSON.stringify(docs));
        activeDocumentName = doc.name;
        renderDocList();
    }

    function renderDocList() {
        const docs = getStoredDocs();
        docList.innerHTML = "";

        if (docs.length === 0) {
            docList.appendChild(docListEmpty);
            indexedCountBadge.textContent = "0";
            topbarStatus.textContent = "Ready";
            return;
        }

        indexedCountBadge.textContent = docs.length.toString();
        topbarStatus.textContent = `${docs.length} ${docs.length === 1 ? "Document" : "Documents"} Indexed`;

        if (!activeDocumentName && docs.length > 0) {
            activeDocumentName = docs[0].name;
        }

        docs.forEach(d => {
            const item = document.createElement("div");
            item.className = "doc-item";
            item.innerHTML = `
                <span class="doc-item-title" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</span>
                <span class="doc-item-badge">${d.chunks ? `${d.chunks} chunks` : "Indexed"}</span>
            `;
            docList.appendChild(item);
        });
    }

    // ================= FILE SELECTION & UPLOAD =================
    function setFile(file) {
        if (!file) return;

        if (!file.name.toLowerCase().endsWith(".pdf")) {
            showToast("Please select a PDF document.", true);
            return;
        }

        selectedFile = file;
        fileCardName.textContent = file.name;
        fileCardSize.textContent = formatBytes(file.size);

        fileCard.style.display = "flex";
        dropZone.style.display = "none";
        uploadBtn.disabled = false;
        uploadStatus.style.display = "none";
    }

    function resetFileSelection() {
        selectedFile = null;
        pdfInput.value = "";
        fileCard.style.display = "none";
        dropZone.style.display = "block";
        uploadBtn.disabled = true;
    }

    dropZone.addEventListener("click", () => pdfInput.click());
    pdfInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    });

    fileCardRemove.addEventListener("click", resetFileSelection);

    attachBtn.addEventListener("click", () => {
        if (window.innerWidth < 860) {
            sidebar.classList.add("open");
            sidebarOverlay.classList.add("active");
        }
        pdfInput.click();
    });

    uploadBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        const fileToUpload = selectedFile;
        uploadBtn.disabled = true;
        uploadSpinner.style.display = "inline-block";
        uploadBtn.querySelector(".btn-text").textContent = "Processing...";

        uploadStatus.style.display = "block";
        uploadStatus.className = "status-box loading";
        uploadStatus.textContent = "Extracting text and generating embeddings...";

        const formData = new FormData();
        formData.append("file", fileToUpload);

        try {
            const response = await fetch(`${API_BASE}/upload_pdf`, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                const chunks = data.chunks_added !== undefined ? data.chunks_added : "Multiple";
                uploadStatus.className = "status-box success";
                uploadStatus.textContent = `Successfully processed ${fileToUpload.name} (${chunks} chunks).`;

                showToast(`Indexed ${chunks} chunks.`);

                saveDocRecord({
                    name: fileToUpload.name,
                    chunks: chunks,
                    timestamp: Date.now()
                });

                resetFileSelection();
                if (currentView === "analytics") {
                    loadAnalyticsDashboard();
                }
            } else {
                throw new Error(data.detail || "Failed to process document.");
            }
        } catch (err) {
            uploadStatus.className = "status-box error";
            uploadStatus.textContent = err.message || "Upload failed. Check backend connection.";
            showToast("Upload failed", true);
        } finally {
            uploadBtn.disabled = false;
            uploadSpinner.style.display = "none";
            uploadBtn.querySelector(".btn-text").textContent = "Process Document";
        }
    });

    // ================= MESSAGE RENDERING WITH INLINE EVAL =================
    function appendUserMessage(text) {
        if (emptyState) emptyState.style.display = "none";

        const row = document.createElement("div");
        row.className = "msg-row user";

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble-user";
        bubble.textContent = text;

        row.appendChild(bubble);
        chatFeed.appendChild(row);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function appendAiMessage(rawText, interactionId = null, isError = false) {
        if (emptyState) emptyState.style.display = "none";

        const row = document.createElement("div");
        row.className = "msg-row ai";

        const card = document.createElement("div");
        card.className = "msg-card-ai";

        // Header
        const header = document.createElement("div");
        header.className = "msg-card-header";
        header.innerHTML = `
            <div class="msg-author-group">
                <span class="msg-author">Assistant</span>
            </div>
            <span class="msg-time">${formatTime()}</span>
        `;

        // Content
        const content = document.createElement("div");
        content.className = "msg-content";

        try {
            content.innerHTML = marked.parse(rawText || "");
        } catch {
            content.textContent = rawText;
        }

        // MathJax equations
        if (window.MathJax && window.MathJax.typesetPromise) {
            MathJax.typesetPromise([content]).catch(() => {});
        }

        // Highlight code
        if (window.hljs) {
            content.querySelectorAll("pre code").forEach(block => {
                hljs.highlightElement(block);
            });
        }

        // Footer with inline evaluation strip & copy button
        const evalStrip = document.createElement("div");
        evalStrip.className = "msg-eval-strip";
        if (interactionId) {
            evalStrip.id = `eval-strip-${interactionId}`;
            evalStrip.innerHTML = `
                <span class="eval-live-pill evaluating" title="Computing Faithfulness, Context Relevance, and Answer Relevance in background...">
                    <span class="tab-pulse-indicator"></span>
                    <span>Evaluating RAG metrics...</span>
                </span>
            `;
        }

        const footer = document.createElement("div");
        footer.className = "msg-card-footer";

        const copyBtn = document.createElement("button");
        copyBtn.className = "copy-text-btn";
        copyBtn.type = "button";
        copyBtn.innerHTML = `
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span>Copy</span>
        `;
        copyBtn.addEventListener("click", () => {
            navigator.clipboard.writeText(rawText).then(() => {
                copyBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
                    <span>Copied</span>
                `;
                setTimeout(() => {
                    copyBtn.innerHTML = `
                        <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                        <span>Copy</span>
                    `;
                }, 1800);
            });
        });

        footer.appendChild(copyBtn);

        card.appendChild(header);
        card.appendChild(content);
        if (interactionId) {
            card.appendChild(evalStrip);
        }
        card.appendChild(footer);

        row.appendChild(card);
        chatFeed.appendChild(row);
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    function showThinking() {
        const id = "think-" + Date.now();
        const row = document.createElement("div");
        row.className = "msg-row ai";
        row.id = id;

        const card = document.createElement("div");
        card.className = "msg-card-ai";
        card.innerHTML = `
            <div class="thinking-box">
                <div class="pulse-dots">
                    <span></span><span></span><span></span>
                </div>
                <span>Searching document context...</span>
            </div>
        `;

        row.appendChild(card);
        chatFeed.appendChild(row);
        chatFeed.scrollTop = chatFeed.scrollHeight;

        return {
            remove: () => document.getElementById(id)?.remove()
        };
    }

    // ================= ASYNC EVALUATION POLLING =================
    function formatMetricScore(score) {
        if (score === null || score === undefined || typeof score !== "number" || isNaN(score)) {
            return "N/A";
        }
        return (score * 100).toFixed(0) + "%";
    }

    function showInlineEvaluationMessage(interactionId, message, type = "evaluating", tooltip = "") {
        const strip = document.getElementById(`eval-strip-${interactionId}`);
        if (!strip) return;
        strip.innerHTML = `
            <span class="eval-live-pill ${type}" title="${escapeHtml(tooltip || message)}">
                <span>${escapeHtml(message)}</span>
            </span>
        `;
        if (type !== "failed") {
            strip.querySelector(".eval-live-pill")?.addEventListener("click", () => {
                switchView("analytics");
            });
        }
    }

    function pollEvaluationStatus(interactionId) {
        if (!interactionId || activePolls.has(interactionId)) return;
        activePolls.add(interactionId);

        let attempts = 0;
        let consecutiveErrors = 0;
        const maxConsecutiveErrors = 6;
        const maxDurationMs = 180000; // 3-minute window
        const startTime = Date.now();

        function getNextDelay(attemptCount) {
            if (attemptCount < 5) return 2000;
            if (attemptCount < 15) return 2500;
            return 3500;
        }

        async function pollStep() {
            // Check if DOM element still exists (e.g. user cleared chat)
            const strip = document.getElementById(`eval-strip-${interactionId}`);
            if (!strip) {
                activePolls.delete(interactionId);
                return;
            }

            const elapsed = Date.now() - startTime;
            if (elapsed >= maxDurationMs) {
                activePolls.delete(interactionId);
                // 3-minute window expired while evaluation is still running on backend
                showInlineEvaluationMessage(
                    interactionId,
                    "Evaluation is taking longer than expected",
                    "evaluating",
                    "Evaluation is still running in the background. You can check RAG Analytics later."
                );
                return;
            }

            attempts++;

            try {
                const res = await fetch(`${API_BASE}/api/evaluations/${interactionId}`);
                if (!res.ok) {
                    consecutiveErrors++;
                    if (consecutiveErrors >= maxConsecutiveErrors && elapsed > 20000) {
                        activePolls.delete(interactionId);
                        showInlineEvaluationMessage(
                            interactionId,
                            "Evaluation unavailable",
                            "failed",
                            `Server returned HTTP ${res.status} when fetching evaluation status.`
                        );
                        return;
                    }
                    setTimeout(pollStep, getNextDelay(attempts));
                    return;
                }

                consecutiveErrors = 0;
                const data = await res.json();
                const status = (data && data.eval_status) ? data.eval_status.toLowerCase() : "";

                if (status === "completed" || status === "partial" || status === "failed") {
                    activePolls.delete(interactionId);
                    updateInlineEvaluationBadge(data, interactionId);

                    // Signal update to Analytics tab
                    analyticsPulseDot.style.display = "inline-block";
                    if (currentView === "analytics") {
                        loadAnalyticsDashboard();
                    }
                    return;
                }

                // Status is still evaluating/pending -> continue polling with progressive delay
                setTimeout(pollStep, getNextDelay(attempts));

            } catch (err) {
                consecutiveErrors++;
                if (consecutiveErrors >= maxConsecutiveErrors && elapsed > 20000) {
                    activePolls.delete(interactionId);
                    showInlineEvaluationMessage(
                        interactionId,
                        "Evaluation unavailable",
                        "failed",
                        "Network connection error while fetching evaluation status."
                    );
                    return;
                }
                setTimeout(pollStep, getNextDelay(attempts));
            }
        }

        setTimeout(pollStep, getNextDelay(0));
    }

    function updateInlineEvaluationBadge(record, fallbackId = null) {
        const id = (record && (record.id || record.interaction_id)) || fallbackId;
        if (!id) return;
        const strip = document.getElementById(`eval-strip-${id}`);
        if (!strip) return;

        if (record && record.eval_status === "failed") {
            const errorMsg = record.eval_error || "Evaluation failed";
            strip.innerHTML = `
                <span class="eval-live-pill failed" title="${escapeHtml(errorMsg)}">
                    <span>Evaluation unavailable</span>
                </span>
            `;
            return;
        }

        const fStr = formatMetricScore(record ? record.faithfulness : null);
        const cStr = formatMetricScore(record ? record.context_relevance : null);
        const aStr = formatMetricScore(record ? record.answer_relevance : null);

        const isPartial = record && record.eval_status === "partial";
        const titleText = isPartial
            ? "Partial evaluation (some metrics may be N/A). Click to view in RAG Analytics."
            : "Click to view detailed evaluation in RAG Analytics";

        strip.innerHTML = `
            <span class="eval-live-pill completed" title="${escapeHtml(titleText)}">
                <span>Faithfulness: <strong class="eval-score-num">${fStr}</strong></span>
                <span>•</span>
                <span>Context Rel: <strong class="eval-score-num">${cStr}</strong></span>
                <span>•</span>
                <span>Answer Rel: <strong class="eval-score-num">${aStr}</strong></span>
            </span>
        `;

        strip.querySelector(".eval-live-pill")?.addEventListener("click", () => {
            switchView("analytics");
        });
    }

    // ================= QUERY DISPATCH =================
    async function submitQuery() {
        const question = userInput.value.trim();
        if (!question || isProcessing) return;

        isProcessing = true;
        userInput.value = "";
        resizeInput();
        sendBtn.disabled = true;

        appendUserMessage(question);
        const thinking = showThinking();

        try {
            const res = await fetch(`${API_BASE}/query`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    chat_history: chatHistory,
                    session_id: sessionId,
                    document_name: activeDocumentName
                })
            });

            const data = await res.json();
            thinking.remove();

            if (res.ok) {
                appendAiMessage(data.answer, data.interaction_id);
                chatHistory.push({ role: "user", content: question });
                chatHistory.push({ role: "assistant", content: data.answer });

                // Poll evaluation in background for live updates
                if (data.interaction_id) {
                    pollEvaluationStatus(data.interaction_id);
                }
            } else {
                appendAiMessage(`**Error:** ${data.detail || "Unable to retrieve an answer."}`, null, true);
            }
        } catch (err) {
            thinking.remove();
            appendAiMessage(
                "**Connection Error:** Unable to reach the backend server. Please verify the FastAPI service is running.",
                null,
                true
            );
        } finally {
            isProcessing = false;
            sendBtn.disabled = userInput.value.trim().length === 0;
            userInput.focus();
        }
    }

    // ================= INPUT RESIZE & EVENTS =================
    function resizeInput() {
        userInput.style.height = "auto";
        const h = Math.min(userInput.scrollHeight, 150);
        userInput.style.height = (h > 24 ? h : 24) + "px";
        sendBtn.disabled = userInput.value.trim().length === 0 || isProcessing;
    }

    userInput.addEventListener("input", resizeInput);
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });

    sendBtn.addEventListener("click", submitQuery);

    // Prompt Chips
    document.querySelectorAll(".prompt-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const prompt = btn.getAttribute("data-prompt");
            if (prompt) {
                userInput.value = prompt;
                resizeInput();
                submitQuery();
            }
        });
    });

    // Clear Chat
    clearChatBtn.addEventListener("click", () => {
        if (chatHistory.length === 0) return;
        chatFeed.innerHTML = "";
        chatHistory = [];
        if (emptyState) {
            emptyState.style.display = "flex";
            chatFeed.appendChild(emptyState);
        }
        showToast("Chat cleared.");
    });

    // Mobile Drawer
    function toggleDrawer(open) {
        sidebar.classList.toggle("open", open);
        sidebarOverlay.classList.toggle("active", open);
    }

    mobileToggleBtn.addEventListener("click", () => toggleDrawer(true));
    mobileCloseBtn.addEventListener("click", () => toggleDrawer(false));
    sidebarOverlay.addEventListener("click", () => toggleDrawer(false));

    // ================= ANALYTICS DASHBOARD LOGIC =================
    async function loadAnalyticsDashboard() {
        const selectedDoc = analyticsDocFilter.value || "all";
        await Promise.all([
            fetchAnalyticsSummary(selectedDoc),
            fetchEvaluationHistory(selectedDoc)
        ]);
    }

    async function fetchAnalyticsSummary(docName = "all") {
        try {
            let url = `${API_BASE}/api/analytics?`;
            if (docName && docName !== "all") {
                url += `document_name=${encodeURIComponent(docName)}&`;
            }

            const res = await fetch(url);
            if (!res.ok) return;

            const data = await res.json();

            // Populate document scope dropdown if needed
            if (data.documents && data.documents.length > 0) {
                const existingVal = analyticsDocFilter.value;
                analyticsDocFilter.innerHTML = '<option value="all">All Documents</option>';
                data.documents.forEach(doc => {
                    const opt = document.createElement("option");
                    opt.value = doc;
                    opt.textContent = doc;
                    analyticsDocFilter.appendChild(opt);
                });
                if (existingVal && data.documents.includes(existingVal)) {
                    analyticsDocFilter.value = existingVal;
                }
            }

            // Overview Counts
            statTotalQuestions.textContent = data.total_questions || 0;
            statEvaluatedQuestions.textContent = data.evaluated_questions || 0;
            statPendingQuestions.textContent = data.pending_questions || 0;
            statFailedQuestions.textContent = data.failed_questions || 0;

            if (navEvalBadge) {
                if (data.total_questions > 0) {
                    navEvalBadge.textContent = data.total_questions.toString();
                    navEvalBadge.style.display = "inline-block";
                } else {
                    navEvalBadge.style.display = "none";
                }
            }

            // Latencies
            if (data.avg_total_latency !== null) {
                statAvgLatency.textContent = (data.avg_total_latency * 1000).toFixed(0) + " ms";
                statTotalLatency.textContent = (data.avg_total_latency * 1000).toFixed(0) + " ms";
            } else {
                statAvgLatency.textContent = "—";
                statTotalLatency.textContent = "—";
            }

            statRetrievalLatency.textContent = data.avg_retrieval_latency !== null
                ? (data.avg_retrieval_latency * 1000).toFixed(0) + " ms"
                : "—";

            statGenerationLatency.textContent = data.avg_generation_latency !== null
                ? (data.avg_generation_latency * 1000).toFixed(0) + " ms"
                : "—";

            // Composite Overall Score (Average of all 3 metrics)
            if (metricOverallScoreValue) {
                if (data.avg_overall_score !== null && data.avg_overall_score !== undefined) {
                    const overallPct = (data.avg_overall_score * 100).toFixed(1);
                    metricOverallScoreValue.textContent = `${overallPct}%`;
                    metricOverallScoreValue.className = "composite-score-val";
                    if (metricOverallScoreBar) {
                        metricOverallScoreBar.style.width = `${Math.min(100, Math.max(0, data.avg_overall_score * 100))}%`;
                    }
                    if (metricOverallScoreCount) {
                        metricOverallScoreCount.textContent = `${data.evaluated_questions || 0} questions evaluated`;
                    }
                } else {
                    metricOverallScoreValue.textContent = "No evaluations yet";
                    metricOverallScoreValue.className = "composite-score-val no-eval";
                    if (metricOverallScoreBar) {
                        metricOverallScoreBar.style.width = "0%";
                    }
                    if (metricOverallScoreCount) {
                        metricOverallScoreCount.textContent = "0 questions evaluated";
                    }
                }
            }

            // Metric 1: Faithfulness
            renderMetricCard(
                metricFaithfulnessValue,
                metricFaithfulnessBar,
                metricFaithfulnessCount,
                data.avg_faithfulness,
                data.count_faithfulness
            );

            // Metric 2: Context Relevance
            renderMetricCard(
                metricContextRelevanceValue,
                metricContextRelevanceBar,
                metricContextRelevanceCount,
                data.avg_context_relevance,
                data.count_context_relevance
            );

            // Metric 3: Answer Relevance
            renderMetricCard(
                metricAnswerRelevanceValue,
                metricAnswerRelevanceBar,
                metricAnswerRelevanceCount,
                data.avg_answer_relevance,
                data.count_answer_relevance
            );

        } catch (err) {
            console.error("Failed to fetch analytics summary:", err);
        }
    }

    function renderMetricCard(valueEl, barEl, countEl, score, count) {
        if (score === null || score === undefined) {
            valueEl.textContent = "No evaluations yet";
            valueEl.className = "metric-score-value no-eval";
            barEl.style.width = "0%";
            barEl.className = "metric-progress-bar";
            countEl.textContent = "0 evaluated";
            return;
        }

        const pct = (score * 100).toFixed(1) + "%";
        valueEl.textContent = pct;
        valueEl.className = "metric-score-value";
        barEl.style.width = Math.min(100, Math.max(0, score * 100)) + "%";

        const scoreClass = getScoreClass(score);
        barEl.className = `metric-progress-bar score-${scoreClass}`;
        countEl.textContent = `${count} evaluated`;
    }

    async function fetchEvaluationHistory(docName = "all") {
        try {
            let url = `${API_BASE}/api/evaluations?limit=50&`;
            if (docName && docName !== "all") {
                url += `document_name=${encodeURIComponent(docName)}&`;
            }

            const res = await fetch(url);
            if (!res.ok) return;

            const records = await res.json();
            evalHistoryList.innerHTML = "";

            if (!records || records.length === 0) {
                evalHistoryList.appendChild(evalHistoryEmpty);
                return;
            }

            records.forEach(r => {
                const card = createHistoryCard(r);
                evalHistoryList.appendChild(card);
            });

        } catch (err) {
            console.error("Failed to fetch evaluation history:", err);
        }
    }

    function createHistoryCard(record) {
        const card = document.createElement("div");
        card.className = "history-card";
        card.id = `history-card-${record.id}`;

        const isEvaluating = record.eval_status === "evaluating" || record.eval_status === "pending";

        // Score formatters
        const fChip = formatScoreChip("Faithfulness", record.faithfulness, isEvaluating);
        const cChip = formatScoreChip("Context Rel", record.context_relevance, isEvaluating);
        const aChip = formatScoreChip("Answer Rel", record.answer_relevance, isEvaluating);

        const statusClass = record.eval_status || "completed";

        card.innerHTML = `
            <div class="history-card-top">
                <span class="history-question">${escapeHtml(record.question)}</span>
                <span class="history-time">${formatDate(record.created_at)}</span>
            </div>

            <div class="history-answer-block">
                <div class="history-answer-text truncated" id="ans-text-${record.id}">
                    ${escapeHtml(record.answer)}
                </div>
                ${record.answer && record.answer.length > 180 ? `
                    <button class="history-toggle-answer" type="button" data-id="${record.id}">
                        Show full answer
                    </button>
                ` : ""}
            </div>

            <div class="history-scores-bar">
                ${fChip}
                ${cChip}
                ${aChip}
                <button class="trace-toggle-btn" type="button" data-trace-id="${record.id}">
                    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                    <span>Trace & Chunks</span>
                </button>
                <span class="status-badge ${statusClass}">${statusClass}</span>
            </div>

            <!-- Trace Drawer (Accordion) -->
            <div class="history-trace-drawer" id="trace-drawer-${record.id}">
                ${renderTraceDrawerContent(record)}
            </div>
        `;

        // Toggle full answer
        const toggleAnsBtn = card.querySelector(".history-toggle-answer");
        if (toggleAnsBtn) {
            toggleAnsBtn.addEventListener("click", () => {
                const ansText = card.querySelector(`#ans-text-${record.id}`);
                const isTruncated = ansText.classList.toggle("truncated");
                toggleAnsBtn.textContent = isTruncated ? "Show full answer" : "Show less";
            });
        }

        // Toggle trace drawer
        const toggleTraceBtn = card.querySelector(".trace-toggle-btn");
        if (toggleTraceBtn) {
            toggleTraceBtn.addEventListener("click", () => {
                const drawer = card.querySelector(`#trace-drawer-${record.id}`);
                const isOpen = drawer.classList.toggle("open");
                toggleTraceBtn.querySelector("span").textContent = isOpen ? "Hide Trace" : "Trace & Chunks";
            });
        }

        return card;
    }

    function formatScoreChip(label, score, isEvaluating) {
        if (isEvaluating) {
            return `<span class="score-chip evaluating"><strong>${label}:</strong> Evaluating...</span>`;
        }
        if (score === null || score === undefined) {
            return `<span class="score-chip unavailable"><strong>${label}:</strong> Unavailable</span>`;
        }
        const cls = getScoreClass(score);
        const pct = (score * 100).toFixed(0) + "%";
        return `<span class="score-chip ${cls}"><strong>${label}:</strong> ${pct}</span>`;
    }

    function renderTraceDrawerContent(record) {
        let chunksHtml = "";
        if (record.retrieved_chunks && record.retrieved_chunks.length > 0) {
            chunksHtml = record.retrieved_chunks.map((c, i) => `
                <div class="chunk-card">
                    <div class="chunk-meta">
                        <span class="chunk-source-badge">${escapeHtml(c.source || 'Document')} ${c.page !== null && c.page !== undefined ? `(Page ${c.page + 1})` : ''}</span>
                        <span>Chunk #${i + 1}</span>
                    </div>
                    <div class="chunk-text">${escapeHtml(c.text)}</div>
                </div>
            `).join("");
        } else {
            chunksHtml = `<div style="font-size: 0.75rem; color: var(--text-muted);">No separate chunk metadata recorded for this query.</div>`;
        }

        let evalDetailsHtml = "";
        if (record.eval_details && record.eval_details.faithfulness && record.eval_details.faithfulness.statements) {
            const stmts = record.eval_details.faithfulness.statements;
            if (stmts.length > 0) {
                evalDetailsHtml = `
                    <div class="trace-block">
                        <span class="trace-block-title">Faithfulness Claim Verification</span>
                        <div class="statements-list">
                            ${stmts.map(s => `
                                <div class="statement-row ${s.supported ? 'supported' : 'unsupported'}">
                                    <span>${s.supported ? '✓' : '✗'}</span>
                                    <span>${escapeHtml(s.statement)} ${s.reason ? `<em style="opacity: 0.7;">— ${escapeHtml(s.reason)}</em>` : ''}</span>
                                </div>
                            `).join("")}
                        </div>
                    </div>
                `;
            }
        }

        const latenciesHtml = `
            <div class="trace-block">
                <span class="trace-block-title">Latency Diagnostics</span>
                <div style="font-size: 0.75rem; color: var(--text-muted); display: flex; gap: 1.5rem;">
                    <span>Retrieval: <strong>${record.retrieval_latency !== null ? (record.retrieval_latency * 1000).toFixed(0) + ' ms' : '—'}</strong></span>
                    <span>Synthesis: <strong>${record.generation_latency !== null ? (record.generation_latency * 1000).toFixed(0) + ' ms' : '—'}</strong></span>
                    <span>Total: <strong>${record.total_latency !== null ? (record.total_latency * 1000).toFixed(0) + ' ms' : '—'}</strong></span>
                </div>
            </div>
        `;

        return `
            ${latenciesHtml}
            ${evalDetailsHtml}
            <div class="trace-block">
                <span class="trace-block-title">Retrieved Context Chunks (${record.retrieved_chunks ? record.retrieved_chunks.length : 0})</span>
                <div class="chunks-grid">
                    ${chunksHtml}
                </div>
            </div>
        `;
    }

    analyticsDocFilter.addEventListener("change", () => {
        loadAnalyticsDashboard();
    });

    analyticsRefreshBtn.addEventListener("click", () => {
        loadAnalyticsDashboard();
        showToast("Dashboard refreshed.");
    });

    analyticsClearBtn.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to clear historical evaluation records?")) return;

        try {
            const res = await fetch(`${API_BASE}/api/evaluations`, { method: "DELETE" });
            if (res.ok) {
                showToast("Evaluation history cleared.");
                loadAnalyticsDashboard();
            }
        } catch {
            showToast("Failed to clear evaluation history.", true);
        }
    });

    // Auto-refresh timer when Analytics view is open
    setInterval(() => {
        if (currentView === "analytics") {
            loadAnalyticsDashboard();
        }
    }, 7000);

    // ================= INITIALIZE =================
    function init() {
        renderDocList();
        checkConnection();
        resizeInput();
        fetchAnalyticsSummary();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
