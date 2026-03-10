(function () {
  const BACKEND_URL = "https://sarvam-transcription-api.onrender.com";

  const els = {
    apiKey: document.getElementById("sarvam-api-key"),
    mode: document.getElementById("transcribe-mode"),
    languageCode: document.getElementById("language-code"),
    speakerCount: document.getElementById("speaker-count"),
    files: document.getElementById("audio-files"),
    run: document.getElementById("run-transcription"),
    downloadZip: document.getElementById("download-zip"),
    status: document.getElementById("tool-status"),
    fileList: document.getElementById("file-list"),
    resultList: document.getElementById("result-list"),
    selectionSummary: document.getElementById("selection-summary"),
    resultSummary: document.getElementById("result-summary"),
    uploadFill: document.getElementById("upload-progress-fill"),
    uploadLabel: document.getElementById("upload-progress-label"),
    uploadText: document.getElementById("upload-progress-text"),
    modelFill: document.getElementById("model-progress-fill"),
    modelLabel: document.getElementById("model-progress-label"),
    modelText: document.getElementById("model-progress-text"),
  };

  if (!els.run) {
    return;
  }

  let selectedFiles = [];
  let latestResults = [];
  let latestZipBlob = null;
  let busy = false;
  let activeJobId = null;

  function setStatus(message) {
    els.status.textContent = message;
  }

  function setProgress(kind, percent, message) {
    const clamped = Math.max(0, Math.min(100, Number(percent) || 0));
    if (kind === "upload") {
      els.uploadFill.style.width = `${clamped}%`;
      els.uploadLabel.textContent = `${Math.round(clamped)}%`;
      els.uploadText.textContent = message;
      return;
    }
    els.modelFill.style.width = `${clamped}%`;
    els.modelLabel.textContent = `${Math.round(clamped)}%`;
    els.modelText.textContent = message;
  }

  function resetProgress() {
    setProgress("upload", 0, "Waiting to upload.");
    setProgress("model", 0, "Waiting for browser upload to finish.");
  }

  function formatBytes(bytes) {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let size = bytes;
    let idx = 0;
    while (size >= 1024 && idx < units.length - 1) {
      size /= 1024;
      idx += 1;
    }
    return `${size.toFixed(size >= 10 || idx === 0 ? 0 : 1)} ${units[idx]}`;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function downloadBlob(blob, fileName) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  function stripCostHeader(text) {
    const lines = text.split("\n");
    const delimiter = "================================================================================";
    if (lines.length > 10 && lines[0] === delimiter) {
      const bodyStart = lines.findIndex((line, idx) => idx > 0 && line === "" && lines[idx - 1] === delimiter);
      if (bodyStart !== -1) {
        return lines.slice(bodyStart + 1).join("\n").trim();
      }
    }
    return text.trim();
  }

  function renderFileList() {
    els.selectionSummary.textContent = `${selectedFiles.length} file${selectedFiles.length === 1 ? "" : "s"}`;
    if (selectedFiles.length === 0) {
      els.fileList.className = "transcribe-list transcribe-empty";
      els.fileList.textContent = "No files selected.";
      return;
    }

    els.fileList.className = "transcribe-list";
    els.fileList.innerHTML = selectedFiles.map((file) => `
      <div class="transcribe-file">
        <div class="transcribe-file-row">
          <div>
            <div class="transcribe-file-name">${escapeHtml(file.name)}</div>
            <div class="transcribe-file-meta">${formatBytes(file.size)} • ${escapeHtml(file.type || "audio file")}</div>
          </div>
          <span class="transcribe-badge pending">selected</span>
        </div>
      </div>
    `).join("");
  }

  function renderResults() {
    if (latestResults.length === 0) {
      els.resultSummary.textContent = "No completed transcripts yet.";
      els.resultList.className = "transcribe-list transcribe-empty";
      els.resultList.textContent = "Run a job to generate downloadable transcripts.";
      els.downloadZip.disabled = true;
      return;
    }

    const successCount = latestResults.filter((result) => result.ok).length;
    els.resultSummary.textContent = `${successCount}/${latestResults.length} files ready`;
    els.resultList.className = "transcribe-list";
    els.resultList.innerHTML = latestResults.map((result, idx) => {
      if (!result.ok) {
        return `
          <div class="transcribe-result">
            <div class="transcribe-result-row">
              <div>
                <div class="transcribe-result-name">${escapeHtml(result.fileName)}</div>
                <div class="transcribe-file-meta">${escapeHtml(result.error || "Unknown error")}</div>
              </div>
              <span class="transcribe-badge error">failed</span>
            </div>
          </div>
        `;
      }

      return `
        <div class="transcribe-result">
          <div class="transcribe-result-row">
            <div>
              <div class="transcribe-result-name">${escapeHtml(result.fileName)}</div>
              <div class="transcribe-file-meta">${result.lineCount} lines</div>
            </div>
            <span class="transcribe-badge success">ready</span>
          </div>
          <div class="transcribe-result-snippet">${escapeHtml(result.preview)}</div>
          <div class="transcribe-result-actions">
            <a href="#" data-download-index="${idx}">Download txt</a>
          </div>
        </div>
      `;
    }).join("");

    els.downloadZip.disabled = !latestZipBlob || successCount === 0;
  }

  async function parseZipResults(blob) {
    const zip = await JSZip.loadAsync(blob);
    const entries = [];
    const files = Object.keys(zip.files).sort();

    for (const name of files) {
      if (name.endsWith("/") || name === "manifest.json" || !name.endsWith(".txt")) {
        continue;
      }
      const text = await zip.files[name].async("string");
      entries.push({
        ok: true,
        fileName: name,
        outputName: name,
        text,
        lineCount: text.split("\n").filter(Boolean).length,
        preview: stripCostHeader(text).slice(0, 220) + (text.length > 220 ? "..." : ""),
        blob: new Blob([text], { type: "text/plain;charset=utf-8" }),
      });
    }

    return entries;
  }

  function createJobWithUploadProgress(backendUrl, formData) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${backendUrl}/transcribe/jobs`);
      xhr.responseType = "json";

      xhr.upload.addEventListener("progress", (event) => {
        if (!event.lengthComputable) {
          setProgress("upload", 10, "Uploading audio to backend...");
          return;
        }
        const percent = (event.loaded / event.total) * 100;
        setProgress(
          "upload",
          percent,
          `Uploading ${formatBytes(event.loaded)} of ${formatBytes(event.total)} to backend.`
        );
      });

      xhr.addEventListener("load", () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          setProgress("upload", 100, "Upload complete.");
          resolve(xhr.response);
          return;
        }
        const payload = xhr.response;
        const detail = payload && payload.detail ? payload.detail : `${xhr.status} ${xhr.statusText}`;
        reject(new Error(detail));
      });

      xhr.addEventListener("error", () => {
        reject(new Error("Network error while uploading audio."));
      });

      xhr.addEventListener("abort", () => {
        reject(new Error("Upload was aborted."));
      });

      xhr.send(formData);
    });
  }

  async function pollJobUntilComplete(backendUrl, jobId) {
    while (true) {
      const response = await fetch(`${backendUrl}/transcribe/jobs/${jobId}`);
      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try {
          const payload = await response.json();
          detail = payload.detail || detail;
        } catch (_err) {
          // ignore
        }
        throw new Error(detail);
      }

      const payload = await response.json();
      const progress = payload.progress || 0;
      const message = payload.message || "Waiting for Sarvam.";
      setProgress("model", progress, message);
      setStatus(message);

      if (payload.state === "completed") {
        return payload;
      }
      if (payload.state === "failed") {
        throw new Error(payload.error || payload.message || "Sarvam job failed.");
      }
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
    }
  }

  async function downloadCompletedZip(backendUrl, jobId) {
    const response = await fetch(`${backendUrl}/transcribe/jobs/${jobId}/download`);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (_err) {
        // ignore
      }
      throw new Error(detail);
    }
    return response.blob();
  }

  async function runBatchTranscription() {
    if (busy) {
      return;
    }

    const backendUrl = BACKEND_URL;
    if (selectedFiles.length === 0) {
      setStatus("Choose at least one audio file.");
      return;
    }

    const seenNames = new Set();
    for (const file of selectedFiles) {
      if (seenNames.has(file.name)) {
        setStatus(`Duplicate filename detected: ${file.name}. Rename duplicates before upload.`);
        return;
      }
      seenNames.add(file.name);
    }

    busy = true;
    activeJobId = null;
    latestResults = [];
    latestZipBlob = null;
    renderResults();
    resetProgress();
    els.run.disabled = true;
    els.downloadZip.disabled = true;

    try {
      const form = new FormData();
      if ((els.apiKey.value || "").trim()) {
        form.append("api_key", els.apiKey.value.trim());
      }
      form.append("mode", els.mode.value);
      form.append("language_code", (els.languageCode.value || "unknown").trim() || "unknown");
      const speakerCount = Number.parseInt(els.speakerCount.value, 10);
      if (Number.isFinite(speakerCount) && speakerCount > 0) {
        form.append("num_speakers", String(speakerCount));
      }
      selectedFiles.forEach((file) => {
        form.append("files", file, file.name);
      });

      setStatus(`Uploading ${selectedFiles.length} file${selectedFiles.length === 1 ? "" : "s"} to backend...`);
      setProgress("model", 0, "Waiting for browser upload to finish.");
      const createdJob = await createJobWithUploadProgress(backendUrl, form);
      activeJobId = createdJob.job_id;

      setProgress("model", createdJob.progress || 5, createdJob.message || "Preparing Sarvam batch job.");
      setStatus(createdJob.message || "Preparing Sarvam batch job.");

      await pollJobUntilComplete(backendUrl, activeJobId);
      latestZipBlob = await downloadCompletedZip(backendUrl, activeJobId);
      latestResults = await parseZipResults(latestZipBlob);
      setProgress("model", 100, "Sarvam job complete.");
      renderResults();
      setStatus(`Completed. ${latestResults.length} transcript${latestResults.length === 1 ? "" : "s"} ready.`);
    } catch (error) {
      setStatus(`Transcription failed: ${error.message || error}`);
      setProgress("model", 100, `Failed${activeJobId ? ` for job ${activeJobId}.` : "."}`);
    } finally {
      busy = false;
      els.run.disabled = false;
    }
  }

  els.files.addEventListener("change", () => {
    selectedFiles = Array.from(els.files.files || []);
    renderFileList();
  });

  els.run.addEventListener("click", runBatchTranscription);

  els.resultList.addEventListener("click", (event) => {
    const anchor = event.target.closest("[data-download-index]");
    if (!anchor) {
      return;
    }
    event.preventDefault();
    const index = Number.parseInt(anchor.getAttribute("data-download-index"), 10);
    const result = latestResults[index];
    if (!result || !result.ok) {
      return;
    }
    downloadBlob(result.blob, result.outputName);
  });

  els.downloadZip.addEventListener("click", () => {
    if (!latestZipBlob) {
      return;
    }
    downloadBlob(latestZipBlob, `transcripts-${Date.now()}.zip`);
  });

  renderFileList();
  renderResults();
  resetProgress();
})();
