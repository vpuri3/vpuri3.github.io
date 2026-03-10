+++
title = "Batch Audio Transcription"
description = "Upload audio files, run Sarvam batch transcription in the browser, and download txt transcripts."
+++

{{< rawhtml >}}
<div class="transcribe-app">
  <div class="transcribe-hero">
    <p class="transcribe-kicker">Browser Tool</p>
    <h1>Batch audio to text</h1>
    <p class="transcribe-subtitle">
      Upload multiple recordings, run Sarvam batch STT through a small backend, and download one <code>.txt</code> transcript per file.
    </p>
    <p class="transcribe-warning">
      GitHub Pages serves only the UI. The backend handles audio uploads, Sarvam batch jobs, and returns a zip of transcript files.
    </p>
  </div>

  <div class="transcribe-panel">
    <div class="transcribe-grid">
      <label class="transcribe-field">
        <span>Sarvam API key</span>
        <input id="sarvam-api-key" type="password" placeholder="sk_..." autocomplete="off" spellcheck="false">
        <small>Sent only to the transcription backend for this request. Leave blank only if the backend already has its own key configured.</small>
      </label>

      <label class="transcribe-field">
        <span>Mode</span>
        <select id="transcribe-mode">
          <option value="translit" selected>translit</option>
          <option value="transcribe">transcribe</option>
          <option value="translate">translate</option>
          <option value="verbatim">verbatim</option>
          <option value="codemix">codemix</option>
        </select>
        <small><code>translit</code> produces English-alphabet output for Indic speech.</small>
      </label>

      <label class="transcribe-field">
        <span>Language code</span>
        <input id="language-code" list="language-codes" value="unknown" placeholder="unknown">
        <small>Use <code>unknown</code> for auto-detect.</small>
      </label>

      <label class="transcribe-field">
        <span>Speaker hint</span>
        <input id="speaker-count" type="number" min="1" step="1" placeholder="optional">
        <small>Optional hint for diarization.</small>
      </label>
    </div>

    <datalist id="language-codes">
      <option value="unknown"></option>
      <option value="hi-IN"></option>
      <option value="bn-IN"></option>
      <option value="pa-IN"></option>
      <option value="sd-IN"></option>
      <option value="ur-IN"></option>
      <option value="ne-IN"></option>
      <option value="ks-IN"></option>
      <option value="doi-IN"></option>
      <option value="mai-IN"></option>
      <option value="ta-IN"></option>
      <option value="te-IN"></option>
      <option value="mr-IN"></option>
      <option value="gu-IN"></option>
      <option value="kn-IN"></option>
      <option value="ml-IN"></option>
      <option value="od-IN"></option>
      <option value="as-IN"></option>
      <option value="en-IN"></option>
    </datalist>

    <label class="transcribe-dropzone" for="audio-files">
      <input id="audio-files" type="file" accept="audio/*,.mp3,.m4a,.wav,.aac,.flac,.ogg,.opus,.webm,.mp4,.amr" multiple>
      <span class="dropzone-title">Drop audio files here or click to choose</span>
      <span class="dropzone-copy">Batch upload uses a single Sarvam job for all selected files.</span>
    </label>

    <div class="transcribe-actions">
      <button id="run-transcription" class="transcribe-primary" type="button">Run transcription</button>
      <button id="download-zip" class="transcribe-secondary" type="button" disabled>Download all txts</button>
    </div>

    <div class="transcribe-progress-stack">
      <div class="transcribe-progress-card">
        <div class="transcribe-progress-head">
          <h2>Upload</h2>
          <span id="upload-progress-label" class="transcribe-meta">0%</span>
        </div>
        <div class="transcribe-progress-bar" aria-hidden="true">
          <div id="upload-progress-fill" class="transcribe-progress-fill"></div>
        </div>
        <div id="upload-progress-text" class="transcribe-progress-copy">Waiting to upload.</div>
      </div>
    </div>

    <div id="tool-status" class="transcribe-status" aria-live="polite">Waiting for files.</div>
  </div>

  <div class="transcribe-panel">
    <div class="transcribe-section-head">
      <h2>Selected files</h2>
      <span id="selection-summary" class="transcribe-meta">0 files</span>
    </div>
    <div id="file-list" class="transcribe-list transcribe-empty">No files selected.</div>
  </div>

  <div class="transcribe-panel">
    <div class="transcribe-progress-card">
      <div class="transcribe-progress-head">
        <h2>Sarvam job</h2>
        <span id="model-progress-label" class="transcribe-meta">0%</span>
      </div>
      <div class="transcribe-progress-bar" aria-hidden="true">
        <div id="model-progress-fill" class="transcribe-progress-fill"></div>
      </div>
      <div id="model-progress-text" class="transcribe-progress-copy">Waiting for browser upload to finish.</div>
    </div>
  </div>

  <div class="transcribe-panel">
    <div class="transcribe-section-head">
      <h2>Results</h2>
      <span id="result-summary" class="transcribe-meta">No completed transcripts yet.</span>
    </div>
    <div id="result-list" class="transcribe-list transcribe-empty">Run a job to generate downloadable transcripts.</div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js" defer></script>
<script src="/assets/transcribe-tool.js?v=20260310-2" defer></script>
{{< /rawhtml >}}
