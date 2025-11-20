# Deepfake Detection Pipeline

End-to-end, multi-modal deepfake detection system designed to operate at social-media scale.  
The project bundles:

- A **FastAPI backend** that ingests short video clips and runs a fused deepfake detector.
- A **React/Vite frontend** that visualizes the multi-stage analysis pipeline in real time.
- **Pre-trained model weights** (PyTorch + scikit-learn) for visual, audio, temporal, and A/V-sync cues plus a calibrated fusion head.
- A **research notebook** used to train and evaluate the models.

---

## Repository layout

```bash
├─ backend-deepfake-detection-pipeline   # FastAPI service and helpers
├─ frontend-deepfake-detection-pipeline  # React/Vite single-page app
├─ models                                # Saved PyTorch / joblib weights
└─ mugen-trainning-notebook.ipynb        # Notebook used for experimentation
```

---

## Backend (FastAPI)

- **Entry point:**  
  `backend-deepfake-detection-pipeline/main.py`

- **Key helper:**  
  `backend-deepfake-detection-pipeline/utils.py`  
  Contains robust video utilities, including `extract_frames`, which:
  - Writes uploaded bytes to a temp file.
  - Tries to read with OpenCV.
  - If needed, uses `ffmpeg` to transcode to a baseline H.264 MP4.
  - Returns a `torch.Tensor` of shape `[N, 3, H, W]` (values in `[0,1]`) ready for CNN inference.

- **Requirements (backend):**
  - `fastapi`
  - `uvicorn[standard]`
  - `python-multipart`
  - `opencv-python-headless`
  - `numpy`
  - `torch`
  - `joblib`
  - plus any other utilities listed in `requirements.txt`

  Install via:

  ```bash
  cd backend-deepfake-detection-pipeline
  python -m venv .venv
  source .venv/bin/activate        # or .venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```

- **Environment prerequisite:**
  - `ffmpeg` should be available on the system `PATH`.  
    This allows `extract_frames` to repair / transcode unusual video containers before handing them to OpenCV.

- **Run locally:**

  ```bash
  cd backend-deepfake-detection-pipeline
  uvicorn main:app --reload --port 8000
  ```

### API contract

- **Endpoint:** `POST /analyze`

  - **Request:** multipart form with a single field `file` containing the uploaded video.
  - **Response shape:**

    ```json
    {
      "status": "processing" | "complete" | "error",
      "percent": 0-100,
      "stageName": "Queued" | "Extracting frames" | "Running models" | "Final classification",
      "result": {
        "verdict": "Deepfake" | "Authentic",
        "confidence": 0.0-100.0
      },
      "error": "optional error message"
    }
    ```

  - The backend runs the **full 5-way fusion pipeline** (Tiny-LaDeDa + visual CNN + audio CNN + temporal BiLSTM + A/V sync head) using pre-trained weights in `models/` and returns a calibrated probability as a **confidence score**.

- **Future async pattern (optional):**

  If you later move to asynchronous processing, you can:

  - Make `POST /analyze` return `{ "analysisId": "<uuid>" }`.
  - Add `GET /status/{analysisId}` that returns the same `{status, percent, stageName, result?}` shape.  
    The frontend is already structured to support a polling-based status endpoint.

---

## Frontend (React + Vite)

- **Location:**  
  `frontend-deepfake-detection-pipeline`

- **Stack:**
  - React 19
  - Vite
  - Framer Motion for pipeline animations
  - Tailwind CSS via CDN in `index.html`

- **Main component:**  
  `components/DeepfakeDetector.tsx`

  Responsibilities:
  - Drag-and-drop / click-to-upload video files.
  - Calls the `/analyze` endpoint and displays progress (`status`, `percent`, `stageName`).
  - Shows final verdict (“Deepfake” vs “Authentic”) with calibrated confidence and a short textual explanation.
  - Animates the multi-stage pipeline (ingest → compressed-domain hints → tiny gate → heavy models → fusion).

- **Configuration:**
  - Create `.env.local` inside `frontend-deepfake-detection-pipeline` with:

    ```bash
    VITE_API_URL=http://127.0.0.1:8000    # or your deployed backend URL
    # Optional (for future Gemini helpers)
    VITE_GEMINI_API_KEY=your_key_here
    ```

- **Run locally:**

  ```bash
  cd frontend-deepfake-detection-pipeline
  npm install
  npm run dev    # typically serves at http://localhost:3000
  ```

- **Simulation mode:**

  - The UI is built to support a “fake animation” mode so the user always sees a nice pipeline animation.
  - In `App.tsx`, the `simulate` prop controls whether the frontend runs a local simulation in parallel with the real API call.
  - Once you trust backend latency and stability, set `simulate={false}` so that the animation timing is driven purely by actual progress from the backend.

---

## Models

Model weights are stored under the top-level `models/` directory. These are **trained offline** (see the notebook) and loaded lazily by the backend at runtime.

### Current artifacts

- **Visual CNN (frame-level):**  
  `visual_v0_resnet18_best.pth`  
  - Backbone: ResNet-18 style CNN.
  - Input: RGB frames resized to 224×224 (`[3, 224, 224]`).
  - Output: per-frame deepfake probability.
  - The backend aggregates per-frame logits into clip-level statistics:
    - `visual_prob_mean`, `visual_prob_max`, `visual_prob_p90`
    - `visual_num_faces` (approximation of how many face crops / valid detections were used).

- **Tiny-LaDeDa patch gate:**  
  `tiny_ladeda_finetuned.pth`  
  - A very small (~1.3K parameter) CNN operating on face patches / crops.
  - Designed as a **fast, CPU-friendly gate**:
    - Processes many small patches quickly.
    - Early-exits if all patches are confidently “real,” reducing GPU load.
  - Clip-level features:
    - `tiny_prob_mean`, `tiny_prob_max`, `tiny_prob_p90`
    - `tiny_num_files`, `tiny_num_patches`

- **Audio anti-spoof CNN:**  
  `audio_cqcc_cnn_asvspoof_la.pth`  
  - Input: CQCC (Constant-Q Cepstral Coefficients) features derived from the audio track.
  - Pretrained on the ASVspoof 2019 LA scenario and fine-tuned in this pipeline.
  - Output: `audio_prob_fake` (probability that the **audio** is spoofed / manipulated).

- **Temporal BiLSTM:**  
  `temporal_bilstm.pth`  
  - Input: sequence of visual features / logits over time (per-frame or per-face embeddings).
  - Architecture: bidirectional LSTM with a small fully-connected head.
  - Captures **temporal inconsistencies** (e.g., copy-paste artifacts, unnatural motion).
  - Output: `temporal_prob_fake`.

- **Audio–Visual Sync Head:**  
  `av_sync_head.pth`  
  - Input: aligned audio and visual features.
  - Learns a scalar **sync score** measuring how well lip motion matches speech.
  - Output: `sync_prob` (higher → more likely “in-sync” or “consistent”).

- **Fusion heads (logistic regression, scikit-learn):**
  - `clip_fusion_logreg.joblib` — baseline 2-way fusion.
  - `clip_fusion_logreg_3way.joblib` — 3-way fusion variants.
  - `clip_fusion_logreg_4way.joblib` — 4-way fusion variants.
  - `clip_fusion_logreg_5way.joblib` — raw 5-way fusion.
  - `clip_fusion_logreg_5way_calibrated.joblib` — final **calibrated** 5-way fusion bundle, containing:
    - `model` — trained `LogisticRegression` classifier.
    - `feature_cols` — canonical list of feature names (must match runtime order).
    - `temperature` — scalar calibration parameter used to sharpen / smooth probabilities.

  The final 5-way model expects 12 scalar features per clip:

  ```python
  [
      # Tiny gate
      "tiny_prob_mean", "tiny_prob_max", "tiny_prob_p90",
      "tiny_num_files", "tiny_num_patches",

      # Visual
      "visual_prob_mean", "visual_prob_max", "visual_prob_p90",
      "visual_num_faces",

      # Audio / temporal / sync
      "audio_prob_fake", "temporal_prob_fake", "sync_prob",
  ]
  ```

---

## Model architecture & training details

> **Note:** This section summarizes the current research pipeline. Exact hyperparameters and dataset splits are documented in `mugen-trainning-notebook.ipynb`.

### 1. Overall architecture

At a high level, the system is **multi-branch** and **multi-modal**:

1. **Preprocessing / ingest**
   - Extract frames from the uploaded video (up to `N` frames, default 8–16).
   - Extract and normalize audio track.
   - Generate face crops / patches where available.

2. **Branch-wise inference**
   - **Tiny-LaDeDa patch gate** — many lightweight patch predictions.
   - **Visual CNN** — 224×224 frame-level CNN over faces/frames.
   - **Temporal BiLSTM** — sequence model over visual features.
   - **Audio CNN (CQCC)** — spectrogram-style CNN over CQCC features.
   - **A/V Sync Head** — measures audio–lip synchronization.

3. **Feature aggregation**
   - For each branch, aggregate outputs into a small set of robust statistics (mean, max, 90th percentile, counts).
   - Concatenate into a 12-dimensional feature vector per clip.

4. **Fusion & calibration**
   - Pass features through a logistic regression fusion head to obtain a **raw deepfake probability**.
   - Apply **temperature scaling** for better calibration.
   - Map calibrated probability to:
     - Discrete verdict (`Deepfake` vs `Authentic` using a threshold, e.g., 0.5).
     - Human-readable confidence (0–100%).

### 2. Visual branch (ResNet-18)

- **Backbone:** ResNet-18 variant with:
  - Initial convolution + maxpool.
  - Four residual stages with skip connections.
  - Global average pooling + final linear classifier.
- **Training data:** clip frames labeled as real/fake from public deepfake datasets (e.g., DFDC) and internal subsets.
- **Training objective:**
  - Binary cross-entropy / log-loss on frame-level labels.
  - Standard data augmentation: random horizontal flip, color jitter, small crops.
- **Aggregation at inference:**
  - For each clip:
    - Run CNN on selected frames.
    - Compute `visual_prob_mean`, `visual_prob_max`, and `visual_prob_p90`.
    - `visual_num_faces` approximates how many valid face crops contributed.

### 3. Tiny-LaDeDa patch gate

- **Goal:** cheap, CPU-friendly first-pass filter to triage content.
- **Architecture:**
  - Very small CNN (≈1.3K parameters).
  - Convolutions over small face crops / patches.
  - Global pooling + single-logit classifier.
- **Usage:**
  - For each video, sample many small patches.
  - Produce per-patch probabilities; aggregate into:
    - `tiny_prob_mean`, `tiny_prob_max`, `tiny_prob_p90`,
    - `tiny_num_files`, `tiny_num_patches`.
- **Training:**
  - Trained on in-the-wild manipulations: varied compression levels, different source datasets, different synthesis methods.
  - Focus on **high recall** (do not miss deepfakes even if some real content is escalated).

### 4. Audio anti-spoof CNN (CQCC)

- **Input pipeline:**
  - Decode audio track.
  - Resample to a standard sampling rate.
  - Compute CQCC features (similar to MFCC but on the constant-Q transform).
  - Normalize features (per-utterance or global statistics).
- **Architecture:**
  - 2D convolutional network over time–frequency CQCC maps.
  - Several convolution + pooling blocks followed by fully-connected layers.
- **Training:**
  - Initialized and trained on ASVspoof 2019 LA scenario.
  - Objective: classify bona fide vs spoofed audio.
- **Inference output:**
  - Clip-level `audio_prob_fake` scalar.

### 5. Temporal BiLSTM

- **Input:** sequence of visual embeddings / logits over time (one per frame or per face track).
- **Architecture:**
  - Bi-directional LSTM with hidden units sufficient to model temporal dependencies.
  - Last hidden state or pooled representation fed into a small MLP.
- **Objective:**
  - Detect temporal inconsistencies such as:
    - Sudden artifact bursts.
    - Unrealistic motion patterns.
- **Output:**
  - `temporal_prob_fake` scalar per clip.

### 6. Audio–Visual Sync Head

- **Input features:**
  - Audio embeddings for short windows.
  - Visual embeddings for matching frames (e.g., mouth-region features).
- **Architecture:**
  - Shallow network that predicts whether a given audio–visual tuple is “in-sync”.
- **Training:**
  - Positive pairs: correct A/V alignment.
  - Negative pairs: misaligned or shuffled combinations.
- **Output:**
  - `sync_prob` (higher suggests audio and video are consistent; anomalies can indicate re-dubbed or manipulated content).

### 7. Fusion models and calibration

- **Feature space:**
  - Each clip is represented by 12 scalar features:

    ```text
    Tiny gate:
      tiny_prob_mean, tiny_prob_max, tiny_prob_p90,
      tiny_num_files, tiny_num_patches

    Visual:
      visual_prob_mean, visual_prob_max, visual_prob_p90,
      visual_num_faces

    Audio/Temporal/Sync:
      audio_prob_fake, temporal_prob_fake, sync_prob
    ```

- **Fusion models:**
  - Multiple logistic regression heads trained over different modality subsets (2-way, 3-way, 4-way) for **ablation** and robustness.
  - The main production head is the **5-way calibrated fusion model** stored in `clip_fusion_logreg_5way_calibrated.joblib`.

- **Training procedure (fusion):**
  1. Generate clip-level features on a validation dataset (e.g., a curated subset of DFDC).
  2. Train `LogisticRegression` on these features with:
     - Class weights to handle imbalance (`real` vs `fake` counts).
     - Standardization / normalization where appropriate.
  3. Evaluate with:
     - **AUC**, **Accuracy**, **F1-score**, confusion matrix.
  4. Apply **temperature scaling**:
     - Learn scalar `T` on a held-out set to minimize Brier score.
     - Replace probabilities `p` with:

       p_cal = sigmoid( log(p / (1-p)) / T )

  - In your calibration run, an optimal temperature around `T ≈ 0.3679` was found and stored inside the calibrated bundle.

  - Example experimental result (subset):
    - On a small DFDC subset of 162 clips (151 fake, 11 real), 5-fold cross-validation with the 5-way fusion features yields **near-perfect metrics** (AUC ≈ 1.0, ACC ≈ 1.0, F1 ≈ 1.0).  
    - **Important caveat:** this is on a limited subset and should not be interpreted as a global, production-level benchmark. Performance on broader, unseen distributions will be lower and should be re-evaluated.

---

## Notebook

`mugen-trainning-notebook.ipynb` documents experimentation and training:

- Data preparation and feature extraction for:
  - Tiny-LaDeDa patch features.
  - Visual CNN logits.
  - Audio CQCC features.
  - Temporal sequence features.
  - A/V sync scores.
- Fusion training scripts:
  - Training 2-way, 3-way, 4-way, and 5-way logistic regression heads.
  - Cross-validation and ablation study.
  - Calibration (temperature scaling) and metric computation.
- Visualization:
  - ROC curves for different modality combinations.
  - Confusion matrices.
  - Distribution plots for features by class (real vs fake).

Use this notebook as the **source of truth** for reproducibility and further research.

---

## Development workflow

1. **Start backend:**

   ```bash
   cd backend-deepfake-detection-pipeline
   uvicorn main:app --reload --port 8000
   ```

2. **Start frontend:**

   ```bash
   cd frontend-deepfake-detection-pipeline
   npm install
   npm run dev
   ```

3. **Configure environment:**
   - Confirm `VITE_API_URL` in `.env.local` (frontend).
   - Confirm model files exist in `../models/` relative to the backend root.
   - Make sure `ffmpeg` is installed and accessible.

4. **Run end-to-end:**
   - Open `http://localhost:3000`.
   - Upload a short MP4 (H.264) video clip.
   - Watch the pipeline animation and monitor:
     - `Queued` → `Extracting frames` → `Running models` → `Final classification`.
   - Review the returned verdict and confidence in both the frontend and backend logs.

---

## Testing & future improvements

- **Unit tests:**
  - Add tests for `utils.extract_frames` with:
    - Valid videos
    - Corrupted / truncated streams
    - Non-video inputs
  - Mock model inference to test `/analyze` without requiring GPUs.

- **Integration tests:**
  - Use `pytest` + `httpx` or `requests` to POST sample videos into `/analyze`.
  - Validate response schema, latency bounds, and basic correctness.

- **Scalability:**
  - Offload heavy inference to workers (Celery / RQ / custom GPU service).
  - Use `POST /analyze` → `analysisId` + `GET /status/{id}` to decouple upload and analysis.
  - Consider batching clips and caching features.

- **Observability:**
  - Add structured logging (JSON logs).
  - Track per-branch latencies and model errors.
  - Add Prometheus / OpenTelemetry metrics for monitoring in production.

- **Model improvements:**
  - Fine-tune on more diverse, highly compressed user-generated content.
  - Expand sync modeling to multi-person and overlapping speech.
  - Explore lightweight transformers for temporal modeling.
  - Periodically recalibrate fusion and thresholds based on live feedback.

- **UX enhancements:**
  - Show branch-wise confidence breakdown (visual vs audio vs temporal vs sync).
  - Provide short textual explanations (e.g., “Visual artifacts detected” vs “Audio inconsistency detected”).

---

This README should now give reviewers, judges, and collaborators a clear end-to-end view of what your system does, how the models are structured and trained, and how to run it locally.
