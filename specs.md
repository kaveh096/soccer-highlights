# **Project: Sunday Soccer Highlights Engine**

## **1. Overview**

A specialized video processing pipeline designed to extract soccer highlights from 2-hour continuous recordings. The system focuses on "shots on target" by identifying audio-visual peaks and extracting the preceding context.

## **2. Target Use Case**

  - **Activity:** 16-player Sunday morning soccer on a half-court.
  - **Goal:** Generate highlights focusing on goals and shots on target for entertainment and tactical analysis.
  - **Workflow:** Capture (Action Cam) -> Upload (Old Laptop/Web) -> Process (Cloud/Local) -> Share (Mobile).

## **3. Hardware Specification (Current Setup)**

  - **Camera:** DJI Osmo Action 4 (Essential Combo).
  - **Storage:** 128GB SanDisk Extreme MicroSD (U3/V30 rated).
  - **Mounting:** Reused heavy-duty Nikon Tripod.
  - **Power:** Continuous 2-hour recording via external USB-C Power Bank (user-owned).
  - **Positioning:** Fixed sideline placement near the opponent's goal box (targeting the attacking third).

## **4. Software Requirements & Constraints**

### **Constraints**

  - **Flexible Budget:** Open to low-cost subscriptions or API usage fees (e.g., a few dollars per game).
  - **Compute:** The primary user laptop is an older Windows machine; heavy rendering should be offloaded to efficient scripts (FFmpeg) or cloud-based processing.
  - **Configurability:** System parameters must be easily adjustable via a config file or environment variables.

### **Functional Requirements**

  - **Bulk Ingestion:** Handle a single ~60GB-80GB video file (4K/2.7K).
  - **Audio-Peak Detection:** Scan the audio track for decibel spikes (clapping, ball-striking, cheering) to identify "events."
  - **Configurable Temporal Slicing:**
    - Adjustable look-back window (e.g., initially 30s-60s for testing).
    - Adjustable "post-peak" buffer.
  - **Collapsing/Deduplication:** Merge overlapping highlights into a single continuous segment.
  - **Output Options:**
    - Option A: Individual clipped files (one per highlight).
    - Option B: A single concatenated highlight reel.
  - **Metadata:** Generate a timestamp log of all detected events.

## **5. Implementation Design Suggestions**

### **Phase 1: The Audio-Trigger Script**

  - **Engine:** Python with FFmpeg (via subprocess or wrapper).
  - **Logic:**
    1. Extract audio to a lightweight .wav.
    2. Use NumPy or SciPy to find timestamps where volume exceeds a dynamic threshold.
    3. Use FFmpeg for "stream copy" slicing (no re-encoding) to extract segments based on config.
    4. Provide choice to either output separate files or concat them.

### **Phase 2: Vision AI Integration**

  - **Google Cloud Video Intelligence API:** Use "Shot Detection" and "Object Tracking" to refine event accuracy.
  - **AWS Rekognition:** Action recognition to detect "Kicking" or "Goal" celebrations.
  - **GPT-4o (Video):** Analyze specific audio-detected segments to confirm the event type before final export.

## **6. Future Upgrades**

  - **Cloud Hosting:** Migrate the pipeline to a cloud-based folder listener (e.g., S3 Bucket + Lambda) for automated processing upon upload.
  - **Interactive UI:** A simple web-based player to review and prune detected highlights before final rendering.

## **7. Deployment Plan**

  - **Code Hosting:** GitHub.
  - **Execution:** Locally via CLI (laptop-assisted) or as a remote cloud task.
