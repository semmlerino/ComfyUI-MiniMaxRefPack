# MiniMax / Hailuo Video Prompting Best Practices

A concise reference for prompting MiniMax video generation models (Video-01, Hailuo, and H3 / H3-Context-IR multimodal architectures).

---

## 1. Core Prompt Architectures

### A. Basic Text-to-Video (T2V) Formula
For short, single-shot clips without audio fields:
> `[Subject & Micro-action]` + `[Environment & Setting]` + `[Camera Motion & Framing]` + `[Lighting & Mood]`

*Example:*  
`A ceramicist's clay-dusted hands center a spinning vase on a pottery wheel. Warm afternoon sunlight cuts through dust motes across a rustic studio table. The camera pushes in slowly with small amplitude toward the clay. Natural earthy tones.`

---

### B. The Standard 3-Field Schema (H3 / Production Native)
MiniMax H3 was trained on a strict three-field multimodal structure. Use these exact field labels in order:

```text
integrated_multimodal_description: [Shot 1] Cinematic, close-up shot. A weathered fisherman in a yellow slicker pulls a heavy wet rope against the gunwale. The camera trucks right with small amplitude at slow speed following his hands. Rain beats against the wooden deck.

overall_soundscape: Howling ocean wind, wooden hull creaking, heavy rainfall, and rhythmic ocean surf crashing against the boat.

non_diegetic_music: N/A
```

* **`integrated_multimodal_description`**: The complete visual script, shot sequence, camera moves, subject actions, and diegetic sounds/dialogue.
* **`overall_soundscape`**: 1–4 sentences of continuous diegetic ambient audio (room tone, weather, foley, breathing). **No dialogue or music.**
* **`non_diegetic_music`**: Score heard only by the audience. Name specific instruments and tempo (e.g., *Sparse acoustic cello, 70 bpm*). Set to `N/A` if no music is desired.

---

### C. Full Reference-to-Video (Ref2VA / 6-Section Schema)
Used for advanced multi-reference conditioning (combining images, clips, and audio tracks):

1. **`subject_definitions:`**  
   Define each asset and its role:
   - `<Subject 1> is the woman in <Picture 1>, wearing the dark wool coat.`
   - `<Subject 2> is the street corner in <Picture 2>.`
   - `<Audio 1> is the voice-timbre reference for <Subject 1> (S1).`
2. **`summary:`**  
   Bracketed task prefix followed by one summary sentence:  
   `[reference generation + audio reference] A woman walks down a rain-slicked avenue speaking on a payphone.`
3. **`retention_analysis:`**  
   1:1 match with `subject_definitions` using valid retention markers:
   - *Visual markers:* `fully_preserved` | `partially_preserved` | `attribute_transfer` | `weak_reference`
   - *Audio markers:* `fully_copy` | `partially_copy` | `reference` | `weak_reference`  
   *(Note: voice timbre matching is always `reference`, never `fully_copy`).*
4. **`detailed_description:`** (or `integrated_multimodal_description:`): Style line, then shot-by-shot timeline (`[Shot 1]`, `[Shot 2]`).
5. **`overall_soundscape:`** Ambient foley and environmental noise.
6. **`non_diegetic_music:`** Instrumentation/tempo or `N/A`.

---

## 2. Shot Timing & Temporal Pacing

* **Shot Budget**:
  * **≤ 7 seconds**: 1–2 shots max.
  * **8–11 seconds**: 2–3 shots.
  * **12–15 seconds**: 3–4 shots.
* **Cut Timing**:
  * The final cut should occur at least **1.5 seconds** before clip end.
  * Peak physical action should land at **55%–65%** into clip duration.
* **Cut Syntax**:
  Format cut lines strictly with timestamps:
  ```text
  [Shot 1] Style declaration. Initial framing and action...
  [Shot 2] At 00:04.500, the shot cuts to a medium profile view...
  ```
  *(Allowed cut verbs: `the camera cuts to`, `the shot cuts to`, `the shot transitions to`, `the shot changes to`, `the shot switches to`).*

---

## 3. Camera Movement Vocabulary

MiniMax understands fixed cinematic camera primitives. Compose using: **[Movement] + [Amplitude] + [Speed]**.

| Movement Type | Modifiers | Example Usage |
| :--- | :--- | :--- |
| `Push In`, `Pull Out` | `with small amplitude` | *"The camera pushes in with small amplitude at slow speed toward the doorway."* |
| `Zoom In`, `Zoom Out` | `with large amplitude` | *"The camera zooms in with small amplitude."* |
| `Pan Left`, `Pan Right` | `at slow speed` | *"The camera pans right at slow speed across the desk."* |
| `Truck Left`, `Truck Right` | `at fast speed` | *"The camera trucks left tracking alongside the runner."* |
| `Tilt Up`, `Tilt Down` | *(Omit for medium)* | *"The camera tilts up from the boots to the face."* |
| `Pedestal Up`, `Pedestal Down` | | *"The camera pedestals up above the crowd."* |
| `Arc Shot`, `Tracking Shot` | | *"An arc shot rotates slowly around the subject."* |
| `Static Shot` | | *"Static shot, locked camera with no motion."* |
| `Shake Slightly`, `Shake Strongly` | | *"Handheld camera shakes slightly with natural movement."* |
| `POV`, `Roll CW / CCW` | | *"POV perspective moving forward."* |

---

## 4. Dialogue & Audio Syntax

* **Speech Tag**: Enclose spoken lines in `<d>[Language] Text</d>`.
* **Speaker ID**: Assign stable IDs `(S1)`, `(S2)` in the identity clause immediately before the speaking verb:
  ```text
  The young woman (S1) smiles and says: <d>[English] We can finally leave.</d>
  ```
* **Dialogue Speed**: Natural cadence is **2.5 to 3 English words per second**. Keep lines concise to maintain accurate lip-sync.
* **Off-Screen Voiceover**: Use `says in an off-screen voiceover` and explicitly note that on-screen lips remain closed.
* **Separation Rule**:
  * Dialogue and diegetic music (radios, singers) belong inside `integrated_multimodal_description`.
  * Ambient room tone, impacts, wind, footsteps belong in `overall_soundscape`.
  * Audience score belongs in `non_diegetic_music`.

---

## 5. Physical Realism vs. Banned Buzzwords

MiniMax renders physical geometry, light, and organic texture—not post-processing metadata.

### What Generates Realism
* **Skin**: Specify pores across the nose/cheeks, uneven skin flush, fine downy hair catching light, natural forehead sheen, neck pulse, freckles, micro-expressions.
* **Hair & Cloth**: Individual flyaways, damp strands, fabric weave, soft creases, seam tension, realistic drape under gravity.
* **Lighting**: Specify **one dominant motivated light source** (angle, softness, color temperature) and one subtle fill. Describe how light falls on flesh or surfaces.
* **Framing**: State frame occupancy and which parts are severed by frame edges rather than abstract metric distances.

### What Degrades Generation (Banned / Negative List)
* **Damage Words (destroys skin texture)**:  
  `grain`, `film grain`, `noise`, `compression artifacts`, `macroblocking`, `VHS`, `lo-fi`, `crushed blacks`, `blown highlights`.
* **Empty Hype Buzzwords**:  
  `photorealistic`, `hyperrealistic`, `4K`, `8K`, `masterpiece`, `best quality`, `unbelievable detail`.
* **Vague Aesthetic Labels**:  
  `cinematic` (as an empty filler adjective), `ethereal`, `dreamy`, `beautifully framed`.
* **Vague Music Descriptors**:  
  `moody`, `tense`, `uplifting`, `epic` *(replace with concrete instruments: cello, muted timpani, synthesizers)*.
* **Negative Prompting in Body**:  
  Avoid `no blur, no extra limbs, no watermark`. Use positive spatial descriptions instead.

---

## 6. Image-to-Video (I2V) & Plate Ground Truth

* **Respect the Plate**: The initial frame is ground truth. Do not re-describe identical static elements (wardrobe, background furniture) unless they undergo transformation.
* **Focus on Delta (Motion)**: Explicitly describe the motion path, micro-actions, gaze direction, and camera trajectory originating from the initial frame.
* **Eye Contact**: Clearly direct gaze (e.g., *"maintains steady eye contact with the camera lens throughout"* or *"eyes remain locked on the documents, never glancing up"*).
