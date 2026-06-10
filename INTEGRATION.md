# Integrating with figbox (for the garret-notes Claude)

This is a handoff brief. **figbox** is a small local HTTP service living in
`~/repos/figure-extraction` that detects figure/table bounding boxes in PDFs
using DocLayout-YOLO (DocStructBench) over ONNX Runtime. No torch at runtime,
~500MB resident, CPU-only, fine on the M1. It is meant to be the **local,
offline, no-quota replacement for the Gemini call** in `/ingest` Step 5 Tier 2
(`.claude/skills/ingest/extract-figures.py`).

It is already set up and verified: the model is exported (`model.onnx`) and the
service runs. You do **not** need to touch the figure-extraction repo — just
call it over HTTP.

## What it does and (importantly) does NOT do

- **Does:** takes a PDF, returns bounding boxes per page for layout classes
  (figure, table, captions, etc.), with both pixel and PDF-point coordinates.
- **Does NOT:** crop images, return image bytes, or read caption *text*. It
  gives you boxes; **you do the cropping and the caption OCR.** This split was a
  deliberate decision — figbox stays a pure stateless detector.

So your job on the garret-notes side: call `/detect`, pair each figure/table
box with its caption box, render the page and crop the figure with PIL, OCR the
caption box with tesseract, and write the same `manifest.json` the existing
Gemini script produces — so the rest of Step 5 is unchanged.

## Why prefer it over the Gemini path

DocLayout-YOLO nails the **text-dense vector schematic** case the SKILL.md flags
Gemini as unreliable on. Verified: AlexNet's Fig 2 architecture diagram detected
at **conf 0.97** with a tight box; the two result tables cleanly separated; works
on scanned PDFs too. It's local (no API key, no per-model daily quota, no sending
page images to Google), so it should be the **first Tier-2 attempt**, with Gemini
and manual crop as fallbacks.

## Running / health

```bash
# in ~/repos/figure-extraction
uv run uvicorn server:app --port 8000
```

Health check (use this to decide whether to fall back):

```bash
curl -s localhost:8000/health        # {"ok": true} when the model is loaded
```

It currently has no autostart. Treat "connection refused" as **service down →
fall back to Gemini/manual**, and print how to start it. (We can wire a launchd
login agent later if you want it always-on.)

## The API

### `POST /detect`

Multipart form upload, field name **`file`** = the PDF. Query params:

| param     | default    | meaning                                                              |
|-----------|------------|---------------------------------------------------------------------|
| `dpi`     | 200        | internal render dpi for detection (72–400). 150 is faster, fine for figures. |
| `conf`    | 0.25       | confidence threshold. Use **0.3**; faint old scans → ~0.2 then filter. |
| `classes` | `figure`   | comma-separated, or `all`. Request `figure,table,figure_caption,table_caption`. |
| `pages`   | all        | 1-indexed, e.g. `1-9` or `5,8`.                                      |

Classes available: `title, plain_text, abandon, figure, figure_caption, table,
table_caption, table_footnote, isolate_formula, formula_caption`.

### Response shape (real output, AlexNet p5 — the architecture schematic + its caption)

```json
{
  "dpi": 150,
  "pages": [
    {
      "page": 5,
      "size_px": [1275, 1650],
      "size_pt": [612.0, 792.0],
      "detections": [
        {
          "class": "figure",
          "confidence": 0.9682,
          "bbox_px": [222.3, 168.4, 1052.8, 425.2],
          "bbox_pt": [106.7, 80.83, 505.34, 204.1]
        },
        {
          "class": "figure_caption",
          "confidence": 0.9552,
          "bbox_px": [223.0, 448.3, 1053.4, 561.4],
          "bbox_pt": [107.04, 215.18, 505.63, 269.47]
        }
      ]
    }
  ]
}
```

- `bbox_px` = `[x1, y1, x2, y2]` in pixels **at the requested `dpi`**.
- `bbox_pt` = the same box in **PDF points (72/inch), dpi-independent** — use
  these. `px_at_any_dpi = pt * (your_dpi / 72)`. This lets you detect cheaply at
  `dpi=150` but render+crop sharply at `dpi=300`.
- Boxes are the **graphic only** (caption excluded) — exactly what you want for a
  tight figure crop. The caption is a separate `figure_caption` detection.

## Integration recipe

1. **One call per PDF.** POST the whole PDF with
   `classes=figure,table,figure_caption,table_caption&conf=0.3&pages=...`.
   stdlib is enough (multipart POST to the `file` field); `requests`/`httpx` are
   also already in the venv if you prefer — but note figbox is meant to *remove*
   the google-genai dependency, so don't lean on google-genai's transitive httpx.

2. **Pair captions.** For each `figure` box, pick the `figure_caption` box with
   the most horizontal overlap and the smallest vertical gap (captions sit
   directly below — see the example: caption y 215→269pt right under figure y
   80→204pt, same x-span). Use `table_caption` for `table` boxes (often *above*
   the table). Match each caption once.

3. **Render + crop** (poppler + PIL, same as the Gemini script — no pymupdf in
   that venv). Render the page once at your crop dpi with
   `pdftoppm -png -r <dpi> -f P -l P`, convert each box `pt → px` (`* dpi/72`),
   pad ~1% and clamp to page bounds, `Image.crop`, save. Crop the **figure
   graphic only** (the chosen contract was *not* to bake the caption into the
   image).

4. **Caption text via tesseract.** Crop the caption box from the same render and
   `tesseract <crop>.png stdout`; collapse whitespace → `caption_text`. Derive
   `label` from it with a regex like `^\s*(Figure|Fig\.?|Table)\s*\.?\s*(\d+)`
   → `"Figure 2"` / `"Table 1"` (figbox gives the class, not the printed label).

5. **Emit the same `manifest.json` schema** the Gemini script writes, into the
   same `/tmp/figures-<pdf-stem>/` dir, so the rest of Step 5 (read manifest →
   pick high-value figure → verify with one Read → copy to `attachments/`) is
   untouched:

   ```json
   {
     "page": 5,
     "index": 1,
     "label": "Figure 2",
     "kind": "figure",
     "caption_text": "Figure 2: An illustration of the architecture ...",
     "box_px": [/* at YOUR crop dpi, matching the saved PNG */],
     "file": "/tmp/figures-<stem>/<stem>-p005-fig1.png"
   }
   ```

6. **Service URL + fallback.** Read `FIGBOX_URL` (default
   `http://localhost:8000`). On connection refused / non-200, print a clear
   "figbox not running — start it with `cd ~/repos/figure-extraction && uv run
   uvicorn server:app --port 8000`, or falling back to Gemini" and exit non-zero
   so the skill drops to the Gemini/manual path.

## Where it slots into SKILL.md (Step 5, Tier 2)

Tier 1 (ar5iv asset download) is unchanged — still the default for arxiv. Tier 2
(render + crop, for scanned/non-arxiv/higher-res) gains figbox as the **first**
attempt:

> First try figbox (local DocLayout-YOLO) — POST the PDF to the service, crop
> from the returned boxes, OCR captions with tesseract. It's offline, has no
> quota, and unlike Gemini it reliably catches text-dense architecture/method
> schematics. If figbox is down or returns nothing useful, fall back to
> `extract-figures.py` (Gemini), then to manual crop.

## Gotchas

- **`figure_caption` is a separate box**, not attached to the figure — you must
  pair them (step 2).
- **No printed label** from the model — derive `"Figure 2"` from the OCR'd
  caption (step 4).
- **conf 0.3** drops faint false positives (a scanned page gave a 0.28 phantom
  figure); lower to ~0.2 only for faint old scans, then filter downstream.
- **Multi-figure pages**: `detections` is a flat per-page list — index them
  `fig1, fig2, …` per page like the Gemini script's `index`.
- **Detect cheap, crop sharp**: detect at `dpi=150`, but use `bbox_pt` to render
  the actual crop at `dpi=300` for a crisp attachment.
- **`IMGSZ` env var** on the service (default 1024) trades inference speed for
  small-text accuracy; figures are large objects and barely suffer at 800.
