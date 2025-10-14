# Shotlist & Storyboarding Tool

This repository now ships with a lightweight Python helper that orchestrates a
three-stage visual development workflow powered by the OpenAI Images API and
Sora 2 for video generation.

## Features

1. **Sketch Expansion** – Upload a rough thumbnail or storyboard sketch and the
   tool will request four cleaned-up black and white frames using
   `gpt-image-1`. Character and environment references can be attached so that
   generated frames stay consistent from shot to shot.
2. **Colorization Pass** – Feed the black and white frames back into the tool
   together with a style reference. The tool generates cinematic, color boards
   tailored to the uploaded look development reference.
3. **Animatic Generation** – The curated frames are passed to Sora 2 to produce a
   moving preview. Frame rate and duration can be tweaked per sequence.

## File Locations

| Component | Path |
|-----------|------|
| Python module | `tools/shotlist_storyboard_tool.py` |
| Output directory | `./outputs` (created automatically) |

## Usage

Install the `openai` Python package and set the `OPENAI_API_KEY` environment
variable. Then run:

```bash
python tools/shotlist_storyboard_tool.py \
  path/to/sketch_1.png path/to/sketch_2.png \
  --characters assets/hero_turnaround.png \
  --environments assets/city_street.png \
  --style assets/color_key.png \
  --scene-prompt "Dawn chase sequence with dynamic lighting" \
  --fps 24 \
  --duration 8
```

Outputs will be written to the `outputs/` directory, including intermediate
black and white frames, final color frames, and either an MP4 animatic or a JSON
file pointing to the Sora-rendered video URL.

## Integration Notes

The module is intentionally decoupled from Toolkit-specific APIs. It can be
imported into custom `tk-desktop` or `tk-shotgun` commands, or used inside DCC
hooks where artists need quick access to storyboarding assistance.
