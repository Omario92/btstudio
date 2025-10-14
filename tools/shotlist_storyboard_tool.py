"""Shotlist and Storyboarding automation utilities.

This module provides a high level orchestration class that can be reused inside
ShotGrid/Shotgun Toolkit configurations or executed as a standalone CLI.  The
implementation focuses on a three-stage creative pipeline:

1. Upload hand-drawn sketches that are expanded into four black and white
   storyboard frames via the OpenAI Images API.
2. Upscale and recolor those frames according to an uploaded style reference in
   order to obtain production-ready color boards.
3. Hand off the curated set of frames to Sora 2 in order to synthesize a motion
   preview.

The code relies on dependency injection of OpenAI clients so that it can be unit
tested without performing real network calls.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from openai import OpenAI


@dataclass
class ReferenceAsset:
    """Represents a user supplied visual reference.

    Attributes:
        path: Path to the image on disk.
        description: Free form text describing the intent of the asset.
        category: Either "character", "environment", or "style".  This value is
            appended to prompts so that model generations stay on-model.
    """

    path: Path
    description: str
    category: str

    def as_prompt_fragment(self) -> str:
        """Return a natural language fragment summarizing the asset."""
        return f"{self.category.title()} reference '{self.path.name}': {self.description}."

    def as_image_payload(self) -> Dict[str, Any]:
        """Return a base64 encoded payload that can be attached to API calls."""
        return {
            "name": self.path.name,
            "image": base64.b64encode(self.path.read_bytes()).decode("utf-8"),
        }


@dataclass
class StoryboardResult:
    """Holds metadata about generated images or videos."""

    assets: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShotlistStoryboardTool:
    """Automates storyboard generation across sketch, color, and video stages."""

    def __init__(
        self,
        client: OpenAI,
        sora_client: Optional[OpenAI] = None,
        output_dir: Path | str = Path("./outputs"),
        frames_per_sketch: int = 4,
        bw_model: str = "gpt-image-1",
        color_model: str = "gpt-image-1",
        sora_model: str = "sora-2.0",
    ) -> None:
        self.client = client
        self.sora_client = sora_client or client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_per_sketch = frames_per_sketch
        self.bw_model = bw_model
        self.color_model = color_model
        self.sora_model = sora_model

    # ------------------------------------------------------------------
    # High level orchestration
    # ------------------------------------------------------------------
    def run_pipeline(
        self,
        sketches: Sequence[Path | str],
        characters: Sequence[ReferenceAsset] = (),
        environments: Sequence[ReferenceAsset] = (),
        style_reference: Optional[ReferenceAsset] = None,
        scene_prompt: Optional[str] = None,
        fps: int = 24,
        duration_seconds: int = 6,
    ) -> Dict[str, StoryboardResult]:
        """Execute the full storyboard workflow.

        Args:
            sketches: Paths to rough storyboard sketches.
            characters: Character reference assets that will be embedded inside
                prompts to maintain consistent character design.
            environments: Environment reference assets used for consistency.
            style_reference: Optional color/style guide.
            scene_prompt: Additional natural language instructions about tone,
                framing, or camera moves.
            fps: Desired video framerate for the Sora render.
            duration_seconds: Duration of the synthesized motion test.

        Returns:
            A mapping with keys "bw", "color", and "video" pointing to
            :class:`StoryboardResult` instances for each stage.
        """

        sketches = [Path(p) for p in sketches]
        bw_result = self.generate_black_and_white_storyboards(
            sketches=sketches,
            characters=characters,
            environments=environments,
            scene_prompt=scene_prompt,
        )

        color_result = self.generate_color_storyboards(
            bw_result.assets,
            style_reference=style_reference,
            characters=characters,
            environments=environments,
            scene_prompt=scene_prompt,
        )

        video_result = self.generate_animatic(
            frames=color_result.assets,
            fps=fps,
            duration_seconds=duration_seconds,
            scene_prompt=scene_prompt,
        )

        return {"bw": bw_result, "color": color_result, "video": video_result}

    # ------------------------------------------------------------------
    # Stage 1 - B/W storyboards
    # ------------------------------------------------------------------
    def generate_black_and_white_storyboards(
        self,
        sketches: Sequence[Path],
        characters: Sequence[ReferenceAsset] = (),
        environments: Sequence[ReferenceAsset] = (),
        scene_prompt: Optional[str] = None,
        size: str = "1024x576",
    ) -> StoryboardResult:
        """Expand sketch uploads into polished black and white storyboard frames."""

        fragments = [asset.as_prompt_fragment() for asset in characters]
        fragments += [asset.as_prompt_fragment() for asset in environments]
        if scene_prompt:
            fragments.append(scene_prompt)

        prompt = (
            "Generate cinematic storyboard frames based on uploaded sketches. "
            "Use clean lines, preserve character likeness, and keep the tone filmic."
        )
        if fragments:
            prompt += "\nAdditional guidance: " + " ".join(fragments)

        outputs: List[Path] = []
        metadata: Dict[str, Any] = {"prompt": prompt, "sketches": [str(p) for p in sketches]}

        for sketch in sketches:
            payload = [self._load_image(sketch, "sketch")] + [a.as_image_payload() for a in characters + tuple(environments)]
            response = self.client.images.generate(
                model=self.bw_model,
                prompt=prompt,
                n=self.frames_per_sketch,
                size=size,
                image=payload,
                style="sketch",
            )
            generated_paths = self._persist_image_set(
                response,
                prefix=f"{sketch.stem}_bw",
            )
            outputs.extend(generated_paths)
            metadata[str(sketch)] = self._extract_image_metadata(response)

        return StoryboardResult(outputs, metadata)

    # ------------------------------------------------------------------
    # Stage 2 - Colorized frames
    # ------------------------------------------------------------------
    def generate_color_storyboards(
        self,
        frames: Sequence[Path | str],
        style_reference: Optional[ReferenceAsset] = None,
        characters: Sequence[ReferenceAsset] = (),
        environments: Sequence[ReferenceAsset] = (),
        scene_prompt: Optional[str] = None,
        size: str = "1536x864",
    ) -> StoryboardResult:
        """Recolor black and white boards according to the provided style guide."""

        frames = [Path(p) for p in frames]
        fragments = [asset.as_prompt_fragment() for asset in characters]
        fragments += [asset.as_prompt_fragment() for asset in environments]
        if style_reference:
            fragments.append(style_reference.as_prompt_fragment())
        if scene_prompt:
            fragments.append(scene_prompt)

        prompt = (
            "Transform storyboard line art into fully lit cinematic frames using the "
            "uploaded style reference. Maintain composition, lighting continuity, "
            "and consistent characters across all frames."
        )
        if fragments:
            prompt += "\nAdditional guidance: " + " ".join(fragments)

        reference_payload: List[Dict[str, Any]] = []
        if style_reference:
            reference_payload.append(style_reference.as_image_payload())

        reference_payload.extend(
            [asset.as_image_payload() for asset in characters + tuple(environments)]
        )

        outputs: List[Path] = []
        metadata: Dict[str, Any] = {"prompt": prompt, "frames": [str(p) for p in frames]}

        for frame in frames:
            payload = [self._load_image(frame, "frame")] + reference_payload
            response = self.client.images.generate(
                model=self.color_model,
                prompt=prompt,
                n=1,
                size=size,
                image=payload,
                style="cinematic",
            )
            generated_paths = self._persist_image_set(
                response,
                prefix=f"{frame.stem}_color",
            )
            outputs.extend(generated_paths)
            metadata[str(frame)] = self._extract_image_metadata(response)

        return StoryboardResult(outputs, metadata)

    # ------------------------------------------------------------------
    # Stage 3 - Video synthesis
    # ------------------------------------------------------------------
    def generate_animatic(
        self,
        frames: Sequence[Path | str],
        fps: int = 24,
        duration_seconds: int = 6,
        scene_prompt: Optional[str] = None,
    ) -> StoryboardResult:
        """Trigger a Sora 2 render using the generated color boards as guidance."""

        frames = [Path(p) for p in frames]
        storyboard_payload = [self._load_image(frame, "frame") for frame in frames]
        prompt = (
            "Create a smooth cinematic animatic that respects the uploaded storyboard "
            "frames. Use gentle camera moves and match the emotional tone of the "
            "scene."
        )
        if scene_prompt:
            prompt += "\nScene direction: " + scene_prompt

        response = self.sora_client.videos.generate(
            model=self.sora_model,
            prompt=prompt,
            fps=fps,
            duration=duration_seconds,
            storyboard_frames=storyboard_payload,
        )

        video_path = self._persist_video_response(response)
        metadata = self._extract_video_metadata(response)
        metadata.update({"prompt": prompt, "fps": fps, "duration_seconds": duration_seconds})

        return StoryboardResult([video_path], metadata)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _persist_image_set(self, response: Any, prefix: str) -> List[Path]:
        """Write OpenAI image responses to disk and return their paths."""

        outputs: List[Path] = []
        for idx, data in enumerate(getattr(response, "data", [])):
            filename = f"{prefix}_{idx + 1:02d}.png"
            output_path = self.output_dir / filename
            if hasattr(data, "b64_json") and data.b64_json:
                output_path.write_bytes(base64.b64decode(data.b64_json))
            elif isinstance(data, dict) and "b64_json" in data:
                output_path.write_bytes(base64.b64decode(data["b64_json"]))
            else:
                raise ValueError("Unsupported image response format: " + repr(data))
            outputs.append(output_path)
        return outputs

    def _persist_video_response(self, response: Any) -> Path:
        """Persist Sora responses (base64 or URL) to disk."""

        data = getattr(response, "data", None)
        if isinstance(data, list) and data:
            item = data[0]
        else:
            item = response

        video_path = self.output_dir / "storyboard_preview.mp4"

        if hasattr(item, "b64_video") and item.b64_video:
            video_path.write_bytes(base64.b64decode(item.b64_video))
        elif isinstance(item, dict) and "b64_video" in item:
            video_path.write_bytes(base64.b64decode(item["b64_video"]))
        elif isinstance(item, dict) and "url" in item:
            # Store the URL as metadata JSON if the asset is hosted remotely.
            video_path = self.output_dir / "storyboard_preview.json"
            video_path.write_text(json.dumps({"url": item["url"]}, indent=2))
        else:
            raise ValueError("Unsupported video response format: " + repr(item))

        return video_path

    def _extract_image_metadata(self, response: Any) -> Dict[str, Any]:
        """Pull confidence or seed metadata from an image response."""

        metadata: Dict[str, Any] = {}
        for idx, data in enumerate(getattr(response, "data", [])):
            container: Dict[str, Any]
            if hasattr(data, "model_output"):
                container = getattr(data, "model_output")  # type: ignore[assignment]
            elif isinstance(data, dict):
                container = data
            else:
                continue
            metadata[str(idx)] = {
                key: container[key]
                for key in ("seed", "finish_reason", "revised_prompt")
                if key in container
            }
        return metadata

    def _extract_video_metadata(self, response: Any) -> Dict[str, Any]:
        """Pull useful metadata from a Sora response."""

        data = getattr(response, "data", None)
        if isinstance(data, list) and data:
            item = data[0]
        else:
            item = response

        if hasattr(item, "model_output"):
            payload = getattr(item, "model_output")
        elif isinstance(item, dict):
            payload = item
        else:
            payload = {}

        metadata = {
            key: payload[key]
            for key in ("id", "status", "seed", "duration")
            if isinstance(payload, dict) and key in payload
        }
        return metadata

    def _load_image(self, path: Path, role: str) -> Dict[str, Any]:
        """Encode an on-disk image so that it can be uploaded."""

        return {
            "name": f"{role}_{path.name}",
            "image": base64.b64encode(path.read_bytes()).decode("utf-8"),
        }


def _build_assets(paths: Iterable[str], category: str) -> List[ReferenceAsset]:
    """Helper that converts raw paths into :class:`ReferenceAsset` objects."""

    return [
        ReferenceAsset(path=Path(p), description=f"Reference asset for {category}", category=category)
        for p in paths
    ]


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Basic CLI entrypoint that demonstrates how to call the tool."""

    import argparse

    parser = argparse.ArgumentParser(description="Generate a storyboard pipeline with OpenAI + Sora")
    parser.add_argument("sketches", nargs="+", help="Paths to sketch images")
    parser.add_argument("--characters", nargs="*", default=[], help="Paths to character reference images")
    parser.add_argument("--environments", nargs="*", default=[], help="Paths to environment reference images")
    parser.add_argument("--style", help="Optional path to a style reference image")
    parser.add_argument("--scene-prompt", help="Optional text describing the scene")
    parser.add_argument("--output", default="./outputs", help="Directory for generated assets")
    parser.add_argument("--fps", type=int, default=24, help="Target frames per second for the animatic")
    parser.add_argument("--duration", type=int, default=6, help="Animatic duration in seconds")

    args = parser.parse_args(argv)

    client = OpenAI()
    tool = ShotlistStoryboardTool(client=client, output_dir=args.output)

    characters = _build_assets(args.characters, "character")
    environments = _build_assets(args.environments, "environment")
    style_reference = (
        ReferenceAsset(path=Path(args.style), description="Color and lighting style", category="style")
        if args.style
        else None
    )

    tool.run_pipeline(
        sketches=[Path(p) for p in args.sketches],
        characters=characters,
        environments=environments,
        style_reference=style_reference,
        scene_prompt=args.scene_prompt,
        fps=args.fps,
        duration_seconds=args.duration,
    )


if __name__ == "__main__":
    main()
