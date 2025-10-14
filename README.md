# btstudio

BTStudio Config with custom Shotlist/Storyboarding automation helpers.

## Shotlist/Storyboarding Tool

The `tools/shotlist_storyboard_tool.py` module automates a three step creative
workflow:

1. Generate four black and white storyboard frames per sketch using the OpenAI
   Images API.
2. Convert those frames into color cinematics guided by a user-supplied style
   reference for look consistency.
3. Compile the final frames into an animatic with Sora 2 so teams can review
   timing and motion direction.

Refer to `docs/shotlist_storyboard.md` for setup instructions, CLI usage, and
integration notes.
