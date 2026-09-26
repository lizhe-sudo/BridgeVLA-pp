"""Video-only overlays for Codex-controlled RLBench episodes."""

import math
import os
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont


GREEN = (45, 220, 90)
RED = (245, 55, 55)
BLUE = (55, 130, 255)
WHITE = (240, 243, 248)
MUTED = (165, 176, 190)
PANEL = (15, 20, 28)


def orientation_error_degrees(target_quaternion, actual_quaternion):
    """Return the shortest relative rotation angle for XYZW quaternions."""
    if target_quaternion is None or actual_quaternion is None:
        return None
    target = np.asarray(target_quaternion, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_quaternion, dtype=np.float64).reshape(-1)
    if target.shape != (4,) or actual.shape != (4,):
        return None
    target_norm = float(np.linalg.norm(target))
    actual_norm = float(np.linalg.norm(actual))
    if target_norm <= 1e-12 or actual_norm <= 1e-12:
        return None
    dot = abs(float(np.dot(target / target_norm, actual / actual_norm)))
    return math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot))))


def _font(size, bold=False):
    candidates = (
        ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in candidates:
        if os.path.isfile(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _xyz(value):
    if value is None:
        return "N/A"
    try:
        values = [float(x) for x in value[:3]]
    except (TypeError, ValueError, IndexError):
        return "N/A"
    return "[" + ", ".join(f"{x:+.3f}" for x in values) + "]"


def _state(value):
    if value is None:
        return "N/A"
    return "OPEN" if bool(value) else "CLOSED"


def _project(camera, point):
    if point is None:
        return None, False
    projected = camera.project_xyz(point)
    if projected is None:
        return None, False
    x, y, depth = projected
    inside = depth > 0 and 0 <= x < camera.width and 0 <= y < camera.height
    return (int(round(x)), int(round(y))), inside


def _marker(draw, camera, point, color, label, radius=10, font=None):
    xy, inside = _project(camera, point)
    if not inside:
        return False
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=color, outline=(255, 255, 255), width=2)
    draw.text((x + radius + 5, y - 12), label, font=font, fill=color,
              stroke_width=2, stroke_fill=(0, 0, 0))
    return True


def _line(draw, camera, start, end, color, width=3):
    p1, visible1 = _project(camera, start)
    p2, visible2 = _project(camera, end)
    if visible1 and visible2:
        draw.line((*p1, *p2), fill=color, width=width)


def render_recording_view(rgb, camera, phase, current_xyz=None,
                          target_xyz=None, actual_xyz=None,
                          target_history=None, panel=None, failure=None):
    """Render one video frame; input RGB is never modified in place."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8).copy(), mode="RGB")
    draw = ImageDraw.Draw(image)
    marker_font = _font(max(14, camera.height // 48), bold=True)
    target_history = target_history or []

    previous = None
    for index, waypoint in enumerate(target_history[:-1]):
        xy, visible = _project(camera, waypoint)
        if visible:
            if previous is not None:
                draw.line((*previous, *xy), fill=(150, 65, 65), width=2)
            draw.ellipse((xy[0] - 5, xy[1] - 5, xy[0] + 5, xy[1] + 5),
                         fill=(175, 65, 65))
        previous = xy if visible else None

    if phase == "AFTER EXECUTION":
        _line(draw, camera, actual_xyz, target_xyz, BLUE, width=3)
        _marker(draw, camera, actual_xyz, BLUE, "ACTUAL", font=marker_font)
        _marker(draw, camera, target_xyz, RED, "TARGET", font=marker_font,
                radius=12)
    else:
        _line(draw, camera, current_xyz, target_xyz, GREEN, width=3)
        _marker(draw, camera, current_xyz, GREEN, "CURRENT", font=marker_font)
        _marker(draw, camera, target_xyz, RED, "TARGET", font=marker_font,
                radius=12)
        if phase == "EXECUTION" and actual_xyz is not None:
            _marker(draw, camera, actual_xyz, GREEN, "EEF", radius=7,
                    font=marker_font)

    panel = dict(panel or {})
    lines = [
        ("Task", str(panel.get("task", ""))),
        ("Instruction", str(panel.get("instruction", ""))),
        ("Step", str(panel.get("step", ""))),
        ("Model", str(panel.get("model", "gpt-6-luna"))),
        ("Reasoning", str(panel.get("reasoning", "max"))),
        ("Current XYZ", _xyz(current_xyz)),
        ("Target XYZ", _xyz(target_xyz)),
        ("Actual XYZ", _xyz(actual_xyz)),
        ("Position error", panel.get("position_error_m", "N/A")),
        ("Orientation error", panel.get("orientation_error_deg", "N/A")),
        ("Gripper before", _state(panel.get("gripper_before"))),
        ("Luna command", _state(panel.get("gripper_command"))),
        ("Gripper after", _state(panel.get("gripper_after"))),
        ("Planner returned", panel.get("planner_returned", "N/A")),
        ("Target reached", panel.get("target_reached", "N/A")),
        ("Reward", panel.get("reward", "N/A")),
        ("Success", panel.get("success", "N/A")),
        ("Codex latency", panel.get("policy_latency_seconds", "N/A")),
        ("Input tokens", panel.get("input_tokens", "N/A")),
        ("Output tokens", panel.get("output_tokens", "N/A")),
        ("Reasoning tokens", panel.get("reasoning_tokens", "N/A")),
    ]
    canvas = Image.new("RGB", (camera.width + camera.panel_width,
                                camera.height), PANEL)
    canvas.paste(image, (0, 0))
    info = ImageDraw.Draw(canvas)
    title_font = _font(max(18, camera.height // 38), bold=True)
    value_font = _font(max(12, camera.height // 60))
    panel_x = camera.width + 18
    max_chars = max(24, int((camera.panel_width - 36) / max(6, camera.height / 115)))
    info.text((panel_x, 16), phase, font=title_font, fill=WHITE)
    y = 58
    for label, value in lines:
        wrapped = textwrap.wrap(f"{label}: {value}", width=max_chars) or [""]
        for part in wrapped[:3]:
            info.text((panel_x, y), part, font=value_font, fill=WHITE)
            y += max(16, camera.height // 45)
        y += max(1, camera.height // 360)

    target_xy, target_visible = _project(camera, target_xyz)
    if target_xyz is not None and not target_visible:
        info.text((18, 16), "TARGET OFF-SCREEN", font=title_font,
                  fill=(255, 215, 90), stroke_width=2, stroke_fill=(0, 0, 0))
    if failure:
        banner = f"EXECUTION FAILED: {failure}"
        info.rectangle((0, camera.height - 54, camera.width, camera.height),
                       fill=(120, 20, 20))
        info.text((18, camera.height - 42), banner[:100], font=title_font,
                  fill=WHITE)
    return np.asarray(canvas)


def render_title_card(width, height, details):
    image = Image.new("RGB", (width, height), (13, 20, 31))
    draw = ImageDraw.Draw(image)
    title = _font(max(28, height // 18), bold=True)
    body = _font(max(17, height // 36))
    bold = _font(max(18, height // 34), bold=True)
    x = int(width * 0.08)
    y = int(height * 0.12)
    draw.text((x, y), "GPT-6 Luna Max", font=title, fill=WHITE)
    y += int(height * 0.09)
    draw.text((x, y), "RLBench Closed-Loop Control", font=bold,
              fill=(80, 190, 255))
    y += int(height * 0.12)
    text_lines = [
        f"Task: {details.get('task', '')}",
        f"Episode: {details.get('episode', '')}",
        f"Instruction: {details.get('instruction', '')}",
        f"Model: {details.get('model', 'gpt-6-luna')}",
        f"Reasoning: {details.get('reasoning', 'max')}",
        "Action space: Absolute EEF Pose",
        "Policy views: Front | Left Shoulder | Right Shoulder | Wrist",
        "Recording view: Fixed third-person",
        f"Azimuth: {details.get('azimuth_deg', 225.0):.1f} degrees",
        f"Elevation: {details.get('elevation_deg', 30.0):.1f} degrees",
    ]
    for line in text_lines:
        wrapped = textwrap.wrap(line, width=max(42, width // 15)) or [""]
        for part in wrapped:
            draw.text((x, y), part, font=body, fill=WHITE)
            y += int(height * 0.047)
    return np.asarray(image)


def render_summary_card(width, height, details):
    image = Image.new("RGB", (width, height), (13, 20, 31))
    draw = ImageDraw.Draw(image)
    title = _font(max(30, height // 17), bold=True)
    body = _font(max(18, height // 34))
    x = int(width * 0.08)
    y = int(height * 0.13)
    draw.text((x, y), "Episode Result", font=title, fill=WHITE)
    y += int(height * 0.12)
    fields = [
        ("Task", details.get("task", "")),
        ("Success", "YES" if details.get("success") else "NO"),
        ("Steps executed", details.get("steps", 0)),
        ("Final reward", details.get("reward", 0.0)),
        ("Termination", details.get("termination", "unknown")),
        ("Total Codex latency", f"{details.get('total_latency_seconds', 0.0):.2f} s"),
        ("Total input tokens", details.get("input_tokens", 0)),
        ("Total output tokens", details.get("output_tokens", 0)),
        ("Total reasoning tokens", details.get("reasoning_tokens", 0)),
    ]
    for label, value in fields:
        draw.text((x, y), f"{label}: {value}", font=body, fill=WHITE)
        y += int(height * 0.075)
    return np.asarray(image)
