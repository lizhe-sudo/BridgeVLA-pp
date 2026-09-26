"""Video-only overlays for Codex-controlled RLBench episodes."""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


GREEN = (45, 220, 90)
RED = (245, 55, 55)
BLUE = (55, 130, 255)
WHITE = (240, 243, 248)
PANEL = (15, 20, 28)

POLICY_CAMERA_LABELS = {
    "front": "FRONT",
    "left_shoulder": "LEFT SHOULDER",
    "right_shoulder": "RIGHT SHOULDER",
    "wrist": "WRIST",
}


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


def _project(camera, point):
    if point is None:
        return None, False
    projected = camera.project_xyz(point)
    if projected is None:
        return None, False
    x, y, depth = projected
    inside = depth > 0 and 0 <= x < camera.width and 0 <= y < camera.height
    return (int(round(x)), int(round(y))), inside


def _marker(draw, camera, point, color, label, radius=8, font=None):
    xy, inside = _project(camera, point)
    if not inside:
        return False
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=color, outline=(255, 255, 255), width=2)
    draw.text((x + radius + 4, y - 11), label, font=font, fill=color,
              stroke_width=2, stroke_fill=(0, 0, 0))
    return True


def _line(draw, camera, start, end, color, width=3):
    p1, visible1 = _project(camera, start)
    p2, visible2 = _project(camera, end)
    if visible1 and visible2:
        draw.line((*p1, *p2), fill=color, width=width)


def render_policy_view(rgb, camera, camera_name, phase, step_index,
                       current_xyz=None, target_xyz=None, actual_xyz=None,
                       target_history=None, panel=None, failure=None):
    """Render one of the actual policy camera views with video-only overlays."""
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8).copy(), mode="RGB")
    if image.size != (camera.width, camera.height):
        image = image.resize((camera.width, camera.height), Image.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")
    marker_font = _font(max(13, camera.height // 40), bold=True)
    title_font = _font(max(17, camera.height // 30), bold=True)
    small_font = _font(max(12, camera.height // 48))
    target_history = target_history or []

    previous = None
    for waypoint in target_history[:-1]:
        xy, visible = _project(camera, waypoint)
        if visible:
            if previous is not None:
                draw.line((*previous, *xy), fill=(180, 75, 75, 220), width=2)
            draw.ellipse((xy[0] - 4, xy[1] - 4, xy[0] + 4, xy[1] + 4),
                         fill=(190, 75, 75, 230))
        previous = xy if visible else None

    if phase == "READY":
        pass
    elif phase == "AFTER EXECUTION":
        _line(draw, camera, actual_xyz, target_xyz, BLUE, width=3)
        _marker(draw, camera, actual_xyz, BLUE, "ACTUAL", font=marker_font)
        _marker(draw, camera, target_xyz, RED, "TARGET", radius=10,
                font=marker_font)
    else:
        _line(draw, camera, current_xyz, target_xyz, GREEN, width=3)
        _marker(draw, camera, current_xyz, GREEN, "CURRENT", font=marker_font)
        _marker(draw, camera, target_xyz, RED, "TARGET", radius=10,
                font=marker_font)
        if phase == "EXECUTION" and actual_xyz is not None:
            _marker(draw, camera, actual_xyz, GREEN, "EEF", radius=6,
                    font=marker_font)

    panel = dict(panel or {})
    label = POLICY_CAMERA_LABELS.get(camera_name, camera_name.upper())
    header = f"{label}  |  {phase}  |  STEP {step_index}"
    draw.rectangle((0, 0, camera.width, max(34, camera.height // 12)),
                   fill=(5, 10, 18, 190))
    draw.text((12, 6), header, font=title_font, fill=WHITE)

    status_parts = []
    for key, label in (("planner_returned", "PLAN"),
                       ("target_reached", "TARGET"),
                       ("reward", "REWARD")):
        value = panel.get(key)
        if value is not None:
            status_parts.append(f"{label}: {value}")
    if panel.get("position_error_m") not in (None, "N/A"):
        status_parts.append(f"ERROR: {panel['position_error_m']}")
    if status_parts:
        footer_text = "  |  ".join(status_parts)
        draw.rectangle((0, camera.height - 30, camera.width, camera.height),
                       fill=(5, 10, 18, 190))
        draw.text((12, camera.height - 27), footer_text[:100],
                  font=small_font, fill=WHITE)

    target_xy, target_visible = _project(camera, target_xyz)
    if target_xyz is not None and not target_visible:
        draw.rounded_rectangle((8, camera.height - 34, camera.width - 8,
                                camera.height - 8), radius=6,
                               fill=(35, 28, 8, 210))
        draw.text((14, camera.height - 31), "TARGET OFF-SCREEN",
                  font=small_font, fill=(255, 215, 90))

    if failure:
        draw.rectangle((0, camera.height - 40, camera.width, camera.height),
                       fill=(120, 20, 20, 225))
        draw.text((12, camera.height - 34),
                  f"EXECUTION FAILED: {str(failure)[:64]}",
                  font=small_font, fill=WHITE)
    return np.asarray(image)


def render_policy_grid(rgb_views, cameras, phase, step_index,
                       current_xyz=None, target_xyz=None, actual_xyz=None,
                       target_history=None, panel=None, failure=None):
    """Compose the four views sent to Luna into one 2x2 video frame."""
    names = ("front", "left_shoulder", "right_shoulder", "wrist")
    if tuple(rgb_views) != names or tuple(cameras) != names:
        raise ValueError("video views must match the four ordered policy cameras")
    tile_size = cameras["front"].width
    frame = Image.new("RGB", (tile_size * 2, tile_size * 2), PANEL)
    placements = {
        "front": (0, 0),
        "left_shoulder": (tile_size, 0),
        "right_shoulder": (0, tile_size),
        "wrist": (tile_size, tile_size),
    }
    for name in names:
        tile = render_policy_view(
            rgb_views[name], cameras[name], name, phase, step_index,
            current_xyz=current_xyz, target_xyz=target_xyz,
            actual_xyz=actual_xyz, target_history=target_history,
            panel=panel, failure=failure,
        )
        frame.paste(Image.fromarray(tile, mode="RGB"), placements[name])
    return np.asarray(frame)
