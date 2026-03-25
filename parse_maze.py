#!/usr/bin/env python3
"""
Parse a Gazebo SDF world file to extract wall positions from the
'rbtc_class_labyrinth' model (and optionally the 'rbtc_class_wall' arena
boundary) and render a text-based grid map of the maze.
"""

import xml.etree.ElementTree as ET
import math

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SDF_FILE = "src/robotics_subject/worlds/labyrinth.world"
ROBOT_SPAWN = (-1.0, 0.0)               # world x, y
RESOLUTION = 0.10                        # metres per cell

# Models to extract walls from:  (model_name, include_in_grid)
MODELS = [
    ("rbtc_class_labyrinth", True),   # maze walls -> included in grid
    ("rbtc_class_wall", False),       # outer arena boundary -> listed but NOT in grid
]

# ---------------------------------------------------------------------------
# Color classifier
# ---------------------------------------------------------------------------
def classify_color(r, g, b, a):
    """Return a human-readable color name from RGBA floats."""
    if r >= 0.9 and g < 0.1 and b < 0.1:
        return "red"
    if r < 0.1 and g >= 0.9 and b < 0.1:
        return "green"
    if r < 0.1 and g < 0.1 and b >= 0.9:
        return "blue"
    if r >= 0.9 and g >= 0.9 and b < 0.1:
        return "yellow"
    if r < 0.1 and g < 0.1 and b < 0.1:
        return "black"
    return f"({r},{g},{b},{a})"


def get_link_pose(link_el):
    """Return the <pose> that is a *direct child* of <link>, skipping any
    <pose> that lives inside <collision> or <visual>."""
    for child in link_el:
        if child.tag == "pose":
            return child
    return None

# ---------------------------------------------------------------------------
# Parse SDF
# ---------------------------------------------------------------------------
tree = ET.parse(SDF_FILE)
root = tree.getroot()

walls = []

for model_name, include_in_grid in MODELS:
    # Locate the <model> element (only inside <world>, not inside <state>)
    model = None
    for world in root.iter("world"):
        for m in world.findall("model"):
            if m.get("name") == model_name:
                model = m
                break
        if model is not None:
            break

    if model is None:
        print(f"[WARNING] Model '{model_name}' not found -- skipping")
        continue

    # Model pose
    model_pose_el = model.find("pose")
    if model_pose_el is not None:
        mp = list(map(float, model_pose_el.text.split()))
        model_ox, model_oy = mp[0], mp[1]
    else:
        model_ox, model_oy = 0.0, 0.0

    # Extract walls from each <link>
    for link in model.findall("link"):
        name = link.get("name", "unknown")

        # -- box size (from collision geometry) --
        box_size_el = link.find(".//collision/geometry/box/size")
        if box_size_el is None:
            continue
        size_parts = list(map(float, box_size_el.text.split()))
        width, depth, height = size_parts[0], size_parts[1], size_parts[2]

        # -- link pose (direct child <pose>) --
        link_pose_el = get_link_pose(link)
        if link_pose_el is None:
            continue
        pose_parts = list(map(float, link_pose_el.text.split()))
        lx, ly, lz = pose_parts[0], pose_parts[1], pose_parts[2]
        roll, pitch, yaw = pose_parts[3], pose_parts[4], pose_parts[5]

        # -- color (from visual/material/ambient) --
        ambient_el = link.find(".//visual/material/ambient")
        if ambient_el is not None:
            rgba = list(map(float, ambient_el.text.split()))
            color = classify_color(*rgba)
        else:
            color = "grey"

        # -- world coordinates (model offset + link local pose) --
        wx = lx + model_ox
        wy = ly + model_oy

        # -- wall endpoints --
        half_len = width / 2.0
        half_thick = depth / 2.0

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        x1 = wx + (-half_len) * cos_yaw
        y1 = wy + (-half_len) * sin_yaw
        x2 = wx + ( half_len) * cos_yaw
        y2 = wy + ( half_len) * sin_yaw

        walls.append({
            "model": model_name,
            "name": name,
            "width": width,
            "depth": depth,
            "height": height,
            "local_pose": (lx, ly, lz, roll, pitch, yaw),
            "world_center": (wx, wy),
            "yaw": yaw,
            "color": color,
            "endpoints": (x1, y1, x2, y2),
            "half_thick": half_thick,
            "include_in_grid": include_in_grid,
        })

# ---------------------------------------------------------------------------
# Print wall list
# ---------------------------------------------------------------------------
print("=" * 120)
print(f"{'Model':<24} {'Name':<14} {'Length':>6} {'Thick':>5} {'World Center':>20} "
      f"{'Yaw':>8} {'Start (x,y)':>20} {'End (x,y)':>20} {'Color':<8}")
print("=" * 120)
for w in sorted(walls, key=lambda w: (w["model"], w["name"])):
    cx, cy = w["world_center"]
    x1, y1, x2, y2 = w["endpoints"]
    print(f"{w['model']:<24} {w['name']:<14} {w['width']:6.2f} {w['depth']:5.2f} "
          f"({cx:8.3f}, {cy:8.3f}) {w['yaw']:8.4f} "
          f"({x1:8.3f}, {y1:8.3f}) ({x2:8.3f}, {y2:8.3f}) {w['color']:<8}")
print(f"\nTotal walls: {len(walls)}")

# ---------------------------------------------------------------------------
# Build grid map
# ---------------------------------------------------------------------------
grid_walls = [w for w in walls if w["include_in_grid"]]

all_x = []
all_y = []
for w in grid_walls:
    x1, y1, x2, y2 = w["endpoints"]
    all_x.extend([x1, x2])
    all_y.extend([y1, y2])

# Add robot spawn to bounds
all_x.append(ROBOT_SPAWN[0])
all_y.append(ROBOT_SPAWN[1])

margin = 0.5
min_x = min(all_x) - margin
max_x = max(all_x) + margin
min_y = min(all_y) - margin
max_y = max(all_y) + margin

cols = int(math.ceil((max_x - min_x) / RESOLUTION)) + 1
rows = int(math.ceil((max_y - min_y) / RESOLUTION)) + 1

grid = [[' ' for _ in range(cols)] for _ in range(rows)]


def world_to_grid(wx, wy):
    col = int(round((wx - min_x) / RESOLUTION))
    row = int(round((max_y - wy) / RESOLUTION))
    return row, col


def draw_wall_on_grid(wall):
    x1, y1, x2, y2 = wall["endpoints"]
    half_t = wall["half_thick"]

    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return
    steps = max(int(length / (RESOLUTION * 0.5)), 2)

    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    px, py = -dy, dx

    thick_steps = max(int(half_t / (RESOLUTION * 0.5)), 1)

    for i in range(steps + 1):
        t = i / steps
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)
        for j in range(-thick_steps, thick_steps + 1):
            tt = j * (half_t / thick_steps) if thick_steps > 0 else 0
            wx_ = cx + tt * px
            wy_ = cy + tt * py
            r, c = world_to_grid(wx_, wy_)
            if 0 <= r < rows and 0 <= c < cols:
                grid[r][c] = '#'


for w in grid_walls:
    draw_wall_on_grid(w)

# Mark robot spawn
rr, rc = world_to_grid(ROBOT_SPAWN[0], ROBOT_SPAWN[1])
if 0 <= rr < rows and 0 <= rc < cols:
    grid[rr][rc] = 'R'

# ---------------------------------------------------------------------------
# Print the grid
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("MAZE MAP  (resolution = {:.2f} m/cell,  '#' = wall,  'R' = robot spawn)".format(RESOLUTION))
print("=" * 80)
for row in grid:
    print(''.join(row))

print(f"\nGrid size: {cols} cols x {rows} rows")
print(f"World bounds: x=[{min_x:.2f}, {max_x:.2f}]  y=[{min_y:.2f}, {max_y:.2f}]")
print(f"Robot spawn world coords: {ROBOT_SPAWN}")
