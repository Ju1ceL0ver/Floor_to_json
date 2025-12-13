import cv2
import numpy as np
import math
import json
import matplotlib.pyplot as plt

BACKGROUND = 0
WALL = 1
WINDOW = 2
DOOR = 3


def clean_mask(mask, k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def seg_len(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def snap_ortho(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return [p2[0], p1[1]] if abs(dx) >= abs(dy) else [p1[0], p2[1]]


def merge_colinear(segments, coord_tol=2, gap=4):
    merged = []
    for orient in ("h", "v"):
        segs = [s for s in segments if s[2] == orient]
        if not segs:
            continue
        if orient == "h":
            segs = [([min(a[0], b[0]), a[1]], [max(a[0], b[0]), a[1]]) for a, b, _ in segs]
            segs.sort(key=lambda s: (s[0][1], s[0][0]))
            collapsed = []
            i = 0
            while i < len(segs):
                base_y = segs[i][0][1]
                group = [segs[i]]
                j = i + 1
                while j < len(segs) and abs(segs[j][0][1] - base_y) <= coord_tol:
                    group.append(segs[j])
                    j += 1
                y_new = int(round(sum(s[0][1] for s in group) / len(group)))
                for s in group:
                    collapsed.append(([s[0][0], y_new], [s[1][0], y_new]))
                i = j
            segs = collapsed
            segs.sort(key=lambda s: (s[0][1], s[0][0]))
            i = 0
            while i < len(segs):
                y = segs[i][0][1]
                x1, x2 = segs[i][0][0], segs[i][1][0]
                j = i + 1
                while j < len(segs) and abs(segs[j][0][1] - y) <= coord_tol:
                    nx1, nx2 = segs[j][0][0], segs[j][1][0]
                    if nx1 - x2 <= gap:
                        x2 = max(x2, nx2)
                        j += 1
                    else:
                        break
                merged.append(([x1, y], [x2, y], orient))
                i = j
        else:
            segs = [([a[0], min(a[1], b[1])], [a[0], max(a[1], b[1])]) for a, b, _ in segs]
            segs.sort(key=lambda s: (s[0][0], s[0][1]))
            collapsed = []
            i = 0
            while i < len(segs):
                base_x = segs[i][0][0]
                group = [segs[i]]
                j = i + 1
                while j < len(segs) and abs(segs[j][0][0] - base_x) <= coord_tol:
                    group.append(segs[j])
                    j += 1
                x_new = int(round(sum(s[0][0] for s in group) / len(group)))
                for s in group:
                    collapsed.append(([x_new, s[0][1]], [x_new, s[1][1]]))
                i = j
            segs = collapsed
            segs.sort(key=lambda s: (s[0][0], s[0][1]))
            i = 0
            while i < len(segs):
                x = segs[i][0][0]
                y1, y2 = segs[i][0][1], segs[i][1][1]
                j = i + 1
                while j < len(segs) and abs(segs[j][0][0] - x) <= coord_tol:
                    ny1, ny2 = segs[j][0][1], segs[j][1][1]
                    if ny1 - y2 <= gap:
                        y2 = max(y2, ny2)
                        j += 1
                    else:
                        break
                merged.append(([x, y1], [x, y2], orient))
                i = j
    return merged


def bridge_segments(segments, bridge_gap=8, coord_tol=2):
    bridged = []
    for orient in ("h", "v"):
        segs = [s for s in segments if s[2] == orient]
        if not segs:
            continue
        segs.sort(key=lambda s: (s[0][1], s[0][0]) if orient == "h" else (s[0][0], s[0][1]))
        i = 0
        while i < len(segs):
            p1, p2, _ = segs[i]
            x1, y1 = p1
            x2, y2 = p2
            j = i + 1
            while j < len(segs):
                n1, n2, _ = segs[j]
                if orient == "h":
                    if abs(n1[1] - y1) > coord_tol or n1[0] - x2 > bridge_gap:
                        break
                    x2 = max(x2, n2[0])
                else:
                    if abs(n1[0] - x1) > coord_tol or n1[1] - y2 > bridge_gap:
                        break
                    y2 = max(y2, n2[1])
                j += 1
            bridged.append(([x1, y1], [x2, y2], orient))
            i = j
    return bridged


def reduce_segments(segments, coord_tol=2, max_keep=None):
    reduced = []
    for orient in ("h", "v"):
        segs = [s for s in segments if s[2] == orient]
        if not segs:
            continue
        key = (lambda s: s[0][1]) if orient == "h" else (lambda s: s[0][0])
        segs.sort(key=key)
        i = 0
        while i < len(segs):
            base = key(segs[i])
            group = [segs[i]]
            j = i + 1
            while j < len(segs) and abs(key(segs[j]) - base) <= coord_tol:
                group.append(segs[j])
                j += 1
            group_sorted = sorted(group, key=lambda s: seg_len(s[0], s[1]), reverse=True)
            keep = group_sorted if max_keep is None else group_sorted[:max_keep]
            reduced.extend(keep)
            i = j
    return reduced


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _project_to_wall(center, walls):
    cx, cy = center
    best = None
    for w in walls:
        (x1, y1), (x2, y2) = w["points"]
        if x1 == x2:
            px = x1
            py = _clamp(cy, min(y1, y2), max(y1, y2))
        else:
            py = y1
            px = _clamp(cx, min(x1, x2), max(x1, x2))
        dist = math.hypot(px - cx, py - cy)
        if best is None or dist < best[2]:
            best = (w, (px, py), dist)
    return best


def _grow_along_wall_mask(center, wall, wall_mask):
    h, w = wall_mask.shape
    cx, cy = center
    (wx1, wy1), (wx2, wy2) = wall["points"]
    if wx1 == wx2:
        x = _clamp(wx1, 0, w - 1)
        cy = _clamp(cy, 0, h - 1)
        while cy > 0 and wall_mask[cy - 1, x] > 0:
            cy -= 1
        y_min = cy
        cy = center[1]
        cy = _clamp(cy, 0, h - 1)
        while cy + 1 < h and wall_mask[cy + 1, x] > 0:
            cy += 1
        y_max = cy
        return y_min, y_max
    else:
        y = _clamp(wy1, 0, h - 1)
        cx = _clamp(cx, 0, w - 1)
        while cx > 0 and wall_mask[y, cx - 1] > 0:
            cx -= 1
        x_min = cx
        cx = center[0]
        cx = _clamp(cx, 0, w - 1)
        while cx + 1 < w and wall_mask[y, cx + 1] > 0:
            cx += 1
        x_max = cx
        return x_min, x_max


def _ray_to_wall(center, wall_mask, dx, dy):
    h, w = wall_mask.shape
    x, y = center
    last_free = [x, y]
    while True:
        x += dx
        y += dy
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        if wall_mask[y, x] > 0:
            return last_free
        last_free = [x, y]


def extract_openings(opening_mask, walls, wall_mask, prefix="o", min_area=4, min_len=4, mode="ray"):
    if opening_mask.max() == 0 or not walls:
        return []
    opening_mask = cv2.dilate(opening_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    cnts, _ = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = []
    oid = 1
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        wall_data = _project_to_wall((cx, cy), walls)
        if wall_data is None:
            continue
        wall, proj, _ = wall_data
        (wx1, wy1), (wx2, wy2) = wall["points"]
        if mode == "perp":
            if wx1 == wx2:
                x_min, x_max = _grow_along_wall_mask((cx, cy), wall, wall_mask)
                if (x_max - x_min) < min_len:
                    mid = cx
                    x_min = _clamp(mid - min_len // 2, 0, wall_mask.shape[1] - 1)
                    x_max = _clamp(x_min + min_len, 0, wall_mask.shape[1] - 1)
                p1 = [x_min, cy]
                p2 = [x_max, cy]
            else:
                y_min, y_max = _grow_along_wall_mask((cx, cy), wall, wall_mask)
                if (y_max - y_min) < min_len:
                    mid = cy
                    y_min = _clamp(mid - min_len // 2, 0, wall_mask.shape[0] - 1)
                    y_max = _clamp(y_min + min_len, 0, wall_mask.shape[0] - 1)
                p1 = [cx, y_min]
                p2 = [cx, y_max]
        elif mode == "ray":
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
            best = None
            for dx, dy in dirs:
                p1 = _ray_to_wall((cx, cy), wall_mask, dx, dy)
                p2 = _ray_to_wall((cx, cy), wall_mask, -dx, -dy)
                if p1 is None or p2 is None:
                    continue
                length = seg_len(p1, p2)
                if length < min_len:
                    continue
                if best is None or length < best[2]:
                    best = (p1, p2, length)
            if best is None:
                continue
            p1, p2, _ = best
        else:
            if wx1 == wx2:
                y_min, y_max = _grow_along_wall_mask((cx, cy), wall, wall_mask)
                if (y_max - y_min) < min_len:
                    mid = cy
                    y_min = _clamp(mid - min_len // 2, 0, wall_mask.shape[0] - 1)
                    y_max = _clamp(y_min + min_len, 0, wall_mask.shape[0] - 1)
                p1 = [wx1, y_min]
                p2 = [wx1, y_max]
            else:
                x_min, x_max = _grow_along_wall_mask((cx, cy), wall, wall_mask)
                if (x_max - x_min) < min_len:
                    mid = cx
                    x_min = _clamp(mid - min_len // 2, 0, wall_mask.shape[1] - 1)
                    x_max = _clamp(x_min + min_len, 0, wall_mask.shape[1] - 1)
                p1 = [x_min, wy1]
                p2 = [x_max, wy1]
        if seg_len(p1, p2) < 2:
            continue
        items.append({"id": f"{prefix}{oid}", "points": [p1, p2]})
        oid += 1
    return items


def _skeletonize(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    img = mask.copy()
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            done = True
    return skel


def extract_walls(mask, min_length=4, angle_tol=12, coord_tol=2, gap=8, hough_thresh=8):
    mask = clean_mask(mask)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    skel = _skeletonize(mask)
    lines = cv2.HoughLinesP(skel, rho=1, theta=np.pi / 180, threshold=hough_thresh, minLineLength=min_length, maxLineGap=gap)
    if lines is None:
        return []
    segments = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = x2 - x1, y2 - y1
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if (angle > angle_tol) and (abs(angle - 90) > angle_tol):
            continue
        p2 = snap_ortho([int(x1), int(y1)], [int(x2), int(y2)])
        p1 = [int(x1), int(y1)]
        if seg_len(p1, p2) < min_length:
            continue
        if p1[0] == p2[0]:
            orient = "v"
            if p1[1] > p2[1]:
                p1, p2 = p2, p1
        else:
            orient = "h"
            if p1[0] > p2[0]:
                p1, p2 = p2, p1
        segments.append((p1, p2, orient))
    merged = merge_colinear(segments, coord_tol=coord_tol, gap=gap)
    merged = bridge_segments(merged, bridge_gap=8, coord_tol=coord_tol)
    merged = reduce_segments(merged, coord_tol=coord_tol, max_keep=None)
    walls = []
    for i, (p1, p2, _) in enumerate(merged, 1):
        walls.append({"id": f"w{i}", "points": [p1, p2]})
    return walls


def find_gap_doors(walls, opening_mask, min_gap=3, tol=2, prefix="d"):
    h, w = opening_mask.shape
    doors = []
    did = 1
    vert = {}
    horiz = {}
    for wseg in walls:
        (x1, y1), (x2, y2) = wseg["points"]
        if x1 == x2:
            x = x1
            y_start, y_end = sorted([y1, y2])
            vert.setdefault(x, []).append((y_start, y_end))
        elif y1 == y2:
            y = y1
            x_start, x_end = sorted([x1, x2])
            horiz.setdefault(y, []).append((x_start, x_end))
    for x, segs in vert.items():
        segs.sort()
        for i in range(len(segs) - 1):
            y_end = segs[i][1]
            y_next_start = segs[i + 1][0]
            gap = y_next_start - y_end - 1
            if gap < min_gap:
                continue
            y0 = y_end + 1
            y1m = y_next_start - 1
            x0 = max(0, x - tol)
            x1 = min(w - 1, x + tol)
            region = opening_mask[y0:y1m + 1, x0:x1 + 1]
            if region.size == 0 or region.max() == 0:
                continue
            doors.append({"id": f"{prefix}{did}", "points": [[x, y0], [x, y1m]]})
            did += 1
    for y, segs in horiz.items():
        segs.sort()
        for i in range(len(segs) - 1):
            x_end = segs[i][1]
            x_next_start = segs[i + 1][0]
            gap = x_next_start - x_end - 1
            if gap < min_gap:
                continue
            x0 = x_end + 1
            x1 = x_next_start - 1
            y0 = max(0, y - tol)
            y1m = min(h - 1, y + tol)
            region = opening_mask[y0:y1m + 1, x0:x1 + 1]
            if region.size == 0 or region.max() == 0:
                continue
            doors.append({"id": f"{prefix}{did}", "points": [[x0, y], [x1, y]]})
            did += 1
    return doors


def build_json(pred_mask, image_name="image.png"):
    wall_mask = (pred_mask == WALL).astype(np.uint8)
    walls = extract_walls(wall_mask)
    opening_mask = (pred_mask == DOOR).astype(np.uint8)
    doors = find_gap_doors(walls, opening_mask, min_gap=1, tol=3, prefix="d")
    if not doors:
        doors = extract_openings(opening_mask, walls, wall_mask, prefix="d", mode="ray")
    windows = []
    return {"meta": {"source": image_name}, "walls": walls, "doors": doors, "windows": windows}


def visualize_pipeline(image, gt_mask, pred_mask, json_data):
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    for w in json_data["walls"]:
        p1, p2 = w["points"]
        cv2.line(vis, tuple(p1), tuple(p2), (255, 0, 0), 2)
    for d in json_data.get("doors", []):
        p1, p2 = d["points"]
        cv2.line(vis, tuple(p1), tuple(p2), (0, 0, 255), 3)
    for win in json_data.get("windows", []):
        p1, p2 = win["points"]
        cv2.line(vis, tuple(p1), tuple(p2), (0, 255, 0), 3)
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(1, 4, 2)
    plt.title("GT")
    plt.imshow(gt_mask)
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.title("Pred")
    plt.imshow(pred_mask)
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.title("Postprocess")
    plt.imshow(vis)
    plt.axis("off")
    plt.show()


def visualize_batch(samples, save_path=None):
    rows = len(samples)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    if rows == 1:
        axes = np.array([axes])
    for i, (image, gt_mask, pred_mask, json_data) in enumerate(samples):
        if image.ndim == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image
        vis = img_rgb.copy()
        for w in json_data["walls"]:
            p1, p2 = w["points"]
            cv2.line(vis, tuple(p1), tuple(p2), (255, 0, 0), 2)
        for d in json_data.get("doors", []):
            p1, p2 = d["points"]
            cv2.line(vis, tuple(p1), tuple(p2), (0, 0, 255), 3)
        for win in json_data.get("windows", []):
            p1, p2 = win["points"]
            cv2.line(vis, tuple(p1), tuple(p2), (0, 255, 0), 3)
        axes[i, 0].imshow(img_rgb, cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(gt_mask)
        axes[i, 1].axis("off")
        axes[i, 1].set_title("GT")
        axes[i, 2].imshow(pred_mask)
        axes[i, 2].axis("off")
        axes[i, 2].set_title("Pred")
        axes[i, 3].imshow(vis)
        axes[i, 3].axis("off")
        axes[i, 3].set_title("Postprocess")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
