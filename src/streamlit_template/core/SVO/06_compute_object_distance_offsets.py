# Purpose: Compute per-frame 3D pairwise offsets and distances between detected objects.
# Notes: Uses RGB + depth with camera intrinsics and writes CSV/JSON outputs.

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from _paper_config import DEPTH_PATH, OBJECT_MODEL_PATH, OBJECTS_PATH, RGB_PATH, ensure_output_dirs, intrinsics_path
from _paper_utils import load_intrinsics, median_valid_depth, to_camera_xyz

OBJECT_CONF_THRESHOLD = float(os.environ.get('VILMA_OBJECT_CONF', '0.8'))
ANCHOR_CLASS = os.environ.get('VILMA_ANCHOR_CLASS', 'shaft').strip().lower()


def _extract_detections(result) -> list[dict]:
    """Extract detection centers/classes/conf from OBB or axis-aligned boxes."""
    detections = []

    if result.obb is not None and len(result.obb) > 0:
        centers = result.obb.xywhr.detach().cpu().numpy()[:, :2]
        confs = result.obb.conf.detach().cpu().numpy()
        classes = result.obb.cls.detach().cpu().numpy().astype(int)
        for i, (uv, conf, cls_id) in enumerate(zip(centers, confs, classes)):
            detections.append(
                {
                    'det_id': i,
                    'u': float(uv[0]),
                    'v': float(uv[1]),
                    'conf': float(conf),
                    'cls_id': int(cls_id),
                }
            )
        return detections

    if result.boxes is not None and len(result.boxes) > 0:
        centers = result.boxes.xywh.detach().cpu().numpy()[:, :2]
        confs = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for i, (uv, conf, cls_id) in enumerate(zip(centers, confs, classes)):
            detections.append(
                {
                    'det_id': i,
                    'u': float(uv[0]),
                    'v': float(uv[1]),
                    'conf': float(conf),
                    'cls_id': int(cls_id),
                }
            )
    return detections


def _label_for(model: YOLO, cls_id: int) -> str:
    names = getattr(model, 'names', None)
    if isinstance(names, dict):
        return str(names.get(cls_id, f'class_{cls_id}'))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return f'class_{cls_id}'


def main() -> None:
    """Main entry point for object-object distance offset computation."""
    ensure_output_dirs()

    intr = load_intrinsics(intrinsics_path())
    model = YOLO(str(OBJECT_MODEL_PATH))

    rgb_files = sorted(
        p.name for p in RGB_PATH.iterdir()
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
    )
    depth_files = sorted(p.name for p in DEPTH_PATH.glob('*.npy'))
    if len(rgb_files) != len(depth_files):
        raise RuntimeError(f'RGB and depth counts do not match: {len(rgb_files)} RGB vs {len(depth_files)} depth.')

    rows = []
    shaft_rows = []
    summary_stats: dict[str, list[float]] = {}
    shaft_summary_stats: dict[str, list[float]] = {}

    for frame_idx, (rgb_name, depth_name) in enumerate(
        tqdm(list(zip(rgb_files, depth_files)), desc='Object offset computation')
    ):
        rgb_path = RGB_PATH / rgb_name
        depth_path = DEPTH_PATH / depth_name
        try:
            rgb = cv2.imread(str(rgb_path))
        except FileNotFoundError:
            continue
        if rgb is None:
            continue
        try:
            depth = np.load(depth_path)
        except FileNotFoundError:
            continue

        result = model(rgb, verbose=False, conf=OBJECT_CONF_THRESHOLD)[0]
        detections = _extract_detections(result)

        det_points = []
        for det in detections:
            u_i = int(round(det['u']))
            v_i = int(round(det['v']))
            z = median_valid_depth(depth, u=u_i, v=v_i, half=2)
            if z is None:
                continue
            xyz = to_camera_xyz(u_i, v_i, z, intr)
            det_points.append(
                {
                    'det_id': det['det_id'],
                    'cls_id': det['cls_id'],
                    'label': _label_for(model, det['cls_id']),
                    'conf': det['conf'],
                    'u': u_i,
                    'v': v_i,
                    'xyz': xyz,
                }
            )

        if len(det_points) < 2:
            continue

        for i in range(len(det_points) - 1):
            for j in range(i + 1, len(det_points)):
                a = det_points[i]
                b = det_points[j]
                delta = b['xyz'] - a['xyz']
                dist = float(np.linalg.norm(delta))

                rows.append(
                    {
                        'frame_idx': frame_idx,
                        'rgb_file': rgb_name,
                        'depth_file': depth_name,
                        'obj_a_id': a['det_id'],
                        'obj_a_label': a['label'],
                        'obj_a_conf': a['conf'],
                        'obj_a_u': a['u'],
                        'obj_a_v': a['v'],
                        'obj_b_id': b['det_id'],
                        'obj_b_label': b['label'],
                        'obj_b_conf': b['conf'],
                        'obj_b_u': b['u'],
                        'obj_b_v': b['v'],
                        'dx_m': float(delta[0]),
                        'dy_m': float(delta[1]),
                        'dz_m': float(delta[2]),
                        'distance_m': dist,
                    }
                )

                a_label_lower = a['label'].strip().lower()
                b_label_lower = b['label'].strip().lower()
                if a_label_lower == ANCHOR_CLASS and b_label_lower != ANCHOR_CLASS:
                    shaft_rows.append(
                        {
                            'frame_idx': frame_idx,
                            'rgb_file': rgb_name,
                            'depth_file': depth_name,
                            'anchor_label': a['label'],
                            'anchor_id': a['det_id'],
                            'anchor_conf': a['conf'],
                            'anchor_u': a['u'],
                            'anchor_v': a['v'],
                            'other_label': b['label'],
                            'other_id': b['det_id'],
                            'other_conf': b['conf'],
                            'other_u': b['u'],
                            'other_v': b['v'],
                            'dx_m': float(delta[0]),
                            'dy_m': float(delta[1]),
                            'dz_m': float(delta[2]),
                            'distance_m': dist,
                        }
                    )
                    shaft_summary_stats.setdefault(b['label'], []).append(dist)
                elif b_label_lower == ANCHOR_CLASS and a_label_lower != ANCHOR_CLASS:
                    inv_delta = a['xyz'] - b['xyz']
                    shaft_rows.append(
                        {
                            'frame_idx': frame_idx,
                            'rgb_file': rgb_name,
                            'depth_file': depth_name,
                            'anchor_label': b['label'],
                            'anchor_id': b['det_id'],
                            'anchor_conf': b['conf'],
                            'anchor_u': b['u'],
                            'anchor_v': b['v'],
                            'other_label': a['label'],
                            'other_id': a['det_id'],
                            'other_conf': a['conf'],
                            'other_u': a['u'],
                            'other_v': a['v'],
                            'dx_m': float(inv_delta[0]),
                            'dy_m': float(inv_delta[1]),
                            'dz_m': float(inv_delta[2]),
                            'distance_m': float(np.linalg.norm(inv_delta)),
                        }
                    )
                    shaft_summary_stats.setdefault(a['label'], []).append(float(np.linalg.norm(inv_delta)))

                key = ' | '.join(sorted([a['label'], b['label']]))
                summary_stats.setdefault(key, []).append(dist)

    csv_path = OBJECTS_PATH / 'object_pair_offsets_3d.csv'
    json_path = OBJECTS_PATH / 'object_pair_offsets_summary.json'
    shaft_csv_path = OBJECTS_PATH / 'shaft_to_other_offsets_3d.csv'
    shaft_json_path = OBJECTS_PATH / 'shaft_to_other_offsets_summary.json'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'frame_idx',
        'rgb_file',
        'depth_file',
        'obj_a_id',
        'obj_a_label',
        'obj_a_conf',
        'obj_a_u',
        'obj_a_v',
        'obj_b_id',
        'obj_b_label',
        'obj_b_conf',
        'obj_b_u',
        'obj_b_v',
        'dx_m',
        'dy_m',
        'dz_m',
        'distance_m',
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    shaft_fieldnames = [
        'frame_idx',
        'rgb_file',
        'depth_file',
        'anchor_label',
        'anchor_id',
        'anchor_conf',
        'anchor_u',
        'anchor_v',
        'other_label',
        'other_id',
        'other_conf',
        'other_u',
        'other_v',
        'dx_m',
        'dy_m',
        'dz_m',
        'distance_m',
    ]
    with open(shaft_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=shaft_fieldnames)
        writer.writeheader()
        for row in shaft_rows:
            writer.writerow(row)

    summary = {
        'confidence_threshold': OBJECT_CONF_THRESHOLD,
        'frames_total': len(rgb_files),
        'pairwise_rows': len(rows),
        'pairs': {
            key: {
                'count': len(vals),
                'mean_distance_m': float(np.mean(vals)),
                'min_distance_m': float(np.min(vals)),
                'max_distance_m': float(np.max(vals)),
            }
            for key, vals in summary_stats.items()
        },
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    shaft_summary = {
        'anchor_class': ANCHOR_CLASS,
        'confidence_threshold': OBJECT_CONF_THRESHOLD,
        'frames_total': len(rgb_files),
        'rows': len(shaft_rows),
        'others': {
            key: {
                'count': len(vals),
                'mean_distance_m': float(np.mean(vals)),
                'min_distance_m': float(np.min(vals)),
                'max_distance_m': float(np.max(vals)),
            }
            for key, vals in shaft_summary_stats.items()
        },
    }
    with open(shaft_json_path, 'w', encoding='utf-8') as f:
        json.dump(shaft_summary, f, indent=2)

    print('Object distance offsets complete.')
    print(f'Rows written: {len(rows)}')
    print(f'CSV: {csv_path}')
    print(f'Summary: {json_path}')
    print(f'Shaft rows written: {len(shaft_rows)}')
    print(f'Shaft CSV: {shaft_csv_path}')
    print(f'Shaft summary: {shaft_json_path}')


if __name__ == '__main__':
    main()