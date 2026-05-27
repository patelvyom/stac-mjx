"""Post-run diagnostics for STAC outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stac_mjx import io

_SEVERITY_RANK = {"": 0, "info": 1, "warn": 2, "critical": 3}
_FRAME_COLUMNS = [
    "frame",
    "marker_rmse_mm",
    "worst_kp_name",
    "worst_kp_residual_mm",
    "root_pos_step_mm",
    "root_geodesic_step_deg",
    "root_quat_dot_raw",
    "qpos_step_rms",
    "qpos_step_max_abs",
    "qpos_step_max_abs_name",
    "qvel_max_abs",
    "qvel_max_abs_name",
    "is_chunk_boundary",
    "event_severity",
    "event_metrics",
]
_KEYPOINT_COLUMNS = [
    "kp_name",
    "mean_residual_mm",
    "p95_residual_mm",
    "max_residual_mm",
    "max_frame",
]
_EVENT_COLUMNS = [
    "metric",
    "severity",
    "start_frame",
    "end_frame",
    "peak_frame",
    "peak_value",
    "threshold",
    "association",
]


def diagnostics_prefix_for_data_path(data_path: str | Path) -> Path:
    """Return the default diagnostics file prefix for an H5 output path."""
    data_path = Path(data_path)
    return data_path.with_name(f"{data_path.stem}_diagnostics")


def compute_stac_diagnostics(
    data: io.StacData,
    cfg: Any,
    source_h5: str | Path | None = None,
    output_prefix: str | Path | None = None,
    write: bool = False,
) -> dict[str, Any]:
    """Compute post-run diagnostics, optionally writing files to disk."""
    qpos = np.asarray(data.qpos)
    marker_sites = np.asarray(data.marker_sites)
    kp_data = np.asarray(data.kp_data)
    n_frames = min(int(qpos.shape[0]), int(marker_sites.shape[0]), int(kp_data.shape[0]))
    if n_frames <= 0:
        raise ValueError("Diagnostics require at least one frame")
    qpos = qpos[:n_frames]
    marker_sites = marker_sites[:n_frames]
    n_keypoints = int(marker_sites.shape[1])
    kp_xyz = kp_data[:n_frames].reshape(n_frames, n_keypoints, 3)
    kp_names = list(getattr(data, "kp_names", [])) or [
        f"kp_{i}" for i in range(n_keypoints)
    ]
    qpos_names = list(getattr(data, "names_qpos", []))
    if len(qpos_names) != qpos.shape[1]:
        qpos_names = [f"qpos_{i}" for i in range(qpos.shape[1])]

    residual = kp_xyz - marker_sites
    keypoint_residual_mm = np.linalg.norm(residual, axis=2) * 1000.0
    marker_rmse_mm = np.sqrt(np.mean(np.square(residual), axis=(1, 2))) * 1000.0
    worst_kp_idx = np.argmax(keypoint_residual_mm, axis=1)
    worst_kp_name = np.array([kp_names[i] for i in worst_kp_idx], dtype=object)
    worst_kp_residual_mm = keypoint_residual_mm[np.arange(n_frames), worst_kp_idx]

    root_pos_step_mm = np.full(n_frames, np.nan)
    root_geodesic_step_deg = np.full(n_frames, np.nan)
    root_quat_dot_raw = np.full(n_frames, np.nan)
    if qpos.shape[1] >= 7 and n_frames > 1:
        root_pos_step_mm[:-1] = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1)
        root_pos_step_mm[:-1] *= 1000.0
        quat = qpos[:, 3:7]
        quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = quat / np.where(quat_norm > 0.0, quat_norm, 1.0)
        dot_raw = np.sum(quat[:-1] * quat[1:], axis=1)
        root_quat_dot_raw[:-1] = dot_raw
        root_geodesic_step_deg[:-1] = np.degrees(
            2.0 * np.arccos(np.clip(np.abs(dot_raw), -1.0, 1.0))
        )

    qpos_step_rms = np.full(n_frames, np.nan)
    qpos_step_max_abs = np.full(n_frames, np.nan)
    qpos_step_max_abs_name = np.array([""] * n_frames, dtype=object)
    if n_frames > 1:
        qpos_diff = np.diff(qpos, axis=0)
        qpos_step_rms[:-1] = np.sqrt(np.mean(np.square(qpos_diff), axis=1))
        qpos_abs = np.abs(qpos_diff)
        qpos_max_idx = np.argmax(qpos_abs, axis=1)
        qpos_step_max_abs[:-1] = qpos_abs[np.arange(n_frames - 1), qpos_max_idx]
        qpos_step_max_abs_name[:-1] = [qpos_names[i] for i in qpos_max_idx]

    qvel = np.asarray(getattr(data, "qvel", np.array([])))
    qvel_max_abs = np.full(n_frames, np.nan)
    qvel_max_abs_name = np.array([""] * n_frames, dtype=object)
    if qvel.ndim == 2 and qvel.shape[0] > 0 and qvel.shape[1] > 0:
        n_qvel_frames = min(n_frames, int(qvel.shape[0]))
        qvel_abs = np.abs(qvel[:n_qvel_frames])
        qvel_max_idx = np.argmax(qvel_abs, axis=1)
        qvel_max_abs[:n_qvel_frames] = qvel_abs[
            np.arange(n_qvel_frames), qvel_max_idx
        ]
        qvel_names = _qvel_names(qpos_names, int(qvel.shape[1]))
        qvel_max_abs_name[:n_qvel_frames] = [qvel_names[i] for i in qvel_max_idx]

    chunk_size = int(_get(_get(cfg, "stac"), "n_frames_per_clip", 0) or 0)
    is_chunk_boundary = np.zeros(n_frames, dtype=bool)
    if chunk_size > 0 and n_frames > 1:
        frame_idx = np.arange(n_frames)
        is_chunk_boundary = ((frame_idx + 1) % chunk_size == 0) & (
            frame_idx < n_frames - 1
        )

    thresholds = _thresholds_from_cfg(cfg)
    metric_values = {
        "marker_rmse_mm": marker_rmse_mm,
        "root_pos_step_mm": root_pos_step_mm,
        "root_geodesic_step_deg": root_geodesic_step_deg,
        "boundary_root_pos_step_mm": np.where(
            is_chunk_boundary, root_pos_step_mm, np.nan
        ),
        "boundary_root_geodesic_step_deg": np.where(
            is_chunk_boundary, root_geodesic_step_deg, np.nan
        ),
        "qpos_step_rms": qpos_step_rms,
        "qpos_step_max_abs": qpos_step_max_abs,
        "qvel_max_abs": qvel_max_abs,
    }
    metric_assoc = {
        "marker_rmse_mm": (worst_kp_name, worst_kp_residual_mm),
        "root_pos_step_mm": ("root", root_pos_step_mm),
        "root_geodesic_step_deg": ("root", root_geodesic_step_deg),
        "boundary_root_pos_step_mm": ("root", root_pos_step_mm),
        "boundary_root_geodesic_step_deg": ("root", root_geodesic_step_deg),
        "qpos_step_rms": (qpos_step_max_abs_name, qpos_step_max_abs),
        "qpos_step_max_abs": (qpos_step_max_abs_name, qpos_step_max_abs),
        "qvel_max_abs": (qvel_max_abs_name, qvel_max_abs),
    }

    threshold_events = []
    info_events = []
    diagnostics_cfg = _get(_get(cfg, "stac"), "diagnostics")
    top_n_events = int(_get(diagnostics_cfg, "top_n_events", 50) or 50)
    for metric, values in metric_values.items():
        warn, critical = thresholds.get(metric, (None, None))
        assoc_names, assoc_values = metric_assoc[metric]
        metric_threshold_events, metric_info_events = _metric_events(
            metric,
            values,
            warn,
            critical,
            top_n_events,
            assoc_names,
            assoc_values,
        )
        threshold_events.extend(metric_threshold_events)
        info_events.extend(metric_info_events)

    info_events = sorted(info_events, key=lambda e: (e["_rank"], e["metric"]))[
        :top_n_events
    ]
    for event in info_events:
        event.pop("_rank", None)
    events = threshold_events + info_events
    events = sorted(
        events,
        key=lambda e: (
            -_SEVERITY_RANK[e["severity"]],
            e["start_frame"],
            e["metric"],
        ),
    )

    event_severity = np.array([""] * n_frames, dtype=object)
    event_metrics = [set() for _ in range(n_frames)]
    for event in events:
        start = int(event["start_frame"])
        end = int(event["end_frame"])
        for frame in range(start, end + 1):
            if _SEVERITY_RANK[event["severity"]] > _SEVERITY_RANK[event_severity[frame]]:
                event_severity[frame] = event["severity"]
            event_metrics[frame].add(event["metric"])
    event_metric_text = np.array(
        [";".join(sorted(metrics)) for metrics in event_metrics], dtype=object
    )

    frame_metrics = {
        "frame": np.arange(n_frames, dtype=int),
        "marker_rmse_mm": marker_rmse_mm,
        "worst_kp_name": worst_kp_name,
        "worst_kp_residual_mm": worst_kp_residual_mm,
        "root_pos_step_mm": root_pos_step_mm,
        "root_geodesic_step_deg": root_geodesic_step_deg,
        "root_quat_dot_raw": root_quat_dot_raw,
        "qpos_step_rms": qpos_step_rms,
        "qpos_step_max_abs": qpos_step_max_abs,
        "qpos_step_max_abs_name": qpos_step_max_abs_name,
        "qvel_max_abs": qvel_max_abs,
        "qvel_max_abs_name": qvel_max_abs_name,
        "is_chunk_boundary": is_chunk_boundary,
        "event_severity": event_severity,
        "event_metrics": event_metric_text,
    }

    keypoint_metrics = []
    for i, name in enumerate(kp_names):
        values = keypoint_residual_mm[:, i]
        keypoint_metrics.append(
            {
                "kp_name": name,
                "mean_residual_mm": float(np.mean(values)),
                "p95_residual_mm": float(np.percentile(values, 95)),
                "max_residual_mm": float(np.max(values)),
                "max_frame": int(np.argmax(values)),
            }
        )
    keypoint_metrics = sorted(
        keypoint_metrics, key=lambda row: row["max_residual_mm"], reverse=True
    )

    summary = {
        "source_h5": "" if source_h5 is None else str(source_h5),
        "n_frames": n_frames,
        "n_keypoints": n_keypoints,
        "chunk_size": chunk_size,
        "thresholds": {
            metric: {"warn": warn, "critical": critical}
            for metric, (warn, critical) in thresholds.items()
        },
        "metrics": {
            "marker_rmse_mm": _metric_stats(marker_rmse_mm),
            "root_pos_step_mm": _metric_stats(root_pos_step_mm),
            "root_geodesic_step_deg": _metric_stats(root_geodesic_step_deg),
            "boundary_root_pos_step_mm": _metric_stats(
                metric_values["boundary_root_pos_step_mm"]
            ),
            "boundary_root_geodesic_step_deg": _metric_stats(
                metric_values["boundary_root_geodesic_step_deg"]
            ),
            "qpos_step_rms": _metric_stats(qpos_step_rms),
            "qpos_step_max_abs": _metric_stats(qpos_step_max_abs),
            "qvel_max_abs": _metric_stats(qvel_max_abs),
            "root_quat_dot_raw": _root_quat_stats(root_quat_dot_raw),
        },
        "events": {
            "count": len(events),
            "info_count": sum(event["severity"] == "info" for event in events),
            "warn_count": sum(event["severity"] == "warn" for event in events),
            "critical_count": sum(event["severity"] == "critical" for event in events),
            "top": events[:10],
        },
        "worst_keypoints": keypoint_metrics[:10],
    }

    diagnostics = {
        "summary": summary,
        "frame_metrics": frame_metrics,
        "keypoint_metrics": keypoint_metrics,
        "events": events,
        "metadata": {"frame_columns": _FRAME_COLUMNS},
    }
    if write:
        prefix = output_prefix
        if prefix is None:
            if source_h5 is None:
                raise ValueError("source_h5 or output_prefix is required when write=True")
            prefix = diagnostics_prefix_for_data_path(source_h5)
        diagnostics["output_paths"] = write_stac_diagnostics(diagnostics, prefix)
    return diagnostics


def write_stac_diagnostics(
    diagnostics: dict[str, Any],
    output_prefix: str | Path,
) -> dict[str, str]:
    """Write summary, frame metrics, keypoint metrics, and events files."""
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_prefix.with_name(f"{output_prefix.name}_summary.json"),
        "frame_metrics": output_prefix.with_name(
            f"{output_prefix.name}_frame_metrics.csv"
        ),
        "keypoint_metrics": output_prefix.with_name(
            f"{output_prefix.name}_keypoint_metrics.csv"
        ),
        "events": output_prefix.with_name(f"{output_prefix.name}_events.csv"),
    }
    with paths["summary"].open("w") as f:
        json.dump(_json_ready(diagnostics["summary"]), f, indent=2)
    _write_frame_metrics(paths["frame_metrics"], diagnostics["frame_metrics"])
    _write_rows(paths["keypoint_metrics"], diagnostics["keypoint_metrics"], _KEYPOINT_COLUMNS)
    _write_rows(paths["events"], diagnostics["events"], _EVENT_COLUMNS)
    return {key: str(path) for key, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for post-run diagnostics."""
    parser = argparse.ArgumentParser(description="Compute STAC-MJX output diagnostics.")
    parser.add_argument("data_path", help="Path to a STAC output HDF5 file.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="File prefix for diagnostics outputs. Defaults beside the H5 file.",
    )
    args = parser.parse_args(argv)

    cfg, data = io.load_stac_data(args.data_path)
    diagnostics = compute_stac_diagnostics(
        data,
        cfg,
        source_h5=args.data_path,
        output_prefix=args.output_prefix,
        write=True,
    )
    for path in diagnostics["output_paths"].values():
        print(path)
    return 0


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _thresholds_from_cfg(cfg: Any) -> dict[str, tuple[float | None, float | None]]:
    defaults = {
        "marker_rmse_mm": (10.0, 25.0),
        "root_pos_step_mm": (6.0, 15.0),
        "root_geodesic_step_deg": (4.0, 12.0),
        "boundary_root_pos_step_mm": (8.0, 15.0),
        "boundary_root_geodesic_step_deg": (6.0, 12.0),
        "qpos_step_rms": (None, None),
        "qpos_step_max_abs": (None, None),
        "qvel_max_abs": (None, None),
    }
    threshold_cfg = _get(_get(_get(cfg, "stac"), "diagnostics"), "thresholds")
    thresholds = {}
    for metric, default in defaults.items():
        limit = _get(threshold_cfg, metric)
        warn = _none_or_float(_get(limit, "warn", default[0]))
        critical = _none_or_float(_get(limit, "critical", default[1]))
        thresholds[metric] = (warn, critical)
    return thresholds


def _none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _qvel_names(qpos_names: list[str], n_qvel: int) -> list[str]:
    if len(qpos_names) == n_qvel:
        return qpos_names
    if len(qpos_names) >= 7 and n_qvel == len(qpos_names) - 1:
        return ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"] + qpos_names[7:]
    return [f"qvel_{i}" for i in range(n_qvel)]


def _metric_stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "max_frame": None,
        }
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "max_frame": _nan_arg(values, np.nanargmax),
    }


def _root_quat_stats(values: np.ndarray) -> dict[str, float | int | None]:
    if not np.any(np.isfinite(values)):
        return {"min": None, "min_frame": None, "negative_count": 0}
    return {
        "min": float(np.nanmin(values)),
        "min_frame": int(np.nanargmin(values)),
        "negative_count": int(np.sum(values < 0.0)),
    }


def _metric_events(
    metric: str,
    values: np.ndarray,
    warn: float | None,
    critical: float | None,
    top_n_events: int,
    assoc_names: Any,
    assoc_values: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finite = np.isfinite(values)
    threshold_mask = np.zeros(values.shape, dtype=bool)
    threshold_events = []
    if warn is not None:
        threshold_mask = finite & (values >= warn)
        threshold_events = _threshold_event_windows(
            metric, values, threshold_mask, warn, critical, assoc_names, assoc_values
        )

    top_idx = np.flatnonzero(finite & ~threshold_mask)
    if top_idx.size == 0:
        return threshold_events, []
    top_idx = top_idx[np.argsort(values[top_idx])[::-1][:top_n_events]]
    info_mask = np.zeros(values.shape, dtype=bool)
    info_mask[top_idx] = True
    info_events = _event_windows(
        metric, values, info_mask, "info", None, assoc_names, assoc_values
    )
    peak_order = {int(frame): rank for rank, frame in enumerate(top_idx)}
    for event in info_events:
        event["_rank"] = peak_order.get(int(event["peak_frame"]), top_n_events)
    return threshold_events, info_events


def _threshold_event_windows(
    metric: str,
    values: np.ndarray,
    mask: np.ndarray,
    warn: float,
    critical: float | None,
    assoc_names: Any,
    assoc_values: np.ndarray,
) -> list[dict[str, Any]]:
    events = []
    for event in _event_windows(metric, values, mask, "warn", warn, assoc_names, assoc_values):
        if critical is not None and event["peak_value"] >= critical:
            event["severity"] = "critical"
            event["threshold"] = critical
        events.append(event)
    return events


def _event_windows(
    metric: str,
    values: np.ndarray,
    mask: np.ndarray,
    severity: str,
    threshold: float | None,
    assoc_names: Any,
    assoc_values: np.ndarray,
) -> list[dict[str, Any]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    events = []
    for group in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        peak_frame = int(group[np.nanargmax(values[group])])
        association = assoc_names
        if not isinstance(assoc_names, str):
            association = assoc_names[peak_frame]
        assoc_value = assoc_values[peak_frame]
        if np.isfinite(assoc_value):
            association = f"{association}:{float(assoc_value):.6g}"
        events.append(
            {
                "metric": metric,
                "severity": severity,
                "start_frame": int(group[0]),
                "end_frame": int(group[-1]),
                "peak_frame": peak_frame,
                "peak_value": float(values[peak_frame]),
                "threshold": threshold,
                "association": str(association),
            }
        )
    return events


def _nan_arg(values: np.ndarray, arg_fn: Any) -> int | None:
    if not np.any(np.isfinite(values)):
        return None
    return int(arg_fn(values))


def _write_frame_metrics(path: Path, frame_metrics: dict[str, np.ndarray]) -> None:
    n_frames = len(frame_metrics["frame"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FRAME_COLUMNS)
        writer.writeheader()
        for i in range(n_frames):
            writer.writerow(
                {column: _csv_value(frame_metrics[column][i]) for column in _FRAME_COLUMNS}
            )


def _write_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
