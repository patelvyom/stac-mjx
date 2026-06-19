"""Fit diagnostics and self-contained HTML report generation."""

from __future__ import annotations

import base64
import html
import io as pyio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

from stac_mjx import io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIT_DIAGNOSTICS_GROUP = "fit_diagnostics"
SCHEMA_VERSION = 2

ROBUST_Z_EVENT_THRESHOLD = 8.0
BOUNDARY_RATIO_EVENT_THRESHOLD = 3.0
EVENT_TAIL_PERCENTILE = 99.7
EPS = 1e-12

PLOT_COLORS = {
    "figure_bg": "#fcfcfd",
    "axes_bg": "#ffffff",
    "grid": "#e6e8f0",
    "spine": "#d7dbe7",
    "text": "#1f2430",
    "muted": "#6f768a",
    "threshold": "#736422",
    "event": "#f0986e",
    "fit": "#5477c4",
    "source": "#cc6f47",
    "motion": "#71b436",
    "root": "#bd569b",
    "boundary": "#b8a037",
    "p95_bar": "#a3befa",
    "max_bar": "#f0986e",
    "share_bar": "#a3d576",
    "hist_edge": "#ffffff",
    "boundary_interior": "#cedffe",
    "boundary_chunk": "#ffbda1",
}

STRING_COLUMNS = {
    "metric_name",
    "score_name",
    "status",
    "domain",
    "unit",
    "description",
    "interpretation",
    "research_basis",
    "event_category",
    "association",
    "focus",
    "keypoint_name",
    "marker_max_residual_name",
    "source_keypoint_max_interp_error_name",
    "fit_marker_max_interp_error_name",
    "qpos_step_max_abs_name",
    "title",
    "url",
    "applies_to",
}

BOOL_COLUMNS = {"is_chunk_boundary", "is_chunk_boundary_event"}

INT_COLUMNS = {
    "frame",
    "absolute_frame",
    "start_frame",
    "end_frame",
    "peak_frame",
    "absolute_start_frame",
    "absolute_end_frame",
    "absolute_peak_frame",
    "duration_frames",
    "count",
    "event_count",
    "fit_event_count",
    "source_event_count",
    "max_frame",
    "max_absolute_frame",
    "marker_max_residual_index",
    "source_keypoint_max_interp_error_index",
    "fit_marker_max_interp_error_index",
    "qpos_step_max_abs_index",
    "keypoint_index",
    "source_temporal_event_count",
    "fit_temporal_event_count",
    "negative_count",
    "min_frame",
}

METRIC_DEFINITIONS = [
    {
        "metric_name": "marker_rmse_mm",
        "unit": "mm",
        "domain": "fit_error",
        "description": "Per-frame RMS distance between observed keypoints and fitted marker sites.",
        "interpretation": "Primary fit metric; high values mean the model markers are not matching the keypoint observations.",
        "research_basis": "OpenSim IK RMS marker error; 3D pose localization error distributions.",
    },
    {
        "metric_name": "marker_mean_residual_mm",
        "unit": "mm",
        "domain": "fit_error",
        "description": "Per-frame mean Euclidean residual over keypoints.",
        "interpretation": "MPJPE-like view of marker fit, less dominated by a single bad keypoint than the max residual.",
        "research_basis": "Mean per-joint position error in 3D pose estimation.",
    },
    {
        "metric_name": "marker_max_residual_mm",
        "unit": "mm",
        "domain": "fit_error",
        "description": "Largest single keypoint residual in a frame.",
        "interpretation": "Localizes the frame and keypoint most responsible for a bad fit.",
        "research_basis": "OpenSim IK maximum marker error and per-landmark error inspection.",
    },
    {
        "metric_name": "marker_rmse_pct_scale",
        "unit": "% body scale",
        "domain": "fit_error",
        "description": "Marker RMSE divided by robust pose scale.",
        "interpretation": "Scale-normalized fit error for comparing animals, models, or calibrated units.",
        "research_basis": "Pose metrics commonly normalize by object, body, or keypoint scale.",
    },
    {
        "metric_name": "source_keypoint_temporal_score",
        "unit": "robust z",
        "domain": "source_keypoint_quality",
        "description": "Largest per-keypoint robust temporal interpolation outlier score in a frame.",
        "interpretation": "High values point to implausible keypoint jumps before fitting.",
        "research_basis": "DeepLabCut jump/fitting outlier heuristics; Anipose 2D filtering.",
    },
    {
        "metric_name": "source_keypoint_interp_rmse_mm",
        "unit": "mm",
        "domain": "source_keypoint_quality",
        "description": "Per-frame RMS distance between observed keypoints and linear interpolation from adjacent observed frames.",
        "interpretation": "Raw keypoint temporal roughness; use with robust score to avoid hard dataset-specific thresholds.",
        "research_basis": "Pose continuity checks and trajectory filtering.",
    },
    {
        "metric_name": "fit_marker_temporal_score",
        "unit": "robust z",
        "domain": "fit_motion_quality",
        "description": "Largest per-marker robust temporal interpolation outlier score in the fitted marker trajectory.",
        "interpretation": "High values indicate fitted motion discontinuities after IK.",
        "research_basis": "Motion continuity and posture plausibility losses used in pose-estimation refinement.",
    },
    {
        "metric_name": "fit_marker_interp_rmse_mm",
        "unit": "mm",
        "domain": "fit_motion_quality",
        "description": "Per-frame RMS distance between fitted marker sites and linear interpolation from adjacent fitted frames.",
        "interpretation": "Fitted marker roughness; compare to source roughness to separate IK artifacts from input data artifacts.",
        "research_basis": "Trajectory smoothness and filtering diagnostics.",
    },
    {
        "metric_name": "root_pos_step_mm",
        "unit": "mm/frame",
        "domain": "root_continuity",
        "description": "Frame-to-frame translation step of the root qpos.",
        "interpretation": "Large isolated values flag root translation jumps.",
        "research_basis": "IK continuity diagnostics.",
    },
    {
        "metric_name": "root_geodesic_step_deg",
        "unit": "deg/frame",
        "domain": "root_continuity",
        "description": "Frame-to-frame geodesic angle of the root quaternion after hemisphere alignment.",
        "interpretation": "Large isolated values flag root orientation jumps; raw quaternion sign flips are tracked separately.",
        "research_basis": "Rigid-body orientation continuity diagnostics.",
    },
    {
        "metric_name": "boundary_marker_step_ratio",
        "unit": "ratio",
        "domain": "chunk_boundary",
        "description": "Fitted marker step at a chunk boundary divided by the interior p95 fitted marker step.",
        "interpretation": "Values above 3 suggest chunking or stitching is amplifying motion at boundaries.",
        "research_basis": "Streaming IK boundary artifact diagnosis.",
    },
    {
        "metric_name": "boundary_root_pos_step_ratio",
        "unit": "ratio",
        "domain": "chunk_boundary",
        "description": "Root translation step at a chunk boundary divided by the interior p95 root translation step.",
        "interpretation": "Values above 3 suggest root discontinuity at chunk boundaries.",
        "research_basis": "Streaming IK boundary artifact diagnosis.",
    },
    {
        "metric_name": "boundary_root_geodesic_step_ratio",
        "unit": "ratio",
        "domain": "chunk_boundary",
        "description": "Root orientation step at a chunk boundary divided by the interior p95 root orientation step.",
        "interpretation": "Values above 3 suggest root orientation discontinuity at chunk boundaries.",
        "research_basis": "Streaming IK boundary artifact diagnosis.",
    },
    {
        "metric_name": "qpos_step_rms",
        "unit": "qpos units/frame",
        "domain": "solver_context",
        "description": "Frame-to-frame RMS step over qpos after root-quaternion sign alignment.",
        "interpretation": "Context metric for debugging which generalized coordinates move abruptly.",
        "research_basis": "IK trajectory continuity diagnostics.",
    },
]

RESEARCH_SOURCES = [
    {
        "title": "SLEAP: A deep learning system for multi-animal pose tracking",
        "url": "https://www.nature.com/articles/s41592-022-01426-1",
        "applies_to": "mAP, landmark localization distributions, body-size-normalized pose accuracy.",
    },
    {
        "title": "DeepLabCut user guide",
        "url": "https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html",
        "applies_to": "RMSE evaluation, p-cutoff confidence filtering, jump and fitting outlier heuristics.",
    },
    {
        "title": "OpenSim inverse kinematics best practices",
        "url": "https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53090489/_Inverse%2BKinematics%2BBest%2BPractices",
        "applies_to": "RMS and maximum marker errors for IK review and marker/model troubleshooting.",
    },
    {
        "title": "Lightning Pose: improved animal pose estimation",
        "url": "https://experiments.springernature.com/articles/10.1038/s41592-024-02319-1",
        "applies_to": "Motion continuity, posture plausibility, and confidence/outlier-aware pose refinement.",
    },
    {
        "title": "Anipose documentation",
        "url": "https://anipose.readthedocs.io/en/latest/",
        "applies_to": "Score thresholds, temporal filtering, triangulation smoothness, and spatial constraints.",
    },
]


def generate_fit_report(
    h5_path: str | Path,
    report_path: str | Path | None = None,
    *,
    frame_start: int = 0,
) -> Path:
    """Write fit diagnostics into an IK H5 file and generate an HTML report."""
    h5_path = Path(h5_path)
    if report_path is None:
        report_path = h5_path.with_name(f"{h5_path.stem}_fit_report.html")
    report_path = Path(report_path)

    cfg, data = io.load_stac_data(h5_path)
    diagnostics = _compute_fit_diagnostics(data, cfg, h5_path, frame_start)
    _write_fit_diagnostics_to_h5(h5_path, diagnostics)

    html_text = _render_html_report(diagnostics)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def _compute_fit_diagnostics(
    data, cfg, h5_path: Path, frame_start: int
) -> dict[str, Any]:
    qpos = np.asarray(data.qpos, dtype=float)
    marker_sites = np.asarray(data.marker_sites, dtype=float)
    kp_data = np.asarray(data.kp_data, dtype=float)
    qvel = np.asarray(data.qvel, dtype=float)
    n_frames = int(qpos.shape[0])
    n_keypoints = int(marker_sites.shape[1]) if marker_sites.ndim == 3 else 0
    frame = np.arange(n_frames, dtype=np.int64)
    absolute_frame = frame + int(frame_start)
    chunk_size = int(getattr(cfg.stac, "n_frames_per_clip", 0) or 0)

    if marker_sites.ndim != 3:
        raise ValueError("marker_sites must have shape (frames, keypoints, xyz).")
    if kp_data.shape[0] != n_frames:
        raise ValueError("kp_data and qpos have different frame counts.")
    if kp_data.shape[1] != n_keypoints * 3:
        raise ValueError("kp_data columns must match marker_sites keypoints * 3.")

    keypoint_names = list(getattr(data, "kp_names", []))
    if len(keypoint_names) != n_keypoints:
        keypoint_names = [f"keypoint_{i}" for i in range(n_keypoints)]
    qpos_names = list(getattr(data, "names_qpos", []))
    if len(qpos_names) != qpos.shape[1]:
        qpos_names = [f"qpos_{i}" for i in range(qpos.shape[1])]

    kp_xyz = kp_data.reshape(n_frames, n_keypoints, 3)
    pose_scale_mm = _pose_scale_mm(kp_xyz)

    residual_xyz = kp_xyz - marker_sites
    residual_mm = np.linalg.norm(residual_xyz, axis=2) * 1000.0
    marker_rmse_mm = np.sqrt(np.nanmean(residual_xyz**2, axis=(1, 2))) * 1000.0
    marker_mean_residual_mm = np.nanmean(residual_mm, axis=1)
    marker_max_residual_index, marker_max_residual_mm = _row_nanargmax(residual_mm)
    marker_max_residual_name = np.array(
        [keypoint_names[i] if i >= 0 else "" for i in marker_max_residual_index],
        dtype=object,
    )
    marker_rmse_pct_scale = _pct_scale(marker_rmse_mm, pose_scale_mm)

    root_pos_step_mm = _nan_series(n_frames)
    root_geodesic_step_deg = _nan_series(n_frames)
    root_quat_dot_raw = _nan_series(n_frames)
    if qpos.shape[1] >= 3 and n_frames > 1:
        root_pos_step_mm[:-1] = (
            np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * 1000.0
        )
    if qpos.shape[1] >= 7 and n_frames > 1:
        quat = qpos[:, 3:7]
        quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = quat / np.where(quat_norm > 0.0, quat_norm, 1.0)
        dot_raw = np.sum(quat[:-1] * quat[1:], axis=1)
        root_quat_dot_raw[:-1] = dot_raw
        root_geodesic_step_deg[:-1] = np.degrees(
            2.0 * np.arccos(np.clip(np.abs(dot_raw), -1.0, 1.0))
        )

    qpos_for_diagnostics = qpos.copy()
    if qpos_for_diagnostics.shape[1] >= 7 and n_frames > 1:
        diagnostic_quat = qpos_for_diagnostics[:, 3:7].copy()
        quat_norm = np.linalg.norm(diagnostic_quat, axis=1, keepdims=True)
        diagnostic_quat = diagnostic_quat / np.where(quat_norm > 0.0, quat_norm, 1.0)
        for frame_idx in range(1, n_frames):
            if np.dot(diagnostic_quat[frame_idx - 1], diagnostic_quat[frame_idx]) < 0.0:
                diagnostic_quat[frame_idx] *= -1.0
                qpos_for_diagnostics[frame_idx, 3:7] *= -1.0

    qpos_step_rms = _nan_series(n_frames)
    qpos_step_max_abs = _nan_series(n_frames)
    qpos_step_max_abs_index = np.full(n_frames, -1, dtype=np.int64)
    qpos_step_max_abs_name = np.array([""] * n_frames, dtype=object)
    if n_frames > 1 and qpos.shape[1] > 0:
        qpos_step = np.diff(qpos_for_diagnostics, axis=0)
        qpos_step_abs = np.abs(qpos_step)
        qpos_step_rms[:-1] = np.sqrt(np.nanmean(qpos_step**2, axis=1))
        max_index = np.nanargmax(
            np.where(np.isfinite(qpos_step_abs), qpos_step_abs, -np.inf), axis=1
        )
        qpos_step_max_abs[:-1] = qpos_step_abs[
            np.arange(qpos_step_abs.shape[0]), max_index
        ]
        qpos_step_max_abs_index[:-1] = max_index
        qpos_step_max_abs_name[:-1] = [qpos_names[i] for i in max_index]

    marker_step_rmse_mm = _nan_series(n_frames)
    if n_frames > 1:
        marker_step = np.diff(marker_sites, axis=0)
        marker_step_rmse_mm[:-1] = (
            np.sqrt(np.nanmean(marker_step**2, axis=(1, 2))) * 1000.0
        )

    source_interp_error_mm = np.full((n_frames, n_keypoints), np.nan, dtype=float)
    fit_interp_error_mm = np.full((n_frames, n_keypoints), np.nan, dtype=float)
    source_keypoint_interp_rmse_mm = _nan_series(n_frames)
    fit_marker_interp_rmse_mm = _nan_series(n_frames)
    if n_frames > 2:
        source_interp = kp_xyz[1:-1] - 0.5 * (kp_xyz[:-2] + kp_xyz[2:])
        source_interp_error_mm[1:-1] = np.linalg.norm(source_interp, axis=2) * 1000.0
        source_keypoint_interp_rmse_mm[1:-1] = (
            np.sqrt(np.nanmean(source_interp**2, axis=(1, 2))) * 1000.0
        )

        fit_interp = marker_sites[1:-1] - 0.5 * (marker_sites[:-2] + marker_sites[2:])
        fit_interp_error_mm[1:-1] = np.linalg.norm(fit_interp, axis=2) * 1000.0
        fit_marker_interp_rmse_mm[1:-1] = (
            np.sqrt(np.nanmean(fit_interp**2, axis=(1, 2))) * 1000.0
        )

    source_score_by_keypoint = _robust_score_by_column(source_interp_error_mm)
    fit_score_by_keypoint = _robust_score_by_column(fit_interp_error_mm)
    source_keypoint_temporal_score = _row_nanmax(source_score_by_keypoint)
    fit_marker_temporal_score = _row_nanmax(fit_score_by_keypoint)

    (
        source_keypoint_max_interp_error_index,
        source_keypoint_max_interp_error_mm,
    ) = _row_nanargmax(source_interp_error_mm)
    source_keypoint_max_interp_error_name = np.array(
        [
            keypoint_names[i] if i >= 0 else ""
            for i in source_keypoint_max_interp_error_index
        ],
        dtype=object,
    )

    (
        fit_marker_max_interp_error_index,
        fit_marker_max_interp_error_mm,
    ) = _row_nanargmax(fit_interp_error_mm)
    fit_marker_max_interp_error_name = np.array(
        [
            keypoint_names[i] if i >= 0 else ""
            for i in fit_marker_max_interp_error_index
        ],
        dtype=object,
    )

    qvel_max_abs = _nan_series(n_frames)
    qvel_max_abs_index = np.full(n_frames, -1, dtype=np.int64)
    qvel_max_abs_name = np.array([""] * n_frames, dtype=object)
    if qvel.ndim == 2 and qvel.shape[0] == n_frames and qvel.shape[1] > 0:
        qvel_abs = np.abs(qvel)
        max_qvel_index = np.nanargmax(
            np.where(np.isfinite(qvel_abs), qvel_abs, -np.inf), axis=1
        )
        qvel_max_abs = qvel_abs[np.arange(n_frames), max_qvel_index]
        qvel_max_abs_index = max_qvel_index.astype(np.int64)
        qvel_max_abs_name = np.array(
            [
                qpos_names[i] if i < len(qpos_names) else f"qvel_{i}"
                for i in max_qvel_index
            ],
            dtype=object,
        )

    is_chunk_boundary = np.zeros(n_frames, dtype=bool)
    if chunk_size > 0:
        for boundary in range(chunk_size - 1, max(n_frames - 1, 0), chunk_size):
            is_chunk_boundary[boundary] = True

    boundary_marker_step_ratio = _boundary_ratio(marker_step_rmse_mm, is_chunk_boundary)
    boundary_root_pos_step_ratio = _boundary_ratio(root_pos_step_mm, is_chunk_boundary)
    boundary_root_geodesic_step_ratio = _boundary_ratio(
        root_geodesic_step_deg, is_chunk_boundary
    )

    boundary_marker_step_mm = np.where(is_chunk_boundary, marker_step_rmse_mm, np.nan)
    boundary_root_pos_step_mm = np.where(is_chunk_boundary, root_pos_step_mm, np.nan)
    boundary_root_geodesic_step_deg = np.where(
        is_chunk_boundary, root_geodesic_step_deg, np.nan
    )

    fit_error_score = _robust_score(marker_rmse_mm)
    root_pos_step_score = _robust_score(root_pos_step_mm)
    root_geodesic_step_score = _robust_score(root_geodesic_step_deg)
    root_continuity_score = _row_nanmax(
        np.vstack([root_pos_step_score, root_geodesic_step_score]).T
    )

    metric_values = {
        "marker_rmse_mm": marker_rmse_mm,
        "marker_mean_residual_mm": marker_mean_residual_mm,
        "marker_max_residual_mm": marker_max_residual_mm,
        "marker_rmse_pct_scale": marker_rmse_pct_scale,
        "source_keypoint_temporal_score": source_keypoint_temporal_score,
        "source_keypoint_interp_rmse_mm": source_keypoint_interp_rmse_mm,
        "fit_marker_temporal_score": fit_marker_temporal_score,
        "fit_marker_interp_rmse_mm": fit_marker_interp_rmse_mm,
        "root_pos_step_mm": root_pos_step_mm,
        "root_geodesic_step_deg": root_geodesic_step_deg,
        "boundary_marker_step_ratio": boundary_marker_step_ratio,
        "boundary_root_pos_step_ratio": boundary_root_pos_step_ratio,
        "boundary_root_geodesic_step_ratio": boundary_root_geodesic_step_ratio,
        "boundary_marker_step_mm": boundary_marker_step_mm,
        "boundary_root_pos_step_mm": boundary_root_pos_step_mm,
        "boundary_root_geodesic_step_deg": boundary_root_geodesic_step_deg,
        "qpos_step_rms": qpos_step_rms,
    }

    threshold_rows = []
    thresholds_by_metric = {}
    for metric_name, values in metric_values.items():
        threshold = _threshold_for_metric(metric_name, values)
        thresholds_by_metric[metric_name] = {"threshold": threshold}
        threshold_rows.append({"metric_name": metric_name, "threshold": threshold})

    event_metric_names = [
        "marker_rmse_mm",
        "marker_max_residual_mm",
        "source_keypoint_temporal_score",
        "fit_marker_temporal_score",
        "root_pos_step_mm",
        "root_geodesic_step_deg",
        "boundary_marker_step_ratio",
        "boundary_root_pos_step_ratio",
        "boundary_root_geodesic_step_ratio",
    ]
    events = []
    for metric_name in event_metric_names:
        events.extend(
            _events_for_metric(
                metric_name,
                metric_values[metric_name],
                thresholds_by_metric[metric_name]["threshold"],
                frame_start,
                marker_max_residual_name,
                marker_max_residual_mm,
                source_keypoint_max_interp_error_name,
                source_keypoint_max_interp_error_mm,
                fit_marker_max_interp_error_name,
                fit_marker_max_interp_error_mm,
                qpos_step_max_abs_name,
                is_chunk_boundary,
            )
        )
    events = sorted(events, key=lambda row: (row["start_frame"], row["metric_name"]))
    fit_events = [event for event in events if event["event_category"] != "source_data"]
    source_events = [
        event for event in events if event["event_category"] == "source_data"
    ]

    event_count = _event_count_series(events, n_frames)
    fit_event_count = _event_count_series(fit_events, n_frames)
    source_event_count = _event_count_series(source_events, n_frames)

    summary_rows = [
        _summary_for_metric(metric_name, values, thresholds_by_metric[metric_name])
        for metric_name, values in metric_values.items()
    ]
    root_quat_summary = _root_quat_summary(root_quat_dot_raw)
    if root_quat_summary:
        summary_rows.append(root_quat_summary)

    keypoint_rows = _keypoint_rows(
        residual_mm,
        source_interp_error_mm,
        source_score_by_keypoint,
        fit_interp_error_mm,
        fit_score_by_keypoint,
        keypoint_names,
        frame_start,
    )

    diagnostic_rows = _diagnostic_score_rows(
        summary_rows,
        keypoint_rows,
        marker_rmse_mm,
        source_keypoint_temporal_score,
        fit_error_score,
        fit_events,
        source_events,
        boundary_marker_step_ratio,
        boundary_root_pos_step_ratio,
        boundary_root_geodesic_step_ratio,
        pose_scale_mm,
    )
    verdict, primary_driver = _fit_verdict(diagnostic_rows, fit_events, source_events)

    frames = {
        "frame": frame,
        "absolute_frame": absolute_frame,
        "marker_rmse_mm": marker_rmse_mm,
        "marker_mean_residual_mm": marker_mean_residual_mm,
        "marker_max_residual_index": marker_max_residual_index,
        "marker_max_residual_name": marker_max_residual_name,
        "marker_max_residual_mm": marker_max_residual_mm,
        "marker_rmse_pct_scale": marker_rmse_pct_scale,
        "fit_error_score": fit_error_score,
        "source_keypoint_temporal_score": source_keypoint_temporal_score,
        "source_keypoint_interp_rmse_mm": source_keypoint_interp_rmse_mm,
        "source_keypoint_max_interp_error_index": (
            source_keypoint_max_interp_error_index
        ),
        "source_keypoint_max_interp_error_name": (
            source_keypoint_max_interp_error_name
        ),
        "source_keypoint_max_interp_error_mm": (source_keypoint_max_interp_error_mm),
        "fit_marker_temporal_score": fit_marker_temporal_score,
        "fit_marker_interp_rmse_mm": fit_marker_interp_rmse_mm,
        "fit_marker_max_interp_error_index": fit_marker_max_interp_error_index,
        "fit_marker_max_interp_error_name": fit_marker_max_interp_error_name,
        "fit_marker_max_interp_error_mm": fit_marker_max_interp_error_mm,
        "marker_step_rmse_mm": marker_step_rmse_mm,
        "root_pos_step_mm": root_pos_step_mm,
        "root_geodesic_step_deg": root_geodesic_step_deg,
        "root_pos_step_score": root_pos_step_score,
        "root_geodesic_step_score": root_geodesic_step_score,
        "root_continuity_score": root_continuity_score,
        "root_quat_dot_raw": root_quat_dot_raw,
        "qpos_step_rms": qpos_step_rms,
        "qpos_step_max_abs": qpos_step_max_abs,
        "qpos_step_max_abs_index": qpos_step_max_abs_index,
        "qpos_step_max_abs_name": qpos_step_max_abs_name,
        "qvel_max_abs": qvel_max_abs,
        "qvel_max_abs_index": qvel_max_abs_index,
        "qvel_max_abs_name": qvel_max_abs_name,
        "is_chunk_boundary": is_chunk_boundary,
        "boundary_marker_step_mm": boundary_marker_step_mm,
        "boundary_marker_step_ratio": boundary_marker_step_ratio,
        "boundary_root_pos_step_mm": boundary_root_pos_step_mm,
        "boundary_root_pos_step_ratio": boundary_root_pos_step_ratio,
        "boundary_root_geodesic_step_deg": boundary_root_geodesic_step_deg,
        "boundary_root_geodesic_step_ratio": boundary_root_geodesic_step_ratio,
        "event_count": event_count,
        "fit_event_count": fit_event_count,
        "source_event_count": source_event_count,
    }

    generated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "fit_quality_report",
        "generated_at": generated_at,
        "source_h5": str(h5_path),
        "frame_start": int(frame_start),
        "n_frames": n_frames,
        "n_keypoints": n_keypoints,
        "chunk_size": chunk_size,
        "pose_scale_mm": pose_scale_mm,
        "verdict": verdict,
        "primary_driver": primary_driver,
        "event_count": len(events),
        "fit_event_count": len(fit_events),
        "source_event_count": len(source_events),
        "fit_event_frame_fraction_pct": _frame_fraction_pct(fit_event_count > 0),
        "source_event_frame_fraction_pct": _frame_fraction_pct(source_event_count > 0),
    }

    return {
        "metadata": metadata,
        "frames": frames,
        "summary": _rows_to_columns(summary_rows),
        "summary_rows": summary_rows,
        "keypoints": _rows_to_columns(keypoint_rows),
        "keypoint_rows": keypoint_rows,
        "events": _rows_to_columns(events, columns=_event_columns()),
        "event_rows": events,
        "fit_events": _rows_to_columns(fit_events, columns=_event_columns()),
        "fit_event_rows": fit_events,
        "source_events": _rows_to_columns(source_events, columns=_event_columns()),
        "source_event_rows": source_events,
        "thresholds": _rows_to_columns(threshold_rows),
        "threshold_rows": threshold_rows,
        "diagnostic_scores": _rows_to_columns(diagnostic_rows),
        "diagnostic_score_rows": diagnostic_rows,
        "metric_definitions": _rows_to_columns(METRIC_DEFINITIONS),
        "metric_definition_rows": METRIC_DEFINITIONS,
        "sources": _rows_to_columns(RESEARCH_SOURCES),
        "source_rows": RESEARCH_SOURCES,
    }


def _pose_scale_mm(kp_xyz: np.ndarray) -> float:
    if kp_xyz.shape[0] == 0 or kp_xyz.shape[1] < 2:
        return np.nan
    step = max(1, kp_xyz.shape[0] // 2000)
    sampled = kp_xyz[::step] * 1000.0
    spans = []
    for points in sampled:
        finite = np.all(np.isfinite(points), axis=1)
        points = points[finite]
        if points.shape[0] < 2:
            continue
        diff = points[:, None, :] - points[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        spans.append(float(np.nanpercentile(distances, 95)))
    if not spans:
        return np.nan
    return float(np.nanmedian(spans))


def _pct_scale(values: np.ndarray, scale_mm: float) -> np.ndarray:
    if not np.isfinite(scale_mm) or scale_mm <= 0.0:
        return np.full_like(np.asarray(values, dtype=float), np.nan)
    return np.asarray(values, dtype=float) / scale_mm * 100.0


def _nan_series(n_frames: int) -> np.ndarray:
    return np.full(n_frames, np.nan, dtype=float)


def _row_nanargmax(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, -np.inf)
    indexes = np.argmax(safe_values, axis=1).astype(np.int64)
    no_value = ~np.any(finite, axis=1)
    max_values = safe_values[np.arange(values.shape[0]), indexes]
    indexes[no_value] = -1
    max_values = max_values.astype(float)
    max_values[no_value] = np.nan
    return indexes, max_values


def _row_nanmax(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, -np.inf)
    out = np.max(safe_values, axis=1).astype(float)
    out[~np.any(finite, axis=1)] = np.nan
    return out


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 8:
        return np.nan, np.nan
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= EPS:
        q25, q75 = np.percentile(finite, [25, 75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= EPS:
        scale = np.nan
    return center, scale


def _robust_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center, scale = _robust_center_scale(values)
    score = np.full(values.shape, np.nan, dtype=float)
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= EPS:
        return score
    score = (values - center) / scale
    score[~np.isfinite(values)] = np.nan
    return np.maximum(score, 0.0)


def _robust_score_by_column(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scores = np.full(values.shape, np.nan, dtype=float)
    if values.ndim != 2:
        return scores
    for column in range(values.shape[1]):
        scores[:, column] = _robust_score(values[:, column])
    return scores


def _threshold_for_metric(metric_name: str, values: np.ndarray) -> float:
    if metric_name.endswith("_ratio"):
        return BOUNDARY_RATIO_EVENT_THRESHOLD
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size < 8:
        return np.nan
    if metric_name.endswith("_score"):
        return float(
            max(ROBUST_Z_EVENT_THRESHOLD, np.percentile(valid, EVENT_TAIL_PERCENTILE))
        )
    center, scale = _robust_center_scale(valid)
    threshold = center + 8.0 * scale if np.isfinite(scale) else np.nan
    tail = float(np.percentile(valid, EVENT_TAIL_PERCENTILE))
    if np.isfinite(threshold):
        return float(max(threshold, tail))
    return tail


def _boundary_ratio(values: np.ndarray, is_chunk_boundary: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ratio = np.full(values.shape, np.nan, dtype=float)
    interior = values[(~is_chunk_boundary) & np.isfinite(values)]
    boundary = is_chunk_boundary & np.isfinite(values)
    if interior.size < 8:
        return ratio
    baseline = float(np.percentile(interior, 95))
    if not np.isfinite(baseline) or baseline <= EPS:
        return ratio
    ratio[boundary] = values[boundary] / baseline
    return ratio


def _events_for_metric(
    metric_name: str,
    values: np.ndarray,
    threshold: float,
    frame_start: int,
    marker_max_residual_name: np.ndarray,
    marker_max_residual_mm: np.ndarray,
    source_keypoint_max_interp_error_name: np.ndarray,
    source_keypoint_max_interp_error_mm: np.ndarray,
    fit_marker_max_interp_error_name: np.ndarray,
    fit_marker_max_interp_error_mm: np.ndarray,
    qpos_step_max_abs_name: np.ndarray,
    is_chunk_boundary: np.ndarray,
) -> list[dict[str, Any]]:
    if not np.isfinite(threshold):
        return []
    values = np.asarray(values, dtype=float)
    threshold_mask = np.isfinite(values) & (values >= threshold)
    events = []
    index = 0
    while index < values.shape[0]:
        if not threshold_mask[index]:
            index += 1
            continue
        start = index
        while index + 1 < values.shape[0] and threshold_mask[index + 1]:
            index += 1
        end = index
        segment = values[start : end + 1]
        peak_offset = int(np.nanargmax(segment))
        peak_frame = start + peak_offset
        peak_value = float(values[peak_frame])
        focus = _event_focus(
            metric_name,
            peak_frame,
            marker_max_residual_name,
            marker_max_residual_mm,
            source_keypoint_max_interp_error_name,
            source_keypoint_max_interp_error_mm,
            fit_marker_max_interp_error_name,
            fit_marker_max_interp_error_mm,
            qpos_step_max_abs_name,
            is_chunk_boundary,
        )
        events.append(
            {
                "metric_name": metric_name,
                "event_category": _event_category(metric_name),
                "start_frame": int(start),
                "end_frame": int(end),
                "duration_frames": int(end - start + 1),
                "peak_frame": int(peak_frame),
                "absolute_start_frame": int(frame_start + start),
                "absolute_end_frame": int(frame_start + end),
                "absolute_peak_frame": int(frame_start + peak_frame),
                "peak_value": peak_value,
                "threshold": float(threshold),
                "severity_score": _event_severity(metric_name, peak_value, threshold),
                "is_chunk_boundary_event": bool(
                    np.any(is_chunk_boundary[start : end + 1])
                ),
                "focus": focus,
                "association": focus,
                "interpretation": _event_interpretation(metric_name, focus),
            }
        )
        index += 1
    return events


def _event_category(metric_name: str) -> str:
    if metric_name.startswith("source_keypoint_"):
        return "source_data"
    if metric_name.startswith("boundary_"):
        return "chunk_boundary"
    if metric_name.startswith("root_"):
        return "root_continuity"
    if metric_name.startswith("fit_marker_"):
        return "fit_motion"
    return "fit_error"


def _event_severity(metric_name: str, peak_value: float, threshold: float) -> float:
    if metric_name.endswith("_score"):
        return float(peak_value)
    if threshold == 0.0 or not np.isfinite(threshold):
        return np.nan
    return float(peak_value / threshold)


def _event_focus(
    metric_name: str,
    peak_frame: int,
    marker_max_residual_name: np.ndarray,
    marker_max_residual_mm: np.ndarray,
    source_keypoint_max_interp_error_name: np.ndarray,
    source_keypoint_max_interp_error_mm: np.ndarray,
    fit_marker_max_interp_error_name: np.ndarray,
    fit_marker_max_interp_error_mm: np.ndarray,
    qpos_step_max_abs_name: np.ndarray,
    is_chunk_boundary: np.ndarray,
) -> str:
    if metric_name.startswith("marker_"):
        return (
            f"{marker_max_residual_name[peak_frame]} "
            f"({marker_max_residual_mm[peak_frame]:.3g} mm)"
        )
    if metric_name.startswith("source_keypoint_"):
        return (
            f"{source_keypoint_max_interp_error_name[peak_frame]} "
            f"({source_keypoint_max_interp_error_mm[peak_frame]:.3g} mm)"
        )
    if metric_name.startswith("fit_marker_"):
        return (
            f"{fit_marker_max_interp_error_name[peak_frame]} "
            f"({fit_marker_max_interp_error_mm[peak_frame]:.3g} mm)"
        )
    if metric_name.startswith("boundary_") or bool(is_chunk_boundary[peak_frame]):
        return "chunk boundary"
    if metric_name == "qpos_step_rms":
        return str(qpos_step_max_abs_name[peak_frame])
    return "root trajectory"


def _event_interpretation(metric_name: str, focus: str) -> str:
    if metric_name.startswith("source_keypoint_"):
        return f"Inspect source keypoint data around {focus}; the raw trajectory has an adaptive temporal outlier."
    if metric_name.startswith("fit_marker_"):
        return f"Inspect IK smoothness around {focus}; fitted marker motion is rougher than its local baseline."
    if metric_name.startswith("boundary_"):
        return "Inspect chunk stitching, context overlap, and warm starts at this boundary."
    if metric_name.startswith("root_"):
        return "Inspect root initialization, trunk/root keypoints, and continuity regularization near this frame."
    return f"Inspect fit residuals and marker offsets for {focus}."


def _event_count_series(events: list[dict[str, Any]], n_frames: int) -> np.ndarray:
    counts = np.zeros(n_frames, dtype=np.int64)
    for event in events:
        counts[int(event["start_frame"]) : int(event["end_frame"]) + 1] += 1
    return counts


def _summary_for_metric(
    metric_name: str, values: np.ndarray, thresholds: dict[str, float]
) -> dict[str, Any]:
    valid = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(valid)
    finite = valid[finite_mask]
    center, scale = _robust_center_scale(finite)
    if finite.size == 0:
        return {
            "metric_name": metric_name,
            "count": 0,
            "mean": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "max_frame": -1,
            "robust_center": center,
            "robust_scale": scale,
            "event_frame_fraction_pct": np.nan,
            **thresholds,
        }
    finite_indexes = np.flatnonzero(finite_mask)
    max_offset = int(np.argmax(finite))
    threshold = thresholds.get("threshold", np.nan)
    if np.isfinite(threshold):
        event_frame_fraction_pct = _frame_fraction_pct(valid >= threshold)
    else:
        event_frame_fraction_pct = np.nan
    return {
        "metric_name": metric_name,
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(finite[max_offset]),
        "max_frame": int(finite_indexes[max_offset]),
        "robust_center": center,
        "robust_scale": scale,
        "event_frame_fraction_pct": event_frame_fraction_pct,
        **thresholds,
    }


def _root_quat_summary(values: np.ndarray) -> dict[str, Any] | None:
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if finite.size == 0:
        return None
    finite_indexes = np.flatnonzero(finite_mask)
    min_offset = int(np.argmin(finite))
    return {
        "metric_name": "root_quat_dot_raw",
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "max_frame": int(finite_indexes[int(np.argmax(finite))]),
        "min": float(finite[min_offset]),
        "min_frame": int(finite_indexes[min_offset]),
        "negative_count": int(np.sum(finite < 0.0)),
        "robust_center": np.nan,
        "robust_scale": np.nan,
        "event_frame_fraction_pct": np.nan,
        "threshold": np.nan,
    }


def _keypoint_rows(
    residual_mm: np.ndarray,
    source_interp_error_mm: np.ndarray,
    source_score_by_keypoint: np.ndarray,
    fit_interp_error_mm: np.ndarray,
    fit_score_by_keypoint: np.ndarray,
    keypoint_names: list[str],
    frame_start: int,
) -> list[dict[str, Any]]:
    rows = []
    total_sq = float(np.nansum(residual_mm**2))
    for index, name in enumerate(keypoint_names):
        values = residual_mm[:, index]
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        source_score = source_score_by_keypoint[:, index]
        fit_score = fit_score_by_keypoint[:, index]
        source_error = source_interp_error_mm[:, index]
        fit_error = fit_interp_error_mm[:, index]
        source_event_count = int(
            np.sum(
                np.isfinite(source_score) & (source_score >= ROBUST_Z_EVENT_THRESHOLD)
            )
        )
        fit_event_count = int(
            np.sum(np.isfinite(fit_score) & (fit_score >= ROBUST_Z_EVENT_THRESHOLD))
        )
        if finite.size == 0:
            rows.append(
                {
                    "keypoint_index": index,
                    "keypoint_name": name,
                    "mean_residual_mm": np.nan,
                    "median_residual_mm": np.nan,
                    "p95_residual_mm": np.nan,
                    "p99_residual_mm": np.nan,
                    "max_residual_mm": np.nan,
                    "max_frame": -1,
                    "max_absolute_frame": -1,
                    "residual_contribution_pct": np.nan,
                    "source_temporal_score_p95": _nanpercentile(source_score, 95),
                    "source_temporal_score_max": _nanmax(source_score),
                    "source_temporal_event_count": source_event_count,
                    "source_interp_max_mm": _nanmax(source_error),
                    "fit_temporal_score_p95": _nanpercentile(fit_score, 95),
                    "fit_temporal_score_max": _nanmax(fit_score),
                    "fit_temporal_event_count": fit_event_count,
                    "fit_interp_max_mm": _nanmax(fit_error),
                }
            )
            continue
        finite_indexes = np.flatnonzero(finite_mask)
        max_offset = int(np.argmax(finite))
        max_frame = int(finite_indexes[max_offset])
        contribution = (
            float(np.nansum(values**2) / total_sq * 100.0) if total_sq > EPS else np.nan
        )
        rows.append(
            {
                "keypoint_index": index,
                "keypoint_name": name,
                "mean_residual_mm": float(np.mean(finite)),
                "median_residual_mm": float(np.percentile(finite, 50)),
                "p95_residual_mm": float(np.percentile(finite, 95)),
                "p99_residual_mm": float(np.percentile(finite, 99)),
                "max_residual_mm": float(finite[max_offset]),
                "max_frame": max_frame,
                "max_absolute_frame": int(max_frame + frame_start),
                "residual_contribution_pct": contribution,
                "source_temporal_score_p95": _nanpercentile(source_score, 95),
                "source_temporal_score_max": _nanmax(source_score),
                "source_temporal_event_count": source_event_count,
                "source_interp_max_mm": _nanmax(source_error),
                "fit_temporal_score_p95": _nanpercentile(fit_score, 95),
                "fit_temporal_score_max": _nanmax(fit_score),
                "fit_temporal_event_count": fit_event_count,
                "fit_interp_max_mm": _nanmax(fit_error),
            }
        )
    return rows


def _nanpercentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    return float(np.percentile(finite, percentile))


def _nanmax(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan
    return float(np.max(finite))


def _diagnostic_score_rows(
    summary_rows: list[dict[str, Any]],
    keypoint_rows: list[dict[str, Any]],
    marker_rmse_mm: np.ndarray,
    source_keypoint_temporal_score: np.ndarray,
    fit_error_score: np.ndarray,
    fit_events: list[dict[str, Any]],
    source_events: list[dict[str, Any]],
    boundary_marker_step_ratio: np.ndarray,
    boundary_root_pos_step_ratio: np.ndarray,
    boundary_root_geodesic_step_ratio: np.ndarray,
    pose_scale_mm: float,
) -> list[dict[str, Any]]:
    summary = {row["metric_name"]: row for row in summary_rows}
    marker = summary.get("marker_rmse_mm", {})
    marker_pct = summary.get("marker_rmse_pct_scale", {})
    source_score = summary.get("source_keypoint_temporal_score", {})
    source_interp = summary.get("source_keypoint_interp_rmse_mm", {})
    source_interp_pct = (
        source_interp.get("p95", np.nan) / pose_scale_mm * 100.0
        if np.isfinite(pose_scale_mm) and pose_scale_mm > 0.0
        else np.nan
    )

    high_fit = np.isfinite(fit_error_score) & (
        fit_error_score >= ROBUST_Z_EVENT_THRESHOLD
    )
    high_source = np.isfinite(source_keypoint_temporal_score) & (
        source_keypoint_temporal_score >= ROBUST_Z_EVENT_THRESHOLD
    )
    if np.any(high_fit):
        fit_source_overlap_pct = float(np.mean(high_source[high_fit]) * 100.0)
    else:
        top_count = max(1, int(np.ceil(np.isfinite(marker_rmse_mm).sum() * 0.01)))
        finite = np.where(np.isfinite(marker_rmse_mm), marker_rmse_mm, -np.inf)
        top_idx = np.argsort(finite)[-top_count:]
        fit_source_overlap_pct = float(np.mean(high_source[top_idx]) * 100.0)

    top_keypoints = sorted(
        keypoint_rows,
        key=lambda row: (
            row["residual_contribution_pct"]
            if np.isfinite(row["residual_contribution_pct"])
            else -1.0
        ),
        reverse=True,
    )
    top3_share = float(
        np.nansum(
            [row.get("residual_contribution_pct", np.nan) for row in top_keypoints[:3]]
        )
    )
    boundary_ratios = np.array(
        [
            _nanmax(boundary_marker_step_ratio),
            _nanmax(boundary_root_pos_step_ratio),
            _nanmax(boundary_root_geodesic_step_ratio),
        ],
        dtype=float,
    )
    max_boundary_ratio = _nanmax(boundary_ratios)
    fit_event_frame_fraction = _event_frame_fraction(
        fit_events, marker_rmse_mm.shape[0]
    )
    source_event_frame_fraction = _event_frame_fraction(
        source_events, marker_rmse_mm.shape[0]
    )

    rows = [
        _score_row(
            "marker_rmse_p95_mm",
            marker.get("p95", np.nan),
            "mm",
            _absolute_fit_status(marker.get("p95", np.nan)),
            "Primary absolute fit accuracy reference.",
        ),
        _score_row(
            "marker_rmse_p95_pct_scale",
            marker_pct.get("p95", np.nan),
            "% body scale",
            _scale_fit_status(marker_pct.get("p95", np.nan)),
            f"Scale-normalized fit accuracy; pose scale is {_fmt(pose_scale_mm)} mm.",
        ),
        _score_row(
            "fit_event_frame_fraction_pct",
            fit_event_frame_fraction,
            "% frames",
            _fraction_status(fit_event_frame_fraction),
            "Fraction of frames covered by high-severity fit, motion, root, or boundary events.",
        ),
        _score_row(
            "source_event_frame_fraction_pct",
            source_event_frame_fraction,
            "% frames",
            _fraction_status(source_event_frame_fraction),
            "Fraction of frames covered by high-severity source keypoint temporal events.",
        ),
        _score_row(
            "source_interp_rmse_p95_pct_scale",
            source_interp_pct,
            "% body scale",
            _source_scale_status(source_interp_pct),
            "Scale-normalized p95 keypoint temporal interpolation error.",
        ),
        _score_row(
            "source_temporal_score_p99",
            source_score.get("p99", np.nan),
            "robust z",
            "context",
            "Whether raw keypoints have a heavy temporal-outlier tail.",
        ),
        _score_row(
            "fit_source_overlap_pct",
            fit_source_overlap_pct,
            "% high-fit frames",
            "context",
            "Share of high-fit-error frames that also have source keypoint temporal outliers.",
        ),
        _score_row(
            "max_boundary_step_ratio",
            max_boundary_ratio,
            "ratio",
            _boundary_status(max_boundary_ratio),
            "Largest chunk-boundary step relative to the interior p95 step.",
        ),
        _score_row(
            "top3_keypoint_residual_share_pct",
            top3_share,
            "% residual energy",
            "context",
            "Residual energy concentration in the top three keypoints.",
        ),
    ]
    return rows


def _score_row(
    score_name: str, value: float, unit: str, status: str, interpretation: str
) -> dict[str, Any]:
    return {
        "score_name": score_name,
        "value": float(value) if np.isfinite(value) else np.nan,
        "unit": unit,
        "status": status,
        "interpretation": interpretation,
    }


def _absolute_fit_status(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 10.0:
        return "pass"
    if value <= 20.0:
        return "review"
    return "poor"


def _scale_fit_status(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 4.0:
        return "pass"
    if value <= 8.0:
        return "review"
    return "poor"


def _source_scale_status(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 1.0:
        return "pass"
    if value <= 2.0:
        return "review"
    return "poor"


def _fraction_status(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value <= 2.0:
        return "pass"
    if value <= 5.0:
        return "review"
    return "poor"


def _boundary_status(value: float) -> str:
    if not np.isfinite(value):
        return "pass"
    if value < BOUNDARY_RATIO_EVENT_THRESHOLD:
        return "pass"
    if value < 2.0 * BOUNDARY_RATIO_EVENT_THRESHOLD:
        return "review"
    return "poor"


def _fit_verdict(
    diagnostic_rows: list[dict[str, Any]],
    fit_events: list[dict[str, Any]],
    source_events: list[dict[str, Any]],
) -> tuple[str, str]:
    statuses = {row["score_name"]: row["status"] for row in diagnostic_rows}
    if any(status == "poor" for status in statuses.values()):
        return "poor", _primary_driver(diagnostic_rows, fit_events, source_events)
    if any(status == "review" for status in statuses.values()):
        return "review", _primary_driver(diagnostic_rows, fit_events, source_events)
    if fit_events:
        return "pass", f"isolated fit events; top focus: {_event_driver(fit_events)}"
    if source_events:
        return (
            "pass",
            f"fit passes; source keypoint events remain: {_event_driver(source_events)}",
        )
    return "pass", "no high-severity fit or source events"


def _primary_driver(
    diagnostic_rows: list[dict[str, Any]],
    fit_events: list[dict[str, Any]],
    source_events: list[dict[str, Any]],
) -> str:
    problem_scores = [
        row
        for row in diagnostic_rows
        if row["status"] in {"poor", "review"} and np.isfinite(row["value"])
    ]
    if problem_scores:
        return f"{problem_scores[0]['score_name']} is {problem_scores[0]['status']}"
    if fit_events:
        return _event_driver(fit_events)
    if source_events:
        return _event_driver(source_events)
    return "no high-severity events"


def _event_driver(events: list[dict[str, Any]]) -> str:
    top = _rank_events(events)[:1]
    if not top:
        return "no high-severity events"
    event = top[0]
    return (
        f"{event['metric_name']} at frames "
        f"{event['absolute_start_frame']}-{event['absolute_end_frame']}"
    )


def _event_frame_fraction(events: list[dict[str, Any]], n_frames: int) -> float:
    if n_frames <= 0:
        return np.nan
    covered = np.zeros(n_frames, dtype=bool)
    for event in events:
        covered[int(event["start_frame"]) : int(event["end_frame"]) + 1] = True
    return float(np.mean(covered) * 100.0)


def _frame_fraction_pct(mask: np.ndarray) -> float:
    mask = np.asarray(mask)
    if mask.size == 0:
        return np.nan
    return float(np.mean(mask) * 100.0)


def _rank_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda row: (
            row["severity_score"] if np.isfinite(row["severity_score"]) else -1.0,
            row["duration_frames"],
        ),
        reverse=True,
    )


def _rows_to_columns(
    rows: list[dict[str, Any]], columns: list[str] | None = None
) -> dict[str, list[Any]]:
    if columns is None:
        columns = []
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
    return {
        column: [row.get(column, _empty_value(column)) for row in rows]
        for column in columns
    }


def _empty_value(column: str) -> Any:
    if (
        column.endswith("_frame")
        or column.endswith("_count")
        or column.endswith("_index")
    ):
        return -1
    return np.nan


def _event_columns() -> list[str]:
    return [
        "metric_name",
        "event_category",
        "start_frame",
        "end_frame",
        "duration_frames",
        "peak_frame",
        "absolute_start_frame",
        "absolute_end_frame",
        "absolute_peak_frame",
        "peak_value",
        "threshold",
        "severity_score",
        "is_chunk_boundary_event",
        "focus",
        "association",
        "interpretation",
    ]


def _write_fit_diagnostics_to_h5(h5_path: Path, diagnostics: dict[str, Any]) -> None:
    tmp_name = f"_{FIT_DIAGNOSTICS_GROUP}_tmp"
    backup_name = f"_{FIT_DIAGNOSTICS_GROUP}_old"
    with h5py.File(h5_path, "a") as h5:
        if tmp_name in h5:
            del h5[tmp_name]
        if backup_name in h5:
            del h5[backup_name]

        root = h5.create_group(tmp_name)
        for key, value in diagnostics["metadata"].items():
            root.attrs[key] = value

        for group_name in [
            "frames",
            "summary",
            "keypoints",
            "events",
            "fit_events",
            "source_events",
            "thresholds",
            "diagnostic_scores",
            "metric_definitions",
            "sources",
        ]:
            group = root.create_group(group_name)
            _write_columns(group, diagnostics[group_name])

        if FIT_DIAGNOSTICS_GROUP in h5:
            h5.move(FIT_DIAGNOSTICS_GROUP, backup_name)
        try:
            h5.move(tmp_name, FIT_DIAGNOSTICS_GROUP)
        except Exception:
            if backup_name in h5 and FIT_DIAGNOSTICS_GROUP not in h5:
                h5.move(backup_name, FIT_DIAGNOSTICS_GROUP)
            raise
        if backup_name in h5:
            del h5[backup_name]


def _write_columns(group, columns: dict[str, Any]) -> None:
    for key, values in columns.items():
        _write_dataset(group, key, values)


def _write_dataset(group, key: str, values: Any) -> None:
    array = np.asarray(values)
    if key in STRING_COLUMNS or array.dtype.kind in {"O", "U"}:
        text = np.array(
            ["" if value is None else str(value) for value in values], dtype=object
        )
        group.create_dataset(key, data=text, dtype=h5py.string_dtype(encoding="utf-8"))
        return
    if key in BOOL_COLUMNS or array.dtype.kind == "b":
        array = array.astype(np.bool_)
    elif key in INT_COLUMNS:
        array = array.astype(np.int64)
    compression = "gzip" if array.ndim > 0 and array.size > 0 else None
    group.create_dataset(key, data=array, compression=compression)


def _render_html_report(diagnostics: dict[str, Any]) -> str:
    metadata = diagnostics["metadata"]
    summary_rows = diagnostics["summary_rows"]
    keypoint_rows = diagnostics["keypoint_rows"]
    fit_event_rows = diagnostics["fit_event_rows"]
    source_event_rows = diagnostics["source_event_rows"]
    diagnostic_score_rows = diagnostics["diagnostic_score_rows"]
    frames = diagnostics["frames"]
    summary_by_metric = {row["metric_name"]: row for row in summary_rows}
    scores = {row["score_name"]: row for row in diagnostic_score_rows}
    top_keypoints = sorted(
        keypoint_rows,
        key=lambda row: (
            row["residual_contribution_pct"]
            if np.isfinite(row["residual_contribution_pct"])
            else -1.0
        ),
        reverse=True,
    )
    top_fit_events = _rank_events(fit_event_rows)[:50]
    top_source_events = _rank_events(source_event_rows)[:50]
    top_fit_frames = _top_frame_rows(frames, limit=50, event_prefix="fit_")
    top_source_frames = _top_frame_rows(frames, limit=50, event_prefix="source_")

    plots = {
        "fit_source_timeline": _plot_fit_source_timeline(
            frames, summary_by_metric, fit_event_rows + source_event_rows
        ),
        "keypoint_residuals": _plot_keypoint_residuals(top_keypoints),
        "continuity_timeline": _plot_continuity_timeline(
            frames, summary_by_metric, fit_event_rows
        ),
        "boundary_comparison": _plot_boundary_comparison(frames),
        "fit_distribution": _plot_fit_distribution(frames, summary_by_metric),
    }

    cards = _summary_cards(metadata, scores, top_keypoints)
    summary = _technical_summary(metadata, scores, top_keypoints)
    fit_interpretation = _fit_interpretation(scores, top_keypoints)
    source_interpretation = _source_interpretation(scores, source_event_rows)
    continuity_interpretation = _continuity_interpretation(scores, fit_event_rows)
    next_steps = _next_steps(metadata, scores, top_keypoints)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STAC-MJX Fit Diagnostics Report</title>
<style>
{_report_css()}
</style>
</head>
<body>
<main>
<header class="report-header">
  <div>
    <p class="eyebrow">STAC-MJX fit diagnostics</p>
    <h1>{_escape(Path(metadata["source_h5"]).name)}</h1>
    <p class="muted">Generated {_escape(metadata["generated_at"])}. Frames {_escape(metadata["frame_start"])}-{_escape(metadata["frame_start"] + metadata["n_frames"] - 1)}; robust pose scale {_fmt(metadata["pose_scale_mm"])} mm.</p>
  </div>
  <div class="verdict verdict-{_escape(metadata["verdict"])}">
    <span>Verdict</span>
    <strong>{_escape(metadata["verdict"])}</strong>
    <small>{_escape(metadata["primary_driver"])}</small>
  </div>
</header>

<section class="summary">
  <h2>Technical summary</h2>
  {summary}
</section>

<section class="cards">
{cards}
</section>

<section>
  <h2>Fit residuals are concentrated by keypoint</h2>
  <p>{fit_interpretation}</p>
  {_image(plots["keypoint_residuals"], "Keypoint residual ranking")}
  {_html_table(top_keypoints[:12], ["keypoint_name", "residual_contribution_pct", "p95_residual_mm", "p99_residual_mm", "max_residual_mm", "max_absolute_frame", "source_temporal_event_count", "fit_temporal_event_count"])}
</section>

<section>
  <h2>Source keypoint quality explains only the frames where it overlaps fit error</h2>
  <p>{source_interpretation}</p>
  {_image(plots["fit_source_timeline"], "Fit and source quality timeline")}
  <h3>Top source-data problem regions</h3>
  {_html_table(top_source_events, ["metric_name", "absolute_start_frame", "absolute_end_frame", "duration_frames", "absolute_peak_frame", "peak_value", "threshold", "severity_score", "focus", "interpretation"])}
</section>

<section>
  <h2>IK continuity and chunk boundaries isolate solver artifacts</h2>
  <p>{continuity_interpretation}</p>
  {_image(plots["continuity_timeline"], "Root and fitted marker continuity timeline")}
  {_image(plots["boundary_comparison"], "Chunk boundary comparison")}
  <h3>Top fit-quality problem regions</h3>
  {_html_table(top_fit_events, ["metric_name", "event_category", "absolute_start_frame", "absolute_end_frame", "duration_frames", "absolute_peak_frame", "peak_value", "threshold", "severity_score", "focus", "interpretation"])}
</section>

<section>
  <h2>Distribution checks keep absolute thresholds in context</h2>
  <p>Absolute marker errors are shown beside scale-normalized and adaptive baselines. Use the absolute millimeter values for practical inspection and the percent-of-scale values when comparing different animals, scales, or datasets.</p>
  {_image(plots["fit_distribution"], "Fit error distribution")}
  {_html_table(diagnostic_score_rows, ["score_name", "value", "unit", "status", "interpretation"])}
</section>

<section>
  <h2>Audit frames</h2>
  <p>These are the highest-priority frames for visual inspection. The first table is fit-driven; the second is source-data-driven.</p>
  <h3>Top fit frames</h3>
  {_html_table(top_fit_frames, ["frame", "absolute_frame", "fit_event_count", "marker_rmse_mm", "marker_max_residual_name", "marker_max_residual_mm", "source_keypoint_temporal_score", "fit_marker_temporal_score", "root_pos_step_mm", "root_geodesic_step_deg", "boundary_marker_step_ratio"])}
  <h3>Top source frames</h3>
  {_html_table(top_source_frames, ["frame", "absolute_frame", "source_event_count", "source_keypoint_max_interp_error_name", "source_keypoint_max_interp_error_mm", "source_keypoint_temporal_score", "marker_rmse_mm", "marker_max_residual_name", "marker_max_residual_mm"])}
</section>

<section>
  <h2>Scope, data, and metric definitions</h2>
  <p>Diagnostics are computed from the exported IK HDF5 arrays: qpos, marker_sites, kp_data, qvel when present, keypoint names, and the STAC config. Robust adaptive thresholds use each run's own finite values; absolute references remain visible but do not alone determine the verdict.</p>
  {_html_table(diagnostics["metric_definition_rows"], ["metric_name", "unit", "domain", "description", "interpretation"])}
</section>

<section>
  <h2>Methodology and research basis</h2>
  <p>The report uses RMS and max marker residuals for IK fit, MPJPE-like mean residuals for pose-estimation comparability, robust temporal keypoint outlier scores for source-data quality, fitted-marker temporal roughness for IK artifacts, and chunk-boundary ratios for streaming artifacts. Adaptive event thresholds use robust center plus scale and the observed upper tail; score metrics use robust z thresholds.</p>
  {_html_table(diagnostics["source_rows"], ["title", "applies_to", "url"])}
</section>

<section>
  <h2>Limitations, uncertainty, and robustness checks</h2>
  <p>No ground-truth 3D labels or pose-estimator confidence values are available in this HDF5, so source keypoint quality is inferred from temporal self-consistency rather than direct label accuracy. Uniformly bad fits can look internally consistent, which is why the report keeps absolute and scale-normalized fit errors beside adaptive thresholds.</p>
</section>

<section>
  <h2>Recommended next steps</h2>
  {next_steps}
</section>
</main>
</body>
</html>"""


def _summary_cards(
    metadata: dict[str, Any],
    scores: dict[str, dict[str, Any]],
    top_keypoints: list[dict[str, Any]],
) -> str:
    worst_keypoint = top_keypoints[0] if top_keypoints else {}
    card_rows = [
        (
            "Trajectory",
            f"{metadata['n_frames']} x {metadata['n_keypoints']}",
            "frames x keypoints",
        ),
        (
            "Marker RMSE p95",
            _fmt(scores.get("marker_rmse_p95_mm", {}).get("value")),
            "mm",
        ),
        (
            "Scale-normalized p95",
            _fmt(scores.get("marker_rmse_p95_pct_scale", {}).get("value")),
            "% body scale",
        ),
        (
            "Fit event frames",
            _fmt(scores.get("fit_event_frame_fraction_pct", {}).get("value")),
            "% frames",
        ),
        (
            "Source event frames",
            _fmt(scores.get("source_event_frame_fraction_pct", {}).get("value")),
            "% frames",
        ),
        (
            "Top keypoint share",
            _fmt(worst_keypoint.get("residual_contribution_pct")),
            worst_keypoint.get("keypoint_name", "n/a"),
        ),
    ]
    return "\n".join(f"""<article class="card">
  <span>{_escape(label)}</span>
  <strong>{_escape(value)}</strong>
  <small>{_escape(detail)}</small>
</article>""" for label, value, detail in card_rows)


def _technical_summary(
    metadata: dict[str, Any],
    scores: dict[str, dict[str, Any]],
    top_keypoints: list[dict[str, Any]],
) -> str:
    marker_mm = scores.get("marker_rmse_p95_mm", {}).get("value")
    marker_pct = scores.get("marker_rmse_p95_pct_scale", {}).get("value")
    fit_fraction = scores.get("fit_event_frame_fraction_pct", {}).get("value")
    source_fraction = scores.get("source_event_frame_fraction_pct", {}).get("value")
    source_interp_pct = scores.get("source_interp_rmse_p95_pct_scale", {}).get("value")
    overlap = scores.get("fit_source_overlap_pct", {}).get("value")
    boundary = scores.get("max_boundary_step_ratio", {}).get("value")
    top = top_keypoints[0] if top_keypoints else {}
    return f"""<ul>
  <li><strong>Overall verdict: {_escape(metadata["verdict"])}.</strong> The primary driver is {_escape(metadata["primary_driver"])}.</li>
  <li><strong>Fit accuracy:</strong> marker RMSE p95 is {_fmt(marker_mm)} mm ({_fmt(marker_pct)}% of robust pose scale), with {_fmt(fit_fraction)}% of frames covered by high-severity fit events.</li>
  <li><strong>Source keypoints:</strong> p95 temporal interpolation error is {_fmt(source_interp_pct)}% of pose scale, and {_fmt(source_fraction)}% of frames have high-severity source temporal events; {_fmt(overlap)}% of high-fit-error frames overlap those source events.</li>
  <li><strong>Where to focus:</strong> the top residual contributor is {_escape(top.get("keypoint_name", "n/a"))}, carrying {_fmt(top.get("residual_contribution_pct"))}% of residual energy. The largest chunk-boundary amplification ratio is {_fmt(boundary)}.</li>
</ul>"""


def _fit_interpretation(
    scores: dict[str, dict[str, Any]], top_keypoints: list[dict[str, Any]]
) -> str:
    marker = scores.get("marker_rmse_p95_mm", {})
    marker_pct = scores.get("marker_rmse_p95_pct_scale", {})
    top_names = ", ".join(row["keypoint_name"] for row in top_keypoints[:3])
    return (
        f"Marker RMSE p95 is {_fmt(marker.get('value'))} mm "
        f"({_fmt(marker_pct.get('value'))}% of pose scale). The highest residual "
        f"concentration is in {top_names or 'n/a'}, so visual review should start "
        "there before changing global solver settings."
    )


def _source_interpretation(
    scores: dict[str, dict[str, Any]], source_events: list[dict[str, Any]]
) -> str:
    source_fraction = scores.get("source_event_frame_fraction_pct", {}).get("value")
    source_interp_pct = scores.get("source_interp_rmse_p95_pct_scale", {}).get("value")
    overlap = scores.get("fit_source_overlap_pct", {}).get("value")
    if source_events:
        driver = _event_driver(source_events)
        return (
            f"Source keypoint temporal interpolation p95 is {_fmt(source_interp_pct)}% of pose scale, and high-severity outliers cover {_fmt(source_fraction)}% of frames. "
            f"{_fmt(overlap)}% of high-fit-error frames overlap source outliers, "
            f"so keypoint cleanup is most relevant near {driver}."
        )
    return (
        f"Source keypoint temporal interpolation p95 is {_fmt(source_interp_pct)}% of pose scale, and high-severity outliers cover {_fmt(source_fraction)}% of frames. "
        "The fit problems are not primarily explained by obvious keypoint jumps."
    )


def _continuity_interpretation(
    scores: dict[str, dict[str, Any]], fit_events: list[dict[str, Any]]
) -> str:
    boundary = scores.get("max_boundary_step_ratio", {}).get("value")
    boundary_status = scores.get("max_boundary_step_ratio", {}).get("status")
    boundary_text = (
        f"The maximum chunk-boundary ratio is {_fmt(boundary)} ({boundary_status})."
    )
    boundary_events = [
        event for event in fit_events if event["event_category"] == "chunk_boundary"
    ]
    if boundary_events:
        return f"{boundary_text} Boundary events are present; inspect context overlap, warm starts, and stitching at the listed frames."
    return f"{boundary_text} No high-severity chunk-boundary events were detected."


def _next_steps(
    metadata: dict[str, Any],
    scores: dict[str, dict[str, Any]],
    top_keypoints: list[dict[str, Any]],
) -> str:
    top = top_keypoints[0] if top_keypoints else {}
    source_fraction = scores.get("source_event_frame_fraction_pct", {}).get("value")
    overlap = scores.get("fit_source_overlap_pct", {}).get("value")
    boundary = scores.get("max_boundary_step_ratio", {}).get("value")
    items = []
    if (
        np.isfinite(source_fraction)
        and source_fraction > 0.5
        and np.isfinite(overlap)
        and overlap >= 25.0
    ):
        items.append(
            f"Review source keypoint tracks around the source event frames, starting with {top.get('keypoint_name', 'the top residual keypoint')}."
        )
    if np.isfinite(boundary) and boundary >= BOUNDARY_RATIO_EVENT_THRESHOLD:
        items.append(
            "Rerun or inspect the same slice with different context-frame and chunk-size settings to confirm boundary sensitivity."
        )
    if metadata["verdict"] != "pass":
        items.append(
            "Inspect the top fit frames visually, then decide whether marker offsets, keypoint weights, or root/trunk initialization should change."
        )
    if not items:
        items.append(
            "Use this slice as a baseline and compare against offset slices before changing fitting parameters."
        )
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _top_frame_rows(
    frames: dict[str, Any], limit: int, event_prefix: str = ""
) -> list[dict[str, Any]]:
    n_frames = len(frames["frame"])
    count_key = f"{event_prefix}event_count"
    score = (
        np.nan_to_num(frames[count_key], nan=0.0) * 1e6
        + np.nan_to_num(frames["marker_rmse_mm"], nan=0.0)
        + np.nan_to_num(frames["marker_max_residual_mm"], nan=0.0)
        + np.nan_to_num(frames["source_keypoint_temporal_score"], nan=0.0)
        + np.nan_to_num(frames["fit_marker_temporal_score"], nan=0.0)
    )
    indexes = np.argsort(score)[::-1]
    indexes = indexes[np.asarray(frames[count_key])[indexes] > 0]
    indexes = indexes[: min(limit, n_frames)]
    rows = []
    for index in indexes:
        rows.append(
            {
                "frame": int(frames["frame"][index]),
                "absolute_frame": int(frames["absolute_frame"][index]),
                count_key: int(frames[count_key][index]),
                "marker_rmse_mm": float(frames["marker_rmse_mm"][index]),
                "marker_max_residual_name": str(
                    frames["marker_max_residual_name"][index]
                ),
                "marker_max_residual_mm": float(
                    frames["marker_max_residual_mm"][index]
                ),
                "source_keypoint_temporal_score": float(
                    frames["source_keypoint_temporal_score"][index]
                ),
                "source_keypoint_max_interp_error_name": str(
                    frames["source_keypoint_max_interp_error_name"][index]
                ),
                "source_keypoint_max_interp_error_mm": float(
                    frames["source_keypoint_max_interp_error_mm"][index]
                ),
                "fit_marker_temporal_score": float(
                    frames["fit_marker_temporal_score"][index]
                ),
                "root_pos_step_mm": float(frames["root_pos_step_mm"][index]),
                "root_geodesic_step_deg": float(
                    frames["root_geodesic_step_deg"][index]
                ),
                "boundary_marker_step_ratio": float(
                    frames["boundary_marker_step_ratio"][index]
                ),
            }
        )
    return rows


def _plot_fit_source_timeline(
    frames: dict[str, Any],
    summary_by_metric: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    _style_figure(fig)
    x = np.asarray(frames["frame"])
    _plot_series(
        axes[0],
        x,
        frames["marker_rmse_mm"],
        "Marker RMSE (mm)",
        "marker_rmse_mm",
        summary_by_metric,
        events,
        "Fit residual trend",
        "Per-frame RMS residual; shaded regions are high-severity events.",
    )
    _plot_series(
        axes[1],
        x,
        frames["source_keypoint_temporal_score"],
        "Source temporal score",
        "source_keypoint_temporal_score",
        summary_by_metric,
        events,
        "",
        "",
    )
    _plot_series(
        axes[2],
        x,
        frames["fit_marker_temporal_score"],
        "Fit temporal score",
        "fit_marker_temporal_score",
        summary_by_metric,
        events,
        "",
        "",
    )
    axes[2].set_xlabel("Frame")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _figure_data_uri(fig)


def _plot_continuity_timeline(
    frames: dict[str, Any],
    summary_by_metric: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    _style_figure(fig)
    x = np.asarray(frames["frame"])
    _plot_series(
        axes[0],
        x,
        frames["root_pos_step_mm"],
        "Root position step (mm/frame)",
        "root_pos_step_mm",
        summary_by_metric,
        events,
        "Root and fitted-marker continuity",
        "Root translation, root orientation, and fitted-marker step highlight IK continuity failures.",
    )
    _plot_series(
        axes[1],
        x,
        frames["root_geodesic_step_deg"],
        "Root orientation step (deg/frame)",
        "root_geodesic_step_deg",
        summary_by_metric,
        events,
        "",
        "",
    )
    _plot_series(
        axes[2],
        x,
        frames["marker_step_rmse_mm"],
        "Fitted marker step (mm/frame)",
        "marker_step_rmse_mm",
        summary_by_metric,
        events,
        "",
        "",
    )
    axes[2].set_xlabel("Frame")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _figure_data_uri(fig)


def _plot_keypoint_residuals(rows: list[dict[str, Any]]) -> str:
    rows = rows[: min(18, len(rows))]
    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.36)))
    _style_figure(fig)
    _style_axes(ax)
    if rows:
        names = [row["keypoint_name"] for row in rows][::-1]
        p95 = [row["p95_residual_mm"] for row in rows][::-1]
        share = [row["residual_contribution_pct"] for row in rows][::-1]
        y = np.arange(len(rows))
        ax.barh(
            y - 0.18,
            p95,
            height=0.35,
            label="p95 residual (mm)",
            color=PLOT_COLORS["p95_bar"],
            edgecolor="#2e4780",
            linewidth=0.8,
            alpha=0.95,
        )
        ax2 = ax.twiny()
        ax2.barh(
            y + 0.18,
            share,
            height=0.35,
            label="residual share (%)",
            color=PLOT_COLORS["share_bar"],
            edgecolor="#386411",
            linewidth=0.8,
            alpha=0.82,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("p95 residual (mm)")
        ax2.set_xlabel("Residual energy share (%)")
        ax.legend(loc="lower right", frameon=True, fontsize=8)
        ax2.legend(loc="upper right", frameon=True, fontsize=8)
        _style_axes(ax2)
        ax2.grid(False)
    _add_chart_header(
        fig,
        ax,
        "Keypoints to inspect first",
        "Ranked by residual contribution; p95 residual shows practical fit size.",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return _figure_data_uri(fig)


def _plot_boundary_comparison(frames: dict[str, Any]) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    _style_figure(fig)
    items = [
        ("marker_step_rmse_mm", "Fitted marker step", "mm/frame"),
        ("root_pos_step_mm", "Root position step", "mm/frame"),
        ("root_geodesic_step_deg", "Root orientation step", "deg/frame"),
    ]
    is_boundary = np.asarray(frames["is_chunk_boundary"], dtype=bool)
    for ax, (key, title, unit) in zip(axes, items):
        _style_axes(ax)
        values = np.asarray(frames[key], dtype=float)
        boundary = values[is_boundary & np.isfinite(values)]
        interior = values[(~is_boundary) & np.isfinite(values)]
        if boundary.size or interior.size:
            box = ax.boxplot(
                [interior, boundary],
                labels=["interior", "boundary"],
                showfliers=False,
                patch_artist=True,
            )
            for patch, color, edge in zip(
                box["boxes"],
                [
                    PLOT_COLORS["boundary_interior"],
                    PLOT_COLORS["boundary_chunk"],
                ],
                ["#2e4780", "#804126"],
            ):
                patch.set_facecolor(color)
                patch.set_alpha(0.9)
                patch.set_edgecolor(edge)
                patch.set_linewidth(0.8)
            for median in box["medians"]:
                median.set_color(PLOT_COLORS["text"])
                median.set_linewidth(1.1)
        ax.set_title(title)
        ax.set_ylabel(unit)
    _add_chart_header(
        fig,
        axes[0],
        "Chunk boundary amplification",
        "Boundary frames are compared with interior frames; large boundary boxes indicate stitching artifacts.",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return _figure_data_uri(fig)


def _plot_fit_distribution(
    frames: dict[str, Any], summary_by_metric: dict[str, dict[str, Any]]
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _style_figure(fig)
    items = [
        ("marker_rmse_mm", "Marker RMSE (mm)"),
        ("marker_rmse_pct_scale", "Marker RMSE (% body scale)"),
    ]
    for ax, (key, title) in zip(axes, items):
        _style_axes(ax)
        values = np.asarray(frames[key], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(
                values,
                bins=60,
                color=PLOT_COLORS["fit"],
                edgecolor=PLOT_COLORS["hist_edge"],
                linewidth=0.35,
                alpha=0.88,
            )
            threshold = summary_by_metric.get(key, {}).get("threshold", np.nan)
            if np.isfinite(threshold):
                ax.axvline(
                    threshold,
                    color=PLOT_COLORS["threshold"],
                    linestyle="--",
                    linewidth=1.1,
                )
        ax.set_title(title)
    _add_chart_header(
        fig,
        axes[0],
        "Fit error distribution",
        "Absolute and scale-normalized distributions catch uniformly bad fits that adaptive event thresholds can miss.",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return _figure_data_uri(fig)


def _plot_series(
    ax,
    x: np.ndarray,
    y: Any,
    label: str,
    metric_name: str,
    summary_by_metric: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    title: str,
    subtitle: str,
) -> None:
    y = np.asarray(y, dtype=float)
    x_plot, y_plot = _downsample(x, y)
    _style_axes(ax)
    for event in events:
        if event["metric_name"] != metric_name:
            continue
        ax.axvspan(
            event["start_frame"],
            event["end_frame"],
            color=PLOT_COLORS["event"],
            alpha=0.14,
            linewidth=0,
            zorder=0,
        )
    ax.plot(
        x_plot,
        y_plot,
        color=_metric_color(metric_name),
        linewidth=1.0,
        zorder=3,
    )
    ax.set_ylabel(label)
    summary = summary_by_metric.get(metric_name, {})
    threshold = summary.get("threshold", np.nan)
    if np.isfinite(threshold):
        threshold_line = ax.axhline(
            threshold,
            color=PLOT_COLORS["threshold"],
            linestyle="--",
            linewidth=1.0,
            label="adaptive event threshold",
            zorder=2,
        )
        ax.legend(handles=[threshold_line], loc="upper right", frameon=True, fontsize=8)
    if title and subtitle:
        _add_chart_header(fig=ax.figure, ax=ax, title=title, subtitle=subtitle)


def _metric_color(metric_name: str) -> str:
    if metric_name.startswith("source_keypoint_"):
        return PLOT_COLORS["source"]
    if metric_name.startswith("fit_marker_") or metric_name == "marker_step_rmse_mm":
        return PLOT_COLORS["motion"]
    if metric_name.startswith("root_"):
        return PLOT_COLORS["root"]
    if metric_name.startswith("boundary_"):
        return PLOT_COLORS["boundary"]
    return PLOT_COLORS["fit"]


def _style_figure(fig) -> None:
    fig.patch.set_facecolor(PLOT_COLORS["figure_bg"])


def _style_axes(ax) -> None:
    ax.set_facecolor(PLOT_COLORS["axes_bg"])
    ax.grid(True, color=PLOT_COLORS["grid"], alpha=0.8, linewidth=0.6)
    ax.tick_params(colors=PLOT_COLORS["muted"], labelsize=9)
    ax.xaxis.label.set_color(PLOT_COLORS["text"])
    ax.yaxis.label.set_color(PLOT_COLORS["text"])
    ax.title.set_color(PLOT_COLORS["text"])
    for side, spine in ax.spines.items():
        spine.set_color(PLOT_COLORS["spine"])
        spine.set_linewidth(0.8)
        if side in {"top", "right"}:
            spine.set_visible(False)


def _add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=PLOT_COLORS["text"],
    )
    fig.text(
        left,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=PLOT_COLORS["muted"],
    )


def _downsample(
    x: np.ndarray, y: np.ndarray, max_points: int = 20000
) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] <= max_points:
        return x, y
    indexes = np.linspace(0, x.shape[0] - 1, max_points).astype(int)
    return x[indexes], y[indexes]


def _figure_data_uri(fig) -> str:
    buffer = pyio.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image(src: str, alt: str) -> str:
    return f'<img class="plot" src="{src}" alt="{_escape(alt)}">'


def _html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return '<p class="empty">No rows.</p>'
    header = "".join(f"<th>{_escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{_escape(_fmt(row.get(column)))}</td>" for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"""<div class="table-wrap"><table>
<thead><tr>{header}</tr></thead>
<tbody>{''.join(body_rows)}</tbody>
</table></div>"""


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return "n/a"
        abs_value = abs(float(value))
        if abs_value >= 1000.0:
            return f"{float(value):,.1f}"
        if abs_value >= 10.0:
            return f"{float(value):.2f}"
        if abs_value >= 1.0:
            return f"{float(value):.3f}"
        return f"{float(value):.4g}"
    return str(value)


def _report_css() -> str:
    return """
:root {
  color-scheme: light;
  --surface: #f6f8fb;
  --panel: #ffffff;
  --ink: #1f2430;
  --muted: #6f768a;
  --line: #d7dbe7;
  --blue: #5477c4;
  --gold: #b8a037;
  --orange: #cc6f47;
  --olive: #71b436;
  --review: #b8a037;
  --poor: #cc6f47;
  --pass: #386411;
}
body {
  margin: 0;
  background: var(--surface);
  color: var(--ink);
  font-family: Aptos, Inter, Segoe UI, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.5;
}
main {
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px 24px 56px;
}
.report-header {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: flex-start;
  margin-bottom: 18px;
}
.eyebrow {
  margin: 0 0 8px;
  color: var(--blue);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}
h1 {
  margin: 0;
  font-size: 30px;
  line-height: 1.15;
}
h2 {
  margin: 0 0 12px;
  font-size: 20px;
}
h3 {
  margin: 16px 0 8px;
  font-size: 15px;
}
p {
  margin: 0 0 12px;
}
.muted {
  color: var(--muted);
}
.verdict {
  min-width: 230px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
}
.verdict span, .card span {
  display: block;
  color: var(--muted);
  font-size: 12px;
}
.verdict strong {
  display: block;
  margin: 4px 0;
  font-size: 26px;
  text-transform: uppercase;
}
.verdict-pass strong { color: var(--pass); }
.verdict-review strong { color: var(--review); }
.verdict-poor strong { color: var(--poor); }
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 8px;
  margin-bottom: 14px;
  background: transparent;
  border: 0;
  padding: 0;
}
.card {
  background: #fbfcfd;
  border: 1px solid #e6e8f0;
  border-radius: 6px;
  padding: 10px 12px;
}
.card strong {
  display: block;
  margin: 2px 0;
  font-size: 20px;
  line-height: 1.15;
  color: var(--ink);
}
.card small {
  color: var(--muted);
  font-size: 11px;
}
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  margin: 14px 0;
}
.summary ul, section ul {
  margin: 0;
  padding-left: 20px;
}
.summary li, section li {
  margin: 6px 0;
}
.plot {
  width: 100%;
  height: auto;
  display: block;
  margin: 8px 0 12px;
}
.table-wrap {
  max-height: 560px;
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 7px 9px;
  text-align: left;
  white-space: nowrap;
}
th {
  position: sticky;
  top: 0;
  background: #f0f3f8;
  z-index: 1;
}
.empty {
  color: var(--muted);
}
@media (max-width: 820px) {
  main {
    padding: 22px 14px 42px;
  }
  .report-header {
    display: block;
  }
  .verdict {
    margin-top: 16px;
  }
  th, td {
    white-space: normal;
  }
}
"""
