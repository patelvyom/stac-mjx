"""User-level API to run stac."""

from jax import Array
import time
from omegaconf import DictConfig
from stac_mjx import diagnostics, io, utils
from stac_mjx.config import compose_config
from stac_mjx.stac import Stac
from pathlib import Path

from jaxtyping import Float


def load_stac_config(
    config_dir: Path | str,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load and validate a STAC config from a Hydra config directory.

    Args:
        config_dir: Absolute path to config directory.
        config_name: Name of the Hydra config to load.
        overrides: Optional Hydra override list.

    Returns:
        Validated STAC configuration.
    """
    cfg = compose_config(config_dir, config_name=config_name, overrides=overrides)
    print("Config loaded and validated.")
    return cfg


def run_stac(
    cfg: DictConfig,
    kp_data: Float[Array, "n_frames n_keypoints_xyz"],
    kp_names: list[str],
    base_path: Path | None = None,
) -> tuple[str, str | None]:
    """Run the full skeletal registration pipeline.

    Runs calibration (unless skipped), then IK using the production jaxls
    q optimization path (unless skipped), optionally infers velocities,
    and saves results to HDF5.

    Args:
        cfg: STAC configuration.
        kp_data: Flattened mocap keypoint data.
        kp_names: Ordered keypoint names matching kp_data columns.
        base_path: Base path for resolving relative file paths. Defaults to cwd.

    Returns:
        Tuple of (calibration output path, IK output path or None).

    Raises:
        ValueError: If kp_data columns don't match kp_names * 3.
    """
    if base_path is None:
        base_path = Path.cwd()

    expected_cols = len(kp_names) * 3
    if kp_data.shape[1] != expected_cols:
        raise ValueError(
            f"kp_data has {kp_data.shape[1]} columns but expected {expected_cols} "
            f"({len(kp_names)} keypoints × 3). "
            f"Ensure kp_data is shaped (n_frames, n_keypoints * 3) and that "
            f"kp_names length matches the number of keypoints in kp_data."
        )

    start_time = time.time()

    calibration_path = base_path / cfg.stac.calibration_path
    ik_path = base_path / cfg.stac.ik_path
    xml_path = base_path / cfg.model.MJCF_PATH
    stac = Stac(xml_path, cfg, kp_names)

    if not cfg.stac.skip_calibration:
        kps = kp_data[: cfg.stac.n_calibration_frames]
        print(f"Running calibration. Mocap data shape: {kps.shape}")
        calibration_data = stac.calibrate(kps)
        print(f"saving data to {calibration_path}", flush=True)
        io.save_data_to_h5(
            config=cfg, file_path=calibration_path, **calibration_data.as_dict()
        )
        _write_diagnostics_if_enabled(cfg, calibration_data, calibration_path)
    else:
        print(
            "Skipping calibration. To change this behavior, set cfg.stac.skip_calibration to False."
        )

    if cfg.stac.skip_ik:
        print(
            "Skipping IK phase. To change this behavior, set cfg.stac.skip_ik to False."
        )
        return calibration_path, None

    print("Running IK")
    _, calibration_data = io.load_stac_data(calibration_path)

    offsets = calibration_data.offsets

    print(f"kp_data shape: {kp_data.shape}")
    ik_data = stac.run_ik(kp_data, offsets)

    print(f"Final qpos shape: {ik_data.qpos.shape}")
    if cfg.stac.infer_qvels:
        t_vel = time.time()
        qvels = utils.compute_velocity_from_kinematics(
            qpos_trajectory=ik_data.qpos,
            dt=stac._mj_model.opt.timestep,
            freejoint=stac._freejoint,
        )
        ik_data.qvel = qvels
        print(f"Finished compute velocity in {time.time() - t_vel} seconds")

    print(
        f"Saving data to {ik_path}. Finished in {(time.time() - start_time)/60:.2f} minutes"
    )
    io.save_data_to_h5(config=cfg, file_path=ik_path, **ik_data.as_dict())
    diagnostic_data = _write_diagnostics_if_enabled(cfg, ik_data, ik_path)
    _render_diagnostics_overlay_if_enabled(
        cfg,
        stac,
        ik_data,
        ik_path,
        diagnostic_data,
        base_path,
    )
    return calibration_path, ik_path


def _write_diagnostics_if_enabled(
    cfg: DictConfig, data: io.StacData, path: Path
) -> dict | None:
    diagnostics_cfg = getattr(getattr(cfg, "stac", None), "diagnostics", None)
    if diagnostics_cfg is None or not getattr(diagnostics_cfg, "enabled", True):
        return None
    result = diagnostics.compute_stac_diagnostics(
        data,
        cfg,
        source_h5=path,
        write=True,
    )
    print(f"Diagnostics saved to {result['output_paths']['summary']}", flush=True)
    return result


def _render_diagnostics_overlay_if_enabled(
    cfg: DictConfig,
    stac: Stac,
    data: io.StacData,
    ik_path: Path,
    diagnostic_data: dict | None,
    base_path: Path,
) -> None:
    diagnostics_cfg = getattr(getattr(cfg, "stac", None), "diagnostics", None)
    if diagnostics_cfg is None or not getattr(diagnostics_cfg, "render_overlay", False):
        return
    if diagnostic_data is None:
        diagnostic_data = diagnostics.compute_stac_diagnostics(data, cfg, source_h5=ik_path)

    overlay_path = getattr(diagnostics_cfg, "overlay_path", None)
    if overlay_path is None:
        overlay_path = ik_path.with_name(f"{ik_path.stem}_diagnostic_overlay.mp4")
    else:
        overlay_path = base_path / overlay_path
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    start_frame = int(getattr(diagnostics_cfg, "overlay_start_frame", 0) or 0)
    n_frames = getattr(diagnostics_cfg, "overlay_n_frames", None)
    if n_frames is None:
        n_frames = int(data.qpos.shape[0]) - start_frame
    n_frames = min(int(n_frames), int(data.qpos.shape[0]) - start_frame)
    if n_frames <= 0:
        raise ValueError("Diagnostics overlay requested with no frames to render")

    stac.render(
        data.qpos,
        data.kp_data,
        data.offsets,
        n_frames=n_frames,
        save_path=overlay_path,
        start_frame=start_frame,
        height=int(getattr(diagnostics_cfg, "overlay_height", 480) or 480),
        width=int(getattr(diagnostics_cfg, "overlay_width", 640) or 640),
        show_marker_error=True,
        diagnostics=diagnostic_data,
    )
    print(f"Diagnostics overlay saved to {overlay_path}", flush=True)
