import bisect
import io
import json
import subprocess
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import click
import imageio.v2 as imageio_v2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    from skimage import measure
except Exception:  # pragma: no cover - map raster conversion is best-effort.
    measure = None

try:
    from termcolor import cprint
except Exception:  # pragma: no cover - keep the converter runnable in lean envs.
    def cprint(text, color=None, attrs=None):
        print(text)


NuScenesToRdsCamera = {
    "front": "CAM_FRONT",
    "front_left": "CAM_FRONT_LEFT",
    "front_right": "CAM_FRONT_RIGHT",
    "side_left": "CAM_BACK_LEFT",
    "side_right": "CAM_BACK_RIGHT",
}

SourceFps = 20
TargetFps = 20
IndexScaleRatio = int(TargetFps / SourceFps)
LidarChannel = "LIDAR_TOP"
MapMaskResolution = 0.1


class FFMpegRawVideoWriter:
    def __init__(self, output_path: Union[str, Path], fps: int):
        self.output_path = Path(output_path)
        self.fps = fps
        self.process: Optional[subprocess.Popen] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None

    def _start(self, frame: np.ndarray) -> None:
        self.height, self.width = frame.shape[:2]
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            self.output_path.as_posix(),
        ]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def append_data(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB uint8 frame with shape HxWx3, got {frame.shape}")
        frame = np.ascontiguousarray(frame)

        if self.process is None:
            self._start(frame)

        assert self.process is not None and self.process.stdin is not None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(
                f"Video frame size changed from {self.width}x{self.height} "
                f"to {frame.shape[1]}x{frame.shape[0]}"
            )
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process is None:
            return
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self.output_path}:\n{stderr}")


def _json_dumps(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def _encode_npy(data: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, data)
    return buffer.getvalue()


def encode_dict_to_npz_bytes(data_dict: Dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **data_dict)
    return buffer.getvalue()


def _encode_tar_value(key: str, value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if key.endswith(".npy"):
        return _encode_npy(np.asarray(value))
    if key.endswith(".json"):
        return _json_dumps(value)
    if key.endswith(".txt"):
        return str(value).encode("utf-8")
    raise ValueError(f"Unsupported tar member type for key: {key}")


def write_to_tar(sample: Dict[str, Any], output_file: Union[str, Path]) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sample_key = sample["__key__"]
    with tarfile.open(output_file, "w") as tar:
        for key, value in sample.items():
            if key == "__key__":
                continue
            member_name = f"{sample_key}.{key}"
            data = _encode_tar_value(key, value)
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    cprint(f"Saved {output_file}", "green")


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def transform_matrix(translation: Iterable[float], rotation_wxyz: Iterable[float]) -> np.ndarray:
    q = list(rotation_wxyz)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    mat[:3, 3] = np.asarray(translation, dtype=np.float64)
    return mat


def nearest_by_timestamp(records: List[Dict[str, Any]], timestamp: int) -> Dict[str, Any]:
    if not records:
        raise ValueError("nearest_by_timestamp() received an empty record list")

    timestamps = [record["timestamp"] for record in records]
    pos = bisect.bisect_left(timestamps, timestamp)
    if pos == 0:
        return records[0]
    if pos == len(records):
        return records[-1]
    before = records[pos - 1]
    after = records[pos]
    if abs(before["timestamp"] - timestamp) <= abs(after["timestamp"] - timestamp):
        return before
    return after


class NuScenesTables:
    def __init__(self, dataroot: Union[str, Path], version: str = "v1.0-mini"):
        self.dataroot = Path(dataroot)
        self.version = version
        self.table_root = self.dataroot / version
        if not self.table_root.exists():
            raise FileNotFoundError(f"nuScenes table folder not found: {self.table_root}")

        self.scene = load_json(self.table_root / "scene.json")
        self.sample = load_json(self.table_root / "sample.json")
        self.sample_data = load_json(self.table_root / "sample_data.json")
        self.calibrated_sensor = load_json(self.table_root / "calibrated_sensor.json")
        self.sensor = load_json(self.table_root / "sensor.json")
        self.ego_pose = load_json(self.table_root / "ego_pose.json")
        self.sample_annotation = load_json(self.table_root / "sample_annotation.json")
        self.instance = load_json(self.table_root / "instance.json")
        self.category = load_json(self.table_root / "category.json")
        self.attribute = load_json(self.table_root / "attribute.json")
        self.log = load_json(self.table_root / "log.json")
        self.map = load_json(self.table_root / "map.json")

        self.sample_by_token = {item["token"]: item for item in self.sample}
        self.calibrated_sensor_by_token = {item["token"]: item for item in self.calibrated_sensor}
        self.sensor_by_token = {item["token"]: item for item in self.sensor}
        self.ego_pose_by_token = {item["token"]: item for item in self.ego_pose}
        self.instance_by_token = {item["token"]: item for item in self.instance}
        self.category_by_token = {item["token"]: item for item in self.category}
        self.attribute_by_token = {item["token"]: item for item in self.attribute}
        self.log_by_token = {item["token"]: item for item in self.log}

        self.sample_token_to_annotations: Dict[str, List[Dict[str, Any]]] = {}
        self.annotation_by_token: Dict[str, Dict[str, Any]] = {}
        for annotation in self.sample_annotation:
            self.annotation_by_token[annotation["token"]] = annotation
            self.sample_token_to_annotations.setdefault(annotation["sample_token"], []).append(annotation)

        self.sample_token_to_data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for sample_data in self.sample_data:
            channel = self.get_channel(sample_data)
            self.sample_token_to_data.setdefault(sample_data["sample_token"], {}).setdefault(channel, []).append(sample_data)

        for channel_to_records in self.sample_token_to_data.values():
            for records in channel_to_records.values():
                records.sort(key=lambda item: item["timestamp"])

    def get_channel(self, sample_data: Dict[str, Any]) -> str:
        calibrated = self.calibrated_sensor_by_token[sample_data["calibrated_sensor_token"]]
        sensor = self.sensor_by_token[calibrated["sensor_token"]]
        return sensor["channel"]

    def get_sample_tokens_for_scene(self, scene_record: Dict[str, Any]) -> List[str]:
        tokens = []
        token = scene_record["first_sample_token"]
        while token:
            tokens.append(token)
            token = self.sample_by_token[token]["next"]
        return tokens

    def get_sample_data_for_scene(self, scene_record: Dict[str, Any], channel: str) -> List[Dict[str, Any]]:
        records = []
        sample_tokens = set(self.get_sample_tokens_for_scene(scene_record))
        for sample_token in sample_tokens:
            records.extend(self.sample_token_to_data.get(sample_token, {}).get(channel, []))
        records.sort(key=lambda item: item["timestamp"])
        return records

    def get_sensor_to_ego(self, sample_data: Dict[str, Any]) -> np.ndarray:
        calibrated = self.calibrated_sensor_by_token[sample_data["calibrated_sensor_token"]]
        return transform_matrix(calibrated["translation"], calibrated["rotation"])

    def get_ego_to_world(self, sample_data: Dict[str, Any]) -> np.ndarray:
        ego_pose = self.ego_pose_by_token[sample_data["ego_pose_token"]]
        return transform_matrix(ego_pose["translation"], ego_pose["rotation"])

    def get_sensor_to_world(self, sample_data: Dict[str, Any]) -> np.ndarray:
        return self.get_ego_to_world(sample_data) @ self.get_sensor_to_ego(sample_data)

    def get_scene_location(self, scene_record: Dict[str, Any]) -> str:
        log_record = self.log_by_token[scene_record["log_token"]]
        return log_record["location"]

    def get_scene_map_path(self, scene_record: Dict[str, Any]) -> Optional[Path]:
        log_token = scene_record["log_token"]
        for map_record in self.map:
            if log_token in map_record["log_tokens"]:
                return self.dataroot / map_record["filename"]
        return None


def category_to_object_type(category_name: str) -> Optional[str]:
    if category_name == "vehicle.car":
        return "Car"
    if category_name.startswith("vehicle.truck") or category_name.startswith("vehicle.bus") or \
            category_name in {"vehicle.trailer", "vehicle.construction"}:
        return "Truck"
    if category_name.startswith("human.pedestrian"):
        return "Pedestrian"
    if category_name in {"vehicle.bicycle", "vehicle.motorcycle"}:
        return "Cyclist"
    if category_name.startswith("movable_object"):
        return "Others"
    return None


def annotation_is_moving(tables: NuScenesTables, annotation: Dict[str, Any], min_moving_speed: float = 0.2) -> bool:
    attribute_names = [
        tables.attribute_by_token[token]["name"]
        for token in annotation.get("attribute_tokens", [])
        if token in tables.attribute_by_token
    ]
    if any(name.endswith(".moving") for name in attribute_names):
        return True
    if any(name.endswith(".stopped") or name.endswith(".parked") or name.endswith(".standing") for name in attribute_names):
        return False

    prev_token = annotation.get("prev", "")
    next_token = annotation.get("next", "")
    if prev_token and next_token and prev_token in tables.annotation_by_token and next_token in tables.annotation_by_token:
        prev_ann = tables.annotation_by_token[prev_token]
        next_ann = tables.annotation_by_token[next_token]
        prev_sample = tables.sample_by_token[prev_ann["sample_token"]]
        next_sample = tables.sample_by_token[next_ann["sample_token"]]
        dt = (next_sample["timestamp"] - prev_sample["timestamp"]) / 1e6
        if dt > 0:
            displacement = np.linalg.norm(np.asarray(next_ann["translation"]) - np.asarray(prev_ann["translation"]))
            return bool(displacement / dt > min_moving_speed)

    return False


def make_minimap_sample(clip_id: str, minimap_name: str, labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "__key__": clip_id,
        f"{minimap_name}.json": {
            "labels": labels,
        },
    }


def polyline_label(vertices: List[List[float]], color: str = "OTHER", style: str = "OTHER") -> Dict[str, Any]:
    return {
        "labelData": {
            "shape3d": {
                "polyline3d": {
                    "vertices": vertices,
                },
                "attributes": [
                    {"name": "colors", "enumsList": {"enumsList": [color]}},
                    {"name": "styles", "enumsList": {"enumsList": [style]}},
                ],
            }
        }
    }


def polygon_label(vertices: List[List[float]]) -> Dict[str, Any]:
    return {
        "labelData": {
            "shape3d": {
                "surface": {
                    "vertices": vertices,
                }
            }
        }
    }


def empty_minimap_labels(minimap_name: str) -> List[Dict[str, Any]]:
    if minimap_name in {"lanelines", "road_boundaries"}:
        return []
    if minimap_name == "crosswalks":
        return []
    return []


def contour_to_world_polyline(
    contour: np.ndarray,
    crop_min_px: int,
    crop_min_py: int,
    image_height: int,
    resolution: float,
    stride: int = 5,
) -> List[List[float]]:
    if len(contour) > 2:
        contour = contour[::stride]
    vertices = []
    for row, col in contour:
        px = crop_min_px + float(col)
        py = crop_min_py + float(row)
        vertices.append([px * resolution, (image_height - py) * resolution, 0.0])
    return vertices


def extract_road_boundaries_from_map(
    map_path: Path,
    ego_positions: np.ndarray,
    margin_m: float = 120.0,
    resolution: float = MapMaskResolution,
) -> List[Dict[str, Any]]:
    if measure is None:
        cprint("skimage is not available; writing empty nuScenes road boundaries.", "yellow")
        return []
    if not map_path or not map_path.exists():
        return []

    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(map_path)
    width, height = image.size

    min_x = max(0.0, float(np.min(ego_positions[:, 0]) - margin_m))
    max_x = min(width * resolution, float(np.max(ego_positions[:, 0]) + margin_m))
    min_y = max(0.0, float(np.min(ego_positions[:, 1]) - margin_m))
    max_y = min(height * resolution, float(np.max(ego_positions[:, 1]) + margin_m))

    crop_min_px = max(0, int(np.floor(min_x / resolution)))
    crop_max_px = min(width, int(np.ceil(max_x / resolution)))
    crop_min_py = max(0, int(np.floor(height - max_y / resolution)))
    crop_max_py = min(height, int(np.ceil(height - min_y / resolution)))

    if crop_max_px <= crop_min_px or crop_max_py <= crop_min_py:
        return []

    crop = image.crop((crop_min_px, crop_min_py, crop_max_px, crop_max_py))
    mask = np.asarray(crop) > 0
    if not np.any(mask):
        return []

    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    labels = []
    for contour in contours:
        if len(contour) < 20:
            continue
        simplified = measure.approximate_polygon(contour, tolerance=3.0)
        if len(simplified) < 2:
            continue
        vertices = contour_to_world_polyline(
            simplified,
            crop_min_px,
            crop_min_py,
            height,
            resolution,
            stride=1,
        )
        if len(vertices) >= 2:
            labels.append(polyline_label(vertices))

    return labels


def convert_nuscenes_hdmap(
    output_root: Path,
    clip_id: str,
    tables: NuScenesTables,
    scene_record: Dict[str, Any],
    target_lidar_records: List[Dict[str, Any]],
    map_margin: float,
    use_map_raster: bool,
) -> None:
    laneline_labels = empty_minimap_labels("lanelines")
    crosswalk_labels = empty_minimap_labels("crosswalks")
    road_boundary_labels: List[Dict[str, Any]] = []

    if use_map_raster:
        ego_positions = np.stack([
            tables.get_ego_to_world(lidar_record)[:3, 3]
            for lidar_record in target_lidar_records
        ])
        road_boundary_labels = extract_road_boundaries_from_map(
            tables.get_scene_map_path(scene_record),
            ego_positions,
            margin_m=map_margin,
        )

    write_to_tar(make_minimap_sample(clip_id, "lanelines", laneline_labels), output_root / "3d_lanelines" / f"{clip_id}.tar")
    write_to_tar(make_minimap_sample(clip_id, "road_boundaries", road_boundary_labels), output_root / "3d_road_boundaries" / f"{clip_id}.tar")
    write_to_tar(make_minimap_sample(clip_id, "crosswalks", crosswalk_labels), output_root / "3d_crosswalks" / f"{clip_id}.tar")


def convert_nuscenes_intrinsics(
    output_root: Path,
    clip_id: str,
    tables: NuScenesTables,
    scene_record: Dict[str, Any],
    camera_names: List[str],
) -> None:
    sample = {"__key__": clip_id}
    for camera_name in camera_names:
        channel = NuScenesToRdsCamera[camera_name]
        camera_records = tables.get_sample_data_for_scene(scene_record, channel)
        if not camera_records:
            raise FileNotFoundError(f"No sample_data found for camera {channel} in {clip_id}")
        camera_record = camera_records[0]
        calib = tables.calibrated_sensor_by_token[camera_record["calibrated_sensor_token"]]
        intrinsic = np.asarray(calib["camera_intrinsic"], dtype=np.float64)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        sample[f"pinhole_intrinsic.{camera_name}.npy"] = np.array(
            [fx, fy, cx, cy, camera_record["width"], camera_record["height"]],
            dtype=np.float32,
        )

    write_to_tar(sample, output_root / "pinhole_intrinsic" / f"{clip_id}.tar")


def convert_nuscenes_pose(
    output_root: Path,
    clip_id: str,
    tables: NuScenesTables,
    scene_record: Dict[str, Any],
    target_lidar_records: List[Dict[str, Any]],
    camera_names: List[str],
) -> None:
    sample_camera_to_world = {"__key__": clip_id}
    sample_vehicle_to_world = {"__key__": clip_id}

    camera_name_to_sensor_to_ego = {}
    for camera_name in camera_names:
        channel = NuScenesToRdsCamera[camera_name]
        camera_records = tables.get_sample_data_for_scene(scene_record, channel)
        if not camera_records:
            raise FileNotFoundError(f"No sample_data found for camera {channel} in {clip_id}")
        camera_name_to_sensor_to_ego[camera_name] = tables.get_sensor_to_ego(camera_records[0])

    for frame_idx, lidar_record in enumerate(target_lidar_records):
        target_frame_idx = frame_idx * IndexScaleRatio
        ego_to_world = tables.get_ego_to_world(lidar_record)
        sample_vehicle_to_world[f"{target_frame_idx:06d}.vehicle_pose.npy"] = ego_to_world.astype(np.float32)

        for camera_name in camera_names:
            camera_to_world = ego_to_world @ camera_name_to_sensor_to_ego[camera_name]
            sample_camera_to_world[f"{target_frame_idx:06d}.pose.{camera_name}.npy"] = camera_to_world.astype(np.float32)

    write_to_tar(sample_camera_to_world, output_root / "pose" / f"{clip_id}.tar")
    write_to_tar(sample_vehicle_to_world, output_root / "vehicle_pose" / f"{clip_id}.tar")


def convert_nuscenes_timestamp(
    output_root: Path,
    clip_id: str,
    target_lidar_records: List[Dict[str, Any]],
) -> None:
    sample = {"__key__": clip_id}
    for frame_idx, lidar_record in enumerate(target_lidar_records):
        sample[f"{frame_idx * IndexScaleRatio:06d}.timestamp_micros.txt"] = str(lidar_record["timestamp"])
    write_to_tar(sample, output_root / "timestamp" / f"{clip_id}.tar")


def convert_nuscenes_bbox(
    output_root: Path,
    clip_id: str,
    tables: NuScenesTables,
    scene_record: Dict[str, Any],
) -> None:
    sample = {"__key__": clip_id}
    sample_tokens = tables.get_sample_tokens_for_scene(scene_record)
    first_sample_timestamp = tables.sample_by_token[sample_tokens[0]]["timestamp"]

    for sample_token in sample_tokens:
        sample_record = tables.sample_by_token[sample_token]
        frame_idx = int(round((sample_record["timestamp"] - first_sample_timestamp) / 1e6 * TargetFps))
        object_info = {}

        annotations = tables.sample_token_to_annotations.get(sample_token, [])
        for annotation in annotations:
            instance = tables.instance_by_token[annotation["instance_token"]]
            category_name = tables.category_by_token[instance["category_token"]]["name"]
            object_type = category_to_object_type(category_name)
            if object_type is None:
                continue

            object_to_world = transform_matrix(annotation["translation"], annotation["rotation"])
            width, length, height = annotation["size"]
            object_lwh = np.array([length, width, height], dtype=np.float32)
            tracking_id = annotation["instance_token"]
            object_info[tracking_id] = {
                "object_to_world": object_to_world.astype(np.float32).tolist(),
                "object_lwh": object_lwh.tolist(),
                "object_is_moving": annotation_is_moving(tables, annotation),
                "object_type": object_type,
            }

        sample[f"{frame_idx:06d}.all_object_info.json"] = object_info

    write_to_tar(sample, output_root / "all_object_info" / f"{clip_id}.tar")


def load_lidar_points(lidar_path: Path) -> np.ndarray:
    points = np.fromfile(lidar_path, dtype=np.float32)
    if points.size % 5 != 0:
        raise ValueError(f"Unexpected nuScenes lidar point shape in {lidar_path}: {points.size} floats")
    return points.reshape(-1, 5)[:, :3]


def convert_nuscenes_lidar(
    output_root: Path,
    clip_id: str,
    tables: NuScenesTables,
    target_lidar_records: List[Dict[str, Any]],
) -> None:
    sample = {"__key__": clip_id}
    cprint("reading lidar and converting to RDS-HQ format, this may take a few minutes...", "yellow")

    for frame_idx, lidar_record in enumerate(target_lidar_records):
        lidar_path = tables.dataroot / lidar_record["filename"]
        if not lidar_path.exists():
            raise FileNotFoundError(f"nuScenes lidar file not found: {lidar_path}")
        lidar_points = load_lidar_points(lidar_path)
        lidar_to_world = tables.get_sensor_to_world(lidar_record)
        sample[f"{frame_idx * IndexScaleRatio:06d}.lidar_raw.npz"] = encode_dict_to_npz_bytes(
            {
                "xyz": lidar_points.astype(np.float32),
                "lidar_to_world": lidar_to_world.astype(np.float32),
            }
        )

    write_to_tar(sample, output_root / "lidar_raw" / f"{clip_id}.tar")


def convert_nuscenes_image(
    output_root: Path,
    clip_id: str,
    tables: NuScenesTables,
    scene_record: Dict[str, Any],
    target_lidar_records: List[Dict[str, Any]],
    camera_names: List[str],
    single_camera: bool = False,
) -> None:
    cprint("reading image and converting to video, this may take a while...", "yellow")

    target_timestamps = [record["timestamp"] for record in target_lidar_records]
    for camera_name in camera_names:
        if single_camera and camera_name != "front":
            continue

        channel = NuScenesToRdsCamera[camera_name]
        camera_records = tables.get_sample_data_for_scene(scene_record, channel)
        if not camera_records:
            raise FileNotFoundError(f"No sample_data found for camera {channel} in {clip_id}")

        output_video_path = output_root / f"pinhole_{camera_name}" / f"{clip_id}.mp4"
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegRawVideoWriter(output_video_path, fps=TargetFps)
        try:
            last_image_path = None
            last_image = None
            for timestamp in target_timestamps:
                camera_record = nearest_by_timestamp(camera_records, timestamp)
                image_path = tables.dataroot / camera_record["filename"]
                if image_path != last_image_path:
                    if not image_path.exists():
                        raise FileNotFoundError(f"nuScenes image file not found: {image_path}")
                    last_image = imageio_v2.imread(image_path)
                    last_image_path = image_path
                writer.append_data(last_image)
        finally:
            writer.close()


def select_camera_names(single_camera: bool, camera_names: Optional[Tuple[str, ...]] = None) -> List[str]:
    if camera_names:
        selected = list(camera_names)
    else:
        selected = ["front"] if single_camera else list(NuScenesToRdsCamera.keys())

    invalid = [name for name in selected if name not in NuScenesToRdsCamera]
    if invalid:
        raise ValueError(f"Invalid camera names: {invalid}. Valid names: {sorted(NuScenesToRdsCamera)}")

    if single_camera and "front" not in selected:
        selected.insert(0, "front")
    if single_camera:
        return ["front"]
    return selected


def convert_nuscenes_scene_to_wds(
    nuscenes_root: Union[str, Path],
    output_wds_path: Union[str, Path],
    scene_name: str,
    version: str = "v1.0-mini",
    single_camera: bool = False,
    camera_names: Optional[Tuple[str, ...]] = None,
    map_margin: float = 120.0,
    use_map_raster: bool = True,
) -> None:
    tables = NuScenesTables(nuscenes_root, version=version)
    scene_records = {scene["name"]: scene for scene in tables.scene}
    if scene_name not in scene_records:
        raise KeyError(f"Scene not found in nuScenes metadata: {scene_name}")

    output_wds_path = Path(output_wds_path)
    clip_id = scene_name
    if (output_wds_path / "lidar_raw" / f"{clip_id}.tar").exists():
        print(f"Skipping {clip_id} because it already exists")
        return

    scene_record = scene_records[scene_name]
    selected_camera_names = select_camera_names(single_camera, camera_names)
    target_lidar_records = tables.get_sample_data_for_scene(scene_record, LidarChannel)
    if not target_lidar_records:
        raise FileNotFoundError(f"No {LidarChannel} sample_data found in scene {scene_name}")

    convert_nuscenes_hdmap(
        output_wds_path,
        clip_id,
        tables,
        scene_record,
        target_lidar_records,
        map_margin=map_margin,
        use_map_raster=use_map_raster,
    )
    convert_nuscenes_intrinsics(output_wds_path, clip_id, tables, scene_record, selected_camera_names)
    convert_nuscenes_pose(output_wds_path, clip_id, tables, scene_record, target_lidar_records, selected_camera_names)
    convert_nuscenes_timestamp(output_wds_path, clip_id, target_lidar_records)
    convert_nuscenes_bbox(output_wds_path, clip_id, tables, scene_record)
    convert_nuscenes_image(output_wds_path, clip_id, tables, scene_record, target_lidar_records, selected_camera_names, single_camera)
    convert_nuscenes_lidar(output_wds_path, clip_id, tables, target_lidar_records)


@click.command()
@click.option("--nuscenes_root", "-i", type=str, required=True, help="nuScenes dataset root, e.g. /data/tianyy/data/v1.0-mini")
@click.option("--output_wds_path", "-o", type=str, required=True, help="Output RDS-HQ path")
@click.option("--version", "-v", type=str, default="v1.0-mini", show_default=True, help="nuScenes metadata version folder")
@click.option("--num_workers", "-n", type=int, default=1, show_default=True, help="Number of worker processes")
@click.option("--scene_name", "-sn", multiple=True, help="Convert only specific scene name(s), e.g. scene-0061")
@click.option("--single_camera", "-s", is_flag=True, help="Convert only the front camera")
@click.option("--camera", "-c", multiple=True, help="RDS camera name to convert: front/front_left/front_right/side_left/side_right")
@click.option("--map_margin", type=float, default=120.0, show_default=True, help="Meters around the ego path used for raster-map boundary extraction")
@click.option("--use_map_raster/--no_map_raster", default=True, show_default=True, help="Extract road_boundaries from nuScenes semantic-prior map PNG")
def main(
    nuscenes_root: str,
    output_wds_path: str,
    version: str,
    num_workers: int,
    scene_name: Tuple[str, ...],
    single_camera: bool,
    camera: Tuple[str, ...],
    map_margin: float,
    use_map_raster: bool,
) -> None:
    tables = NuScenesTables(nuscenes_root, version=version)
    if scene_name:
        all_scene_names = list(scene_name)
    else:
        all_scene_names = [scene["name"] for scene in tables.scene]

    print(f"Found {len(all_scene_names)} nuScenes scenes")

    if num_workers <= 1:
        for name in tqdm(all_scene_names, desc="Converting nuScenes scenes"):
            convert_nuscenes_scene_to_wds(
                nuscenes_root,
                output_wds_path,
                scene_name=name,
                version=version,
                single_camera=single_camera,
                camera_names=camera,
                map_margin=map_margin,
                use_map_raster=use_map_raster,
            )
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                convert_nuscenes_scene_to_wds,
                nuscenes_root=nuscenes_root,
                output_wds_path=output_wds_path,
                scene_name=name,
                version=version,
                single_camera=single_camera,
                camera_names=camera,
                map_margin=map_margin,
                use_map_raster=use_map_raster,
            )
            for name in all_scene_names
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting nuScenes scenes"):
            try:
                future.result()
            except Exception as e:
                print(f"Failed to convert due to error: {e}")


if __name__ == "__main__":
    main()
