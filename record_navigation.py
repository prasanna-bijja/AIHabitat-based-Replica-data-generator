"""Keyboard-driven Habitat capture for building a ground-truth dataset.

Controls:
  w / s   move forward / backward      a / d   move left / right
  r / f   move up / down               j / l   turn left / right
  i / k   look up / down
  space   save the current frame       q / Esc quit
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import quaternion

DEFAULT_SCENE = ("/path_to_/replica/apartment_0/habitat/mesh_semantic.ply") #replace path 

# Habitat camera axes (OpenGL: x right, y up, looks along -z)
# -> OpenCV camera axes (x right, y down, looks along +z).
# T_world_camera_cv = T_world_camera_gl @ GL_TO_CV
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

README_TEXT = """# Habitat ground-truth capture

All poses are camera-to-world (T_world_camera) of the colour sensor, in the
Habitat world frame (+y up, metres). All modalities share the frame index
(000000, 000001, ...). See metadata.json for full details.

| Path | Content |
|---|---|
| rgb/*.png | 8-bit RGB images (DA3 input) |
| depth/*.npy | float32 planar z-depth in metres, 0 = invalid |
| instance/*.npy | int32 per-object instance IDs from the semantic sensor |
| semantic/*.npy | int32 class IDs, 0 = unknown (see semantic_classes.json) |
| poses/*.npy | 4x4 c2w, Habitat/OpenGL camera axes (x right, y up, -z forward) |
| poses_opencv/*.npy | 4x4 c2w, OpenCV camera axes (x right, y down, +z forward) |
| traj_c2w_opencv.txt | one frame per line: 16 numbers of the OpenCV 4x4 c2w, row-major |
| trajectory.txt | TUM format (timestamp tx ty tz qx qy qz qw), OpenGL camera axes |
| trajectory_opencv.txt | TUM format, OpenCV camera axes (use with evo / DA3 comparison) |
| raw_habitat_poses.jsonl | untouched agent + sensor states, quaternions [w, x, y, z] |
| intrinsics.npy / .txt | 3x3 K; cx = (W-1)/2, cy = (H-1)/2 (OpenCV pixel centres) |
| semantic_classes.json | instance ID -> class ID -> class name |
| frames.csv | frame_id, timestamp_s, timestamp_unix_ns |
| metadata.json | scene, camera, conventions, units, capture settings |

For DA3 (world-to-camera, OpenCV): invert the matrices in poses_opencv/.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, default=Path(DEFAULT_SCENE))
    parser.add_argument("--output", type=Path, default=Path("apartment_0_capture2"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--sensor-height", type=float, default=1.30)
    parser.add_argument("--move-step", type=float, default=0.10)
    parser.add_argument("--turn-step", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a non-empty output folder.")
    return parser.parse_args()


def check_output_dir(output, overwrite):
    """Refuse to silently overwrite an earlier recording."""
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output folder '{output}' is not empty.\n"
            "Choose a new --output folder, or pass --overwrite to reuse it."
        )


def write_json(path, data):
    with path.open("w") as file:
        json.dump(data, file, indent=2)


def camera_intrinsics(width, height, hfov_deg):
    """Compute camera intrinsics matrix (square pixels, OpenCV pixel centres)."""
    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float64)


def make_sensor(uuid, sensor_type, args):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [args.height, args.width]
    spec.position = [0.0, args.sensor_height, 0.0]
    spec.hfov = args.hfov
    return spec


def make_simulator(args):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(args.scene)
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        make_sensor("color_sensor", habitat_sim.SensorType.COLOR, args),
        make_sensor("depth_sensor", habitat_sim.SensorType.DEPTH, args),
        make_sensor("semantic_sensor", habitat_sim.SensorType.SEMANTIC, args),
    ]
    move = habitat_sim.agent.ActuationSpec(amount=args.move_step)
    turn = habitat_sim.agent.ActuationSpec(amount=args.turn_step)
    agent_cfg.action_space = {
        name: habitat_sim.agent.ActionSpec(name, move)
        for name in ("move_forward", "move_backward", "move_left",
                     "move_right", "move_up", "move_down")
    }
    agent_cfg.action_space.update({
        name: habitat_sim.agent.ActionSpec(name, turn)
        for name in ("turn_left", "turn_right", "look_up", "look_down")
    })
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def initialize_apartment_0_agent(sim):
    """Known clear viewpoint in apartment_0, with a level camera.

    NOTE: this start pose is specific to apartment_0. For another scene
    (e.g. room_0), check that it is not inside a wall or furniture.
    """
    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array([-2.7389, 0.4793, 1.0766], dtype=np.float32)
    yaw = math.atan2(0.7355, 0.6775)
    state.rotation = quaternion.from_rotation_vector(np.array([0.0, yaw, 0.0]))
    agent.set_state(state, reset_sensors=True)
    return agent


def sensor_pose(agent, sensor_uuid="color_sensor"):
    """Return the camera-to-world pose in OpenGL and OpenCV axes, plus raw states."""
    agent_state = agent.get_state()
    sensor_state = agent_state.sensor_states[sensor_uuid]
    T_gl = np.eye(4, dtype=np.float64)
    T_gl[:3, :3] = quaternion.as_rotation_matrix(sensor_state.rotation)
    T_gl[:3, 3] = np.asarray(sensor_state.position, dtype=np.float64)
    T_cv = T_gl @ GL_TO_CV
    return T_gl, T_cv, agent_state, sensor_state


def quat_wxyz(q):
    return [float(v) for v in quaternion.as_float_array(q)]


def tum_line(timestamp_s, T):
    """TUM line: timestamp tx ty tz qx qy qz qw."""
    t = T[:3, 3]
    w, x, y, z = quaternion.as_float_array(quaternion.from_rotation_matrix(T[:3, :3]))
    return (f"{timestamp_s:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{x:.9f} {y:.9f} {z:.9f} {w:.9f}\n")


def category_name(obj):
    if obj is None or obj.category is None:
        return "unknown"
    try:
        return obj.category.name()
    except TypeError:
        return str(obj.category.name)


def object_instance_id(obj, fallback_id):
    """Prefer semantic_id, which is the value the semantic sensor renders."""
    semantic_id = getattr(obj, "semantic_id", None)
    if semantic_id is not None:
        return int(semantic_id)
    try:
        return int(obj.id.split("_")[-1])
    except (AttributeError, ValueError):
        return fallback_id


def build_semantic_metadata(sim):
    """Map Habitat instance IDs to stable integer class IDs and class names."""
    instance_to_name = {}
    objects = []
    for fallback_id, obj in enumerate(sim.semantic_scene.objects):
        if obj is None:
            continue
        instance_id = object_instance_id(obj, fallback_id)
        name = category_name(obj)
        instance_to_name[instance_id] = name
        objects.append({
            "instance_id": instance_id,
            "habitat_object_id": str(obj.id),
            "category": name,
        })

    names = sorted(set(instance_to_name.values()))
    name_to_class = {name: idx + 1 for idx, name in enumerate(names)}
    instance_to_class = {
        instance_id: name_to_class[name]
        for instance_id, name in instance_to_name.items()
    }
    class_info = {
        "note": "class ID 0 = unknown / unmapped instance",
        "class_id_to_name": {str(v): k for k, v in name_to_class.items()},
        "name_to_class_id": name_to_class,
        "instance_to_class_id": {str(k): v for k, v in sorted(instance_to_class.items())},
        "instance_to_name": {str(k): v for k, v in sorted(instance_to_name.items())},
    }
    return instance_to_class, name_to_class, objects, class_info


def instance_to_semantic(instance_image, mapping):
    semantic = np.zeros(instance_image.shape, dtype=np.int32)
    for instance_id in np.unique(instance_image):
        semantic[instance_image == instance_id] = mapping.get(int(instance_id), 0)
    return semantic


def build_metadata(args, K, name_to_class, objects):
    return {
        "scene": str(args.scene),
        "habitat_sim_version": getattr(habitat_sim, "__version__", "unknown"),
        "created_unix_ns": time.time_ns(),
        "frame_count": 0,
        "image_size": [args.width, args.height],
        "hfov_degrees": args.hfov,
        "depth_unit": "metres",
        "camera": {
            "image_size_wh": [args.width, args.height],
            "hfov_degrees": args.hfov,
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "principal_point_convention": "OpenCV pixel centres at integer coordinates: cx=(W-1)/2, cy=(H-1)/2",
            "sensor_height_m": args.sensor_height,
            "sensors": "colour, depth and semantic sensors are co-located with identical intrinsics",
        },
        "world_frame": {"source": "Habitat world frame", "up_axis": "+y", "units": "metres"},
        "pose_convention": "T_world_camera; Habitat camera looks along local -Z",
        "poses": {
            "source": "colour sensor state (camera at sensor height), not the agent base",
            "type": "camera-to-world (T_world_camera)",
            "poses/*.npy": "4x4, Habitat/OpenGL camera axes (x right, y up, looks along -z)",
            "poses_opencv/*.npy": "4x4, OpenCV camera axes (x right, y down, looks along +z) = T_gl @ diag(1,-1,-1,1)",
            "traj_c2w_opencv.txt": "one line per frame: 16 numbers of the OpenCV 4x4 c2w, row-major",
            "trajectory.txt": "TUM: timestamp tx ty tz qx qy qz qw, OpenGL camera axes",
            "trajectory_opencv.txt": "TUM: timestamp tx ty tz qx qy qz qw, OpenCV camera axes",
            "raw_habitat_poses.jsonl": "untouched agent and sensor states, quaternions as [w, x, y, z]",
            "for_da3": "invert poses_opencv to get world-to-camera (OpenCV)",
        },
        "depth": {
            "format": "float32 .npy",
            "unit": "metres",
            "type": "planar z-depth along the optical axis (not ray distance)",
            "invalid_value": 0.0,
        },
        "instance": {"format": "int32 .npy", "content": "Habitat semantic sensor output (instance IDs)"},
        "semantic": {"format": "int32 .npy", "content": "class IDs, see semantic_classes.json",
                     "unknown_class_id": 0},
        "timestamps": "seconds since recording start (wall clock at key press); unix ns in frames.csv",
        "capture": {"method": "manual keyboard teleoperation",
                    "move_step_m": args.move_step, "turn_step_deg": args.turn_step},
        "class_id_to_name": {str(v): k for k, v in name_to_class.items()},
        "objects": objects,
    }


def prepare_output(output, K, args, name_to_class, objects, class_info):
    for folder in ("rgb", "depth", "instance", "semantic", "poses", "poses_opencv"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    np.save(output / "intrinsics.npy", K)
    np.savetxt(output / "intrinsics.txt", K, fmt="%.10f")
    write_json(output / "semantic_classes.json", class_info)
    write_json(output / "metadata.json", build_metadata(args, K, name_to_class, objects))
    (output / "README.md").write_text(README_TEXT)

    tum_header = "# timestamp_s tx ty tz qx qy qz qw\n"
    (output / "trajectory.txt").write_text(tum_header)
    (output / "trajectory_opencv.txt").write_text(tum_header)
    (output / "traj_c2w_opencv.txt").write_text("")
    (output / "raw_habitat_poses.jsonl").write_text("")
    (output / "frames.csv").write_text("frame_id,timestamp_s,timestamp_unix_ns\n")


def finalize_metadata(output, frame_count):
    path = output / "metadata.json"
    if not path.exists():
        return
    metadata = json.loads(path.read_text())
    metadata["frame_count"] = frame_count
    write_json(path, metadata)


def save_frame(output, frame_id, observations, agent, instance_to_class, start_ns):
    stem = f"{frame_id:06d}"
    now_ns = time.time_ns()
    timestamp_s = (now_ns - start_ns) / 1e9

    rgba = np.asarray(observations["color_sensor"])
    rgb = rgba[..., :3].astype(np.uint8)
    depth = np.asarray(observations["depth_sensor"], dtype=np.float32)
    instance = np.asarray(observations["semantic_sensor"], dtype=np.int32)
    semantic = instance_to_semantic(instance, instance_to_class)
    T_gl, T_cv, agent_state, sensor_state = sensor_pose(agent)

    unmapped = [int(i) for i in np.unique(instance) if int(i) not in instance_to_class]
    if unmapped:
        print(f"  WARNING frame {stem}: instance IDs without a class mapping: {unmapped[:10]}")
    if not np.any(depth > 0):
        print(f"  WARNING frame {stem}: no valid depth pixels")

    # OpenCV expects BGR only at write time; stored PNG displays as RGB normally.
    cv2.imwrite(str(output / "rgb" / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(str(output / "depth_png" / f"{stem}.png"), cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(str(output / "instance" / f"{stem}.png"), cv2.cvtColor(instance, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(str(output / "semantic_png" / f"{stem}.png"), cv2.cvtColor(semantic, cv2.COLOR_RGB2BGR))
    np.save(output / "depth" / f"{stem}.npy", depth)
    np.save(output / "instance" / f"{stem}.npy", instance)
    np.save(output / "semantic" / f"{stem}.npy", semantic)
    np.save(output / "poses" / f"{stem}.npy", T_gl)
    np.save(output / "poses_opencv" / f"{stem}.npy", T_cv)

    with (output / "trajectory.txt").open("a") as file:
        file.write(tum_line(timestamp_s, T_gl))
    with (output / "trajectory_opencv.txt").open("a") as file:
        file.write(tum_line(timestamp_s, T_cv))
    with (output / "traj_c2w_opencv.txt").open("a") as file:
        file.write(" ".join(f"{v:.10f}" for v in T_cv.reshape(-1)) + "\n")
    with (output / "raw_habitat_poses.jsonl").open("a") as file:
        file.write(json.dumps({
            "frame_id": stem,
            "timestamp_s": timestamp_s,
            "agent_position": [float(v) for v in agent_state.position],
            "agent_rotation_wxyz": quat_wxyz(agent_state.rotation),
            "sensor_position": [float(v) for v in sensor_state.position],
            "sensor_rotation_wxyz": quat_wxyz(sensor_state.rotation),
        }) + "\n")
    with (output / "frames.csv").open("a") as file:
        file.write(f"{stem},{timestamp_s:.9f},{now_ns}\n")
    print(f"Saved synchronized frame {stem}")


def depth_preview(depth):
    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        limit = max(float(np.percentile(depth[valid], 95)), 1e-6)
        preview[valid] = np.clip(depth[valid] / limit * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(255 - preview, cv2.COLORMAP_TURBO)


def main():
    args = parse_args()
    if not args.scene.exists():
        raise FileNotFoundError(
            f"Scene not found: {args.scene}\n"
            "Pass the correct file with --scene /path/to/<scene>/habitat/mesh_semantic.ply"
        )
    check_output_dir(args.output, args.overwrite)

    sim = make_simulator(args)
    frame_id = 0
    prepared = False
    try:
        agent = initialize_apartment_0_agent(sim)
        instance_to_class, name_to_class, objects, class_info = build_semantic_metadata(sim)
        K = camera_intrinsics(args.width, args.height, args.hfov)
        prepare_output(args.output, K, args, name_to_class, objects, class_info)
        prepared = True

        key_to_action = {
            ord("w"): "move_forward",
            ord("s"): "move_backward",
            ord("a"): "move_left",
            ord("d"): "move_right",
            ord("r"): "move_up",
            ord("f"): "move_down",
            ord("j"): "turn_left",
            ord("l"): "turn_right",
            ord("i"): "look_up",
            ord("k"): "look_down",
        }
        start_ns = time.time_ns()
        observations = sim.get_sensor_observations()
        print(__doc__)

        while True:
            rgba = np.asarray(observations["color_sensor"])
            rgb = rgba[..., :3].astype(np.uint8)
            color_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            preview = np.hstack((color_bgr, depth_preview(observations["depth_sensor"])))
            cv2.putText(preview, f"saved: {frame_id}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("capture: RGB | depth", preview)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                save_frame(args.output, frame_id, observations, agent,
                           instance_to_class, start_ns)
                frame_id += 1
            elif key in key_to_action:
                observations = sim.step(key_to_action[key])
    finally:
        if prepared:
            finalize_metadata(args.output, frame_id)
        cv2.destroyAllWindows()
        sim.close()


if __name__ == "__main__":
    main()
