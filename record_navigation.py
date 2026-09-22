import argparse
import json
import math
import time
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
from PIL import Image
import quaternion 

DEFAULT_SCENE = ("/media/pbijja/New Volume/habitat-sim/data/scene_datasets/replica/apartment_0/habitat/mesh_semantic.ply")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, default=Path(DEFAULT_SCENE))
    parser.add_argument("--output", type=Path, default=Path("apartment_0_capture"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--sensor-height", type=float, default=0.0)
    parser.add_argument("--move-step", type=float, default=0.10)
    parser.add_argument("--turn-step", type=float, default=5.0)
    return parser.parse_args()


def camera_intrinsics(width, height, hfov_deg):
    """Compute camera intrinsics matrix."""
    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    # cx = width / 2.0
    # cy = height / 2.0

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intrinsics = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]])
    dtype = np.float64
    return intrinsics.astype(dtype)


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
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=args.move_step)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=args.move_step)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=args.move_step)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=args.move_step)
        ),
        "move_up": habitat_sim.agent.ActionSpec(
            "move_up", habitat_sim.agent.ActuationSpec(amount=args.move_step)
        ),
        "move_down": habitat_sim.agent.ActionSpec(
            "move_down", habitat_sim.agent.ActuationSpec(amount=args.move_step)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=args.turn_step)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=args.turn_step)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=args.turn_step)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=args.turn_step)
        ),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def initialize_apartment_0_agent(sim):
    """Known clear viewpoint in apartment_0, with a level camera."""
    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array([-2.7389, 0.4793, 1.0766], dtype=np.float32)
    # Level camera, yaw matching the known useful apartment_0 starting view.
    yaw = math.atan2(0.7355, 0.6775)
    state.rotation = quaternion.from_rotation_vector(np.array([0.0, yaw, 0.0]))
    agent.set_state(state, reset_sensors=True)
    return agent


def sensor_pose_matrix(agent, sensor_uuid="color_sensor"):
    sensor_state = agent.get_state().sensor_states[sensor_uuid]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion.as_rotation_matrix(sensor_state.rotation)
    transform[:3, 3] = np.asarray(sensor_state.position, dtype=np.float64)
    return transform, sensor_state.rotation


def category_name(obj):
    if obj is None or obj.category is None:
        return "unknown"
    try:
        return obj.category.name()
    except TypeError:
        return str(obj.category.name)


def build_semantic_metadata(sim):
    """Map Habitat instance IDs to stable integer class IDs and class names."""
    instance_to_name = {}
    objects = []
    for fallback_id, obj in enumerate(sim.semantic_scene.objects):
        if obj is None:
            continue
        try:
            instance_id = int(obj.id.split("_")[-1])
        except (AttributeError, ValueError):
            instance_id = fallback_id
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
    return instance_to_class, name_to_class, objects


def instance_to_semantic(instance_image, mapping):
    semantic = np.zeros(instance_image.shape, dtype=np.int32)
    for instance_id in np.unique(instance_image):
        semantic[instance_image == instance_id] = mapping.get(int(instance_id), 0)
    return semantic


def prepare_output(output, K, args, name_to_class, objects):
    for folder in ("rgb", "depth", "instance", "semantic", "poses"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    np.save(output / "intrinsics.npy", K)
    np.savetxt(output / "intrinsics.txt", K, fmt="%.10f")
    metadata = {
        "scene": str(args.scene),
        "image_size": [args.width, args.height],
        "hfov_degrees": args.hfov,
        "depth_unit": "metres",
        "pose_convention": "T_world_camera; Habitat camera looks along local -Z",
        "class_id_to_name": {str(v): k for k, v in name_to_class.items()},
        "objects": objects,
    }
    with (output / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
    (output / "trajectory.txt").write_text(
        "# timestamp_s tx ty tz qx qy qz qw\n"
    )
    (output / "frames.csv").write_text(
        "frame_id,timestamp_s,timestamp_unix_ns\n"
    )


def save_frame(output, frame_id, observations, agent, instance_to_class, start_ns):
    stem = f"{frame_id:06d}"
    now_ns = time.time_ns()
    timestamp_s = (now_ns - start_ns) / 1e9

    rgba = np.asarray(observations["color_sensor"])
    rgb = rgba[..., :3].astype(np.uint8)
    depth = np.asarray(observations["depth_sensor"], dtype=np.float32)
    instance = np.asarray(observations["semantic_sensor"], dtype=np.int32)
    semantic = instance_to_semantic(instance, instance_to_class)
    T_world_camera, rotation = sensor_pose_matrix(agent)

    # OpenCV expects BGR only at write time; stored PNG displays as RGB normally.
    cv2.imwrite(str(output / "rgb" / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(output / "depth" / f"{stem}.npy", depth)

    # depth_mm = np.round(depth * 1000.0).astype(np.uint16)
    # cv2.imwrite(str(output / "depth" / f"{stem}.png"), depth_mm)

    np.save(output / "instance" / f"{stem}.npy", instance)
    # cv2.imwrite(
    # str(output / "instance" / f"{stem}.png"),
    # instance.astype(np.uint16)
    # )

    np.save(output / "semantic" / f"{stem}.npy", semantic)
    # cv2.imwrite(
    # str(output / "semantic" / f"{stem}.png"),
    # semantic.astype(np.uint16)
    # )
    np.save(output / "poses" / f"{stem}.npy", T_world_camera)

    position = T_world_camera[:3, 3]
    q = quaternion.as_float_array(rotation)  # [w, x, y, z]
    with (output / "trajectory.txt").open("a") as file:
        file.write(
            # f"{timestamp_s:.9f} "
            # f"{position[0]:.9f} {position[1]:.9f} {position[2]:.9f} "
            # f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            f"{timestamp_s:.9f} {position[0]:.9f} {position[1]:.9f} "
            f"{position[2]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {q[0]:.9f}\n"
        )
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
            "Pass the correct file with --scene /path/to/apartment_0/habitat/mesh_semantic.ply"
        )

    sim = make_simulator(args)
    try:
        agent = initialize_apartment_0_agent(sim)
        instance_to_class, name_to_class, objects = build_semantic_metadata(sim)
        K = camera_intrinsics(args.width, args.height, args.hfov)
        prepare_output(args.output, K, args, name_to_class, objects)

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
        frame_id = 0
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
            cv2.imshow("apartment_0: RGB | depth", preview)
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
        cv2.destroyAllWindows()
        sim.close()


if __name__ == "__main__":
    main()

