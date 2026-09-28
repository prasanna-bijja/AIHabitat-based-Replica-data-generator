# Replica Ground-Truth Data Generation using AI-Habitat

This repository documents how RGB, depth, semantic, and camera pose data were generated from the [Replica dataset](https://github.com/facebookresearch/Replica-Dataset) using [AI-Habitat (Habitat-Sim)](https://github.com/facebookresearch/habitat-sim).

## Why this data was generated

Replica provides photorealistic, densely reconstructed indoor scenes with accurate geometry and semantic annotations, which makes it well suited as a **ground-truth source** for evaluating monocular depth and pose estimation pipelines. The RGB, depth, semantics, camera intrinsics, and per-frame poses collected here are intended to be used as reference ground truth when benchmarking predicted depth and trajectory outputs against known, simulator-accurate values.

## What this pipeline captures

For each recorded frame, the following are saved:

| Modality | Format | Description |
|---|---|---|
| RGB | `.png` | Color image from the agent's camera |
| Depth | `.npy` | Metric depth map (float32, meters) |
| Semantic / Instance masks | `.npy` | Per-pixel instance IDs |
| Camera intrinsics | `intrinsics.json` | Focal length, principal point, FOV (fixed for the session) |
| Camera pose (extrinsics) | `.txt` + `trajectory.json` | 4×4 world-to-camera transform per frame |
| Object metadata | `objects.json` | Instance ID → semantic category mapping |

## Environment Setup

**Platform:** Ubuntu
**Python version:** 3.9

> **⚠️ Version note:** Habitat-Sim's pre-built bindings currently target **Python 3.9**. Using **Python 3.12** caused the simulator window to open in a static, non-interactive state — keyboard input wasn't reaching the simulator and the agent could not be moved. Creating a dedicated Python 3.9 environment resolved this completely. If you hit an unresponsive or frozen viewer window, check your Python version first before debugging anything else.

```bash
conda create -n habitat python=3.9 -y
conda activate habitat
```

### Installing Habitat-Sim

Followed the official [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) installation instructions for Ubuntu:

```bash
conda install habitat-sim -c conda-forge -c aihabitat
```

After installation, verify the simulator launches and responds to input using one of Habitat's bundled test scenes before moving on to Replica:

```bash
python examples/viewer.py --scene /path/to/habitat-test-scenes/skokloster-castle.glb
```

At this stage you should be able to see the scene render and move around using the keyboard.

## Downloading the Replica Dataset

```bash
git clone https://github.com/facebookresearch/Replica-Dataset.git
cd Replica-Dataset
./download.sh /path/to/replica_v1
```

Each downloaded scene follows this structure:

```
apartment_0/
├── mesh.ply                     # raw textured mesh (no semantics)
├── textures/
└── habitat/
    ├── mesh_semantic.ply        # mesh with semantic/instance data — use this for the scene_id
    ├── mesh_semantic.navmesh    # navigation mesh for the agent
    ├── info_semantic.json       # instance ID → category name mapping
    └── replica_stage.stage_config.json
```

> **Important:** point Habitat-Sim's `scene_id` at `habitat/mesh_semantic.ply`, not the top-level `mesh.ply`. The plain mesh carries geometry only — semantic and instance segmentation will not load from it.

## Verifying the Setup

Before recording any data, the following were confirmed for each scene:

- [x] The scene loads without errors
- [x] `sim.pathfinder.is_loaded` returns `True` (navmesh present and valid)
- [x] `sim.semantic_scene` is populated with a non-zero object count
- [x] The agent can be placed at a random navigable point and moved with the keyboard
- [x] RGB, depth, and semantic sensors all render correctly and stay spatially aligned

Only once a scene passed these checks was it used for data recording.

## Recording Data

With the environment verified, a Python script drives the simulator through a sequence of camera poses in a scene (e.g. `apartment_0`), and at each captured step saves the RGB frame, depth map, semantic/instance mask, and the agent's current pose, while the camera intrinsics are computed once and saved for the whole session.

Output is organized per scene as:

```
<output>/
├── rgb/                     000000.png ...   8-bit RGB
├── depth/                   000000.npy ...   float32 planar z-depth (m), 0 = invalid
├── instance/                000000.npy ...   int32 instance IDs
├── semantic/                000000.npy ...   int32 class IDs, 0 = unknown
├── poses/                   000000.npy ...   4x4 camera-to-world, OpenGL camera axes
├── poses_opencv/            000000.npy ...   4x4 camera-to-world, OpenCV camera axes
├── trajectory.txt           TUM format, OpenGL camera axes
├── trajectory_opencv.txt    TUM format, OpenCV camera axes
├── traj_c2w_opencv.txt      16 numbers per line (row-major 4x4), OpenCV
├── raw_habitat_poses.jsonl  untouched agent + sensor states
├── intrinsics.npy / .txt    3x3 camera matrix K
├── semantic_classes.json    instance ID -> class ID -> class name
├── frames.csv               frame index and timestamps
├── metadata.json            scene, camera, conventions, units, settings
└── README.md                description of the capture

```

## Acknowledgements

- [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset) — Facebook Reality Labs Research
- [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) / [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) — Facebook AI Research

