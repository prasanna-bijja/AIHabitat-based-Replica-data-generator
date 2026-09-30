
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from PIL import Image

# ============================ CONFIG ============================
CAPTURE = Path("/path_to_collected_data")
MESH_PATH = Path("/path_to_/replica/apartment_0/habitat/mesh_semantic.ply")  # cached triangulated mesh
POSE_DIR = CAPTURE / "poses_opencv"
DEPTH_DIR = CAPTURE / "depth"
RGB_DIR = CAPTURE / "rgb"            # only used if POINT_COLOR = "rgb"
OUT_DIR = CAPTURE / "videos"

HFOV_DEG = 90.0          # <-- hfov of your Habitat depth sensor
WIN_W, WIN_H = 1280, 720 # video resolution
FPS = 8
HOLD_LAST_S = 2          # freeze the last frame for this many seconds

CROP_CEILING = True      # remove geometry above the cameras so the view isn't blocked
AXIS_SIZE = 0.25         # length of the RGB camera axes (m)

DEPTH_VIEW = "follow"    # "follow": view from behind/above the current camera | "overview": fixed view
ACCUMULATE_POINTS = True # keep points of previous frames (False: show only the current frame)
POINT_COLOR = "frame"    # "frame": colour by frame index (shows alignment) | "rgb": colour from RGB images
VOXEL = 0.02             # downsampling per frame (m), keeps rendering fast
# ================================================================


def natural_sorted(files):
    """Sort frame_2 before frame_10."""
    def key(p):
        nums = re.findall(r"\d+", p.stem)
        return (int(nums[-1]) if nums else -1, p.name)
    return sorted(files, key=key)


from plyfile import PlyData   # add to the imports at the top


def load_ply_with_plyfile(path):
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    V = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    fdata = ply["face"].data
    name = "vertex_indices" if "vertex_indices" in fdata.dtype.names else "vertex_index"
    faces = fdata[name]
    lengths = np.array([len(f) for f in faces])
    tris = []
    if np.any(lengths == 4):                                   # quads -> 2 triangles each
        q = np.vstack(faces[lengths == 4]).astype(np.int32)
        tris.append(np.stack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]], axis=1).reshape(-1, 3))
    if np.any(lengths == 3):
        tris.append(np.vstack(faces[lengths == 3]).astype(np.int32))

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                     o3d.utility.Vector3iVector(np.concatenate(tris)))
    if "red" in v.dtype.names:
        C = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(C)
    return mesh


def load_mesh():
    mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))
    if len(mesh.triangles) == 0:
        print("Open3D could not parse the PLY -> loading with plyfile (takes a minute)...")
        mesh = load_ply_with_plyfile(MESH_PATH)
        cache = Path(__file__).with_name("apt0_semantic_tri.ply")
        o3d.io.write_triangle_mesh(str(cache), mesh)
        print(f"Cached triangulated mesh -> {cache}  (set MESH_PATH to this for faster runs)")
    print(mesh)

    R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))   # Replica Z-up -> Habitat Y-up
    mesh.rotate(R, center=(0, 0, 0))
    return mesh

def crop_above(mesh, max_y):
    bb = mesh.get_axis_aligned_bounding_box()
    lo, hi = bb.get_min_bound(), bb.get_max_bound()
    box = o3d.geometry.AxisAlignedBoundingBox(lo - 1.0, np.array([hi[0] + 1.0, max_y, hi[2] + 1.0]))
    return mesh.crop(box)


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """OpenCV camera-to-world pose looking from eye to target (Y-up world)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def overview_view(poses):
    """Fixed view from above and behind the cameras' average viewing direction."""
    c = poses[:, :3, 3]
    target = c.mean(axis=0) - np.array([0.0, 0.6, 0.0])
    spread = np.ptp(c[:, [0, 2]], axis=0).max()
    d = max(3.0, 1.5 * spread)
    fwd = poses[:, :3, 2].mean(axis=0)
    fwd[1] = 0.0
    fwd = fwd / np.linalg.norm(fwd) if np.linalg.norm(fwd) > 1e-6 else np.array([0.0, 0.0, -1.0])
    eye = target - 0.8 * d * fwd + np.array([0.0, 0.9 * d, 0.0])
    return look_at(eye, target)


def follow_view(T, back=1.8, up=0.6, pitch_deg=15.0):
    """Virtual camera behind and above the real one, tilted slightly down (OpenCV axes)."""
    trans = np.eye(4)
    trans[:3, 3] = [0.0, -up, -back]          # OpenCV: -y is up, -z is backwards
    a = np.deg2rad(-pitch_deg)
    rot = np.eye(4)
    rot[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return T @ trans @ rot


def depth_intrinsics(W, H):
    fx = (W / 2.0) / np.tan(np.deg2rad(HFOV_DEG) / 2)
    return np.array([[fx, 0, (W - 1) / 2.0], [0, fx, (H - 1) / 2.0], [0, 0, 1]])


def backproject(depth, T, color):
    H, W = depth.shape
    K = depth_intrinsics(W, H)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth > 0
    z = depth[valid]
    pts_cam = np.stack([(u[valid] - K[0, 2]) * z / K[0, 0], (v[valid] - K[1, 2]) * z / K[1, 1], z], axis=1)
    pts_world = pts_cam @ T[:3, :3].T + T[:3, 3]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_world))
    if isinstance(color, np.ndarray) and color.ndim == 3:        # RGB image
        pcd.colors = o3d.utility.Vector3dVector(color[valid])
    else:
        pcd.paint_uniform_color(color)
    return pcd.voxel_down_sample(VOXEL) if VOXEL > 0 else pcd


class Recorder:
    def __init__(self, mesh):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=WIN_W, height=WIN_H, visible=True)
        opt = self.vis.get_render_option()
        opt.mesh_show_back_face = True
        opt.point_size = 3.0
        opt.background_color = np.array([1.0, 1.0, 1.0])
        self.vis.add_geometry(mesh)               # sets the scene bounds / clipping planes
        f = (WIN_W / 2) / np.tan(np.deg2rad(60) / 2)  # 60 deg field of view for the video camera
        self.intr = o3d.camera.PinholeCameraIntrinsic(WIN_W, WIN_H, f, f, WIN_W / 2 - 0.5, WIN_H / 2 - 0.5)
        self.frames = []

    def add(self, g):
        self.vis.add_geometry(g, reset_bounding_box=False)

    def remove(self, g):
        self.vis.remove_geometry(g, reset_bounding_box=False)

    def set_view(self, T_c2w):
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = self.intr
        params.extrinsic = np.linalg.inv(T_c2w)   # Open3D wants world-to-camera
        ctr = self.vis.get_view_control()
        try:
            ok = ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        except TypeError:                         # older Open3D without allow_arbitrary
            ok = ctr.convert_from_pinhole_camera_parameters(params)
        if ok is False:
            print("WARNING: view not applied (window size may differ from WIN_W x WIN_H)")

    def grab(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        img = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        self.frames.append(img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2])  # even size for mp4

    def save(self, stem):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        frames = self.frames + [self.frames[-1]] * (FPS * HOLD_LAST_S)
        try:
            with imageio.get_writer(OUT_DIR / f"{stem}.mp4", fps=FPS, macro_block_size=1) as w:
                for fr in frames:
                    w.append_data(fr)
            print("saved", OUT_DIR / f"{stem}.mp4")
        except Exception as e:
            print(f"MP4 failed ({e}); install imageio-ffmpeg. GIF is still written.")
        gif = [Image.fromarray(fr[::2, ::2]) for fr in frames]          # half resolution keeps GIF small
        gif[0].save(OUT_DIR / f"{stem}.gif", save_all=True, append_images=gif[1:],
                    duration=int(1000 / FPS), loop=0)
        print("saved", OUT_DIR / f"{stem}.gif")
        self.vis.destroy_window()


def render_trajectory(mesh, poses, view):
    rec = Recorder(mesh)
    centers = poses[:, :3, 3]
    for i, T in enumerate(poses):
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE)
        axes.transform(T)
        rec.add(axes)
        if i > 0:
            seg = o3d.geometry.LineSet(o3d.utility.Vector3dVector(centers[i - 1:i + 1]),
                                       o3d.utility.Vector2iVector([[0, 1]]))
            seg.paint_uniform_color([0.0, 0.0, 0.0])
            rec.add(seg)
        rec.set_view(view)
        rec.grab()
        print(f"trajectory frame {i + 1}/{len(poses)}")
    rec.save("trajectory")


def render_backprojection(mesh, poses, depth_files, rgb_files, overview):
    rec = Recorder(mesh)
    n = len(poses)
    prev_pcd, markers = None, []
    for i, T in enumerate(poses):
        depth = np.load(depth_files[i]).squeeze()
        H, W = depth.shape

        if POINT_COLOR == "rgb" and rgb_files:
            color = np.asarray(imageio.imread(rgb_files[i]))[..., :3] / 255.0
        else:
            color = [i / max(n - 1, 1), 0.2, 1.0 - i / max(n - 1, 1)]   # blue -> red over time

        pcd = backproject(depth, T, color)
        if not ACCUMULATE_POINTS and prev_pcd is not None:
            rec.remove(prev_pcd)
        rec.add(pcd)
        prev_pcd = pcd

        for m in markers:                          # only the current camera is drawn
            rec.remove(m)
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            W, H, depth_intrinsics(W, H), np.linalg.inv(T), scale=0.3)
        frustum.paint_uniform_color([0.0, 0.0, 0.0])
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=AXIS_SIZE)
        axes.transform(T)
        markers = [frustum, axes]
        for m in markers:
            rec.add(m)

        rec.set_view(follow_view(T) if DEPTH_VIEW == "follow" else overview)
        rec.grab()
        print(f"backprojection frame {i + 1}/{n}")
    rec.save("backprojection")


def main():
    pose_files = natural_sorted(POSE_DIR.glob("*.npy"))
    depth_files = natural_sorted(DEPTH_DIR.glob("*.npy"))
    assert len(pose_files) == len(depth_files) > 0, \
        f"{len(pose_files)} poses vs {len(depth_files)} depth files"
    poses = np.stack([np.load(f) for f in pose_files]).astype(np.float64)

    rgb_files = []
    if POINT_COLOR == "rgb" and RGB_DIR.exists():
        rgb_files = natural_sorted(list(RGB_DIR.glob("*.png")) + list(RGB_DIR.glob("*.jpg")))
        if len(rgb_files) != len(poses):
            print("RGB count does not match poses -> colouring by frame instead")
            rgb_files = []

    mesh = load_mesh()
    if CROP_CEILING:
        mesh = crop_above(mesh, poses[:, 1, 3].max() + 0.4)
    mesh.compute_vertex_normals()

    overview = overview_view(poses)
    render_trajectory(mesh, poses, overview)
    render_backprojection(mesh, poses, depth_files, rgb_files, overview)


if __name__ == "__main__":
    main()
