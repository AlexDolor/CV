import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Пути (можно адаптировать)
# VIDEO_IN = "data/shaky_video.mp4"
# OUT_DIR = "output/vis"
# os.makedirs(OUT_DIR, exist_ok=True)


def save_example_frames_before_after(
    opts,
    frame_indices=(0, 100, 200),
    prefix=""
):
    if prefix == '':
        prefix = f'before_after_{opts.method}'
    
    cap_orig = cv2.VideoCapture(opts.video_in)
    cap_stab = cv2.VideoCapture(opts.video_out_stab)

    if not cap_orig.isOpened() or not cap_stab.isOpened():
        raise RuntimeError("Cant open initial or stabilized video.")

    n_frames = int(min(
        cap_orig.get(cv2.CAP_PROP_FRAME_COUNT),
        cap_stab.get(cv2.CAP_PROP_FRAME_COUNT)
    ))

    frame_indices = [i for i in frame_indices if 0 <= i < n_frames]

    for idx in frame_indices:
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_stab.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret0, frame_orig = cap_orig.read()
        ret1, frame_stab = cap_stab.read()
        if not (ret0 and ret1):
            continue

        side = np.hstack((frame_orig, frame_stab))
        out_path = os.path.join(opts.visualization_dir, f"{prefix}_frame_{idx:04d}.png")
        cv2.imwrite(out_path, side)

    cap_orig.release()
    cap_stab.release()


def plot_trajectory(opts, trajectory, smoothed_trajectory, prefix=""):
    """
    Рисует и сохраняет графики траектории движения камеры:
    - dx, dy, da (угол) + их сглаженные версии.
    """
    if prefix == '':
        prefix = f'traj_{opts.method}'
    t = np.arange(len(trajectory))

    dx = trajectory[:, 0]
    dy = trajectory[:, 1]
    da = trajectory[:, 2]

    dx_s = smoothed_trajectory[:, 0]
    dy_s = smoothed_trajectory[:, 1]
    da_s = smoothed_trajectory[:, 2]

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, dx, label="dx (raw)")
    plt.plot(t, dx_s, label="dx (smoothed)", linewidth=2)
    plt.legend()
    plt.ylabel("dx, pixels")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, dy, label="dy (raw)")
    plt.plot(t, dy_s, label="dy (smoothed)", linewidth=2)
    plt.legend()
    plt.ylabel("dy, pixels")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, da, label="da (raw)")
    plt.plot(t, da_s, label="da (smoothed)", linewidth=2)
    plt.legend()
    plt.ylabel("da, rad")
    plt.xlabel("frame №")
    plt.grid(True)

    plt.tight_layout()
    out_path = os.path.join(opts.visualization_dir, f"{prefix}_dx_dy_da.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Сохранён график траектории: {out_path}")

def transforms_to_frame(transforms):
    dx, dy, da = transforms
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy
    return m

def visualize_warp_example(opts, transforms, transforms_smooth, example_frame=150, prefix=''):
    """
    shows warping example
    saves file with exaple_frame in 
        1. initial video, 
        2. video, warped with transforms
        3. video, warped with transforms_smooth
    Показывает пример warping-а:
    - исходная трансформация (по сырому движению);
    - стабилизированная трансформация (по сглаженной траектории).
    """
    if prefix == '':
        prefix = f'warp_{opts.method}'
    cap = cv2.VideoCapture(opts.video_in)
    if not cap.isOpened():
        raise RuntimeError("Cant open video to visualize warp")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    example_frame = min(example_frame, n_frames - 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, example_frame)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cant read example frame.")

    h, w = frame.shape[:2]

    frame_orig = frame.copy()

    #transform matrices
    m_raw = transforms_to_frame(transforms[example_frame])
    m_s = transforms_to_frame(transforms_smooth[example_frame])

    warped_raw = cv2.warpAffine(frame, m_raw, (w, h))
    warped_s = cv2.warpAffine(frame, m_s, (w, h))

    # initial / warp(raw) / warp(smoothed)
    vis = np.hstack((frame_orig, warped_raw, warped_s))
    out_path = os.path.join(opts.visualization_dir, f"{prefix}_frame_{example_frame:04d}.png")
    cv2.imwrite(out_path, vis)
    cap.release()
    print(f"Warping example saved: {out_path}")