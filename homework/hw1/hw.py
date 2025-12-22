import cv2
import numpy as np
from tqdm import tqdm
# Local imports
from parser import create_parser, print_opts
from visualizations import (  
    save_example_frames_before_after,
    plot_trajectory,
    visualize_warp_example,
)


# params for corner finding
feature_params = dict(
    maxCorners=200,        # max points
    qualityLevel=0.01,     # min quality of corner
    minDistance=30,        # min distance between points
    blockSize=3
)

# -----------------------
# UTILS
# -----------------------

def moving_average(curve, radius):
    """
    movin average on 1D-array.
    radius - window radius. window size = 2*radius+1.
    """
    window_size = 2 * radius + 1
    curve_pad = np.pad(curve, (radius, radius), mode="edge")
    kernel = np.ones(window_size) / window_size
    curve_smooth = np.convolve(curve_pad, kernel, mode="same")
    return curve_smooth[radius:-radius]


def smooth_trajectory(trajectory, radius):
    """
    smoths trajectory (N x 3: dx, dy, da) each column individually.
    """
    smoothed = np.copy(trajectory)
    for i in range(3):
        smoothed[:, i] = moving_average(trajectory[:, i], radius)
    return smoothed


def fix_border(frame, scale=1.04):
    """
    zooms to negate warp effect
    Немного масштабирует изображение, чтобы убрать чёрные края после warp'а.
    scale - scale of zooming
    """
    h, w = frame.shape[:2]
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    return cv2.warpAffine(frame, T, (w, h))


# -----------------------
# optical flow calc
# -----------------------


def estimate_affine_lk(cap, n_frames):
    """
    Estimate affine motion with sparce optical flow (LK (Lucas–Kanade))
    cap - video to analize
    n_frames - number of frames
    """
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    # read first frame
    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Can't read first frame.")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    print("LK Method initiating...")

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    for i in tqdm(range(n_frames - 1)):
        ret, curr = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **lk_params
        )

        if curr_pts is None:
            transforms[i] = transforms[i - 1] if i > 0 else [0, 0, 0]
        else:
            idx = status.flatten() == 1
            prev_pts_good = prev_pts[idx]
            curr_pts_good = curr_pts[idx]

            if len(prev_pts_good) < 10:
                transforms[i] = transforms[i - 1] if i > 0 else [0, 0, 0]
            else:
                #affine matrix 2x3 on point pairs
                m, inliers = cv2.estimateAffine2D(prev_pts_good, curr_pts_good)
                if m is None:
                    transforms[i] = transforms[i - 1] if i > 0 else [0, 0, 0]
                else:
                    dx = m[0, 2]
                    dy = m[1, 2]
                    da = np.arctan2(m[1, 0], m[0, 0])
                    transforms[i] = [dx, dy, da]

        prev_gray = curr_gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    return transforms

def estimate_affine_farneback(cap, n_frames):
    """
    Estimate affine motion with dence optical flow (Farneback)
    cap - video to analize
    n_frames - number of frames
    """
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    # читаем первый кадр
    ret, prev = cap.read()
    if not ret:
        raise RuntimeError("Can't read first frame.")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    print("Farneback Method initiating...")

    # Farneback params
    fb_params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    h, w = prev_gray.shape[:2]

    # move on a grid
    step = 15  # in pixels
    grid_y, grid_x = np.mgrid[step // 2:h:step, step // 2:w:step]  # grid coords

    for i in tqdm(range(n_frames - 1)):
        ret, curr = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # dence flow[y, x] = (dx, dy)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                            fb_params["pyr_scale"],
                                            fb_params["levels"],
                                            fb_params["winsize"],
                                            fb_params["iterations"],
                                            fb_params["poly_n"],
                                            fb_params["poly_sigma"],
                                            fb_params["flags"])

        # only in grid corners
        fx = flow[grid_y, grid_x, 0]
        fy = flow[grid_y, grid_x, 1]

        # previous cords
        prev_pts = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32)
        # new cords = prev + flow
        curr_pts = (np.stack([grid_x + fx, grid_y + fy], axis=-1)
                    .reshape(-1, 2).astype(np.float32))

        # disregard points with small dlow
        motion_mag = np.linalg.norm(curr_pts - prev_pts, axis=1)
        mask_motion = motion_mag > 0.2  # threshold (in pixels)
        prev_pts_good = prev_pts[mask_motion]
        curr_pts_good = curr_pts[mask_motion]

        if len(prev_pts_good) < 20:
            transforms[i] = transforms[i - 1] if i > 0 else [0, 0, 0]
        else:
            m, inliers = cv2.estimateAffine2D(prev_pts_good, curr_pts_good)
            if m is None:
                transforms[i] = transforms[i - 1] if i > 0 else [0, 0, 0]
            else:
                dx = m[0, 2]
                dy = m[1, 2]
                da = np.arctan2(m[1, 0], m[0, 0])
                transforms[i] = [dx, dy, da]

        prev_gray = curr_gray
    return transforms


def main(opts):
    cap = cv2.VideoCapture(opts.video_in)
    if not cap.isOpened():
        raise RuntimeError("Can't open input video.")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_stab = cv2.VideoWriter(opts.video_out_stab, fourcc, fps, (w, h))
    out_side = cv2.VideoWriter(opts.video_out_comare, fourcc, fps, (w * 2, h))

    if opts.method == "lk":
        transforms = estimate_affine_lk(cap, n_frames)
    elif opts.method == "farneback":
        transforms = estimate_affine_farneback(cap, n_frames)
    else:
        raise ValueError(f"opts.method must be 'lk' or 'farneback'. Got {opts.method}")
    
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory, opts.smoothing_radius)
    transforms_smooth = transforms + (smoothed_trajectory - trajectory)

    # WARPING
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Can't read first frame.")

    frame_idx = 0

    print("Apply smooth trajectory and warping...")

    for frame_idx in tqdm(range(n_frames - 1)):
        ret, frame = cap.read()
        if not ret:
            break

        frame_orig = frame.copy()

        if frame_idx < len(transforms_smooth):
            dx, dy, da = transforms_smooth[frame_idx]
        else:
            dx, dy, da = transforms_smooth[-1]

        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        frame_stab = cv2.warpAffine(frame, m, (w, h))
        frame_stab = fix_border(frame_stab)

        out_stab.write(frame_stab)
        out_side.write(np.hstack((frame_orig, frame_stab)))


    cap.release()
    out_stab.release()
    out_side.release()

    print("Done.")
    print(f"Method: {opts.method}")
    print(f"Stabilized video: {opts.video_out_stab}")
    print(f"Comparison:       {opts.video_out_comare}")
    
    # Make visualizations
    save_example_frames_before_after(opts, frame_indices=(0, 50, 150, 250))
    plot_trajectory(opts, trajectory, smoothed_trajectory)
    visualize_warp_example(opts, transforms, transforms_smooth, example_frame=150)


if __name__ == "__main__":
    parser = create_parser()
    opts, _ = parser.parse_known_args()

    opts.video_out_stab = opts.video_out_dir + 'stab_video_' + opts.method + '.mp4'
    opts.video_out_comare = opts.video_out_dir + 'compare_before_after_' + opts.method + '.mp4'
    print_opts(opts)
    main(opts)

