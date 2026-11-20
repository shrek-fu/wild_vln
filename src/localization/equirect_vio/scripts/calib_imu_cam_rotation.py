#!/usr/bin/env python3
"""Estimate the camera-to-IMU rotation (OpenVIO rig.imu_from_camera) of a live equirectangular camera.

Rotate the camera in place by hand around all three axes (yaw, pitch, roll) while this runs. For every image pair
the camera rotation is solved from tracked unit rays (Kabsch + RANSAC, pure rotation), the IMU rotation from the
gyro; the rotation R with w_imu = R w_cam is then solved over all pairs, together with a time offset scan.

Camera frame (OpenVIO EQUIRECTANGULAR): x right, y down, z forward;
u = W (0.5 + atan2(x, z) / 2pi), v = H (0.5 + asin(y) / pi).

Usage: python3 calib_imu_cam_rotation.py [--duration 40] [--image /equirectangular/image] [--imu /imu/data_raw]
"""

import argparse
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2 * np.sin(th)) * w


def kabsch(a, b):
    """R minimizing |b - R a| for row-vector sets a, b."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    return Vt.T @ D @ U.T


def pixels_to_rays(pts, w, h):
    lon = (pts[:, 0] / w - 0.5) * 2 * np.pi
    lat = (pts[:, 1] / h - 0.5) * np.pi
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], axis=1)


def robust_rotation(a, b, rng, iters=80, thresh_deg=0.3):
    n = len(a)
    if n < 10:
        return None
    thresh = np.deg2rad(thresh_deg)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        M = kabsch(a[idx], b[idx])
        err = np.arccos(np.clip(np.sum((a @ M.T) * b, axis=1), -1, 1))
        inl = err < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < max(10, 0.5 * n):
        return None
    return kabsch(a[best], b[best])


class Collector(Node):
    def __init__(self, image_topic, imu_topic, scale):
        super().__init__("calib_imu_cam_rotation")
        self.scale = scale
        self.imu = []
        self.frames = []  # (t, gray)
        self.create_subscription(Imu, imu_topic, self.on_imu,
                                 QoSProfile(depth=2000, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(Image, image_topic, self.on_image,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

    def on_imu(self, m):
        self.imu.append((stamp(m), m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z))

    def on_image(self, m):
        ch = {"rgba8": 4, "bgra8": 4, "rgb8": 3, "bgr8": 3, "mono8": 1}.get(m.encoding)
        if ch is None:
            return
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step // ch if ch else m.width, ch)[:, :m.width]
        gray = img[:, :, 0] if ch == 1 else cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.frames.append((stamp(m), gray))


def visual_rotations(frames, rng):
    """Camera rotation vectors (camera frame) between consecutive frames."""
    out = []
    h, w = frames[0][1].shape
    lat_ok = lambda p: np.abs(p[:, 1] / h - 0.5) < 0.33  # skip poles / tripod
    for (t0, g0), (t1, g1) in zip(frames[:-1], frames[1:]):
        p0 = cv2.goodFeaturesToTrack(g0, 400, 0.01, 12)
        if p0 is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(21, 21), maxLevel=3)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb - p0, axis=2)[:, 0] < 0.5)
        a, b = p0[ok, 0], p1[ok, 0]
        keep = lat_ok(a) & lat_ok(b)
        # Horizontal wrap: tracks never cross the seam with LK, nothing to fix.
        a, b = a[keep], b[keep]
        if len(a) < 20:
            continue
        M = robust_rotation(pixels_to_rays(a, w, h), pixels_to_rays(b, w, h), rng)
        if M is None:
            continue
        # b = M a  =>  camera k+1 orientation relative to k is M^T
        out.append((t0, t1, log_so3(M.T)))
    return out


def gyro_rotation(imu, t0, t1):
    t = imu[:, 0]
    i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
    if i0 == 0 or i1 >= len(t):
        return None
    R = np.eye(3)
    knots = np.concatenate([[t0], t[i0:i1], [t1]])
    for ta, tb in zip(knots[:-1], knots[1:]):
        wm = np.array([np.interp(0.5 * (ta + tb), t, imu[:, k]) for k in (1, 2, 3)])
        R = R @ exp_so3(wm * (tb - ta))
    return log_so3(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--image", default="/equirectangular/image")
    ap.add_argument("--imu", default="/imu/data_raw")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--min-deg", type=float, default=1.0, help="ignore pairs rotating less than this")
    args = ap.parse_args()

    rclpy.init()
    node = Collector(args.image, args.imu, args.scale)
    print(f"Collecting {args.duration:.0f} s — rotate the camera in place around ALL three axes now...")
    end = node.get_clock().now().nanoseconds * 1e-9 + args.duration
    while node.get_clock().now().nanoseconds * 1e-9 < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    frames, imu = node.frames, np.array(node.imu)
    node.destroy_node()
    rclpy.shutdown()
    print(f"frames={len(frames)} imu={len(imu)}")
    if len(frames) < 50 or len(imu) < 100:
        sys.exit("Not enough data")

    rng = np.random.default_rng(0)
    vis = visual_rotations(frames, rng)
    print(f"visual rotations={len(vis)}")

    # Static gyro bias from the calmest 10% of samples
    mag = np.linalg.norm(imu[:, 1:], axis=1)
    bias = imu[mag <= np.percentile(mag, 10), 1:].mean(axis=0)
    imu_c = imu.copy()
    imu_c[:, 1:] -= bias

    best = None
    for dt in np.arange(-0.08, 0.0801, 0.002):
        A, B = [], []
        for t0, t1, wc in vis:
            if np.rad2deg(np.linalg.norm(wc)) < args.min_deg:
                continue
            wi = gyro_rotation(imu_c, t0 + dt, t1 + dt)
            if wi is None:
                continue
            A.append(wc)
            B.append(wi)
        if len(A) < 30:
            continue
        A, B = np.array(A), np.array(B)
        for sign in (1.0, -1.0):
            R = kabsch(sign * A, B)
            cost = np.median(np.linalg.norm(B - sign * A @ R.T, axis=1))
            if best is None or cost < best[0]:
                best = (cost, dt, R, sign * A, B)
    if best is None:
        sys.exit("Not enough rotating pairs — rotate more (and around all axes)")

    cost, dt, R, A, B = best
    sv = np.linalg.svd(A, compute_uv=False)
    print(f"\npairs={len(A)}  time offset (t_imu = t_img + dt): {1e3 * dt:+.1f} ms")
    print(f"median residual {np.rad2deg(cost):.3f} deg/pair; excitation singular values {np.round(sv, 3)}")
    if sv[-1] < 0.1 * sv[0]:
        print("WARNING: one axis was barely rotated; rotate around all three axes and run again.")
    print("\nR (w_imu = R w_cam), rounded to nearest axis permutation for reference:")
    print(np.round(R).astype(int))
    print("\nYAML (rig.imu_from_camera, translation left at 0):")
    print("  imu_from_camera:")
    for i in range(3):
        print(f"    - [{R[i, 0]:.6f}, {R[i, 1]:.6f}, {R[i, 2]:.6f}, 0.0]")
    print("    - [0.0, 0.0, 0.0, 1.0]")
    g_cam = R.T @ np.array([-1.0, 0.0, 0.0])
    print(f"\nCheck: IMU -x (measured 'up' at rest) in camera frame = {np.round(g_cam, 3)} (expect ~[0,-1,0] if upright)")


if __name__ == "__main__":
    main()
