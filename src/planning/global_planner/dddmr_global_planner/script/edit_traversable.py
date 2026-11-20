#!/usr/bin/env python3
"""Offline editor for *_traversable.pcd (xyz + dGraph intensity).

Commands on /traversable_editor/command (std_msgs/String):
  lethal  - mark selected points as blocked (intensity=0.25)
  free    - mark selected points as traversable (intensity=9999)
  save    - write the cloud back to traversable_file
  reload  - reload from disk
"""

import os
import struct

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String


LETHAL_INTENSITY = 0.25
FREE_INTENSITY = 9999.0
MATCH_RADIUS = 0.2


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
    )


class TraversableEditor(Node):
    def __init__(self):
        super().__init__('traversable_editor')
        self.declare_parameter('traversable_file', '')
        self.declare_parameter('global_frame', 'map')
        self.traversable_file = self.get_parameter('traversable_file').get_parameter_value().string_value
        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value

        self._points = []
        self._selected = []

        self.cloud_pub = self.create_publisher(PointCloud2, 'mapground', _latched_qos())
        self.dgraph_pub = self.create_publisher(PointCloud2, 'map/dGraph', _latched_qos())
        self.create_subscription(
            PointCloud2,
            '/point_cloud_selection/selected_points',
            self._on_selected,
            10,
        )
        self.create_subscription(String, '/traversable_editor/command', self._on_command, 10)

        if not self.traversable_file:
            self.get_logger().error('traversable_file is empty')
            return
        if not self._load():
            return
        self._publish()
        self.get_logger().info(
            f'Loaded {len(self._points)} pts from {self.traversable_file}. '
            'Select points in RViz, then Mark Lethal / Mark Free / Save.'
        )

    def _load(self) -> bool:
        path = self.traversable_file
        if not os.path.isfile(path):
            self.get_logger().error(f'File not found: {path}')
            return False
        try:
            self._points = _load_pcd_xyz_i(path)
            return True
        except Exception as exc:
            self.get_logger().error(f'Failed to load {path}: {exc}')
            return False

    def _publish(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.global_frame
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg = point_cloud2.create_cloud(header, fields, self._points)
        self.cloud_pub.publish(msg)
        self.dgraph_pub.publish(msg)

    def _on_selected(self, msg: PointCloud2):
        self._selected = [
            (float(p[0]), float(p[1]), float(p[2]))
            for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        ]
        self.get_logger().info(f'Selected {len(self._selected)} points')

    def _on_command(self, msg: String):
        cmd = (msg.data or '').strip().lower()
        if cmd == 'reload':
            if self._load():
                self._publish()
            return
        if cmd == 'save':
            self._save()
            return
        if cmd in ('lethal', 'free'):
            intensity = LETHAL_INTENSITY if cmd == 'lethal' else FREE_INTENSITY
            n = self._paint(intensity)
            self.get_logger().info(f'Marked {n} points as {cmd}')
            self._publish()
            return
        self.get_logger().warn(f'Unknown command: {msg.data}')

    def _paint(self, intensity: float) -> int:
        if not self._selected:
            self.get_logger().warn('No selected points')
            return 0
        changed = 0
        r2 = MATCH_RADIUS * MATCH_RADIUS
        for sx, sy, sz in self._selected:
            best_i = -1
            best_d = r2
            for i, (x, y, z, _) in enumerate(self._points):
                d = (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2
                if d <= best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                x, y, z, _ = self._points[best_i]
                self._points[best_i] = (x, y, z, intensity)
                changed += 1
        return changed

    def _save(self):
        try:
            _save_pcd_xyz_i(self.traversable_file, self._points)
            self.get_logger().info(f'Saved {len(self._points)} pts to {self.traversable_file}')
        except Exception as exc:
            self.get_logger().error(f'Save failed: {exc}')


def _load_pcd_xyz_i(path: str):
    with open(path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('truncated PCD header')
            header_lines.append(line.decode('ascii', errors='replace').strip())
            if header_lines[-1].upper().startswith('DATA '):
                break
        fields = {}
        for line in header_lines:
            if ' ' not in line:
                continue
            key, rest = line.split(' ', 1)
            fields[key.upper()] = rest
        names = fields.get('FIELDS', 'x y z').split()
        sizes = [int(x) for x in fields.get('SIZE', '4 4 4').split()]
        types = fields.get('TYPE', 'F F F').split()
        width = int(fields.get('WIDTH', '0'))
        height = int(fields.get('HEIGHT', '1'))
        n = int(fields.get('POINTS', str(width * height)))
        data_kind = fields.get('DATA', 'ascii').split()[0].lower()
        idx = {name: i for i, name in enumerate(names)}
        if not all(k in idx for k in ('x', 'y', 'z')):
            raise ValueError(f'PCD missing xyz fields: {names}')
        has_i = 'intensity' in idx
        if data_kind == 'ascii':
            rest = f.read().decode('ascii', errors='replace').split()
            stride = len(names)
            out = []
            for p in range(n):
                row = rest[p * stride:(p + 1) * stride]
                intensity = float(row[idx['intensity']]) if has_i else FREE_INTENSITY
                out.append((float(row[idx['x']]), float(row[idx['y']]), float(row[idx['z']]), intensity))
            return out
        fmt_map = {('F', 4): 'f', ('F', 8): 'd', ('I', 4): 'i', ('U', 4): 'I', ('U', 1): 'B'}
        struct_fmt = '<'
        for size, typ in zip(sizes, types):
            struct_fmt += fmt_map.get((typ, size), 'f')
        step = struct.calcsize(struct_fmt)
        raw = f.read(step * n)
        out = []
        for p in range(n):
            rec = struct.unpack_from(struct_fmt, raw, p * step)
            intensity = float(rec[idx['intensity']]) if has_i else FREE_INTENSITY
            out.append((float(rec[idx['x']]), float(rec[idx['y']]), float(rec[idx['z']]), intensity))
        return out


def _save_pcd_xyz_i(path: str, points):
    n = len(points)
    header = (
        '# .PCD v0.7 - Point Cloud Data file format\n'
        'VERSION 0.7\n'
        'FIELDS x y z intensity\n'
        'SIZE 4 4 4 4\n'
        'TYPE F F F F\n'
        'COUNT 1 1 1 1\n'
        f'WIDTH {n}\n'
        'HEIGHT 1\n'
        'VIEWPOINT 0 0 0 1 0 0 0\n'
        f'POINTS {n}\n'
        'DATA ascii\n'
    )
    with open(path, 'w', encoding='ascii') as f:
        f.write(header)
        for x, y, z, intensity in points:
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {intensity:.6f}\n')


def main():
    rclpy.init()
    node = TraversableEditor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
