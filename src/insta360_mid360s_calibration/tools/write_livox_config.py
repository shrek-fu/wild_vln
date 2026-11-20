#!/usr/bin/env python3
"""Write the Livox config (one or two lidars) from config/rig.yaml. Does not touch livox_ws."""
from __future__ import annotations

import argparse
import ipaddress
import json
import subprocess
from pathlib import Path

import yaml


def local_ipv4() -> list:
    out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True).stdout
    return [ipaddress.ip_interface(line.split()[3]) for line in out.splitlines() if len(line.split()) > 3]


def host_ip(cfg: dict) -> str:
    """rig.host_ip if this machine has it, else this machine's address on the lidars' subnet.

    The same rig.yaml then works on the PC (192.168.1.5) and on the Orin (192.168.1.102)."""
    want = cfg["rig"]["host_ip"]
    addrs = local_ipv4()
    if not addrs or any(str(a.ip) == want for a in addrs):
        return want
    lidar = ipaddress.ip_address(cfg["lidars"][0]["ip"])
    for a in addrs:
        if lidar in a.network:
            print(f"rig.host_ip {want} is not on this machine; using {a.ip} ({a.network}) for the lidars")
            return str(a.ip)
    print(f"WARNING: no local address on the subnet of {lidar}; keeping rig.host_ip {want}")
    return want


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rig", type=Path, default=root / "config" / "rig.yaml")
    ap.add_argument("--output", type=Path, default=root / "config" / "MID360s_dual.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.rig.read_text(encoding="utf-8"))
    livox = cfg["livox"]
    doc = {
        "lidar_summary_info": {"lidar_type": 8},
        "Mid360s": {
            "lidar_net_info": {
                "cmd_data_port": 56100,
                "push_msg_port": 56200,
                "point_data_port": 56300,
                "imu_data_port": 56400,
                "log_data_port": 56500,
            },
            "host_net_info": [
                {
                    "host_ip": host_ip(cfg),
                    "cmd_data_port": 56101,
                    "push_msg_port": 56201,
                    "point_data_port": 56301,
                    "imu_data_port": 56401,
                    "log_data_port": 56501,
                }
            ],
        },
        "lidar_configs": [
            {
                "ip": lidar["ip"],
                "pcl_data_type": int(livox["pcl_data_type"]),
                "pattern_mode": int(livox["pattern_mode"]),
                "extrinsic_parameter": {
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "x": 0,
                    "y": 0,
                    "z": 0,
                },
            }
            for lidar in cfg["lidars"]
        ],
    }
    args.output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
