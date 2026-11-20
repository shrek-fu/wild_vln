#!/usr/bin/env python3
"""Report whether an Insta360 (vendor 2e1a) is on USB. Does not require the SDK."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vendor", default="2e1a")
    args = ap.parse_args()
    vendor = args.vendor.lower()
    found = []
    for dev in Path("/sys/bus/usb/devices").glob("*"):
        vid = dev / "idVendor"
        if not vid.is_file():
            continue
        if vid.read_text(encoding="utf-8").strip().lower() != vendor:
            continue
        pid = (dev / "idProduct").read_text(encoding="utf-8").strip() if (dev / "idProduct").is_file() else "?"
        speed = (dev / "speed").read_text(encoding="utf-8").strip() if (dev / "speed").is_file() else "?"
        product = (dev / "product").read_text(encoding="utf-8").strip() if (dev / "product").is_file() else ""
        busnum = int((dev / "busnum").read_text(encoding="utf-8"))
        devnum = int((dev / "devnum").read_text(encoding="utf-8"))
        node = Path(f"/dev/bus/usb/{busnum:03d}/{devnum:03d}")
        found.append((dev.name, pid, speed, product, node))
    if not found:
        print(f"Insta360 USB vendor {vendor} not found. Plug X4 Air into USB 3.0 and choose Android mode.")
        return 1
    writable = True
    for name, pid, speed, product, node in found:
        usb3 = speed.isdigit() and int(speed) >= 5000
        print(f"found {name} idProduct={pid} speed={speed} product={product!r} usb3={usb3} node={node}")
        if not os.access(node, os.R_OK | os.W_OK):
            writable = False
            print(f"{node} is not writable by this user, so the SDK cannot open the camera. Install the udev rule once:")
            print("  sudo cp config/udev/99-insta360.rules /etc/udev/rules.d/")
            print("  sudo udevadm control --reload-rules && sudo udevadm trigger --attr-match=idVendor=2e1a")
    return 0 if writable else 1


if __name__ == "__main__":
    raise SystemExit(main())
