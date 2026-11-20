"""ROS 2 node: ``topo_map_node`` options plus the ``--hf_*`` options of ``hydra_fake.core``.

Same topics and outputs as the topo map (``objects_4d.json``, ``topo_graph.json``, ``/topo/graph``,
markers, track centres); the graph is in the corrected world frame, survives LIO resets, and carries the
``free_places`` layer and per-object ``approach`` poses.
"""

from __future__ import annotations

import argparse
import sys

from scene_memory.nodes.topo_map_node import _parse_args, main as topo_main

from hydra_fake.core import HydraFakeMapper, add_args


def parse(argv=None) -> argparse.Namespace:
    hf, rest = add_args(argparse.ArgumentParser(add_help=False)).parse_known_args(argv)
    args = _parse_args(rest)
    for k, v in vars(hf).items():
        setattr(args, k, v)
    return args


def main(argv=None) -> None:
    topo_main(parse(sys.argv[1:] if argv is None else argv), mapper_factory=HydraFakeMapper,
              node_name="hydra_fake", mapper_name="hydra_fake")


if __name__ == "__main__":
    main()
