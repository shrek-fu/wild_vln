# apps/

Offline VIO is provided by the sibling OpenVIO binary:

```bash
../openvio/build/bin/vio odometry \
  --dataset_path ../data/raw/session01 \
  --config_path ../config/x4_air_equirect_1440.yaml \
  --output_path ../data/processed/session01/traj.tum
```

Live streaming is the ROS 2 node `equirect_vio_node` under `ros2_ws/src/equirect_vio/`.
