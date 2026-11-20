std_msgs/Header header
string camera_name

# From PreviewParam. delay_timestamp and sweep_time align IMU samples with frames.
int64 delay_timestamp
int64 sweep_time
int32 acceleration_range
int32 gyro_range

uint32 calib_width
uint32 calib_height

# Index 0 is the lower-cx lens, index 1 the other. Angles are degrees, as stored.
float64[2] radius
float64[2] cx
float64[2] cy
float64[2] yaw_deg
float64[2] pitch_deg
float64[2] roll_deg

# Offset v2 lens-center quaternion and translation, in the camera's stored units.
bool has_v2
float64[2] v2_qx
float64[2] v2_qy
float64[2] v2_qz
float64[2] v2_qw
float64[2] v2_tx
float64[2] v2_ty
float64[2] v2_tz

# Offset v3 omnidirectional intrinsics. k1..p2 follow that version's tx,ty,tz.
bool has_v3
float64[2] v3_xi
float64[2] v3_fx
float64[2] v3_fy
float64[2] v3_k1
float64[2] v3_k2
float64[2] v3_k3
float64[2] v3_p1
float64[2] v3_p2

string offset_v1
string offset_v2
string offset_v3
