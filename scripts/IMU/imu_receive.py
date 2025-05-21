import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
import tf2_ros
import json
import numpy as np
import websockets
import asyncio
from scipy.spatial.transform import Rotation as R

IMU_WEBSOCKET_PORT = 8001

class IMUTracker:
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = R.identity()
        self.last_timestamp = None
        self.last_position = np.zeros(3)
        self.last_velocity = np.zeros(3)
        self.last_orientation = R.identity()
        self.velocity_threshold = 0.001  # Threshold to assume zero velocity
        self.alpha = 0.97  # Complementary filter alpha for orientation fusion
        self.last_accel = np.zeros(3)
        self.timestamp_diff = 0.1  # Assumed IMU update rate of 2 seconds

    def update(self, accel, gyro, timestamp):
        accel = np.array(accel)  # Convert accel to numpy array
        gyro = np.array(gyro)  # Convert gyro to numpy array
        
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return self.get_transform()

        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        # Predict position and velocity over the gap since the last update
        if dt > self.timestamp_diff:  # Assuming a gap of ~2 seconds
            predicted_velocity = self.last_velocity + accel * dt
            predicted_position = self.last_position + self.last_velocity * dt + 0.5 * accel * dt ** 2
        else:
            predicted_velocity = self.last_velocity
            predicted_position = self.last_position

        # Update orientation using complementary filter
        accel_angle = self.accel_to_angle(accel)
        gyro_angle = self.last_orientation.as_euler('xyz', degrees=True) + gyro * dt
        gyro_angle = np.mod(gyro_angle, 360)

        fused_angle = self.alpha * gyro_angle + (1 - self.alpha) * accel_angle
        self.orientation = R.from_euler('xyz', fused_angle, degrees=True)

        # Apply gravity compensation
        accel_world = self.orientation.apply(accel)
        #accel_world[2] -= 9.81  # Gravity compensation

        # Zero Velocity Update (ZUPT) to prevent drift when stationary
        if np.linalg.norm(accel_world) < self.velocity_threshold:
            self.velocity = np.zeros(3)  # Reset velocity
            accel_world = np.zeros(3)  # No acceleration, no movement
        else:
            self.velocity = predicted_velocity + accel_world * dt
            self.position = predicted_position + self.velocity * dt
            # Apply a small drift prevention constant
            drift_prevention = 1e-6
            self.position += drift_prevention * np.sign(self.velocity)


        # Update velocity and position
        self.velocity = predicted_velocity + accel_world * dt
        self.position = predicted_position + self.velocity * dt

        return self.get_transform()

    def accel_to_angle(self, accel):
        """Convert accelerometer readings to pitch and roll angles (in degrees)"""
        pitch = np.arctan2(accel[1], np.sqrt(accel[0]**2 + accel[2]**2)) * (180.0 / np.pi)
        roll = np.arctan2(-accel[0], accel[2]) * (180.0 / np.pi)
        return np.array([pitch, roll, 0])  # Using pitch and roll; yaw can be left out for simplicity

    def get_transform(self):
        """Return the transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = self.orientation.as_matrix()
        T[:3, 3] = self.position
        return T

class IMUPublisher(Node):
    def __init__(self):
        super().__init__('imu_publisher')
        self.imu_pub = self.create_publisher(Imu, 'imu_data', 10)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.tracker = IMUTracker()
        self.get_logger().info("IMU Publisher Node Initialized.")

    async def imu_websocket_server(self):
        async def handler(websocket, path):
            self.get_logger().info(f"New WebSocket client connected from {websocket.remote_address}")
            async for message in websocket:
                try:
                    data = json.loads(message)
                    #print(data)
                    accel = [float(v) for v in data["imu"]["Samsung Linear Acceleration Sensor"]["values"]]
                    gyro = [float(v) for v in data["imu"]["ICM42632M Gyroscope"]["values"]]
                    timestamp = float(data["timestamp"])
                    self.get_logger().debug(f"Received IMU data at timestamp {timestamp}")
                    self.get_logger().debug(f"Accel: {accel}, Gyro: {gyro}")

                    # Update IMU tracker and get transformation
                    transform = self.tracker.update(accel, gyro, timestamp)
                    translation = transform[:3, 3].tolist()
                    rotation = R.from_matrix(transform[:3, :3]).as_quat().tolist()

                    # Publish IMU message
                    imu_msg = Imu()
                    imu_msg.header.stamp = self.get_clock().now().to_msg()
                    imu_msg.header.frame_id = "imu_link"
                    imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z = accel
                    imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z = gyro
                    imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w = rotation
                    imu_msg.orientation_covariance = [1e-3] * 9
                    imu_msg.angular_velocity_covariance = [1e-3] * 9
                    imu_msg.linear_acceleration_covariance = [1e-2] * 9

                    self.imu_pub.publish(imu_msg)
                    self.get_logger().debug("Published IMU message.")

                    # Publish TF transform
                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = "world"
                    t.child_frame_id = "imu_base"
                    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = translation
                    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = rotation
                    self.br.sendTransform(t)
                    self.get_logger().debug("Published TF transform.")

                except Exception as e:
                    self.get_logger().error(f"Failed to process IMU data: {e}")

        self.get_logger().info(f"Starting WebSocket server on port {IMU_WEBSOCKET_PORT}")
        server = await websockets.serve(handler, "0.0.0.0", IMU_WEBSOCKET_PORT)
        self.get_logger().info("WebSocket server running.")
        await server.wait_closed()

def main():
    rclpy.init()
    node = IMUPublisher()
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(node.imu_websocket_server())
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down on user interrupt.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
