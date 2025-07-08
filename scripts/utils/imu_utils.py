import asyncio
import threading
import websockets
import json
import numpy as np
from scipy.spatial.transform import Rotation as R


class IMUTracker:
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = R.identity()
        self.last_timestamp = None

    def update(self, accel, gyro, timestamp):
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return self.get_transform()

        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        delta_rot = R.from_rotvec(np.array(gyro) * dt)
        self.orientation = self.orientation * delta_rot

        accel_world = self.orientation.apply(accel)
        accel_world[2] -= 9.81  # Remove gravity

        self.velocity += accel_world * dt
        self.position += self.velocity * dt

        return self.get_transform()

    def get_transform(self):
        T = np.eye(4)
        T[:3, :3] = self.orientation.as_matrix()
        T[:3, 3] = self.position
        return T


class IMU:
    def __init__(self, port=8001):
        self.port = port
        self.latest_data = None
        self.tf_msg = None
        self.imu_tracker = IMUTracker()
        self.lock = threading.Lock()
        self._start_server_in_background()

    def _start_server_in_background(self):
        thread = threading.Thread(target=self._run_server_loop, daemon=True)
        thread.start()

    def _run_server_loop(self):
        asyncio.run(self._main())

    async def _handler(self, websocket):
        async for message in websocket:
            try:
                data = json.loads(message)
                timestamp = data["timestamp"]
                accel = data["imu"]["Samsung Linear Acceleration Sensor"]["values"]
                gyro = data["imu"]["ICM42632M Gyroscope"]["values"]

                imu_msg = {
                    "accel_x": accel[0],
                    "accel_y": accel[1],
                    "accel_z": accel[2],
                    "gyro_x": gyro[0],
                    "gyro_y": gyro[1],
                    "gyro_z": gyro[2],
                }

                transform = self.imu_tracker.update(accel, gyro, timestamp)
                translation = transform[:3, 3].tolist()
                rotation = R.from_matrix(transform[:3, :3]).as_quat().tolist()

                tf_msg = {
                    "transforms": [
                        {
                            "child_frame_id": "imu_link",
                            "transform": {
                                "translation": {
                                    "x": translation[0],
                                    "y": translation[1],
                                    "z": translation[2]
                                },
                                "rotation": {
                                    "x": rotation[0],
                                    "y": rotation[1],
                                    "z": rotation[2],
                                    "w": rotation[3]
                                }
                            }
                        }
                    ]
                }

                with self.lock:
                    self.latest_data = imu_msg
                    self.tf_msg = tf_msg

            except Exception as e:
                print(f"[IMU] Error processing message: {e}")

    async def _main(self):
        print(f"[IMU] WebSocket server running on ws://0.0.0.0:{self.port}")
        async with websockets.serve(self._handler, "0.0.0.0", self.port):
            await asyncio.Future()  # run forever

    def get_data(self):
        with self.lock:
            return self.latest_data

    def get_tf_data(self):
        with self.lock:
            return self.tf_msg
