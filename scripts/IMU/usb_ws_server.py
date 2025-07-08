import asyncio
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from foxglove_websocket.server import FoxgloveServer, FoxgloveServerListener
from foxglove_websocket.types import ChannelId
import websockets


FOXGLOVE_PORT = 8700
IMU_WEBSOCKET_PORT = 8001
SERVER_START_TIME = time.time()

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

        # Orientation update
        delta_rot = R.from_rotvec(np.array(gyro) * dt)
        self.orientation = self.orientation * delta_rot

        # Acceleration update in world frame
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

class Listener(FoxgloveServerListener):
    async def on_subscribe(self, server: FoxgloveServer, channel_id: ChannelId):
        print(f"Client subscribed to channel {channel_id}")

imu_tracker = IMUTracker()

async def imu_websocket_handler(websocket, server, channel_id, tf_channel_id):
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

            transform = imu_tracker.update(accel, gyro, timestamp)
            translation = transform[:3, 3].tolist()
            rotation = R.from_matrix(transform[:3, :3]).as_quat().tolist()

            # Compute consistent timestamp
            elapsed = time.time() - SERVER_START_TIME
            sec = int(elapsed)
            nsec = int((elapsed - sec) * 1e9)
            timestamp_ns = int(elapsed * 1e9)

            tf_msg = {
                "transforms": [
                    {
                        "header": {
                            "stamp": {"sec": sec, "nsec": nsec},
                            "frame_id": "imu_base"
                        },
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

            await server.send_message(
                channel_id,
                timestamp_ns,
                json.dumps(imu_msg).encode("utf-8"),
            )

            await server.send_message(
                tf_channel_id,
                timestamp_ns,
                json.dumps(tf_msg).encode("utf-8"),
            )

        except Exception as e:
            print(f"Error processing IMU data: {e}")

async def start_imu_server():
    async with FoxgloveServer(
        "0.0.0.0", 
        FOXGLOVE_PORT, 
        "IMU Server", 
        capabilities=["clientPublish"],
        supported_encodings=["json"],
    ) as server:
        server.set_listener(Listener())

        imu_channel_id = await server.add_channel(
            {
                "topic": "imu_data",
                "encoding": "json",
                "schemaName": "IMUData",
                "schemaEncoding": "jsonschema",
                "schema": json.dumps({
                    "type": "object",
                    "properties": {
                        "accel_x": {"type": "number"},
                        "accel_y": {"type": "number"},
                        "accel_z": {"type": "number"},
                        "gyro_x": {"type": "number"},
                        "gyro_y": {"type": "number"},
                        "gyro_z": {"type": "number"},
                    },
                }),
            }
        )

        tf_channel_id = await server.add_channel(
            {
                "topic": "/tf",
                "encoding": "json",
                "schemaName": "TFMessage",
                "schemaEncoding": "jsonschema",
                "schema": json.dumps({
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "integer"},
                        "frame_id": {"type": "string"},
                        "child_frame_id": {"type": "string"},
                        "translation": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"}
                            },
                        },
                        "rotation": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "w": {"type": "number"}
                            },
                        }
                    },
                }),
            }
        )

        elapsed = time.time() - SERVER_START_TIME
        sec = int(elapsed)
        nsec = int((elapsed - sec) * 1e9)

        static_tf_msgs = [
            {
                "transforms": [
                    {
                        "header": {
                            "stamp": {"sec": sec, "nsec": nsec},
                            "frame_id": "world"
                        },
                        "child_frame_id": "imu_base",
                        "transform": {
                            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                        }
                    }
                ]
            },
            {
                "transforms": [
                    {
                        "header": {
                            "stamp": {"sec": sec, "nsec": nsec},
                            "frame_id": "imu_base"
                        },
                        "child_frame_id": "imu_link",
                        "transform": {
                            "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                        }
                    }
                ]
            }
        ]

        for tf_msg in static_tf_msgs:
            await server.send_message(
                tf_channel_id,
                int(elapsed * 1e9),
                json.dumps(tf_msg).encode("utf-8"),
            )

        imu_server = websockets.serve(
            lambda websocket: imu_websocket_handler(websocket, server, imu_channel_id, tf_channel_id),
            "0.0.0.0",
            IMU_WEBSOCKET_PORT,
        )




        await imu_server
        print(f"Foxglove server running on ws://localhost:{FOXGLOVE_PORT}")
        print(f"Receiving IMU data on ws://localhost:{IMU_WEBSOCKET_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(start_imu_server())
