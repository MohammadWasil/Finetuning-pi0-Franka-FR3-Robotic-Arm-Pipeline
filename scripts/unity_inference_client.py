from collections import deque
import socket
import struct
import numpy as np
from PIL import Image
import io
import tyro
from openpi_client import websocket_client_policy


def recv_exact(conn: socket.socket, num_bytes: int) -> bytes:
    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = conn.recv(num_bytes - len(buf))
        if not chunk:
            raise ConnectionError("Unity disconnected.")
        buf.extend(chunk)
    return bytes(buf)


def main(
    policy_host: str = "127.0.0.1",
    policy_port: int = 8000,
    unity_host: str = "127.0.0.1",
    unity_port: int = 5005,
    prompt: str = "pick up the red ball and place it on the green platform",
    replan_steps: int = 15,
) -> None:
    """
    Bridge Unity TCP frames/states to openpi's WebSocket policy server and return predicted actions.

    This script connects the Unity simulation on Windows to the openpi AI model running on our cloud GPU.
    
    1. Connects to Both Sides (main startup):
	Opens a WebSocket client (WebsocketClientPolicy) to localhost:8000 (which is SSH-tunneled straight to serve_policy.py GPU)
	Starts a local TCP server on localhost:5005 and waits for Unity to hit Play and connect.

    2.Receives Live Observation Packets from Unity
    Every 1/15th of a second (~66.7 ms), Unity sends a single binary packet containing
    
    3. Executes Chunked Actions
    Calling the 3B π₀ server every frame (~80-120 ms) would lag Unity, so whenever action_queue 
    is empty (or reset_flag == 1), the script sends the decoded images, state, and text prompt 
    to π₀, receives a 50-step action chunk (~3.3 s), and queues the first 15 steps (1.0 s).
    For the next 14 frames, it pops queued actions in < 0.1 ms with zero network delay, 
    then re-observes and replans on step 15.

    4. Sends the Target (Predicted) Action Back to Unity
    Packs the single 8-float action vector (32 bytes) and sends it back over TCP to Unity, 
    which applies those 8 joint targets to the Franka FR3's ArticulationBody drives.

    Args:
        policy_host: The host of the openpi WebSocket policy server (default: localhost).
        policy_port: The port of the openpi WebSocket policy server (default: 8000).
        unity_host: The host of the Unity TCP server (default: localhost).
        unity_port: The port of the Unity TCP server (default: 5005).
        prompt: The text prompt to provide to the policy model for task specification.
        replan_steps: The number of steps to execute before re-observing and replanning.
    """

    print(f"Connecting to openpi policy server at ws://{policy_host}:{policy_port} ...")
    policy = websocket_client_policy.WebsocketClientPolicy(host=policy_host, port=policy_port)
    print(f"Connected to policy server! Metadata: {policy.get_server_metadata()}")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((unity_host, unity_port))
    server_sock.listen(1)
    print(f"Waiting for Unity simulation on {unity_host}:{unity_port} ...")

    conn, addr = server_sock.accept()
    print(f"Unity connected from {addr}")

    action_queue: deque[np.ndarray] = deque()

    try:
        while True:
            # Packet header from Unity:
            # 1 byte reset_flag + 8 floats (state: 32 bytes) + 4B ext_len + 4B wrist_len = 41 bytes
            header = recv_exact(conn, 41)
            reset_flag = header[0]
            state = np.frombuffer(header[1:33], dtype="<f4").copy()
            ext_len, wrist_len = struct.unpack("<II", header[33:41])

            ext_bytes = recv_exact(conn, ext_len)
            wrist_bytes = recv_exact(conn, wrist_len)

            if reset_flag == 1:
                action_queue.clear()

            if len(action_queue) == 0:
                ext_img = np.array(Image.open(io.BytesIO(ext_bytes)).convert("RGB"), dtype=np.uint8)
                wrist_img = np.array(Image.open(io.BytesIO(wrist_bytes)).convert("RGB"), dtype=np.uint8)

                obs = {
                    "observation/image": ext_img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": prompt,
                }
                result = policy.infer(obs)
                action_chunk = np.asarray(result["actions"], dtype=np.float32)  # shape: (50, 8)
                for step_action in action_chunk[:replan_steps]:
                    action_queue.append(step_action)

            next_action = action_queue.popleft().astype("<f4")
            conn.sendall(next_action.tobytes())  # 32 bytes (8 float32 values)
    finally:
        conn.close()
        server_sock.close()


if __name__ == "__main__":
    tyro.cli(main)
