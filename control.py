import socket
import struct
import subprocess
import threading
import time
import random
from pathlib import Path

import av
import numpy as np

from algo.algo_base import TouchAction


class DeviceController:
    serial: str | None
    session_id: str
    video_socket: socket.socket
    control_socket: socket.socket
    server_process: subprocess.Popen
    streaming_collector: threading.Thread
    control_collector: threading.Thread
    device_width: int
    device_height: int
    video_width: int
    video_height: int
    collector_running: bool
    immediate_send_lock: threading.Lock
    latest_frame_lock: threading.Lock
    latest_frame: np.ndarray | None
    decode_fps_lock: threading.Lock
    decode_fps: float
    latest_frame_timestamp_lock: threading.Lock
    latest_frame_timestamp: float | None
    latest_frame_seq_lock: threading.Lock
    latest_frame_seq: int
    stream_crop_rect: tuple[int, int, int, int]

    def __init__(
        self,
        serial: str | None = None,
        port: int = 27188,
        push_server: bool = True,
        server_dir: str | Path | None = None,
        max_fps: int = 60,
        video_bit_rate: int | None = None,
        video_crop: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.serial = serial
        adb = ("adb",) if serial is None else ("adb", "-s", serial)
        self.session_id = format(random.randint(0, 0x7FFFFFFF), "08x")
        if server_dir is None:
            server_dir_path = Path(__file__).resolve().parent
        else:
            server_dir_path = Path(server_dir)

        server_candidates = sorted(server_dir_path.glob("scrcpy-server-v*"))
        if not server_candidates:
            raise FileNotFoundError(f"No scrcpy-server-v* found in {server_dir_path}")
        server_file = str(server_candidates[0])
        server_version = server_file.split("v")[-1]
        if push_server:
            subprocess.run(
                [*adb, "push", server_file, "/data/local/tmp/scrcpy-server.jar"]
            )
        subprocess.run(
            [*adb, "reverse", f"localabstract:scrcpy_{self.session_id}", f"tcp:{port}"]
        )
        skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        skt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        skt.bind(("localhost", port))
        skt.listen(1)
        command_line = [
            *adb,
            "shell",
            "CLASSPATH=/data/local/tmp/scrcpy-server.jar",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            server_version,
            f"scid={self.session_id}",
            "log_level=info",
            "audio=false",
            "clipboard_autosync=false",
            f"max_fps={max_fps}",
        ]
        if video_bit_rate is not None:
            command_line.append(f"video_bit_rate={int(video_bit_rate)}")
        crop_rect = self._normalize_crop(video_crop)
        if crop_rect is not None:
            crop_x, crop_y, crop_w, crop_h = crop_rect
            command_line.append(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")
        self.server_process = subprocess.Popen(command_line)
        self.video_socket, _ = skt.accept()
        self.control_socket, _ = skt.accept()
        subprocess.run(
            [*adb, "reverse", "--remove", f"localabstract:scrcpy_{self.session_id}"]
        )

        self.collector_running = True
        self.immediate_send_lock = threading.Lock()
        self.latest_frame_lock = threading.Lock()
        self.latest_frame = None
        self.decode_fps_lock = threading.Lock()
        self.decode_fps = 0.0
        self.latest_frame_timestamp_lock = threading.Lock()
        self.latest_frame_timestamp = None
        self.latest_frame_seq_lock = threading.Lock()
        self.latest_frame_seq = 0

        def streaming_decoder():
            codec = av.CodecContext.create("h264", "r")
            frame_counter = 0
            window_start = time.perf_counter()
            try:
                while self.collector_running:
                    _pts = self.video_socket.recv(8)
                    size = int.from_bytes(self.video_socket.recv(4), "big")
                    packet_data = bytearray()
                    while len(packet_data) < size:
                        chunk = self.video_socket.recv(size - len(packet_data))
                        if not chunk:
                            raise ConnectionError("scrcpy video socket closed")
                        packet_data.extend(chunk)
                    packets = codec.parse(bytes(packet_data))
                    for packet in packets:
                        frames = codec.decode(packet)
                        for frame in frames:
                            if (
                                self.video_width != frame.width
                                or self.video_height != frame.height
                            ):
                                print(
                                    "[client]",
                                    f"video_size: {self.video_width}x{self.video_height} -> {frame.width}x{frame.height}",
                                )
                                self.video_width = frame.width
                                self.video_height = frame.height
                            try:
                                frame_data = frame.to_ndarray(format="bgr24")
                                frame_ts = time.perf_counter()
                                with self.latest_frame_lock:
                                    self.latest_frame = frame_data
                                with self.latest_frame_timestamp_lock:
                                    self.latest_frame_timestamp = frame_ts
                                with self.latest_frame_seq_lock:
                                    self.latest_frame_seq += 1
                                frame_counter += 1
                                now = time.perf_counter()
                                elapsed = now - window_start
                                if elapsed >= 1.0:
                                    with self.decode_fps_lock:
                                        self.decode_fps = frame_counter / elapsed
                                    frame_counter = 0
                                    window_start = now
                            except Exception:
                                pass
                for frame in codec.decode(None):
                    try:
                        frame_data = frame.to_ndarray(format="bgr24")
                        frame_ts = time.perf_counter()
                        with self.latest_frame_lock:
                            self.latest_frame = frame_data
                        with self.latest_frame_timestamp_lock:
                            self.latest_frame_timestamp = frame_ts
                        with self.latest_frame_seq_lock:
                            self.latest_frame_seq += 1
                    except Exception:
                        pass
            except Exception as e:
                print(e.with_traceback(None))
                self.collector_running = False

        def ctrlmsg_receiver():
            try:
                while self.collector_running:
                    _msg_type = self.control_socket.recv(1)
                    size = int.from_bytes(self.control_socket.recv(4), "big")
                    self.control_socket.recv(size)
            except Exception as e:
                print(e.with_traceback(None))
                self.collector_running = False

        _device_name = self.video_socket.recv(64)
        codec_id = self.video_socket.recv(4).decode()
        self.device_width = int.from_bytes(self.video_socket.recv(4), "big")
        self.device_height = int.from_bytes(self.video_socket.recv(4), "big")
        self.video_width = self.device_width
        self.video_height = self.device_height
        if crop_rect is not None:
            self.stream_crop_rect = crop_rect
        else:
            self.stream_crop_rect = (0, 0, self.device_width, self.device_height)

        print(
            "[client]",
            f"device_size = {self.device_width}x{self.device_height}, codec_id = {codec_id}",
        )

        self.streaming_collector = threading.Thread(
            target=streaming_decoder, daemon=True
        )
        self.streaming_collector.start()

        self.control_collector = threading.Thread(target=ctrlmsg_receiver, daemon=True)
        self.control_collector.start()

    def touch(self, x: int, y: int, action: TouchAction, pointer_id: int) -> None:
        self.control_socket.send(
            struct.pack(
                "!bbQiiHHHII",
                2,  # SC_CONTROL_MSG_TYPE_INJECT_TOUCH_EVENT
                action.value,
                pointer_id,
                x,
                y,
                self.device_width,
                self.device_height,
                0xFFFF,  # pressure
                1,  # action_button: AMOTION_EVENT_BUTTON_PRIMARY
                1,  # buttons: AMOTION_EVENT_BUTTON_PRIMARY
            )
        )

    def tap(self, x: int, y: int, pointer_id: int = 1000, delay: float = 0.1) -> None:
        self.touch(x, y, TouchAction.DOWN, pointer_id)
        time.sleep(delay)
        self.touch(x, y, TouchAction.UP, pointer_id)

    def get_latest_frame(self, copy_frame: bool = True) -> np.ndarray | None:
        with self.latest_frame_lock:
            if self.latest_frame is None:
                return None
            if copy_frame:
                return self.latest_frame.copy()
            return self.latest_frame

    def get_decode_fps(self) -> float:
        with self.decode_fps_lock:
            return self.decode_fps

    def get_latest_frame_timestamp(self) -> float | None:
        with self.latest_frame_timestamp_lock:
            return self.latest_frame_timestamp

    def get_latest_frame_seq(self) -> int:
        with self.latest_frame_seq_lock:
            return self.latest_frame_seq

    def get_stream_crop_rect(self) -> tuple[int, int, int, int]:
        return self.stream_crop_rect

    def get_stream_crop_norm(self) -> tuple[float, float, float, float]:
        crop_x, crop_y, crop_w, crop_h = self.stream_crop_rect
        full_w = max(1, int(self.device_width))
        full_h = max(1, int(self.device_height))
        return (
            crop_x / float(full_w),
            crop_y / float(full_h),
            (crop_x + crop_w) / float(full_w),
            (crop_y + crop_h) / float(full_h),
        )

    def close(self) -> None:
        self.collector_running = False
        try:
            self.control_socket.close()
        except Exception:
            pass
        try:
            self.video_socket.close()
        except Exception:
            pass
        try:
            self.server_process.terminate()
        except Exception:
            pass

    @staticmethod
    def get_devices() -> list[str]:
        ret, output = subprocess.getstatusoutput("adb devices")
        if ret != 0:
            return []
        return [
            serial
            for serial, status in (
                line.split("\t")
                for line in output.splitlines()
                if not line.startswith("*") and line != "List of devices attached"
            )
            if status == "device"
        ]

    @staticmethod
    def _normalize_crop(
        crop: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if crop is None:
            return None
        if len(crop) != 4:
            return None
        x, y, w, h = (int(crop[0]), int(crop[1]), int(crop[2]), int(crop[3]))
        if w <= 0 or h <= 0:
            return None
        if x < 0 or y < 0:
            return None
        return (x, y, w, h)


if __name__ == "__main__":
    print(DeviceController.get_devices())
    controller = DeviceController()
    device_width = controller.device_width
    device_height = controller.device_height

    controller.tap(device_width >> 1, device_height >> 1)
