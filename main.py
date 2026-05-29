"""Gesture-controlled floating video player using MediaPipe Tasks API.

This application tracks the user's left-hand thumb and index finger pinch to
position and size a virtual mobile screen. Videos from the local `videos/`
folder are rendered inside that floating rectangle in real time.
"""

from __future__ import annotations

import math
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# OpenCV wheels often ship without the Wayland Qt plugin on Linux.
# Forcing XCB avoids a startup crash on Arch-based systems using Wayland sessions.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python_tasks
from mediapipe.tasks.python.vision import HandLandmarkerResult
from mediapipe.tasks.python.vision import PoseLandmarkerResult


WINDOW_NAME = "JARVIS Gesture Video Player"
CAMERA_INDEX = 0
FRAME_MARGIN = 20
MIN_RECT_WIDTH = 120
MIN_RECT_HEIGHT = 200
MAX_RECT_WIDTH_RATIO = 0.55
RECT_ASPECT_RATIO = 9 / 16
PINCH_SMOOTHING = 0.22
HAND_LOST_TIMEOUT_SECONDS = 0.75
STATUS_TEXT_DURATION_SECONDS = 2.0
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MAX_HANDS = 2
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
VIDEO_DIR = Path("videos")
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
THUMB_TIP_INDEX = 4
INDEX_FINGER_TIP_INDEX = 8
WRIST_INDEX = 0


@dataclass
class RectangleState:
    """Stores the current rectangle center and size."""

    center: Tuple[int, int]
    width: int
    height: int
    last_updated: float


@dataclass
class DetectionSnapshot:
    """Represents the latest completed async hand detection output."""

    result: Optional[HandLandmarkerResult]
    output_image: Optional[mp.Image]
    timestamp_ms: int


class VideoLibrary:
    """Manages the collection of videos available for playback."""

    def __init__(self, video_dir: Path) -> None:
        """Create the library, ensuring the target folder exists."""
        self.video_dir = video_dir
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.video_paths: List[Path] = []
        self.current_index = 0
        self.refresh()

    def refresh(self) -> None:
        """Rescan the video directory and keep the current selection if possible."""
        previous_path = self.current_path()
        self.video_paths = sorted(
            path for path in self.video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        )

        if not self.video_paths:
            self.current_index = 0
            return

        if previous_path in self.video_paths:
            self.current_index = self.video_paths.index(previous_path)
            return

        self.current_index = min(self.current_index, len(self.video_paths) - 1)

    def current_path(self) -> Optional[Path]:
        """Return the currently selected video path, if available."""
        if not self.video_paths:
            return None
        return self.video_paths[self.current_index]

    def next_video(self) -> Optional[Path]:
        """Advance to the next video, wrapping around the playlist."""
        if not self.video_paths:
            return None
        self.current_index = (self.current_index + 1) % len(self.video_paths)
        return self.current_path()

    def previous_video(self) -> Optional[Path]:
        """Move to the previous video, wrapping around the playlist."""
        if not self.video_paths:
            return None
        self.current_index = (self.current_index - 1) % len(self.video_paths)
        return self.current_path()

    def count(self) -> int:
        """Return the total number of available videos."""
        return len(self.video_paths)


class VideoPlayer:
    """Reads video frames and exposes playback controls for overlay rendering."""

    def __init__(self, library: VideoLibrary) -> None:
        """Initialize the player with a managed video library."""
        self.library = library
        self.capture: Optional[cv2.VideoCapture] = None
        self.current_video_path: Optional[Path] = None
        self.cached_frame: Optional[np.ndarray] = None
        self.is_paused = False
        self.last_error: Optional[str] = None
        self._open_selected_video()

    def refresh_library(self) -> None:
        """Reload the library and reopen the current video selection safely."""
        current_path = self.library.current_path()
        self.library.refresh()
        if self.library.current_path() != current_path:
            self._open_selected_video()
            return

        if self.capture is None and self.library.current_path() is not None:
            self._open_selected_video()

    def next_video(self) -> None:
        """Switch to the next video in the library."""
        self.library.next_video()
        self._open_selected_video()

    def previous_video(self) -> None:
        """Switch to the previous video in the library."""
        self.library.previous_video()
        self._open_selected_video()

    def toggle_pause(self) -> bool:
        """Toggle playback pause state and return the new state."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Return the next displayable frame, looping videos automatically."""
        if self.capture is None:
            return self.cached_frame

        if self.is_paused and self.cached_frame is not None:
            return self.cached_frame.copy()

        success, frame = self.capture.read()
        if not success or frame is None:
            # Looping avoids a dead-end user experience for short videos.
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.capture.read()
            if not success or frame is None:
                self.last_error = f"Could not read video: {self.current_video_name()}"
                return self.cached_frame

        self.last_error = None
        self.cached_frame = frame
        return frame.copy()

    def current_video_name(self) -> str:
        """Return the current video's file name or a fallback label."""
        if self.current_video_path is None:
            return "No video loaded"
        return self.current_video_path.name

    def close(self) -> None:
        """Release any open video capture resource."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _open_selected_video(self) -> None:
        """Open the library's current selection and prepare for playback."""
        selected_path = self.library.current_path()
        self.close()
        self.current_video_path = selected_path
        self.cached_frame = None
        self.last_error = None

        if selected_path is None:
            return

        capture = cv2.VideoCapture(str(selected_path))
        if not capture.isOpened():
            self.last_error = f"Could not open video: {selected_path.name}"
            capture.release()
            return

        self.capture = capture


class ObjectMeasurer:
    """Detect object contours, classify shapes, and estimate real-world size in centimeters."""

    A4_WIDTH_CM = 21.0
    A4_HEIGHT_CM = 29.7

    def __init__(self) -> None:
        """Initialize measurement mode, calibration state, and display settings."""
        self.enabled = False
        self.calibration_requested = False
        self.pixels_per_cm: Optional[float] = None
        self.status_text = "Measurement mode OFF"
        self.status_until = 0.0
        self.min_contour_area = 2500.0

    def toggle_enabled(self) -> bool:
        """Toggle measurement mode and return the new enabled state."""
        self.enabled = not self.enabled
        if self.enabled:
            self._set_status("Measurement mode ON. Press C while A4 paper is visible to calibrate.")
        else:
            self._set_status("Measurement mode OFF")
        return self.enabled

    def request_calibration(self) -> None:
        """Mark calibration to run on the next suitable frame."""
        self.calibration_requested = True
        self._set_status("Calibration requested. Show a full A4 paper clearly to the camera.")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Detect shapes, calibrate if requested, and draw measurements on the frame."""
        if not self.enabled:
            self._draw_status(frame)
            return frame

        contour_data = self._detect_candidate_contours(frame)

        if self.calibration_requested:
            self._calibrate_from_a4(contour_data)

        if self.pixels_per_cm is not None:
            self._draw_measurements(frame, contour_data)
        else:
            self._draw_calibration_hint(frame)

        self._draw_status(frame)
        return frame

    def _detect_candidate_contours(self, frame: np.ndarray) -> List[dict]:
        """Find external contours and compute geometry needed for labeling and measurement."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Mixing thresholding with edges makes flat rectangular objects like books much easier to separate.
        edges = cv2.Canny(blurred, 35, 120)
        threshold_mask = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            8,
        )
        combined_mask = cv2.bitwise_or(edges, threshold_mask)

        kernel = np.ones((5, 5), dtype=np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_data: List[dict] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)
            if w <= 0 or h <= 0:
                continue

            bbox_area = float(w * h)
            fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
            aspect_ratio = w / float(h)
            shape_name = self._classify_shape(approx, contour, fill_ratio)
            if shape_name is None:
                continue
            rectangle_priority = 1 if shape_name in {"Rectangle", "Square", "Quadrilateral"} else 0

            contour_data.append(
                {
                    "contour": contour,
                    "approx": approx,
                    "area": area,
                    "bbox": (x, y, w, h),
                    "shape_name": shape_name,
                    "fill_ratio": fill_ratio,
                    "aspect_ratio": aspect_ratio,
                    "rectangle_priority": rectangle_priority,
                }
            )

        contour_data.sort(
            key=lambda item: (
                item["rectangle_priority"],
                item["fill_ratio"],
                item["area"],
            ),
            reverse=True,
        )
        return contour_data

    def _classify_shape(
        self, approx: np.ndarray, contour: np.ndarray, fill_ratio: float
    ) -> Optional[str]:
        """Classify only strong shape matches and reject weak noisy contours."""
        vertices = len(approx)

        if vertices == 3:
            if fill_ratio < 0.45:
                return None
            return "Triangle"

        if vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                return None
            aspect_ratio = w / float(h)
            if fill_ratio < 0.78:
                return None
            if 0.95 <= aspect_ratio <= 1.05:
                return "Square"
            return "Rectangle"

        if vertices in (5, 6):
            # Perspective can add extra corners to a book, so near-rectangles should not be discarded.
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = float(w * h)
            contour_area = cv2.contourArea(contour)
            local_fill_ratio = contour_area / rect_area if rect_area > 0 else 0.0
            if local_fill_ratio >= 0.84:
                return "Quadrilateral"

        if vertices == 5:
            if fill_ratio < 0.6:
                return None
            return "Pentagon"

        if vertices >= 6:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                return None

            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity >= 0.86 and fill_ratio >= 0.7:
                return "Circle"
            return None

        return None

    def _calibrate_from_a4(self, contour_data: List[dict]) -> None:
        """Estimate pixels-per-centimeter from the largest detected contour assumed to be A4 paper."""
        self.calibration_requested = False

        if not contour_data:
            self._set_status("Calibration failed: no large object detected.")
            return

        reference = next(
            (
                item for item in contour_data
                if item["shape_name"] in {"Rectangle", "Square", "Quadrilateral"}
            ),
            contour_data[0],
        )
        _, _, width_px, height_px = reference["bbox"]

        longer_px = max(width_px, height_px)
        shorter_px = min(width_px, height_px)
        longer_cm = max(self.A4_WIDTH_CM, self.A4_HEIGHT_CM)
        shorter_cm = min(self.A4_WIDTH_CM, self.A4_HEIGHT_CM)

        px_per_cm_width = shorter_px / shorter_cm
        px_per_cm_height = longer_px / longer_cm
        pixels_per_cm = (px_per_cm_width + px_per_cm_height) / 2.0

        if pixels_per_cm <= 0:
            self._set_status("Calibration failed: invalid A4 measurement.")
            return

        self.pixels_per_cm = pixels_per_cm
        self._set_status(f"Calibration successful: {self.pixels_per_cm:.2f} px/cm")

    def _draw_measurements(self, frame: np.ndarray, contour_data: List[dict]) -> None:
        """Draw contour outlines, bounding boxes, and real-world dimensions for each object."""
        for item in contour_data[:3]:
            contour = item["contour"]
            approx = item["approx"]
            x, y, w, h = item["bbox"]
            shape_name = item["shape_name"]

            width_cm = w / self.pixels_per_cm
            height_cm = h / self.pixels_per_cm

            # Contour and box together make detection clearer for users during live camera motion.
            cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 120), 2)

            label_y = max(20, y - 12)
            cv2.putText(
                frame,
                shape_name,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"W: {width_cm:.2f} cm",
                (x, y + h + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"H: {height_cm:.2f} cm",
                (x, y + h + 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )

    def _draw_calibration_hint(self, frame: np.ndarray) -> None:
        """Show calibration guidance until a valid pixels-per-centimeter value exists."""
        cv2.putText(
            frame,
            "Press C to calibrate with A4 paper (21.0cm x 29.7cm)",
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_status(self, frame: np.ndarray) -> None:
        """Render a temporary status message near the bottom of the frame."""
        if time.time() > self.status_until:
            return

        cv2.putText(
            frame,
            self.status_text,
            (15, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _set_status(self, text: str) -> None:
        """Store a short-lived on-screen status message."""
        self.status_text = text
        self.status_until = time.time() + 2.5


class BodySkeletonTracker:
    """Tracks full-body pose landmarks with MediaPipe Tasks API and renders a smoothed 3D-style skeleton."""

    MODEL_CONFIGS = {
        "lite": {
            "path": MODEL_DIR / "pose_landmarker_lite.task",
            "url": (
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            ),
            "detection_confidence": 0.55,
            "presence_confidence": 0.55,
            "tracking_confidence": 0.55,
        },
        "full": {
            "path": MODEL_DIR / "pose_landmarker_full.task",
            "url": (
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            ),
            "detection_confidence": 0.65,
            "presence_confidence": 0.65,
            "tracking_confidence": 0.65,
        },
    }
    SIDE_VIEW_WIDTH = 180
    SIDE_VIEW_HEIGHT = 260
    SIDE_VIEW_MARGIN = 16
    SMOOTHING_ALPHA = 0.35
    LOW_CONFIDENCE_ALPHA = 0.18
    MIN_VISIBILITY = 0.60
    MIN_PRESENCE = 0.60
    RESET_TIMEOUT_SECONDS = 0.45

    HEAD_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
        (0, 2),
        (0, 5),
        (2, 7),
        (5, 8),
    )
    CENTER_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
    )
    LEFT_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
        (11, 13),
        (13, 15),
        (15, 17),
        (17, 19),
        (19, 21),
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),
    )
    RIGHT_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
        (12, 14),
        (14, 16),
        (16, 18),
        (18, 20),
        (20, 22),
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),
    )

    def __init__(self) -> None:
        """Prepare the pose landmarker, callback state, and smoothing memory."""
        self.mode = "full"
        self._snapshot_lock = threading.Lock()
        self._latest_result: Optional[PoseLandmarkerResult] = None
        self._latest_timestamp_ms = -1
        self._processed_timestamp_ms = -1
        self._smoothed_landmarks: Optional[List[dict]] = None
        self._last_pose_time = 0.0
        self._load_model_assets(self.mode)
        self.landmarker = self._create_landmarker()

    def _load_model_assets(self, mode: str) -> None:
        """Ensure the selected model exists locally before landmarker creation."""
        config = self.MODEL_CONFIGS[mode]
        ensure_model_exists(config["path"], config["url"])

    def _create_landmarker(self) -> mp.tasks.vision.PoseLandmarker:
        """Create a pose landmarker configured for LIVE_STREAM mode."""
        config = self.MODEL_CONFIGS[self.mode]
        base_options = mp_python_tasks.BaseOptions(model_asset_path=str(config["path"]))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=float(config["detection_confidence"]),
            min_pose_presence_confidence=float(config["presence_confidence"]),
            min_tracking_confidence=float(config["tracking_confidence"]),
            output_segmentation_masks=False,
            result_callback=self._handle_result,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def toggle_mode(self) -> str:
        """Switch between lite and full pose models and return the active mode."""
        new_mode = "lite" if self.mode == "full" else "full"
        self._load_model_assets(new_mode)

        old_landmarker = self.landmarker
        self.mode = new_mode
        self.landmarker = self._create_landmarker()
        old_landmarker.close()

        self._latest_result = None
        self._latest_timestamp_ms = -1
        self._processed_timestamp_ms = -1
        self._smoothed_landmarks = None
        self._last_pose_time = 0.0
        return self.mode

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Submit the frame for async pose detection and draw the latest available skeleton."""
        timestamp_ms = int(time.time() * 1000)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.landmarker.detect_async(mp_image, timestamp_ms)

        pose_landmarks = self._consume_latest_pose()
        if pose_landmarks is None:
            return frame

        smoothed_landmarks = self._smooth_landmarks(pose_landmarks)
        self._draw_skeleton(frame, smoothed_landmarks)
        self._draw_side_view(frame, smoothed_landmarks)
        return frame

    def _handle_result(
        self,
        result: PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """Receive async pose results and store the latest one safely."""
        del output_image
        with self._snapshot_lock:
            self._latest_result = result
            self._latest_timestamp_ms = timestamp_ms

    def _consume_latest_pose(self) -> Optional[Sequence]:
        """Return the newest pose landmark list once, or None if unavailable."""
        with self._snapshot_lock:
            result = self._latest_result
            timestamp_ms = self._latest_timestamp_ms

        if result is None or timestamp_ms <= self._processed_timestamp_ms:
            return None

        self._processed_timestamp_ms = timestamp_ms
        if not result.pose_landmarks:
            self._smoothed_landmarks = None
            return None

        first_pose = result.pose_landmarks[0]
        if not first_pose:
            self._smoothed_landmarks = None
            return None

        self._last_pose_time = time.time()
        return first_pose

    def _smooth_landmarks(self, landmarks: Sequence) -> List[dict]:
        """Apply exponential smoothing so the skeleton feels stable instead of jittery."""
        if time.time() - self._last_pose_time > self.RESET_TIMEOUT_SECONDS:
            self._smoothed_landmarks = None

        current_points: List[dict] = []
        for landmark in landmarks:
            visibility = float(getattr(landmark, "visibility", 1.0))
            presence = float(getattr(landmark, "presence", 1.0))
            current_points.append(
                {
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "visibility": visibility,
                    "presence": presence,
                }
            )

        if self._smoothed_landmarks is None or len(self._smoothed_landmarks) != len(current_points):
            self._smoothed_landmarks = current_points
            return current_points

        smoothed_points: List[dict] = []
        for previous, current in zip(self._smoothed_landmarks, current_points):
            confidence = min(current["visibility"], current["presence"])
            alpha = (
                self.SMOOTHING_ALPHA
                if confidence >= self.MIN_VISIBILITY
                else self.LOW_CONFIDENCE_ALPHA
            )
            smoothed_points.append(
                {
                    "x": (1.0 - alpha) * previous["x"] + alpha * current["x"],
                    "y": (1.0 - alpha) * previous["y"] + alpha * current["y"],
                    "z": (1.0 - alpha) * previous["z"] + alpha * current["z"],
                    "visibility": max(previous["visibility"] * 0.6, current["visibility"]),
                    "presence": max(previous["presence"] * 0.6, current["presence"]),
                }
            )

        self._smoothed_landmarks = smoothed_points
        return smoothed_points

    def _draw_skeleton(self, frame: np.ndarray, landmarks: List[dict]) -> None:
        """Draw a color-coded body skeleton with line thickness driven by landmark depth."""
        frame_height, frame_width = frame.shape[:2]

        self._draw_connections(
            frame,
            landmarks,
            self.HEAD_CONNECTIONS,
            (255, 255, 255),
            frame_width,
            frame_height,
        )
        self._draw_connections(
            frame,
            landmarks,
            self.CENTER_CONNECTIONS,
            (255, 255, 255),
            frame_width,
            frame_height,
        )
        self._draw_connections(
            frame,
            landmarks,
            self.LEFT_CONNECTIONS,
            (255, 100, 100),
            frame_width,
            frame_height,
        )
        self._draw_connections(
            frame,
            landmarks,
            self.RIGHT_CONNECTIONS,
            (100, 255, 100),
            frame_width,
            frame_height,
        )

        for landmark in landmarks:
            if not self._is_landmark_visible(landmark):
                continue
            px = int(max(0.0, min(1.0, landmark["x"])) * (frame_width - 1))
            py = int(max(0.0, min(1.0, landmark["y"])) * (frame_height - 1))
            joint_radius = 6 if landmark["z"] < -0.15 else 5
            cv2.circle(frame, (px, py), joint_radius, (0, 215, 255), -1)

    def _draw_connections(
        self,
        frame: np.ndarray,
        landmarks: List[dict],
        connections: Sequence[Tuple[int, int]],
        base_color: Tuple[int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Draw a group of skeleton bones with simple depth-based 3D styling."""
        for start_idx, end_idx in connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue

            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if not self._is_landmark_visible(start) or not self._is_landmark_visible(end):
                continue

            start_px = (
                int(max(0.0, min(1.0, start["x"])) * (frame_width - 1)),
                int(max(0.0, min(1.0, start["y"])) * (frame_height - 1)),
            )
            end_px = (
                int(max(0.0, min(1.0, end["x"])) * (frame_width - 1)),
                int(max(0.0, min(1.0, end["y"])) * (frame_height - 1)),
            )

            avg_z = (start["z"] + end["z"]) / 2.0
            thickness = max(1, min(4, int(2 - avg_z * 10)))
            color = self._depth_adjusted_color(base_color, avg_z)
            # Slightly thicker center torso lines help the overall body pose read more clearly.
            if base_color == (255, 255, 255):
                thickness = min(4, thickness + 1)
            cv2.line(frame, start_px, end_px, color, thickness, cv2.LINE_AA)

    def _draw_side_view(self, frame: np.ndarray, landmarks: List[dict]) -> None:
        """Render a compact side-view skeleton using z and y coordinates in the top-right corner."""
        frame_height, frame_width = frame.shape[:2]
        box_width = self.SIDE_VIEW_WIDTH
        box_height = self.SIDE_VIEW_HEIGHT
        box_x = max(0, frame_width - box_width - self.SIDE_VIEW_MARGIN)
        box_y = self.SIDE_VIEW_MARGIN

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            "SIDE VIEW",
            (box_x + 10, box_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        self._draw_side_connections(
            frame,
            landmarks,
            self.HEAD_CONNECTIONS,
            (255, 255, 255),
            box_x,
            box_y,
            box_width,
            box_height,
        )
        self._draw_side_connections(
            frame,
            landmarks,
            self.CENTER_CONNECTIONS,
            (255, 255, 255),
            box_x,
            box_y,
            box_width,
            box_height,
        )
        self._draw_side_connections(
            frame,
            landmarks,
            self.LEFT_CONNECTIONS,
            (255, 100, 100),
            box_x,
            box_y,
            box_width,
            box_height,
        )
        self._draw_side_connections(
            frame,
            landmarks,
            self.RIGHT_CONNECTIONS,
            (100, 255, 100),
            box_x,
            box_y,
            box_width,
            box_height,
        )

        for landmark in landmarks:
            if not self._is_landmark_visible(landmark):
                continue
            side_x = int(box_x + landmark["z"] * 200 + 90)
            side_y = int(box_y + max(0.0, min(1.0, landmark["y"])) * (box_height - 30)) + 24
            side_x = max(box_x + 4, min(box_x + box_width - 4, side_x))
            side_y = max(box_y + 24, min(box_y + box_height - 4, side_y))
            cv2.circle(frame, (side_x, side_y), 4, (0, 215, 255), -1)

    def _draw_side_connections(
        self,
        frame: np.ndarray,
        landmarks: List[dict],
        connections: Sequence[Tuple[int, int]],
        base_color: Tuple[int, int, int],
        box_x: int,
        box_y: int,
        box_width: int,
        box_height: int,
    ) -> None:
        """Draw skeleton connections inside the side-view box."""
        for start_idx, end_idx in connections:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue

            start = landmarks[start_idx]
            end = landmarks[end_idx]
            if not self._is_landmark_visible(start) or not self._is_landmark_visible(end):
                continue

            start_pt = self._side_view_point(start, box_x, box_y, box_width, box_height)
            end_pt = self._side_view_point(end, box_x, box_y, box_width, box_height)
            avg_z = (start["z"] + end["z"]) / 2.0
            thickness = max(1, min(4, int(2 - avg_z * 10)))
            color = self._depth_adjusted_color(base_color, avg_z)
            cv2.line(frame, start_pt, end_pt, color, thickness, cv2.LINE_AA)

    def _side_view_point(
        self,
        landmark: dict,
        box_x: int,
        box_y: int,
        box_width: int,
        box_height: int,
    ) -> Tuple[int, int]:
        """Convert a landmark's z and y values into a clamped side-view point."""
        side_x = int(box_x + landmark["z"] * 200 + 90)
        side_y = int(box_y + max(0.0, min(1.0, landmark["y"])) * (box_height - 30)) + 24
        side_x = max(box_x + 4, min(box_x + box_width - 4, side_x))
        side_y = max(box_y + 24, min(box_y + box_height - 4, side_y))
        return side_x, side_y

    def _depth_adjusted_color(
        self, base_color: Tuple[int, int, int], z_value: float
    ) -> Tuple[int, int, int]:
        """Brighten closer landmarks and darken farther landmarks for a simple depth cue."""
        brightness = max(0.55, min(1.45, 1.05 - z_value * 2.4))
        return tuple(
            max(0, min(255, int(channel * brightness))) for channel in base_color
        )

    def _is_landmark_visible(self, landmark: dict) -> bool:
        """Return True only for landmarks with enough confidence to draw."""
        return (
            landmark["visibility"] >= self.MIN_VISIBILITY
            and landmark["presence"] >= self.MIN_PRESENCE
        )

    def close(self) -> None:
        """Release the pose landmarker cleanly."""
        self.landmarker.close()


class SignLanguageInterpreter:
    """Recognize a small set of static hand signs from MediaPipe hand landmarks."""

    SIGN_HOLD_SECONDS = 1.2

    def __init__(self) -> None:
        """Initialize sign-mode runtime state."""
        self.enabled = False
        self.current_sign = "NONE"
        self.last_sign_time = 0.0

    def toggle_enabled(self) -> bool:
        """Toggle sign mode and return the new state."""
        self.enabled = not self.enabled
        if not self.enabled:
            self.current_sign = "NONE"
        return self.enabled

    def update_from_hand(self, hand_landmarks: Sequence, handedness_label: str) -> Optional[str]:
        """Infer a sign from one detected hand and store it briefly for display."""
        if not self.enabled:
            return None

        sign_name = self._classify_sign(hand_landmarks, handedness_label)
        if sign_name is not None:
            self.current_sign = sign_name
            self.last_sign_time = time.time()
        elif time.time() - self.last_sign_time > self.SIGN_HOLD_SECONDS:
            self.current_sign = "NONE"

        return self.current_sign

    def draw_overlay(self, frame: np.ndarray) -> None:
        """Draw the active sign label on the frame when sign mode is enabled."""
        if not self.enabled:
            return

        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 120), (280, 175), (8, 8, 8), -1)
        cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
        cv2.rectangle(frame, (15, 120), (280, 175), (0, 215, 255), 1)
        cv2.putText(
            frame,
            "SIGN MODE",
            (28, 142),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self.current_sign,
            (28, 166),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )

    def _classify_sign(self, hand_landmarks: Sequence, handedness_label: str) -> Optional[str]:
        """Classify a few stable static signs using finger extension rules."""
        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        index_tip = hand_landmarks[8]
        middle_tip = hand_landmarks[12]
        ring_tip = hand_landmarks[16]
        pinky_tip = hand_landmarks[20]

        index_extended = self._is_finger_extended(hand_landmarks, 8)
        middle_extended = self._is_finger_extended(hand_landmarks, 12)
        ring_extended = self._is_finger_extended(hand_landmarks, 16)
        pinky_extended = self._is_finger_extended(hand_landmarks, 20)
        thumb_extended = self._is_thumb_extended(hand_landmarks, handedness_label)

        pinch_distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)

        if pinch_distance < 0.06 and middle_extended and ring_extended and pinky_extended:
            return "OK"

        if all([index_extended, middle_extended, ring_extended, pinky_extended]) and thumb_extended:
            return "OPEN_PALM"

        if not any([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
            return "FIST"

        if index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "PEACE"

        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "POINT"

        thumb_above_index = thumb_tip.y < thumb_ip.y < index_tip.y
        if thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]) and thumb_above_index:
            return "THUMBS_UP"

        return None

    def _is_finger_extended(self, hand_landmarks: Sequence, tip_index: int) -> bool:
        """Estimate whether a non-thumb finger is extended using tip and PIP y positions."""
        pip_index = tip_index - 2
        mcp_index = tip_index - 3
        tip = hand_landmarks[tip_index]
        pip = hand_landmarks[pip_index]
        mcp = hand_landmarks[mcp_index]
        return tip.y < pip.y < mcp.y

    def _is_thumb_extended(self, hand_landmarks: Sequence, handedness_label: str) -> bool:
        """Estimate thumb extension using horizontal spread, adjusted for handedness."""
        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        index_mcp = hand_landmarks[5]

        if handedness_label == "Left":
            return thumb_tip.x < thumb_ip.x < index_mcp.x
        return thumb_tip.x > thumb_ip.x > index_mcp.x


class GestureRectangleApp:
    """Runs webcam capture, async hand detection, and floating video playback."""

    def __init__(self) -> None:
        """Prepare runtime state, ensure assets exist, and build the landmarker."""
        self.rectangle_state: Optional[RectangleState] = None
        self.smoothed_pinch_distance: Optional[float] = None
        self.status_text = "Add videos to the videos folder, then pinch with your LEFT hand."
        self.status_until = time.time() + STATUS_TEXT_DURATION_SECONDS

        self._latest_snapshot: Optional[DetectionSnapshot] = None
        self._snapshot_lock = threading.Lock()
        self._latest_processed_timestamp_ms = -1

        ensure_model_exists(MODEL_PATH, MODEL_URL)
        self.landmarker = self._create_landmarker()
        self.video_library = VideoLibrary(VIDEO_DIR)
        self.video_player = VideoPlayer(self.video_library)
        self.video_overlay_enabled = True
        self.object_measurer = ObjectMeasurer()
        self.body_tracker = BodySkeletonTracker()
        self.sign_interpreter = SignLanguageInterpreter()
        self.skeleton_enabled = False

        if self.video_library.count() == 0:
            self._set_status(
                "No videos found. Put MP4/MOV/MKV files into the videos folder and press R."
            )

    def run(self) -> None:
        """Start the webcam loop, submit frames asynchronously, and display output."""
        capture = self._open_camera(CAMERA_INDEX)

        try:
            while True:
                success, frame = capture.read()
                if not success or frame is None:
                    self._set_status("Camera frame not available. Retrying...")
                    continue

                frame = cv2.flip(frame, 1)
                timestamp_ms = int(time.time() * 1000)
                self._submit_frame(frame, timestamp_ms)
                annotated_frame = self._render_frame(frame)
                cv2.imshow(WINDOW_NAME, annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if self._handle_keypress(key):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
            self.video_player.close()
            self.landmarker.close()
            self.body_tracker.close()

    def _create_landmarker(self) -> mp.tasks.vision.HandLandmarker:
        """Create a HandLandmarker configured for LIVE_STREAM inference."""
        base_options = mp_python_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            result_callback=self._handle_result,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _open_camera(self, camera_index: int) -> cv2.VideoCapture:
        """Open the camera with a Linux-friendly fallback sequence."""
        capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture = cv2.VideoCapture(camera_index)

        if not capture.isOpened():
            raise RuntimeError(
                "Could not open the webcam. Check permissions, confirm /dev/video* "
                "exists, and make sure another app is not using the camera."
            )

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _submit_frame(self, frame: np.ndarray, timestamp_ms: int) -> None:
        """Send the current frame to MediaPipe for asynchronous inference."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def _handle_result(
        self,
        result: HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        """Receive async inference results and store only the newest snapshot."""
        snapshot = DetectionSnapshot(
            result=result,
            output_image=output_image,
            timestamp_ms=timestamp_ms,
        )
        with self._snapshot_lock:
            self._latest_snapshot = snapshot

    def _render_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply the newest detection result, draw the video overlay, and render UI."""
        snapshot = self._consume_latest_snapshot()
        if snapshot is not None:
            self._apply_detection_result(snapshot.result, frame)
        else:
            self._handle_missing_left_hand()

        self._draw_video_overlay(frame)
        self.object_measurer.process_frame(frame)
        if self.skeleton_enabled:
            self.body_tracker.process_frame(frame)
        self.sign_interpreter.draw_overlay(frame)
        self._draw_rectangle(frame, frame.shape[1], frame.shape[0])
        self._draw_help_panel(frame)
        return frame

    def _consume_latest_snapshot(self) -> Optional[DetectionSnapshot]:
        """Fetch the newest callback result exactly once per timestamp."""
        with self._snapshot_lock:
            snapshot = self._latest_snapshot

        if snapshot is None:
            return None

        if snapshot.timestamp_ms <= self._latest_processed_timestamp_ms:
            return None

        self._latest_processed_timestamp_ms = snapshot.timestamp_ms
        return snapshot

    def _apply_detection_result(
        self, result: Optional[HandLandmarkerResult], frame: np.ndarray
    ) -> None:
        """Draw hand landmarks and update rectangle state from the left hand."""
        if result is None or not result.hand_landmarks:
            self._handle_missing_left_hand()
            return

        frame_height, frame_width = frame.shape[:2]
        left_hand_landmarks: Optional[Sequence] = None

        for hand_landmarks, handedness_list in zip(
            result.hand_landmarks, result.handedness
        ):
            handedness_label = handedness_list[0].category_name if handedness_list else "Unknown"
            hand_points = [
                self._normalized_to_pixel(landmark.x, landmark.y, frame_width, frame_height)
                for landmark in hand_landmarks
            ]

            self._draw_hand_skeleton(frame, hand_points, handedness_label)
            self.sign_interpreter.update_from_hand(hand_landmarks, handedness_label)
            if handedness_label == "Left":
                left_hand_landmarks = hand_landmarks
                self._draw_hand_label(frame, hand_points, "LEFT HAND")

        if left_hand_landmarks is not None:
            self._update_rectangle_from_left_hand(
                frame, left_hand_landmarks, frame_width, frame_height
            )
        else:
            self._handle_missing_left_hand()

    def _update_rectangle_from_left_hand(
        self,
        frame: np.ndarray,
        hand_landmarks: Sequence,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Compute rectangle center and size from the left-hand pinch gesture."""
        thumb_tip = hand_landmarks[THUMB_TIP_INDEX]
        index_tip = hand_landmarks[INDEX_FINGER_TIP_INDEX]

        thumb_px = self._normalized_to_pixel(thumb_tip.x, thumb_tip.y, frame_width, frame_height)
        index_px = self._normalized_to_pixel(index_tip.x, index_tip.y, frame_width, frame_height)
        pinch_center = (
            int((thumb_px[0] + index_px[0]) / 2),
            int((thumb_px[1] + index_px[1]) / 2),
        )
        raw_pinch_distance = math.dist(thumb_px, index_px)

        self.smoothed_pinch_distance = self._smooth_distance(raw_pinch_distance)
        rect_width = self._pinch_distance_to_width(self.smoothed_pinch_distance, frame_width)
        rect_height = max(MIN_RECT_HEIGHT, int(rect_width / RECT_ASPECT_RATIO))
        clamped_center = self._clamp_center_to_frame(
            pinch_center, rect_width, rect_height, frame_width, frame_height
        )

        self.rectangle_state = RectangleState(
            center=clamped_center,
            width=rect_width,
            height=rect_height,
            last_updated=time.time(),
        )

        self._draw_pinch_guides(frame, thumb_px, index_px, pinch_center)
        if self.video_overlay_enabled:
            self._set_status(
                f"Tracking active. Video: {self.video_player.current_video_name()}"
            )
        else:
            self._set_status("Tracking active. Video overlay is OFF.")

    def _handle_missing_left_hand(self) -> None:
        """Hold the previous rectangle briefly so temporary drops do not flicker."""
        if self.rectangle_state is None:
            self._set_status("No left hand detected. Show your LEFT hand to place the screen.")
            return

        age = time.time() - self.rectangle_state.last_updated
        if age > HAND_LOST_TIMEOUT_SECONDS:
            self.rectangle_state = None
            self.smoothed_pinch_distance = None
            self._set_status("Left hand lost. Reacquire the pinch to redraw the screen.")
        else:
            self._set_status("Tracking unstable. Holding last rectangle position.")

    def _draw_video_overlay(self, frame: np.ndarray) -> None:
        """Render the current video frame inside the tracked rectangle."""
        if not self.video_overlay_enabled or self.rectangle_state is None:
            return

        center_x, center_y = self.rectangle_state.center
        half_width = self.rectangle_state.width // 2
        half_height = self.rectangle_state.height // 2
        top_left_x = center_x - half_width
        top_left_y = center_y - half_height
        bottom_right_x = center_x + half_width
        bottom_right_y = center_y + half_height

        frame_height, frame_width = frame.shape[:2]
        top_left_x = max(FRAME_MARGIN, top_left_x)
        top_left_y = max(FRAME_MARGIN, top_left_y)
        bottom_right_x = min(frame_width - FRAME_MARGIN, bottom_right_x)
        bottom_right_y = min(frame_height - FRAME_MARGIN, bottom_right_y)

        if bottom_right_x <= top_left_x or bottom_right_y <= top_left_y:
            return

        overlay_width = bottom_right_x - top_left_x
        overlay_height = bottom_right_y - top_left_y
        video_frame = self.video_player.get_current_frame()

        if video_frame is None:
            self._draw_video_placeholder(
                frame,
                top_left_x,
                top_left_y,
                overlay_width,
                overlay_height,
                "Add videos to the videos folder",
            )
            return

        prepared_frame = self._fit_frame_to_rectangle(video_frame, overlay_width, overlay_height)
        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = prepared_frame

    def _draw_video_placeholder(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        message: str,
    ) -> None:
        """Draw a placeholder when no playable video frame is available."""
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        placeholder[:] = (20, 20, 20)
        cv2.putText(
            placeholder,
            "NO VIDEO",
            (15, max(40, height // 2 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            placeholder,
            message[: max(10, min(40, len(message)))],
            (15, max(70, height // 2 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        frame[y:y + height, x:x + width] = placeholder

    def _fit_frame_to_rectangle(
        self, video_frame: np.ndarray, target_width: int, target_height: int
    ) -> np.ndarray:
        """Resize and crop the video frame so it fills the floating rectangle."""
        source_height, source_width = video_frame.shape[:2]
        if source_height == 0 or source_width == 0:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        scale = max(target_width / source_width, target_height / source_height)
        resized_width = max(1, int(source_width * scale))
        resized_height = max(1, int(source_height * scale))
        resized_frame = cv2.resize(
            video_frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )

        start_x = max(0, (resized_width - target_width) // 2)
        start_y = max(0, (resized_height - target_height) // 2)
        end_x = start_x + target_width
        end_y = start_y + target_height
        return resized_frame[start_y:end_y, start_x:end_x]

    def _handle_keypress(self, key: int) -> bool:
        """Handle keyboard shortcuts and return True when the app should exit."""
        if key in (27, ord("q")):
            return True

        if key == ord("r"):
            previous_count = self.video_library.count()
            self.video_player.refresh_library()
            current_count = self.video_library.count()
            if current_count == 0:
                self._set_status("No videos found after refresh. Add files to videos/ and press R.")
            elif current_count > previous_count:
                self._set_status(f"Library refreshed. Loaded {current_count} videos.")
            else:
                self._set_status(f"Library refreshed. Current: {self.video_player.current_video_name()}")

        elif key == ord("n"):
            if self.video_library.count() == 0:
                self._set_status("No videos available. Add files to videos/ first.")
            else:
                self.video_player.next_video()
                self._set_status(f"Next video: {self.video_player.current_video_name()}")

        elif key == ord("b"):
            if self.video_library.count() == 0:
                self._set_status("No videos available. Add files to videos/ first.")
            else:
                self.video_player.previous_video()
                self._set_status(f"Previous video: {self.video_player.current_video_name()}")

        elif key == ord(" "):
            paused = self.video_player.toggle_pause()
            state = "Paused" if paused else "Playing"
            self._set_status(f"{state}: {self.video_player.current_video_name()}")

        elif key in (ord("v"), ord("V")):
            self.video_overlay_enabled = not self.video_overlay_enabled
            state = "ON" if self.video_overlay_enabled else "OFF"
            self._set_status(f"Video overlay {state}")

        elif key in (ord("m"), ord("M")):
            enabled = self.object_measurer.toggle_enabled()
            state = "ON" if enabled else "OFF"
            self._set_status(f"Measurement mode {state}")

        elif key in (ord("c"), ord("C")):
            self.object_measurer.request_calibration()
            self._set_status("A4 calibration requested. Hold paper clearly in view.")

        elif key in (ord("k"), ord("K")):
            self.skeleton_enabled = not self.skeleton_enabled
            state = "ON" if self.skeleton_enabled else "OFF"
            self._set_status(f"Body skeleton {state}")

        elif key in (ord("p"), ord("P")):
            active_mode = self.body_tracker.toggle_mode()
            self._set_status(f"Skeleton mode: {active_mode.upper()}")

        elif key in (ord("s"), ord("S")):
            enabled = self.sign_interpreter.toggle_enabled()
            state = "ON" if enabled else "OFF"
            self._set_status(f"Sign mode {state}")

        return False

    def _draw_hand_skeleton(
        self, frame: np.ndarray, hand_points: List[Tuple[int, int]], handedness_label: str
    ) -> None:
        """Draw hand landmarks and connections without using mp.solutions."""
        base_color = (30, 144, 255) if handedness_label == "Left" else (255, 120, 30)
        joint_color = (0, 215, 255) if handedness_label == "Left" else (255, 220, 120)

        for start_index, end_index in HAND_CONNECTIONS:
            cv2.line(frame, hand_points[start_index], hand_points[end_index], base_color, 2)

        for point_index, point in enumerate(hand_points):
            radius = 5 if point_index in (THUMB_TIP_INDEX, INDEX_FINGER_TIP_INDEX) else 4
            cv2.circle(frame, point, radius, joint_color, -1)

    def _draw_rectangle(self, frame: np.ndarray, frame_width: int, frame_height: int) -> None:
        """Draw the tracked rectangle or a guide when no hand is active."""
        if self.rectangle_state is None:
            guide_top_left = (FRAME_MARGIN, FRAME_MARGIN)
            guide_bottom_right = (frame_width - FRAME_MARGIN, frame_height - FRAME_MARGIN)
            cv2.rectangle(frame, guide_top_left, guide_bottom_right, (80, 80, 80), 2)
            cv2.putText(
                frame,
                "Pinch with LEFT thumb + index finger",
                (FRAME_MARGIN + 10, FRAME_MARGIN + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )
            return

        center_x, center_y = self.rectangle_state.center
        half_width = self.rectangle_state.width // 2
        half_height = self.rectangle_state.height // 2
        top_left = (center_x - half_width, center_y - half_height)
        bottom_right = (center_x + half_width, center_y + half_height)

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 120), 3)
        cv2.rectangle(frame, top_left, bottom_right, (15, 40, 15), 1)
        cv2.putText(
            frame,
            f"Screen {self.rectangle_state.width}x{self.rectangle_state.height}",
            (top_left[0], max(30, top_left[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )

    def _draw_pinch_guides(
        self,
        frame: np.ndarray,
        thumb_px: Tuple[int, int],
        index_px: Tuple[int, int],
        center_px: Tuple[int, int],
    ) -> None:
        """Draw colored pinch markers so the user can see the control points."""
        cv2.circle(frame, thumb_px, 10, (0, 140, 255), -1)
        cv2.circle(frame, index_px, 10, (255, 180, 0), -1)
        cv2.line(frame, thumb_px, index_px, (255, 255, 255), 2)
        cv2.circle(frame, center_px, 6, (0, 255, 120), -1)

    def _draw_hand_label(
        self, frame: np.ndarray, hand_points: List[Tuple[int, int]], label: str
    ) -> None:
        """Draw a label close to the wrist landmark."""
        wrist_px = hand_points[WRIST_INDEX]
        cv2.putText(
            frame,
            label,
            (wrist_px[0] - 30, wrist_px[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _draw_help_panel(self, frame: np.ndarray) -> None:
        """Render top-of-screen status text and playback shortcuts."""
        panel_height = 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        status = (
            self.status_text
            if time.time() <= self.status_until
            else f"Video: {self.video_player.current_video_name()}"
        )
        cv2.putText(
            frame,
            status,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Library: {self.video_library.count()} videos | V: Video ON/OFF | K: Skeleton ON/OFF | P: Skeleton Mode | S: Sign Mode | N: Next | B: Back | R: Refresh",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 230, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Add your video files into the videos/ folder. Press Q or ESC to exit.",
            (15, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        if self.video_player.last_error:
            cv2.putText(
                frame,
                self.video_player.last_error,
                (15, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (80, 180, 255),
                1,
                cv2.LINE_AA,
            )

    def _smooth_distance(self, raw_distance: float) -> float:
        """Smooth pinch changes so the rectangle size does not jitter heavily."""
        if self.smoothed_pinch_distance is None:
            return raw_distance
        return (
            PINCH_SMOOTHING * raw_distance
            + (1.0 - PINCH_SMOOTHING) * self.smoothed_pinch_distance
        )

    def _pinch_distance_to_width(self, pinch_distance: float, frame_width: int) -> int:
        """Convert pinch distance in pixels into a bounded rectangle width."""
        scaled_width = int(pinch_distance * 3.4)
        max_width = int(frame_width * MAX_RECT_WIDTH_RATIO)
        return max(MIN_RECT_WIDTH, min(scaled_width, max_width))

    def _clamp_center_to_frame(
        self,
        center: Tuple[int, int],
        rect_width: int,
        rect_height: int,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[int, int]:
        """Keep the rectangle fully visible inside the camera frame."""
        half_width = rect_width // 2
        half_height = rect_height // 2
        clamped_x = max(
            FRAME_MARGIN + half_width,
            min(center[0], frame_width - FRAME_MARGIN - half_width),
        )
        clamped_y = max(
            FRAME_MARGIN + half_height,
            min(center[1], frame_height - FRAME_MARGIN - half_height),
        )
        return clamped_x, clamped_y

    def _normalized_to_pixel(
        self, x: float, y: float, frame_width: int, frame_height: int
    ) -> Tuple[int, int]:
        """Convert normalized landmark coordinates into image pixel positions."""
        pixel_x = int(max(0.0, min(1.0, x)) * (frame_width - 1))
        pixel_y = int(max(0.0, min(1.0, y)) * (frame_height - 1))
        return pixel_x, pixel_y

    def _set_status(self, text: str) -> None:
        """Update temporary status text without unnecessary churn."""
        if text != self.status_text:
            self.status_text = text
            self.status_until = time.time() + STATUS_TEXT_DURATION_SECONDS


def ensure_model_exists(model_path: Path, model_url: str) -> None:
    """Download the MediaPipe task model if it is not already present."""
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_file(model_url, model_path)
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not download the MediaPipe hand landmarker model. "
            f"Download it manually from {model_url} and place it at "
            f"{model_path.resolve()}."
        ) from error


def download_file(url: str, destination: Path) -> None:
    """Download a file safely to a temporary name, then atomically replace it."""
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        temp_path.write_bytes(data)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> None:
    """Launch the gesture-controlled floating video player."""
    app = GestureRectangleApp()
    app.run()


if __name__ == "__main__":
    main()
