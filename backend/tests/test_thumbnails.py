from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from backend.errors import ThumbnailExtractionError
from backend.thumbnails import extract_video_thumbnails, thumbnails_manifest


class ThumbnailExtractionTests(unittest.TestCase):
    def test_extracts_source_jpegs_at_exact_requested_segment_starts(self) -> None:
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            Path(command[-1]).write_bytes(b"\xff\xd8real-source-frame\xff\xd9")
            return subprocess.CompletedProcess(command, 0, b"", b"")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            with (
                patch("backend.thumbnails.shutil.which", return_value="/usr/bin/ffmpeg"),
                patch(
                    "backend.thumbnails._probe_video_duration_seconds",
                    return_value=2.0,
                ),
                patch("backend.thumbnails.subprocess.run", side_effect=fake_run),
            ):
                result = extract_video_thumbnails(
                    video, (0.0, 1.25), root / "thumbnails"
                )
            manifest = thumbnails_manifest("a" * 32, result)

            self.assertEqual(len(result.items), 2)
            self.assertEqual(result.unavailable_items, ())
            self.assertEqual([item.requested_time_seconds for item in result.items], [0.0, 1.25])
            self.assertEqual(manifest["captureTimeBasis"], "tribe-segment-start")
            self.assertEqual(manifest["items"][1]["requestedTimeSeconds"], 1.25)
            payload = result.items[0].path.read_bytes()
            self.assertEqual(result.items[0].sha256, hashlib.sha256(payload).hexdigest())
            for command, expected in zip(commands, ("0", "1.25")):
                self.assertLess(command.index("-i"), command.index("-ss"))
                self.assertEqual(command[command.index("-ss") + 1], expected)

    def test_rejects_negative_times_without_clamping_or_fabricating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            with patch("backend.thumbnails.shutil.which", return_value="ffmpeg"):
                with self.assertRaisesRegex(ThumbnailExtractionError, "non-negative"):
                    extract_video_thumbnails(video, (-0.25,), root / "thumbnails")
            self.assertFalse((root / "thumbnails").exists())

    def test_failed_decode_removes_all_partial_stills(self) -> None:
        def fake_run(command, **kwargs):
            if command[command.index("-ss") + 1] == "0":
                Path(command[-1]).write_bytes(b"\xff\xd8first\xff\xd9")
                return subprocess.CompletedProcess(command, 0, b"", b"")
            return subprocess.CompletedProcess(command, 1, b"", b"decode failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            with (
                patch("backend.thumbnails.shutil.which", return_value="ffmpeg"),
                patch(
                    "backend.thumbnails._probe_video_duration_seconds",
                    return_value=10.0,
                ),
                patch("backend.thumbnails.subprocess.run", side_effect=fake_run),
            ):
                with self.assertRaisesRegex(ThumbnailExtractionError, "interval 1"):
                    extract_video_thumbnails(video, (0.0, 1.0), root / "thumbnails")
            self.assertFalse((root / "thumbnails").exists())

    def test_preserves_decoded_stills_and_marks_only_out_of_source_times(self) -> None:
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            requested_time = float(command[command.index("-ss") + 1])
            if requested_time < 2.0:
                Path(command[-1]).write_bytes(b"\xff\xd8source-frame\xff\xd9")
                return subprocess.CompletedProcess(command, 0, b"", b"")
            return subprocess.CompletedProcess(command, 234, b"", b"no packets")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(b"video")
            with (
                patch("backend.thumbnails.shutil.which", return_value="ffmpeg"),
                patch(
                    "backend.thumbnails._probe_video_duration_seconds",
                    return_value=2.0,
                ),
                patch("backend.thumbnails.subprocess.run", side_effect=fake_run),
            ):
                result = extract_video_thumbnails(
                    video, (0.0, 1.0, 2.0, 3.0), root / "thumbnails"
                )
            manifest = thumbnails_manifest("a" * 32, result)

            self.assertEqual([item.index for item in result.items], [0, 1])
            self.assertEqual(
                [item.index for item in result.unavailable_items], [2, 3]
            )
            self.assertEqual(len(commands), 3)
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(
                manifest["unavailableItems"],
                [
                    {
                        "index": 2,
                        "requestedTimeSeconds": 2.0,
                        "reason": "outside-source-video",
                    },
                    {
                        "index": 3,
                        "requestedTimeSeconds": 3.0,
                        "reason": "outside-source-video",
                    },
                ],
            )
            self.assertEqual(
                sorted(path.name for path in (root / "thumbnails").iterdir()),
                ["0000.jpg", "0001.jpg"],
            )


if __name__ == "__main__":
    unittest.main()
