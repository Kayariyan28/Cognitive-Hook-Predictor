from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

from backend.forecast.orchestrator import MultimodalEvidenceOrchestrator
from backend.forecast.jobs import unavailable_behavioral_heads


@dataclass
class StubOutput:
    evidence_kind: str = "learned-visual-representation"
    features = {"vjepa2_1.motion_change": 0.2}

    def public_value(self):
        return {
            "schemaVersion": "creator-forecast-worker-output/1",
            "branch": "vjepa21",
            "features": dict(self.features),
            "behavioralOutcome": False,
        }


class StubAdapter:
    def __init__(self, *, fails=False):
        self.fails = fails
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.fails:
            raise RuntimeError("private worker detail")
        return StubOutput()


class StubScoringRuntime:
    def score_all(self, features, context):
        self.features = dict(features)
        self.context = dict(context)
        result = unavailable_behavioral_heads("not installed for this test metric")
        result["retention5Second"] = {
            "status": "available",
            "value": 0.7,
            "featureContract": {
                "schemaVersion": "creator-forecast-feature-contract/1",
                "sha256": "c" * 64,
                "features": [
                    {
                        "name": "metadata.duration_seconds",
                        "source": "videoMetadata",
                    },
                    {
                        "name": "vjepa2_1.motion_change",
                        "source": "vjepa21",
                    },
                ],
            },
            "targetContract": {
                "targetId": "creator-forecast.continuous-retention-at-5s.v1",
                "platform": "youtube-shorts",
                "domain": "short-form-video",
                "locale": "en-us",
                "evaluationManifestId": "release/retention5/evaluation-v1",
                "evaluationManifestSha256": "d" * 64,
            },
        }
        return result


class ForecastOrchestratorTests(unittest.TestCase):
    def test_runs_configured_evidence_and_withholds_every_behavioral_head(self):
        adapter = StubAdapter()
        stages = []
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "clip.mp4"
            video_path.write_bytes(b"video")
            result = MultimodalEvidenceOrchestrator({"vjepa21": adapter}).run(
                video_path,
                "a" * 64,
                {"durationSeconds": 12.5, "sizeBytes": 5, "contentType": "video/mp4"},
                {"platform": "youtube-shorts"},
                stages.append,
            )

        provider = result["evidence"]["optionalProviders"]["vjepa21"]
        self.assertEqual(provider["status"], "available")
        self.assertFalse(provider["forecastContribution"])
        self.assertFalse(provider["behavioralOutcome"])
        self.assertEqual(adapter.calls[0]["duration_seconds"], 12.5)
        self.assertEqual(
            stages,
            ["checking-vjepa21", "extracting-vjepa21", "assembling-evidence"],
        )
        self.assertTrue(result["behavioralHeads"])
        for head in result["behavioralHeads"].values():
            self.assertEqual(head["status"], "unavailable")
            self.assertIsNone(head["value"])

    def test_worker_failure_is_isolated_and_does_not_leak_error_text(self):
        stages = []
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "clip.mp4"
            video_path.write_bytes(b"video")
            result = MultimodalEvidenceOrchestrator(
                {"vjepa21": StubAdapter(fails=True)}
            ).run(
                video_path,
                "b" * 64,
                {"durationSeconds": 10.0, "sizeBytes": 5, "contentType": None},
                None,
                stages.append,
            )

        provider = result["evidence"]["optionalProviders"]["vjepa21"]
        self.assertEqual(provider["status"], "unavailable")
        self.assertTrue(provider["configured"])
        self.assertNotIn("private worker detail", provider["reason"])
        self.assertEqual(
            set(result["evidence"]["optionalProviders"]),
            {
                "vjepa21",
                "videoLlama21Av",
                "audioModel",
                "semanticModel",
                "measuredAudio",
                "asr",
                "account",
                "trends",
                "competitors",
            },
        )

    def test_successful_head_names_every_consumed_source_and_contract(self):
        adapter = StubAdapter()
        head_runtime = StubScoringRuntime()
        context = {
            "platform": "youtube-shorts",
            "domain": "short-form-video",
            "locale": "en-us",
        }
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "clip.mp4"
            video_path.write_bytes(b"video")
            result = MultimodalEvidenceOrchestrator(
                {"vjepa21": adapter}, head_runtime
            ).run(
                video_path,
                "e" * 64,
                {
                    "durationSeconds": 12.5,
                    "sizeBytes": 5,
                    "contentType": "video/mp4",
                },
                context,
                lambda _stage: None,
            )

        evidence = result["evidence"]
        visual = evidence["optionalProviders"]["vjepa21"]
        self.assertTrue(visual["forecastContribution"])
        self.assertFalse(visual["behavioralOutcome"])
        self.assertEqual(
            visual["forecastUse"],
            {
                "kind": "calibrated-head-feature-input",
                "usedByHeads": ["retention5Second"],
                "featureContracts": [
                    {
                        "head": "retention5Second",
                        "featureContractSha256": "c" * 64,
                        "featureNames": ["vjepa2_1.motion_change"],
                    }
                ],
            },
        )
        metadata = evidence["videoMetadata"]
        self.assertTrue(metadata["forecastContribution"])
        self.assertEqual(
            metadata["forecastUse"]["featureContracts"][0]["featureNames"],
            ["metadata.duration_seconds"],
        )
        creator_context = evidence["creatorContext"]
        self.assertTrue(creator_context["forecastContribution"])
        self.assertEqual(
            creator_context["forecastUse"]["usedByHeads"],
            ["retention5Second"],
        )
        self.assertFalse(evidence["tribe"]["forecastContribution"])


if __name__ == "__main__":
    unittest.main()
