# Historical local TRIBE v2 vision-only verification report

This document records one completed run on 2026-08-18. It is immutable local
run evidence, not a claim about the current checkout or a substitute for a
fresh test run on another machine.

## Outcome

**PASS — a genuine, pinned local TRIBE v2 vision-only inference completed and
produced a valid cortical prediction artifact.** The result contains six
time-indexed frames over the exact 20,484-vertex `fsaverage5` surface. The raw
tensor is finite, non-constant across time, and its independently computed
SHA-256 matches the persisted manifest.

This is specifically a **vision-only ablation**, not the published full
video/audio/text path. The run used the official V-JEPA2 visual extractor and
the released TRIBE cortical encoder; audio and text were disabled. It is also
**not a virality score**. TRIBE predicts an average-subject cortical BOLD
response and has no views, sharing, reach, or engagement head.

## Input and run identity

The input was a locally generated synthetic color-bar test fixture containing
no person, private creator media, or third-party footage. Its digest is
published below so the historical run can be identified unambiguously.

| Item | Verified value |
| --- | --- |
| Input | `work/test-signal-video.mp4` |
| Input SHA-256 | `9e8b5f2b4ddc16fb714092ce010f1caf046cdee984e9f711a99a3d2444cd92b0` |
| Input size | 657,660 bytes |
| Video | H.264, 640 x 360, 24 fps, 144 frames, 6.000 s |
| Audio track | AAC, 6.000 s; present in the file but not used by this ablation |
| Result ID | `4d5b125f8b134566a63ad58515456b10` |
| Manifest creation time | `2026-08-18T12:50:59.254451+00:00` |
| Inference mode | `vision-only` |
| Modalities used | `video` |
| Missing modalities | `audio`, `text` |
| Local execution | V-JEPA2 visual extraction on CPU; TRIBE cortical encoder on MPS |

The feature-cache operation began at 18:09:49 IST and its data artifact was
written at 18:20:58 IST, an elapsed time of approximately **11 min 09 s**. The
brain tensor was persisted at 18:20:59 IST, approximately **1 s** later. These
durations are derived from local filesystem timestamps, not a profiler, so they
should be treated as run evidence rather than a formal benchmark.

## Immutable model provenance

| Component | Repository/model | Immutable revision | Verified weights SHA-256 |
| --- | --- | --- | --- |
| TRIBE v2 code | `facebookresearch/tribev2` | `af58661791a351a448a489042a28f6c37e1c14b7` | N/A |
| TRIBE v2 model | `facebook/tribev2` | `f894e783020944dcd96e5568550afe2aa9743f9f` | `9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321` |
| Visual extractor | `facebook/vjepa2-vitg-fpc64-256` | `875c192b7b704b87d1e1d99345769632dd5f739a` | `f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b` |

The installed TRIBE package resolves to the code commit above. Both cached
weight files were independently hashed and matched the digests shown here. The
worker loads both components by full immutable revision, verifies those hashes
before inference, and was configured with `HF_HUB_OFFLINE=1` for this local
run. The manifest records the same provenance. The TRIBE code and weights are
reported by the project under **CC BY-NC 4.0**; commercial use requires an
appropriate license or permission.

## Tensor verification

The persisted artifact is
`backend/.runtime/results/4d5b125f8b134566a63ad58515456b10/frames.f32`.

| Check | Verified value |
| --- | --- |
| Encoding | little-endian `float32`, time-major |
| Shape | `[6, 20484]` |
| Scalar count | 122,904 |
| Byte length | 491,616 (`6 x 20,484 x 4`) |
| Tensor SHA-256 | `84705afd1086d2d32b9ded09b5620c09300f568dfc01851e78bb52a544247994` |
| Finite values | 122,904 / 122,904 |
| Minimum | -0.67415118 |
| Maximum | 0.77004290 |
| Mean | -0.03430518 |
| Population standard deviation | 0.13391462 |
| Frame times | `[0, 1, 2, 3, 4, 5]` seconds |
| Frame durations | `[1, 1, 1, 1, 1, 1]` seconds |
| Hemodynamic offset | 5 seconds |
| Global display domain | `[-0.37563734, 0, 0.38324644]` (p1 / zero / p99) |

Adjacent-frame L2 deltas across all 20,484 vertices were:

| Transition | L2 delta |
| --- | ---: |
| Frame 0 -> 1 | 1.766438 |
| Frame 1 -> 2 | 2.146267 |
| Frame 2 -> 3 | 2.207066 |
| Frame 3 -> 4 | 6.953980 |
| Frame 4 -> 5 | 1.964350 |

Every one of the 20,484 vertex values changed in each adjacent transition.
This establishes that the six authoritative frames contain real temporal
variation for the prediction map; they are not one tensor repeated six times.

## Browser integration verification

The production build restored the same result through the live local API,
validated its manifest, downloaded the raw float32 artifact, matched its
SHA-256, and rendered frame 1/6 at 0 seconds and frame 5/6 at 4 seconds.

The Destrieux parcel readout changed with the model frame. For example, the
right `S_oc-temp_med_and_Lingual` parcel changed from `+0.391` on frame 1 to
`+0.244` on frame 5; the ordering of the next-ranked parcels also changed.
Selecting that parcel returned to the prediction map and focused its exact 134
right-hemisphere vertices. Browser screenshots were checked locally and are
deliberately excluded from the public repository because they can contain
uploaded-video frames.

The standalone replay control was also exercised on the restored result: it
advanced from frame 1/6 to frame 3/6 in real time, updated the timeline, and
returned to a stable frame when paused. A fresh production-preview tab reported
no console warnings or errors.

At the time of this run, the then-current frontend/backend verification suites
and production build passed. Run the commands in the root README for the
current checkout; this historical report does not assert a current test count.

## MPS acceleration comparison

A separate controlled benchmark replayed the same six-second input through the
same pinned V-JEPA2 snapshot, unchanged official temporal sampling and feature
aggregation, and the same pinned TRIBE cortical encoder. Only V-JEPA2 execution
changed from CPU FP32 to MPS FP16 autocast.

| Check | Verified value |
| --- | ---: |
| CPU reference elapsed | approximately 11 min 09 s |
| MPS FP16-autocast elapsed | 415.2646 s (6 min 55 s) |
| End-to-end local speedup | approximately 1.61x |
| Final cortical-tensor correlation | 0.999999113 |
| Mean absolute difference | 0.000131402 |
| Root mean square difference | 0.000185821 |
| Maximum absolute difference | 0.001320899 |

All six MPS result frames were finite and retained shape `[6, 20484]`. The
fixed global display domain changed from
`[-0.37563734, 0, 0.38324644]` to
`[-0.37557033, 0, 0.38382179]`. These small differences are expected from
autocast and are why `device` and `precision` are now explicit result
provenance and part of the cache identity. A one-window test measured 68.85 s
on CPU versus 16.79 s on MPS, but sustained end-to-end speedup was smaller as
the shared-memory device warmed and throttled. The end-to-end figure is the
more representative local result.

## Surface and spatial mapping

The manifest identifies the bundled surface as `fsaverage5` with mapping ID:

`fsaverage5-half:52eda3c221c9439b320f97663d441390c56a14e69c2cddb7a5073f6d550466cb`

The tensor order is left hemisphere vertices 0-10,241 followed by right
hemisphere vertices 10,242-20,483. The backend serializes the model output
without reordering vertices, fabricating missing vertices, or applying
per-frame normalization. One fixed p1/p99 scale computed over the complete
tensor drives the browser colors.

## Evidence and method

The verification used the saved result manifest and tensor, `shasum -a 256`,
`ffprobe`, and NumPy decoding as little-endian float32. It checked the exact
shape and byte-length equation, all-value finiteness, descriptive statistics,
global display percentiles, and adjacent-frame differences. Provenance was
cross-checked against `backend/requirements.txt`, `backend/config.py`, the
installed TRIBE Git checkout, and the cached model files.

The primary saved evidence was retained locally and is deliberately excluded
from Git because it contains an uploaded video and generated model artifacts:

- `backend/.runtime/results/4d5b125f8b134566a63ad58515456b10/manifest.json`
- `backend/.runtime/results/4d5b125f8b134566a63ad58515456b10/frames.f32`
- `work/real-manifest.json`
- `work/test-signal-video.mp4`

## Interpretation limits

- This run proves the local vision-only path can execute end to end with the
  pinned official models and produce a valid, time-varying cortical tensor.
- It does not prove the full trimodal path; audio and text were not evaluated.
- It does not validate biological accuracy against measured participant fMRI,
  nor does one synthetic test clip establish general predictive performance.
- The output is a predicted average-subject BOLD response, not an individual
  measurement, medical result, or clinical signal.
- It does not predict virality. Any separate content score in the interface is
  a distinct proxy and must not be described as TRIBE output.
