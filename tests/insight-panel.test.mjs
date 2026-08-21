import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  HYPOTHESIS_LABEL,
  INSIGHT_LANES,
  parseCitation,
  resolveCitation,
  resolveJsonPointer,
  validateInsightArtifact,
  validateInsightBundle,
} from "../src/insight/contract.js";
import {
  HOOK_REPORT_SECTION_LABELS,
  TRIBE_STANDING_CAPTION,
  buildAbCompareLink,
  experimentsByEffort,
  phaseCommentaryInClipOrder,
  presentCitation,
  presentProviderState,
  presentUnavailable,
} from "../src/insight/presentation.js";

function bundleFixture({ tribe = "present" } = {}) {
  return {
    schemaVersion: "insight-evidence-bundle/2",
    source: { forecastResultId: "a".repeat(32), tribeResultId: null, window: [0, 3] },
    inputEvidenceHash: "c".repeat(64),
    lanes: {
      measured: {
        status: "present",
        video: { durationSeconds: 21.5, sizeBytes: 4823104, contentType: "video/mp4", sha256: "a".repeat(64) },
        audio: {
          status: "present",
          descriptors: { rms: 0.1372, silent_window_fraction: 0.1875 },
          energyPeaks: [{ index: 0, startSec: 1.44, endSec: 1.504, text: "A relative short-window RMS peak was measured here." }],
        },
      },
      nanollava: {
        status: "present",
        keyframes: [{ index: 0, startSec: 0, endSec: 2.15, text: "{}", parsed: { scene: "a kitchen counter" } }],
        warnings: [],
      },
      ast: {
        status: "present",
        windows: [{ index: 0, startSec: 0, endSec: 10, labels: [{ label: "Speech", modelScore: 0.8125 }] }],
        descriptors: { mean_top_label_score: 0.6425 },
      },
      vjepa: { status: "absent", reason: "No hash-verified V-JEPA 2.1 artifact is registered." },
      asr: { status: "absent", reason: "the transcript branch did not publish evidence for this result" },
      ocr: { status: "absent", reason: "the on-screen-text branch did not publish evidence for this result" },
      context: { status: "present", declared: { platform: "reels" } },
      tribe: tribe === "present"
        ? {
          status: "present",
          rankedBy: "rms",
          provenance: { resultId: "b".repeat(32), modelId: "facebook/tribev2" },
          intervals: [{ index: 0, startSec: 0, endSec: 1.49, durationSec: 1.49, magnitude: 0.4 }],
          phases: [{ id: "early", startSec: 0, endSec: 2.98, responseMagnitude: 0.425 }],
          parcels: [{ key: "left-0", name: "Parcel 0", hemisphere: "left", rms: 0.9 }],
        }
        : { status: "absent", reason: "no TRIBE creator-report descriptor document was supplied" },
    },
  };
}

function artifactFixture(overrides = {}) {
  return {
    schemaVersion: "insight/1",
    insightId: "d".repeat(32),
    generatedAt: "2026-02-01T10:00:05Z",
    source: { forecastResultId: "a".repeat(32), tribeResultId: null, window: null },
    hookReport: {
      windowSeconds: [0, 3],
      whatTheHookContains: [
        { text: "The opening frame shows a kitchen counter.", citations: ["nanollava:/keyframes/0/parsed"] },
      ],
      observations: [
        { text: "The first measured energy peak starts at 1.44 seconds.", citations: ["measured:/audio/energyPeaks/0"] },
      ],
      hypotheses: [
        {
          id: "h1",
          text: "Moving the jar action earlier may change the opening's measured audio structure.",
          label: HYPOTHESIS_LABEL,
          citations: ["measured:/audio/energyPeaks/0"],
        },
      ],
      experiments: [
        {
          id: "x1",
          hypothesisId: "h1",
          edit: "Re-cut so the jar action lands before the measured peak.",
          effort: "high",
          expectedSignalShift: [
            { metricPath: "measured:/audio/descriptors/silent_window_fraction", direction: "decrease" },
          ],
          citations: ["measured:/audio/energyPeaks/0"],
        },
        {
          id: "x2",
          hypothesisId: "h1",
          edit: "Trim the still frame at the start.",
          effort: "low",
          expectedSignalShift: [
            { metricPath: "measured:/audio/descriptors/rms", direction: "increase" },
          ],
          citations: ["ast:/windows/0"],
        },
      ],
    },
    phaseCommentary: [
      { phase: "late", text: "The late phase carries the highest predicted cortical magnitude.", citations: ["tribe:/phases/0"] },
      { phase: "early", text: "The early phase carries the lowest predicted cortical magnitude.", citations: ["tribe:/phases/0"] },
    ],
    tribeNotes: [
      { text: "These values are predicted average-subject cortical BOLD, not audience behavior.", citations: ["tribe:/provenance"] },
    ],
    behavioralOutcome: false,
    limits: "This is a descriptive lane.",
    provenance: {
      provider: "mlx-local",
      modelId: "mlx-community/Qwen2.5-7B-Instruct-4bit",
      modelRevision: "b".repeat(40),
      promptTemplateId: "hook-doctor.v1",
      promptHash: "e".repeat(64),
      temperature: 0,
      inputEvidenceHash: "c".repeat(64),
      outputHash: "f".repeat(64),
      hookOnly: false,
      behavioralOutcome: false,
    },
    ...overrides,
  };
}

test("a valid artifact fixture passes contract validation", () => {
  const artifact = validateInsightArtifact(artifactFixture());
  assert.equal(artifact.schemaVersion, "insight/1");
  assert.equal(artifact.behavioralOutcome, false);
  assert.equal(artifact.hookReport.hypotheses[0].label, HYPOTHESIS_LABEL);
  assert.equal(artifact.provenance.modelRevision, "b".repeat(40));
  assert.ok(Object.isFrozen(artifact));
});

test("every hypothesis is labelled an untested heuristic", () => {
  const payload = artifactFixture();
  payload.hookReport.hypotheses[0].label = "finding";
  assert.throws(() => validateInsightArtifact(payload), /untested heuristic/);
});

test("an artifact that claims a behavioral outcome is refused", () => {
  assert.throws(
    () => validateInsightArtifact(artifactFixture({ behavioralOutcome: true })),
    /behavioralOutcome false/,
  );
});

test("a TRIBE note that does not cite the TRIBE lane is refused", () => {
  const payload = artifactFixture();
  payload.tribeNotes[0].citations = ["ast:/windows/0"];
  assert.throws(() => validateInsightArtifact(payload), /does not cite the TRIBE lane/);
});

test("an experiment referencing an unknown hypothesis is refused", () => {
  const payload = artifactFixture();
  payload.hookReport.experiments[0].hypothesisId = "h9";
  assert.throws(() => validateInsightArtifact(payload), /hypothesis this artifact does not contain/);
});

test("the citation grammar matches the backend grammar", () => {
  assert.deepEqual(parseCitation("measured:/audio/descriptors/rms").lane, "measured");
  assert.deepEqual(parseCitation("vjepa:/windows/0@window(0.0,2.0)").window, [0, 2]);
  assert.equal(parseCitation("measured/audio"), null);
  assert.equal(parseCitation("brain:/x"), null);
  assert.equal(parseCitation("vjepa:/windows/0@window(3.0,1.0)"), null);
  assert.deepEqual([...INSIGHT_LANES].sort(), [
    "asr", "ast", "context", "measured", "nanollava", "ocr", "tribe", "vjepa",
  ]);
});

test("pointers resolve, and dangling pointers resolve to nothing", () => {
  const bundle = bundleFixture();
  assert.equal(resolveJsonPointer(bundle.lanes.measured, "/audio/descriptors/rms"), 0.1372);
  assert.equal(resolveJsonPointer(bundle.lanes.measured, "/audio/descriptors/nope"), undefined);
  assert.equal(resolveJsonPointer(bundle.lanes.ast, "/windows/9"), undefined);
});

test("a citation into an absent lane reveals the reason, never a value", () => {
  const bundle = bundleFixture();
  const chip = presentCitation(bundle, "vjepa:/windows/0");
  assert.equal(chip.status, "absent");
  assert.equal(chip.valueText, null);
  assert.match(chip.reason, /V-JEPA/);
});

test("a resolved citation reveals its evidence value and a seek time", () => {
  const chip = presentCitation(bundleFixture(), "measured:/audio/energyPeaks/0");
  assert.equal(chip.status, "resolved");
  assert.equal(chip.seekSeconds, 1.44);
  assert.equal(chip.lane, "measured");
});

test("a TRIBE citation carries the standing caption", () => {
  const chip = presentCitation(bundleFixture(), "tribe:/intervals/0");
  assert.equal(chip.status, "resolved");
  assert.equal(chip.caption, TRIBE_STANDING_CAPTION);
  assert.match(TRIBE_STANDING_CAPTION, /predicted average-subject cortical BOLD/);
  assert.match(TRIBE_STANDING_CAPTION, /not audience behavior/);
});

test("resolveCitation reports malformed citations separately from unresolved ones", () => {
  const bundle = bundleFixture();
  assert.equal(resolveCitation(bundle, "not-a-citation").status, "malformed");
  assert.equal(resolveCitation(bundle, "measured:/audio/nope").status, "unresolved");
  assert.equal(resolveCitation(bundle, "ocr:/frames/0").status, "absent");
});

test("experiments are ranked by effort, cheapest first", () => {
  const artifact = validateInsightArtifact(artifactFixture());
  assert.deepEqual(
    experimentsByEffort(artifact.hookReport.experiments).map((item) => item.id),
    ["x2", "x1"],
  );
});

test("phase commentary renders in clip order", () => {
  const artifact = validateInsightArtifact(artifactFixture());
  assert.deepEqual(
    phaseCommentaryInClipOrder(artifact.phaseCommentary).map((item) => item.phase),
    ["early", "late"],
  );
});

test("the hook report renders in the contract's order", () => {
  assert.deepEqual(Object.keys(HOOK_REPORT_SECTION_LABELS), [
    "whatTheHookContains",
    "observations",
    "hypotheses",
    "experiments",
  ]);
});

test("every unavailable state is specific rather than a spinner", () => {
  const cases = [
    ["provider_unavailable", /pinned model revision/],
    ["bundle_unavailable", /evidence for this clip is incomplete/],
    ["claim_boundary_violation", /outcome or mental-state claim/],
    ["numeric_not_in_evidence", /number the evidence does not contain/],
    ["citation_unresolvable", /cited evidence this clip does not have/],
    ["output_not_json", /rejected whole rather than repaired/],
  ];
  for (const [reasonCode, pattern] of cases) {
    const presented = presentUnavailable({ unavailable: true, reasonCode, detail: "x" });
    assert.equal(presented.status, "unavailable");
    assert.equal(presented.reasonCode, reasonCode);
    assert.match(presented.message, pattern);
  }
});

test("a claim-boundary rejection names the offending sentence and term", () => {
  const presented = presentUnavailable({
    unavailable: true,
    reasonCode: "claim_boundary_violation",
    rejectionId: "e".repeat(32),
    detail: { term: "go viral", sentence: "This clip will go viral.", itemPath: "/hookReport/observations/0/text" },
  });
  assert.match(presented.detailText, /This clip will go viral\./);
  assert.match(presented.detailText, /go viral/);
  assert.equal(presented.rejectionId, "e".repeat(32));
});

test("provider state explains why generation is offline", () => {
  const offline = presentProviderState({
    provider: { provider: "mlx-local", available: false, reason: "mlx-lm is not installed.", model: { id: null, revision: null } },
  });
  assert.equal(offline.available, false);
  assert.match(offline.message, /mlx-lm is not installed/);

  const missing = presentProviderState({});
  assert.equal(missing.available, false);
  assert.match(missing.message, /did not report a provider/);
});

test("the A/B deep link carries experiment identity into the comparison flow", () => {
  const artifact = validateInsightArtifact(artifactFixture());
  const link = buildAbCompareLink(artifact.hookReport.experiments[1], {
    forecastResultId: "a".repeat(32),
    insightId: artifact.insightId,
  });
  assert.equal(link.action, "open-ab-compare");
  assert.equal(link.experimentId, "x2");
  assert.equal(link.hypothesisId, "h1");
  assert.equal(link.baselineResultId, "a".repeat(32));
  assert.deepEqual(link.expectedSignalShift, [
    { metricPath: "measured:/audio/descriptors/rms", direction: "increase" },
  ]);
  assert.equal(buildAbCompareLink(null), null);
});

test("the evidence bundle contract requires every lane to declare a status", () => {
  const bundle = validateInsightBundle(bundleFixture({ tribe: "absent" }));
  assert.equal(bundle.lanes.tribe.status, "absent");
  const broken = bundleFixture();
  delete broken.lanes.ocr;
  assert.throws(() => validateInsightBundle(broken), /ocr lane/);
});

test("the panel never renders a placeholder score or a fabricated value", async () => {
  const source = await readFile(new URL("../src/components/InsightPanel.jsx", import.meta.url), "utf8");
  const forbidden = ["score", "probability", "estimated", "predicted views", "engagement"];
  for (const term of forbidden) {
    assert.equal(
      source.toLowerCase().includes(term),
      false,
      `InsightPanel must not mention ${term}`,
    );
  }
});

import { EXPERIMENT_STATUS_LABELS, presentExperiment } from "../src/insight/presentation.js";

function experimentRecord(overrides = {}) {
  return {
    schemaVersion: "insight-experiment/1",
    id: "a".repeat(32),
    createdAt: "2026-02-01T10:00:00Z",
    sourceInsightId: "d".repeat(32),
    sourceExperimentId: "x1",
    hypothesisIds: ["h1"],
    edit: "Trim the still frame at the start.",
    effort: "low",
    expectedSignalShift: [
      { metricPath: "measured:/audio/descriptors/silent_window_fraction", direction: "decrease" },
    ],
    status: "compared",
    linkedResultIds: ["a".repeat(32), "b".repeat(32)],
    measuredDeltas: [
      {
        metricPath: "measured:/audio/descriptors/silent_window_fraction",
        expectedDirection: "decrease",
        baselineValue: 0.1875,
        variantValue: 0.1125,
        delta: -0.075,
        observedDirection: "decrease",
        match: "matched",
        reason: null,
      },
    ],
    behavioralOutcome: false,
    ...overrides,
  };
}

test("a tracked experiment shows expected against measured direction", () => {
  const presented = presentExperiment(experimentRecord());
  assert.equal(presented.statusLabel, EXPERIMENT_STATUS_LABELS.compared);
  assert.equal(presented.baselineResultId, "a".repeat(32));
  assert.equal(presented.variantResultId, "b".repeat(32));
  assert.equal(presented.signals[0].expectedDirection, "decrease");
  assert.equal(presented.signals[0].observedDirection, "decrease");
  assert.equal(presented.signals[0].match, "matched");
  assert.match(presented.signals[0].matchLabel, /as expected/);
  assert.equal(presented.behavioralOutcome, false);
});

test("an opposite measured direction is labelled, not hidden", () => {
  const record = experimentRecord();
  record.measuredDeltas[0].observedDirection = "increase";
  record.measuredDeltas[0].match = "opposite";
  const presented = presentExperiment(record);
  assert.equal(presented.signals[0].match, "opposite");
  assert.match(presented.signals[0].matchLabel, /the other way/);
});

test("an experiment with no variant reports every signal as unmeasured", () => {
  const presented = presentExperiment(
    experimentRecord({ status: "proposed", linkedResultIds: ["a".repeat(32)], measuredDeltas: null }),
  );
  assert.equal(presented.statusLabel, EXPERIMENT_STATUS_LABELS.proposed);
  assert.equal(presented.variantResultId, null);
  assert.equal(presented.signals[0].match, "unmeasured");
  assert.equal(presented.signals[0].delta, null);
});

test("an unmeasured metric carries the reason it could not be measured", () => {
  const record = experimentRecord();
  record.measuredDeltas[0] = {
    metricPath: "measured:/audio/descriptors/silent_window_fraction",
    expectedDirection: "decrease",
    baselineValue: 0.1875,
    variantValue: null,
    delta: null,
    observedDirection: null,
    match: "unmeasured",
    reason: "This metric path does not resolve to a number in the variant result.",
  };
  const presented = presentExperiment(record);
  assert.equal(presented.signals[0].match, "unmeasured");
  assert.match(presented.signals[0].reason, /variant result/);
});

test("presentExperiment refuses anything that is not a record", () => {
  assert.equal(presentExperiment(null), null);
  assert.equal(presentExperiment("nope"), null);
});
