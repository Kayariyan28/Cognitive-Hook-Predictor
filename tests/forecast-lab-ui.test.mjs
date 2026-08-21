import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { validateForecastCapabilityReport } from "../src/forecast/client.js";
import {
  FORECAST_METRIC_SPECS,
  metricCanShowNumber,
  presentForecastMetric,
  resolveForecastMetric,
} from "../src/forecast/forecastPresentation.js";

function metric(overrides = {}) {
  return {
    status: "available",
    value: 73,
    scoreKind: "behavioral-probability",
    behavioralOutcome: true,
    validation: { status: "production-calibrated", reason: null },
    ...overrides,
  };
}

function capabilityFixture() {
  const branch = (state, configured = false, role = "feature-only", serverExecutionAvailable = false) => ({
    state,
    configured,
    role,
    evidenceKind: "declared-capability",
    serverExecutionAvailable,
    forecastContribution: false,
    substitution: null,
    readiness: { executable: false },
  });
  const branches = {
    browserLocalMetadata: branch("implemented-client-runtime", true),
    browserLocalVisual: branch("implemented-client-runtime", true),
    browserLocalAudio: branch("implemented-client-runtime", true),
    vjepa21: branch("not-configured"),
    videoLlama21Av: branch("not-configured"),
    audioModel: branch("not-configured"),
    semanticModel: branch("not-configured"),
    measuredAudio: {
      ...branch("ready", true, "measured-audio", true),
      currentlyReady: true,
    },
    asr: branch("not-configured"),
    ocr: branch("not-configured"),
    account: branch("not-configured"),
    trends: branch("not-configured"),
    competitors: branch("not-configured"),
    tribeCortical: {
      ...branch("not_loaded", false, "cortical-only"),
      currentlyReady: false,
    },
  };
  const calibrationHeads = Object.fromEntries([
    "hookStrength",
    "retention5Second",
    "completionProbability",
    "rewatchPotential",
    "sharePotential",
    "predictedEngagement",
    "viralityPotential",
  ].map((key) => [key, { state: "not-configured", configured: false, scoringAvailable: false, artifact: null }]));
  return {
    schemaVersion: "creator-forecast-capabilities/1",
    service: "creator-forecast",
    state: "evidence-job-service",
    scoreGenerationAvailable: false,
    uploadsAccepted: true,
    jobExecution: {
      schemaVersion: "creator-forecast-job-runtime/1",
      state: "ready",
      submissionRoute: "/api/forecast/v1/jobs",
      uploadsAccepted: true,
      authoritativeMediaProbe: true,
      durableIdempotency: "client-uuid-job-identity",
      configuredBranchCount: 1,
      readyBranchCount: 1,
      executableCalibrationHeadCount: 0,
    },
    supportedInput: {
      kind: "short-video",
      durationSeconds: { minimum: 10, maximum: 60, inclusive: true },
      durationValidation: "Authoritative server probe required.",
    },
    availableNowProfile: {
      id: "available-now-evidence/1",
      behavioralForecastAvailable: false,
      implementedDescriptiveProviders: ["browserLocalMetadata", "browserLocalVisual", "browserLocalAudio"],
    },
    registry: { schemaVersion: "creator-forecast-registry/1", immutable: true },
    branches,
    calibrationHeads,
    boundaries: ["No outcome scores are generated."],
  };
}

test("forecast metric values fail closed while preserving a real calibrated zero", () => {
  assert.equal(metricCanShowNumber(metric({ value: 0 })), true);
  assert.equal(presentForecastMetric(metric({ value: 0 }), { unit: "%" }).valueText, "0");
  assert.equal(metricCanShowNumber(metric({ value: null })), false);
  assert.equal(metricCanShowNumber(metric({ value: "" })), false);
  assert.equal(metricCanShowNumber(metric({ status: "withheld", value: 91 })), false);
  assert.equal(metricCanShowNumber(metric({ value: 91, validation: { status: "not-calibrated" } })), false);
  assert.equal(presentForecastMetric(metric({ value: 91, validation: { status: "not-calibrated" } })).valueText, "—");
});

test("validated source-backed context percentiles may render without unlocking behavioral numerics", () => {
  const context = metric({
    value: 0,
    scoreKind: "context-cohort-percentile",
    behavioralOutcome: false,
    validation: { status: "validated-context-snapshot", reason: null },
  });
  const presented = presentForecastMetric(context, { unit: "/100" });
  assert.equal(presented.canShowNumber, true);
  assert.equal(presented.valueText, "0");
  assert.equal(presented.evidenceLabel, "Source-backed context percentile");

  const behavioral = metric({ value: 88, validation: { status: "validated-context-snapshot" } });
  assert.equal(presentForecastMetric(behavioral).canShowValue, false);
});

test("predictedEngagement aliases to first-pass watch and top-level confidence stays provenance gated", () => {
  const watchSpec = FORECAST_METRIC_SPECS.find((item) => item.key === "expectedFirstPassWatch");
  const watch = resolveForecastMetric({ metrics: { predictedEngagement: metric({ value: 64 }) } }, watchSpec);
  assert.equal(watch.value, 64);
  assert.equal(presentForecastMetric(watch, watchSpec).valueText, "64");

  const confidenceSpec = FORECAST_METRIC_SPECS.find((item) => item.key === "confidence");
  const confidence = resolveForecastMetric({
    metrics: {},
    confidence: metric({ value: "medium", scoreKind: "calibration-confidence-category" }),
  }, confidenceSpec);
  assert.equal(presentForecastMetric(confidence, confidenceSpec).valueText, "Medium");

  const uncalibrated = { ...confidence, validation: { status: "not-calibrated" } };
  assert.equal(presentForecastMetric(uncalibrated, confidenceSpec).valueText, "—");
  assert.equal(resolveForecastMetric({ confidence: "high" }, confidenceSpec).status, "unavailable");
});

test("forecast capability status validates the executable job-route shape and deep-freezes it", () => {
  const report = validateForecastCapabilityReport(capabilityFixture());
  assert.equal(report.scoreGenerationAvailable, false);
  assert.equal(report.branches.tribeCortical.forecastContribution, false);
  assert.equal(Object.isFrozen(report), true);
  assert.equal(Object.isFrozen(report.branches), true);
  assert.equal(Object.isFrozen(report.branches.vjepa21), true);
  assert.equal(Object.isFrozen(report.calibrationHeads), true);
  assert.equal(Object.isFrozen(report.calibrationHeads.hookStrength), true);
  assert.throws(() => { report.branches.vjepa21.state = "ready"; }, TypeError);

  const shallowFrozenFixture = Object.freeze(capabilityFixture());
  const deeplyFrozenReport = validateForecastCapabilityReport(shallowFrozenFixture);
  assert.equal(Object.isFrozen(deeplyFrozenReport.branches), true);
  assert.equal(Object.isFrozen(deeplyFrozenReport.branches.vjepa21), true);
  assert.equal(Object.isFrozen(deeplyFrozenReport.calibrationHeads.hookStrength), true);
});

test("Forecast Lab exposes all creator inputs, evidence lanes, requested rows, and fail-closed copy", async () => {
  const component = await readFile(new URL("../src/components/ForecastLab.jsx", import.meta.url), "utf8");
  const presentation = await readFile(new URL("../src/forecast/forecastPresentation.js", import.meta.url), "utf8");
  const styles = await readFile(new URL("../src/forecast-lab.css", import.meta.url), "utf8");
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const uiSource = `${component}\n${presentation}`;

  for (const label of [
    "Intrinsic clip evidence",
    "Context opportunity",
    "Account / distribution",
    "Platform",
    "Caption",
    "Topic",
    "Genre",
    "Locale",
    "Language",
    "Intended post date / time",
    "UTC offset",
    "Account band",
    "Account history summary",
    "Hook Strength",
    "5-sec Retention",
    "Completion Probability",
    "Rewatch Potential",
    "Share Potential",
    "Trend Alignment",
    "Content Saturation",
    "Expected first-pass watch %",
    "Virality Potential",
    "Confidence",
  ]) assert.match(uiSource, new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i"));

  for (const signal of ["Opening signal", "Post-opening continuity", "Pacing", "Ending", "Visual clarity", "Visual stability", "Audio support"]) {
    assert.ok(component.includes(signal), `missing descriptive signal: ${signal}`);
  }
  assert.match(component, /AVAILABLE NOW/);
  assert.match(component, /scoreGenerationAvailable:false/);
  assert.match(component, /no executable production-calibrated head is configured/);
  assert.match(component, /backend capability registry is unreachable/);
  assert.match(component, /no backend capability registry is configured/);
  assert.match(component, /capability verification is pending/);
  assert.match(component, /onRetryCapabilities/);
  assert.match(component, /forecastContribution:false/);
  assert.match(component, /BROWSER PREFLIGHT ELIGIBLE · SERVER PROBE REQUIRED/);
  assert.match(component, /10–60S INCLUSIVE/);
  assert.match(component, /Not predicted attention, retention, engagement or virality/);
  assert.match(component, /Local substitutes are descriptive and never relabeled/);
  assert.match(styles, /\.forecast-lab\s*\{[\s\S]*position:\s*fixed;[\s\S]*inset:\s*0/);
  assert.match(styles, /grid-template-columns:\s*minmax\(225px/);

  assert.match(app, /deriveCreatorForecast\(\{/);
  assert.match(app, /backendCapabilities:\s*capabilityReport/);
  assert.match(app, /creatorInputs, forecastCapabilityState/);
  assert.match(app, /getForecastCapabilityStatus\(\{ signal: controller\.signal \}\)/);
  assert.match(app, /explicitPostInstant\(creatorInputs\.intendedPostAt, creatorInputs\.utcOffset\)/);
  assert.match(app, /The active TRIBE request must finish before you replace the original video/);
  assert.match(app, /Three evidence lanes/);
  assert.match(app, /behavioral values require production calibration/);
  assert.match(app, /onRetryCapabilities=\{retryForecastCapabilities\}/);
});

test("Forecast Lab uses one reachable full-screen scroll surface without clamped explanations", async () => {
  const styles = await readFile(new URL("../src/forecast-lab.css", import.meta.url), "utf8");

  const labRule = styles.match(/\.forecast-lab\s*\{([\s\S]*?)\}/)?.[1] ?? "";
  const topbarRule = styles.match(/\.forecast-lab-topbar\s*\{([\s\S]*?)\}/)?.[1] ?? "";
  const layoutRule = styles.match(/\.forecast-lab-layout\s*\{([\s\S]*?)\}/)?.[1] ?? "";
  const panelRule = styles.match(/\.forecast-evidence-panel\s*\{([\s\S]*?)\}/)?.[1] ?? "";
  const explanationRule = styles.match(/\.forecast-metric-copy span\s*\{([\s\S]*?)\}/)?.[1] ?? "";

  assert.match(labRule, /overflow-y:\s*auto/);
  assert.doesNotMatch(labRule, /overflow:\s*hidden/);
  assert.match(topbarRule, /position:\s*sticky/);
  assert.match(topbarRule, /top:\s*0/);
  assert.match(layoutRule, /height:\s*auto/);
  assert.match(layoutRule, /align-items:\s*start/);
  assert.match(panelRule, /overflow:\s*visible/);
  assert.doesNotMatch(styles, /-webkit-line-clamp/);
  assert.doesNotMatch(explanationRule, /overflow:\s*hidden/);
  assert.doesNotMatch(styles, /\.forecast-(?:context|score|evidence)-panel\s*\{[^}]*overflow-y:\s*auto/s);
});

test("Forecast Lab opens in a readable focused view with expandable panels and draggable text sizing", async () => {
  const component = await readFile(new URL("../src/components/ForecastLab.jsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../src/forecast-lab.css", import.meta.url), "utf8");

  assert.match(component, /useState\("scores"\)/);
  assert.match(component, /Forecast Lab workspace views/);
  for (const label of ["Post context", "Forecast", "Evidence", "All panels"]) {
    assert.ok(component.includes(label), `missing Forecast Lab view: ${label}`);
  }
  assert.match(component, /type="range"[\s\S]*min="100"[\s\S]*max="140"/);
  assert.match(component, /aria-label="Forecast Lab text size"/);
  assert.match(component, /onInput=\{updateTextScale\}/);
  assert.match(component, /onChange=\{updateTextScale\}/);
  assert.match(component, /labRef\.current\?\.scrollTo\(\{ top: 0, behavior: "auto" \}\)/);
  assert.match(component, /hidden=\{workspaceView !== "all" && workspaceView !== "context"\}/);
  assert.match(component, /hidden=\{workspaceView !== "all" && workspaceView !== "scores"\}/);
  assert.match(component, /hidden=\{workspaceView !== "all" && workspaceView !== "evidence"\}/);
  assert.match(styles, /\.forecast-lab-layout:not\(\.forecast-lab-layout--all\)/);
  assert.match(styles, /--forecast-text-scale/);
  assert.match(styles, /cursor:\s*ew-resize/);
  assert.match(styles, /\.forecast-lab-layout > \[hidden\]\s*\{\s*display:\s*none !important/);
  assert.match(styles, /@media \(max-width: 1320px\)[\s\S]*\.forecast-workspace-tools\s*\{[\s\S]*grid-row:\s*2/);
});

test("Forecast Lab runs one durable async evidence job and exposes authoritative provider evidence", async () => {
  const component = await readFile(new URL("../src/components/ForecastLab.jsx", import.meta.url), "utf8");
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const client = await readFile(new URL("../src/forecast/jobClient.js", import.meta.url), "utf8");

  assert.match(app, /forecastJobInFlightRef\.current/);
  assert.match(app, /monitorForecastJob\(initialJob/);
  assert.match(app, /No completion percentage is estimated/);
  assert.match(app, /onResumeForecastJob=\{resumeForecastJob\}/);
  assert.match(app, /onRetryForecastJob=\{retryForecastJob\}/);
  assert.match(app, /idempotencyKey:\s*jobError\?\.idempotencyKey \?\? idempotencyKey/);
  assert.doesNotMatch(app, /setForecastOpen\(false\)[\s\S]{0,120}forecastJobAbortRef\.current\?\.abort/);

  assert.match(component, /Resume this job/);
  assert.match(component, /Close · job continues/);
  assert.match(component, /INTERVAL OBSERVATIONS/);
  assert.match(component, /PINNED MODEL/);
  assert.match(component, /NanoLLaVA keyframes/);
  assert.match(component, /Measured audio/);

  assert.match(client, /creator-forecast-job\/1/);
  assert.match(client, /creator-forecast-result\/1/);
  assert.match(client, /production-calibrated/);
  assert.match(client, /same submission identity cannot create a duplicate job/);
  assert.match(client, /Forecast evidence attempted to invoke or use TRIBE/);
});
