import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const productionFiles = [
  "../src/App.jsx",
  "../src/analysis.js",
  "../src/components/TribeBrainViewer.jsx",
];

test("production UI contains no illustrative brain asset or proxy-as-neuroscience labels", async () => {
  const source = (await Promise.all(productionFiles.map((path) => readFile(new URL(path, import.meta.url), "utf8")))).join("\n");
  for (const forbidden of ["neural-brain.png", "Visual Cortex", "Auditory Cortex", "brainFloat"]) {
    assert.equal(source.includes(forbidden), false, `found forbidden production term: ${forbidden}`);
  }
  await assert.rejects(access(new URL("../public/assets/neural-brain.png", import.meta.url)));
});

test("viewer colors are gated on a ready TRIBE prediction", async () => {
  const viewer = await readFile(new URL("../src/components/TribeBrainViewer.jsx", import.meta.url), "utf8");
  assert.match(viewer, /status !== "ready" \|\| !prediction/);
  assert.match(viewer, /writeNeutralColors/);
  assert.match(viewer, /writeScientificColors/);
  assert.equal(/autoRotate\s*=\s*true/.test(viewer), false);
});

test("vision-only mode is explicit and verification claims are scoped to contract integrity", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(source, /VIDEO-ONLY CORTICAL BOLD/);
  assert.match(source, /audio and text were not modeled/);
  assert.match(source, /contract-validated, hash-checked/);
  assert.equal(source.includes("extracting multimodal features"), false);
  assert.equal(source.includes("GPU WORKER CONNECTED"), false);
});

test("modeled engagement is directional, evidence-scoped, and independent from TRIBE colors", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(source, /scores\?\.engagementScore \?\? result\?\.scores\?\.viralPotential/);
  assert.match(source, /Modeled engagement/);
  assert.match(source, /engagement\?\.components\?\.sustainedAttention/);
  assert.match(source, /coverage/i);
  assert.match(source, /not a probability or guarantee/);
  assert.match(source, /never color(?:s)? the brain/i);
  assert.match(source, /Analyze a clip to measure content signals/);
  assert.match(source, /Very strong modeled signal/);
  assert.equal(source.includes("modeled hold"), false);
  assert.equal(source.includes("Likely early drop"), false);
  assert.equal(source.includes("holds attention"), false);
  assert.equal(source.includes("predicted watch time"), false);
  assert.equal(source.includes("engagement probability"), false);
});

test("Virality Outlook is a prominent local content-signal proxy with explicit evidence limits", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../src/creator-report.css", import.meta.url), "utf8");
  assert.match(source, /Virality outlook/);
  assert.match(source, /result\?\.virality/);
  assert.match(source, /outlook\?\.score, result\?\.scores\?\.viralPotential/);
  assert.match(source, /outlook\?\.evidenceCoverage, result\?\.scores\?\.coverage/);
  assert.match(source, /Content-signal proxy · not TRIBE output · not a calibrated probability/);
  assert.match(source, /Input coverage, not model confidence/);
  assert.match(source, /LOCAL CONTENT MODEL/);
  assert.match(source, /value === null \|\| value === undefined \|\| value === ""/);
  assert.match(source, /Object\.hasOwn\(outlook\.components, key\)/);
  assert.match(styles, /\.virality-outlook-card/);
});

test("main and result surfaces remain reachable instead of hiding evidence at responsive sizes", async () => {
  const main = await readFile(new URL("../src/main.jsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  const reportStyles = await readFile(new URL("../src/creator-report.css", import.meta.url), "utf8");
  const shellRule = styles.match(/\.console-shell\s*\{([\s\S]*?)\}/)?.[1] ?? "";
  const layoutRule = styles.match(/\.console-layout\s*\{([\s\S]*?)\}/)?.[1] ?? "";

  assert.match(main, /import "\.\/styles\.css"/);
  assert.doesNotMatch(main, /styles-runtime\.css/);
  assert.match(shellRule, /height:\s*auto/);
  assert.match(shellRule, /min-height:\s*100svh/);
  assert.match(shellRule, /overflow:\s*visible/);
  assert.doesNotMatch(shellRule, /overflow:\s*hidden/);
  assert.match(layoutRule, /height:\s*auto/);
  assert.match(styles, /\.truth-strip\s*\{[\s\S]*display:\s*grid/);

  for (const hiddenRule of [
    /\.truth-strip,\s*footer\s*\{\s*display:\s*none/,
    /\.clip-heading p\s*\{\s*display:\s*none/,
    /\.tribe-facts\s*\{\s*display:\s*none/,
    /\.model-mode\s*\{\s*display:\s*none/,
  ]) assert.doesNotMatch(styles, hiddenRule);

  for (const hiddenRule of [
    /\.virality-outlook-score p,\s*\.virality-outlook-evidence small\s*\{\s*display:\s*none/,
    /\.tribe-panel \.prediction-tabs button span\s*\{\s*display:\s*none/,
    /\.ab-comparison-table \[role="cell"\] small\s*\{\s*display:\s*none/,
  ]) assert.doesNotMatch(reportStyles, hiddenRule);
  assert.match(reportStyles, /\.ab-comparison-table\s*\{[\s\S]*overflow-x:\s*auto/);
});

test("TRIBE Creator report exposes tensor descriptors and seekable moments without behavioral labels", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../src/creator-report.css", import.meta.url), "utf8");
  assert.match(source, /deriveTribeDescriptors\(prediction, atlasState\.atlas\)/);
  assert.match(source, />Creator report </);
  for (const label of ["Predicted response magnitude", "Largest pattern change", "Pattern continuity", "Spatial distribution", "Clip phases", "Parcel moments", "Evidence boundary"]) {
    assert.ok(source.includes(label), `missing TRIBE descriptor label: ${label}`);
  }
  assert.match(source, /onClick=\{\(\) => onSeek\(frame\.time\)\}/);
  assert.match(source, /onClick=\{\(\) => onSeek\(parcel\.peakMagnitude\.time\)\}/);
  assert.match(source, /148 usable Destrieux hemisphere parcels/);
  assert.match(source, /A\/B tests to run/);
  assert.match(source, /first ~1\.5s/);
  assert.match(source, /Measure actual stayed-to-watch \/ 3s hold/);
  assert.match(source, /Measure actual completion \/ rewatch/);
  assert.match(source, /TRIBE selects moments to test; it does not predict the winner or uplift/);
  assert.equal(source.includes("152 hemispheric parcels"), false);

  const reportStart = source.indexOf("function TribeCreatorReport");
  const reportEnd = source.indexOf("function TribeFrameIntervals", reportStart);
  const reportSource = source.slice(reportStart, reportEnd);
  for (const forbidden of ["attention", "engagement", "virality", "emotion", "memory", "purchase intent", "watch time"]) {
    assert.equal(reportSource.toLowerCase().includes(forbidden), false, `behavioral label leaked into TRIBE report: ${forbidden}`);
  }
  assert.match(reportSource, /usable-cortex RMS/i);
  assert.equal(reportSource.toLowerCase().includes("whole-cortex rms"), false);
  assert.match(styles, /\.prediction-tabs\s*\{[\s\S]*grid-template-columns:\s*repeat\(5/);
  assert.match(styles, /\.creator-report\s*\{[\s\S]*overflow-y:\s*auto/);
  assert.match(styles, /\.creator-test-list/);
});

test("TRIBE Frame intervals ranks every interval with exact timing and an explicit behavioral boundary", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../src/creator-report.css", import.meta.url), "utf8");
  assert.match(source, /import \{ rankTribeFrameIntervals \} from "\.\/tribe\/frameIntervals\.js"/);
  assert.match(source, /rankTribeFrameIntervals\(report, sortMode\)/);
  assert.match(source, />Frame intervals </);
  assert.match(source, />Response magnitude<\/button>/);
  assert.match(source, />Pattern change<\/button>/);
  assert.match(source, /intervals\.map\(\(interval\) =>/);
  assert.match(source, /onClick=\{\(\) => onSeek\(interval\.startTime\)\}/);
  assert.match(source, /interval\.index === activeFrameIndex/);
  assert.match(source, /formatPreciseTime\(interval\.startTime\).*formatPreciseTime\(interval\.endTime\)/s);
  assert.match(source, /RMS \{formatDecimal\(interval\.responseMagnitude\)\}/);
  assert.match(source, /Δ \{formatDecimal\(interval\.changeRms\)\}/);
  assert.match(source, /Largest parcel RMS/);
  assert.match(source, /PEAK RESPONSE/);
  assert.match(source, /LARGEST CHANGE/);
  assert.match(source, /Percentiles are relative to this TRIBE result only/);
  assert.match(source, /Ranked TRIBE-predicted cortical response—not measured attention, watch time, interaction, or virality\./);
  assert.match(source, /Source-video frames are not TRIBE intervals/);
  assert.match(source, /without subtracting 5s/);
  assert.match(styles, /\.frame-interval-list\s*\{[\s\S]*overflow-y:\s*auto/);
  assert.match(styles, /\.frame-interval-list > button\.is-active/);
  assert.match(styles, /\.prediction-stage--intervals/);
});

test("TRIBE Final report translates descriptors into qualitative creator language without displaying metrics", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const helper = await readFile(new URL("../src/tribe/plainLanguageReport.js", import.meta.url), "utf8");
  const styles = await readFile(new URL("../src/creator-report.css", import.meta.url), "utf8");
  assert.match(source, /createTribePlainLanguageReport\(report\)/);
  assert.match(source, />Final report<\/button>/);
  assert.match(source, /FINAL CREATOR READOUT/);
  for (const section of ["CLIP FLOW", "PATTERN OVER TIME", "SPATIAL SPREAD", "CREATOR EXPERIMENT", "What this report can claim"]) {
    assert.ok(source.includes(section), `missing plain-language section: ${section}`);
  }
  assert.match(helper, /Numeric values are never emitted/);
  assert.match(helper, /No separate audio or text input was modeled/);
  assert.match(helper, /does not forecast publishing performance/);
  assert.equal(/displayAtlasLabel|parcel\.name|hemisphere/.test(helper), false);
  assert.match(styles, /\.plain-final-report\s*\{[\s\S]*overflow-y:\s*auto/);
  assert.match(styles, /\.prediction-stage--final/);
});

test("TRIBE submission is single-flight, stale-safe, restorable, and independently polled", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const client = await readFile(new URL("../src/tribe/client.js", import.meta.url), "utf8");
  assert.match(source, /tribeSubmitInFlightRef\.current \|\| tribeState\.status === "predicting"/);
  assert.match(source, /const runToken = \+\+tribeRunTokenRef\.current/);
  assert.ok(source.match(/tribeRunTokenRef\.current !== runToken/g)?.length >= 4);
  assert.match(source, /current\.operation === "predict" && current\.runToken === runToken/);
  assert.match(source, /getTribeWorkerStatus\(\{ signal: controller\.signal \}\)/);
  assert.match(source, /window\.history\.replaceState\(window\.history\.state, "", urlWithTribeResult/);
  assert.match(source, /formatInferenceElapsed\(tribeElapsedSeconds\)/);
  assert.equal(source.includes("Math.round(progress.value)}%"), false);
  for (const stage of ["requesting", "validating", "downloading", "ready"]) assert.match(client, new RegExp(`stage: "${stage}"`));
});
