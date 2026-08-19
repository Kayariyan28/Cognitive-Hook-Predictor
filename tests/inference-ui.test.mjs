import assert from "node:assert/strict";
import test from "node:test";
import {
  describeWorkerStage,
  formatInferenceElapsed,
  labelForTribeStage,
  urlWithTribeFrame,
  urlWithTribeResult,
  urlWithoutTribeResult,
} from "../src/tribe/inferenceUi.js";

test("TRIBE elapsed time is real duration text rather than an invented percentage", () => {
  assert.equal(formatInferenceElapsed(0), "0:00");
  assert.equal(formatInferenceElapsed(65.9), "1:05");
  assert.equal(formatInferenceElapsed(3_661), "1:01:01");
  assert.equal(formatInferenceElapsed(Number.NaN), "0:00");
  assert.equal(formatInferenceElapsed(-4), "0:00");
});

test("selected prediction frames persist as one-based URL values", () => {
  const resultId = "4d5b125f8b134566a63ad58515456b10";
  const next = new URL(urlWithTribeFrame(`http://localhost:4173/?tribeResult=${resultId}#results`, 4));
  assert.equal(next.searchParams.get("tribeFrame"), "5");
  assert.equal(next.hash, "#results");
  assert.throws(() => urlWithTribeFrame("http://localhost:4173/", 0), /persisted TRIBE result/);
  assert.throws(() => urlWithTribeFrame(`http://localhost:4173/?tribeResult=${resultId}`, -1), /zero-based/);
});

test("worker states map to scoped stage copy without completion claims", () => {
  assert.equal(labelForTribeStage("loading"), "Loading pinned model");
  assert.equal(labelForTribeStage("predicting"), "Running model inference");
  assert.equal(describeWorkerStage("predicting").stage, "predicting");
  assert.match(describeWorkerStage("ready").message, /waiting/i);
  assert.equal(describeWorkerStage("unknown").stage, "requesting");
  for (const state of ["not_loaded", "loading", "predicting", "ready", "unconfigured", "unknown"]) {
    assert.equal(describeWorkerStage(state).message.includes("%"), false);
  }
});

test("completed result URLs are restorable and stale frame state is cleared", () => {
  const resultId = "4d5b125f8b134566a63ad58515456b10";
  const next = new URL(urlWithTribeResult("http://localhost:4173/app?mode=video&tribeFrame=5#results", resultId));
  assert.equal(next.searchParams.get("mode"), "video");
  assert.equal(next.searchParams.get("tribeResult"), resultId);
  assert.equal(next.searchParams.has("tribeFrame"), false);
  assert.equal(next.hash, "#results");
  assert.throws(() => urlWithTribeResult("http://localhost:4173/", "not-a-result"), /valid TRIBE result ID/);
});

test("reset URLs remove only TRIBE restoration parameters", () => {
  const next = new URL(urlWithoutTribeResult("http://localhost:4173/app?mode=video&tribeResult=4d5b125f8b134566a63ad58515456b10&tribeFrame=4#results"));
  assert.equal(next.searchParams.get("mode"), "video");
  assert.equal(next.searchParams.has("tribeResult"), false);
  assert.equal(next.searchParams.has("tribeFrame"), false);
  assert.equal(next.hash, "#results");
});
