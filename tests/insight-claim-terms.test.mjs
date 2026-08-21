import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import terms from "../src/insight/claim-terms.json" with { type: "json" };

/**
 * The third enforcement layer, on the frontend side.
 *
 * The prompt asks a model not to make outcome claims and the backend validator
 * refuses them, but neither guards the strings this app ships in its own UI.
 * This check reads the same exported vocabulary and fails when an insight
 * component hard-codes a forbidden claim without stating a limit.
 */

const INSIGHT_SOURCES = Object.freeze([
  "../src/insight/contract.js",
  "../src/insight/presentation.js",
  "../src/insight/client.js",
  "../src/components/InsightPanel.jsx",
]);

const STRING_LITERAL = /"((?:[^"\\\n]|\\.)*)"|'((?:[^'\\\n]|\\.)*)'|`((?:[^`\\]|\\.)*)`/g;

function stringLiterals(source) {
  const literals = [];
  for (const match of source.matchAll(STRING_LITERAL)) {
    const value = match[1] ?? match[2] ?? match[3] ?? "";
    if (value.trim()) literals.push(value);
  }
  return literals;
}

function wordPattern(term) {
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(?<!\\w)${escaped}(?!\\w)`, "i");
}

const FORBIDDEN = Object.freeze([
  ...terms.globalOutcomeTerms,
  ...terms.globalMentalStateTerms,
  ...terms.tribeScopedTerms,
]);

function statesALimit(literal) {
  const lowered = literal.toLowerCase();
  return (
    terms.wholeSentenceLimiters.some((limiter) => lowered.includes(limiter))
    || terms.proximateNegations.some((negation) => wordPattern(negation).test(lowered))
  );
}

test("the exported vocabulary is the shared backend list, not a copy", () => {
  assert.equal(terms.schemaVersion, "insight-claim-terms/1");
  for (const key of [
    "globalOutcomeTerms",
    "globalMentalStateTerms",
    "tribeScopedTerms",
    "wholeSentenceLimiters",
    "proximateNegations",
  ]) {
    assert.ok(Array.isArray(terms[key]) && terms[key].length > 0, `${key} is missing`);
  }
  assert.ok(terms.globalOutcomeTerms.includes("go viral"));
  assert.ok(terms.globalMentalStateTerms.includes("attention"));
  assert.ok(terms.tribeScopedTerms.includes("subconscious"));
  assert.equal(typeof terms.proximityCharacters, "number");
});

test("no insight component hard-codes a forbidden outcome or mental-state claim", async () => {
  const violations = [];
  for (const relative of INSIGHT_SOURCES) {
    const source = await readFile(new URL(relative, import.meta.url), "utf8");
    for (const literal of stringLiterals(source)) {
      if (statesALimit(literal)) continue;
      for (const term of FORBIDDEN) {
        if (wordPattern(term).test(literal)) {
          violations.push(`${relative}: “${literal}” contains “${term}”`);
        }
      }
    }
  }
  assert.deepEqual(violations, [], violations.join("\n"));
});

test("the check catches a forbidden claim when one is introduced", () => {
  const planted = "This opening will help the clip go viral.";
  assert.equal(statesALimit(planted), false);
  assert.ok(FORBIDDEN.some((term) => wordPattern(term).test(planted)));
});

test("the check does not flag a string that states its limit", () => {
  for (const legal of [
    "This is not a virality claim.",
    "TRIBE values are predicted average-subject cortical BOLD, not audience behavior.",
    "Hypotheses are untested heuristic statements.",
    "The hook window opens on a counter.",
  ]) {
    const flagged = !statesALimit(legal) && FORBIDDEN.some((term) => wordPattern(term).test(legal));
    assert.equal(flagged, false, `“${legal}” should not be flagged`);
  }
});
