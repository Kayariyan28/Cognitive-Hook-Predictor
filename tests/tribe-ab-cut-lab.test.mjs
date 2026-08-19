import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("Creator report seeks authoritative frames and opens the real A/B Cut Lab", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(source, /function TribeAbCutLab/);
  assert.match(source, /Open A\/B Cut Lab/);
  assert.match(source, /onSeek\(times\[frameIndex\]\);\s*setView\("map"\)/);
  assert.match(source, /urlWithTribeFrame\(window\.location\.href, frameIndex\)/);
  assert.match(source, /predictWithTribe\(variantFile, options\)/);
  assert.match(source, /getTribeWorkerStatus\(\{ signal: controller\.signal \}\)/);
  assert.match(source, /deriveTribeDescriptors\(variantPrediction, atlas\)/);
  assert.match(source, /compareTribeVariantReports\(primaryReport, variantReport\)/);
  assert.match(source, /tribeSubmitInFlightRef\.current = true;\s*setTribeBusyOwner\("variant"\)/);
  assert.match(source, />Close · analysis keeps running</);
  const runningActions = source.slice(source.indexOf("state.status === \"running\" ? <button", source.indexOf("function TribeAbCutLab")), source.indexOf("{(file || state.comparison)", source.indexOf("function TribeAbCutLab")));
  assert.doesNotMatch(runningActions, /abortRef|cancel|stop waiting/i);
  assert.doesNotMatch(source.slice(source.indexOf("function TribeAbCutLab"), source.indexOf("function TribePanel")), /predict(?:s|ed)? (?:the )?winner/i);
});

test("A/B Cut Lab has responsive, scroll-safe dialog styling", async () => {
  const styles = await readFile(new URL("../src/creator-report.css", import.meta.url), "utf8");
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const dialogRule = styles.match(/\.ab-cut-lab\s*\{([\s\S]*?)\}/)?.[1] ?? "";
  const scrollRule = styles.match(/\.ab-lab-scroll\s*\{([\s\S]*?)\}/)?.[1] ?? "";
  const actionRule = styles.match(/\.ab-lab-actions\s*\{([\s\S]*?)\}/)?.[1] ?? "";

  assert.match(dialogRule, /grid-template-rows:\s*auto minmax\(0, 1fr\) auto/);
  assert.match(dialogRule, /overflow:\s*hidden/);
  assert.match(dialogRule, /resize:\s*both/);
  assert.match(scrollRule, /min-height:\s*0/);
  assert.match(scrollRule, /overflow-y:\s*auto/);
  assert.match(actionRule, /border-top:/);
  assert.match(styles, /\.ab-cut-lab--maximized\s*\{[\s\S]*width:\s*calc\(100dvw - 16px\)[\s\S]*height:\s*calc\(100dvh - 16px\)/);
  assert.match(styles, /@media \(max-height: 760px\) and \(min-width: 761px\)/);
  assert.match(styles, /@media \(max-width: 760px\)[\s\S]*\.ab-cut-lab,[\s\S]*height:\s*calc\(100dvh - 16px\)/);
  assert.match(styles, /\.ab-lab-videos\s*\{[\s\S]*grid-template-columns:\s*1fr 1fr/);
  assert.match(styles, /@media \(max-width: 760px\)[\s\S]*\.ab-lab-videos\s*\{[\s\S]*grid-template-columns:\s*1fr/);
  assert.match(source, /className="ab-lab-scroll"/);
  assert.match(source, /aria-label=\{maximized \? "Restore A\/B Cut Lab window" : "Maximize A\/B Cut Lab window"\}/);
  assert.match(source, /onPointerDown=\{beginDialogDrag\}/);
  assert.match(source, /onDoubleClick=\{\(event\) => !event\.target\.closest\("button"\) && toggleMaximized\(\)\}/);
});

test("A/B Cut Lab text remains readable instead of using micro type", async () => {
  const styles = await readFile(new URL("../src/creator-report.css", import.meta.url), "utf8");

  for (const selector of [
    ".ab-lab-head p",
    ".ab-video-picker small,\n.ab-video-empty small",
    ".ab-lab-status p > span",
    ".ab-comparison-table [role=\"cell\"] strong",
    ".ab-comparison-table [role=\"cell\"] small",
    ".ab-lab-actions button",
  ]) {
    const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const rule = styles.match(new RegExp(`${escaped}\\s*\\{([\\s\\S]*?)\\}`))?.[1] ?? "";
    const sizeValue = rule.match(/font-size:\s*([^;]+)/)?.[1] ?? "";
    const pixelSizes = [...sizeValue.matchAll(/([\d.]+)px/g)].map((match) => Number(match[1]));
    assert.ok(pixelSizes.length > 0 && Math.min(...pixelSizes) >= 11, `${selector} must use at least 11px text, got ${sizeValue || "no size"}`);
  }
});
