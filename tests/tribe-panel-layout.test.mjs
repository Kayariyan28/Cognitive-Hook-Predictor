import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("TRIBE results expose an accessible viewport-filling focus mode", async () => {
  const source = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const panelStart = source.indexOf("function TribePanel");
  const panelEnd = source.indexOf("function MethodModal", panelStart);
  const panel = source.slice(panelStart, panelEnd);

  assert.match(panel, /const \[isExpanded, setIsExpanded\] = useState\(false\)/);
  assert.match(panel, /aria-label=\{isExpanded \? "Collapse TRIBE results" : "Expand TRIBE results to full screen"\}/);
  assert.match(panel, /aria-pressed=\{isExpanded\}/);
  assert.match(panel, /event\.key === "Escape"/);
  assert.match(panel, /tribe-panel--expanded/);
  assert.match(panel, /panelHost\?\.classList\.add\("tribe-panel-host--expanded"\)/);
  assert.match(panel, /panelRef\.current\.scrollTop = 0/);
  assert.match(panel, /\[isExpanded, view\]/);
});

test("expanded TRIBE reports use large type and independent scrolling without touching the brain renderer", async () => {
  const styles = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  const reportStyles = await readFile(new URL("../src/creator-report.css", import.meta.url), "utf8");
  const viewer = await readFile(new URL("../src/components/TribeBrainViewer.jsx", import.meta.url), "utf8");

  assert.match(styles, /\.tribe-panel--expanded\s*\{[\s\S]*position:\s*fixed[\s\S]*height:\s*calc\(100dvh - 24px\)[\s\S]*overflow:\s*clip/);
  assert.match(styles, /\.tribe-panel--expanded \.prediction-tabs button\s*\{[\s\S]*font-size:\s*clamp/);
  assert.match(reportStyles, /\.creator-report\s*\{[\s\S]*overflow-y:\s*auto/);
  assert.match(reportStyles, /\.frame-interval-list\s*\{[\s\S]*overflow-y:\s*auto/);
  assert.match(reportStyles, /\.plain-final-report\s*\{[\s\S]*overflow-y:\s*auto/);
  assert.match(reportStyles, /\.tribe-panel--expanded \.creator-report-head strong[\s\S]*font-size:\s*clamp/);
  assert.match(reportStyles, /\.tribe-panel--expanded \.frame-interval-thumbnail\s*\{[\s\S]*width:\s*118px/);
  assert.doesNotMatch(viewer, /tribe-panel--expanded|isExpanded|Expand TRIBE results/);
});
