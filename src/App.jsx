import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowsInSimple,
  ArrowsOutSimple,
  Brain,
  ChartLineUp,
  Check,
  FileVideo,
  Flask,
  Info,
  List,
  Pause,
  Play,
  ShieldCheck,
  Sparkle,
  UploadSimple,
  VideoCamera,
  WaveSine,
  X,
} from "@phosphor-icons/react";
import { analyzeVideoFile } from "./analysis.js";
import { ForecastLab, INITIAL_CREATOR_INPUTS } from "./components/ForecastLab.jsx";
import { TribeBrainViewer } from "./components/TribeBrainViewer.jsx";
import { getForecastCapabilityStatus, forecastBackendConfiguration } from "./forecast/client.js";
import { deriveCreatorForecast } from "./forecast/creatorForecast.js";
import {
  buildForecastCreatorContext,
  createForecastIdempotencyKey,
  monitorForecastJob,
  submitForecastJob,
} from "./forecast/jobClient.js";
import { displayAtlasLabel, loadDestrieuxAtlas, summarizeFrameByParcel } from "./tribe/atlas.js";
import { getTribeWorkerStatus, loadTribeResult, predictWithTribe, tribeBackendConfiguration } from "./tribe/client.js";
import { deriveTribeDescriptors } from "./tribe/descriptors.js";
import { rankTribeFrameIntervals } from "./tribe/frameIntervals.js";
import { predictionFrameAtMediaTime } from "./tribe/frame.js";
import { describeWorkerStage, formatInferenceElapsed, labelForTribeStage, urlWithTribeFrame, urlWithTribeResult, urlWithoutTribeResult } from "./tribe/inferenceUi.js";
import { createTribePlainLanguageReport } from "./tribe/plainLanguageReport.js";
import { useVideoIntervalThumbnails } from "./tribe/useVideoIntervalThumbnails.js";
import { compareTribeVariantReports } from "./tribe/variantComparison.js";
import "./engagement.css";
import "./creator-report.css";
import "./forecast-lab.css";

const ANALYSIS_STEPS = [
  { value: 12, label: "Reading metadata" },
  { value: 32, label: "Sampling frames" },
  { value: 58, label: "Measuring edit dynamics" },
  { value: 78, label: "Reading audio energy" },
  { value: 94, label: "Computing modeled engagement" },
];

const PROXY_METRICS = [
  { key: "visualSignal", label: "Visual signal", description: "Measured luminance, contrast, color variation, and edge density." },
  { key: "editDynamics", label: "Edit dynamics", description: "Measured frame-to-frame motion and cut cadence." },
  { key: "audioSignal", label: "Audio signal", description: "Measured energy, dynamics, onset variation, and usable sound." },
  { key: "visualStability", label: "Visual stability", description: "Measured exposure variation, edge load, and frame-to-frame movement." },
];

function clamp(value, minimum = 0, maximum = 100) {
  const number = Number(value);
  return Math.min(maximum, Math.max(minimum, Number.isFinite(number) ? number : minimum));
}

function formatTime(seconds = 0) {
  const safe = Math.max(0, Math.round(seconds));
  return `${Math.floor(safe / 60)}:${String(safe % 60).padStart(2, "0")}`;
}

function formatPreciseTime(seconds = 0) {
  const safe = Math.max(0, Number.isFinite(Number(seconds)) ? Number(seconds) : 0);
  const minutes = Math.floor(safe / 60);
  const remaining = (safe % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${remaining}`;
}

function firstFinite(...values) {
  for (const value of values) {
    if (value === null || value === undefined || value === "") continue;
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }
  return null;
}

function formatDecimal(value, digits = 3, signed = false) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const prefix = signed && number > 0 ? "+" : "";
  return `${prefix}${number.toFixed(digits)}`;
}

function explicitPostInstant(localDateTime, utcOffset) {
  const local = typeof localDateTime === "string" ? localDateTime.trim() : "";
  const offset = typeof utcOffset === "string" ? utcOffset.trim().toUpperCase() : "";
  if (!local) return null;
  const offsetMatch = offset.match(/^([+-])(\d{2}):(\d{2})$/);
  const validOffset = offset === "Z" || Boolean(
    offsetMatch
      && Number(offsetMatch[2]) <= 14
      && Number(offsetMatch[3]) <= 59
      && (Number(offsetMatch[2]) < 14 || Number(offsetMatch[3]) === 0),
  );
  const localWithSeconds = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(local) ? `${local}:00` : local;
  return validOffset ? `${localWithSeconds}${offset}` : local;
}

function capabilityBranchExecutable(report, key) {
  const branch = report?.branches?.[key];
  return branch?.configured === true && ["ready", "implemented-client-runtime"].includes(branch.state);
}

function engagementScore(result) {
  const score = result?.scores?.engagementScore ?? result?.scores?.viralPotential;
  return Number.isFinite(score) ? clamp(score) : null;
}

function engagementDescriptor(score) {
  if (score >= 80) return "Very strong modeled signal";
  if (score >= 65) return "Strong modeled signal";
  if (score >= 50) return "Mixed modeled signal";
  if (score >= 35) return "Low modeled signal";
  return "Very low modeled signal";
}

function viralityOutlook(result) {
  const outlook = result?.virality;
  const score = firstFinite(outlook?.score, result?.scores?.viralPotential, result?.scores?.engagementScore);
  const evidenceCoverage = firstFinite(outlook?.evidenceCoverage, result?.scores?.coverage);
  const componentScore = (key, fallback) => (
    outlook?.components && Object.hasOwn(outlook.components, key)
      ? firstFinite(outlook.components[key]?.score)
      : firstFinite(fallback)
  );
  return {
    score: score === null ? null : clamp(score),
    evidenceCoverage: evidenceCoverage === null ? null : clamp(evidenceCoverage),
    interpretation: outlook?.interpretation || (score === null ? "Analyze a clip for a directional outlook" : score >= 75 ? "Strong measured content signals" : score >= 55 ? "Mixed-to-promising content signals" : score >= 35 ? "Mixed measured content signals" : "Limited measured content signals"),
    components: [
      { key: "opening", label: outlook?.components?.opening?.label || "Opening", score: componentScore("opening", result?.scores?.hookScore) },
      { key: "sustain", label: outlook?.components?.sustain?.label || "Sustain", score: componentScore("sustain", result?.scores?.holdRate) },
      { key: "ending", label: outlook?.components?.ending?.label || "Ending", score: componentScore("ending", result?.engagement?.components?.endingStrength) },
    ],
  };
}

function Logo() {
  return (
    <a className="brand" href="#predictor" aria-label="SignalFrame home">
      <span className="brand-icon"><Brain size={15} weight="fill" /></span>
      <span>SignalFrame.</span>
    </a>
  );
}

function Sparkline({ timeline = [], peakTime = 0 }) {
  const points = useMemo(() => {
    if (!timeline.length) return "";
    const maxTime = Math.max(1, ...timeline.map((item) => item.time));
    return timeline.map((item) => {
      const x = 5 + (item.time / maxTime) * 90;
      const y = 78 - clamp(item.score) * 0.58;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
  }, [timeline]);

  return (
    <div className="sparkline" aria-label={timeline.length ? `Modeled engagement signal peaks at ${formatTime(peakTime)}.` : "No engagement signal analyzed yet."}>
      <svg viewBox="0 0 100 84" preserveAspectRatio="none" aria-hidden="true">
        <path d="M5 20 H95 M5 49 H95 M5 78 H95" className="spark-grid" />
        {points ? <polyline points={points} className="spark-path" /> : <path d="M5 60 C25 60 30 42 48 51 S73 67 95 40" className="spark-placeholder" />}
      </svg>
      <div className="spark-months"><span>OPEN</span><span>MIDDLE</span><span>CLOSE</span></div>
    </div>
  );
}

function UploadCard({ file, status, progress, error, tribeRunning, tribeBusyOwner, tribeElapsedSeconds, tribeStage, onPick, onDrop, onAnalyze, onReset }) {
  const inputRef = useRef(null);
  const isAnalyzing = status === "analyzing";
  const actionLocked = isAnalyzing || tribeRunning;
  const actionLabel = isAnalyzing
    ? progress.label
    : tribeRunning
      ? tribeBusyOwner === "variant" ? "Variant B inference running" : `${labelForTribeStage(tribeStage)} · ${formatInferenceElapsed(tribeElapsedSeconds)}`
      : "Run analysis";

  return (
    <section className={`upload-card ${file ? "has-file" : ""}`} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); if (!actionLocked) onDrop(event); }} aria-labelledby="upload-title">
      <div className="card-index">01 / INPUT</div>
      <button className="upload-hit-area" type="button" onClick={() => inputRef.current?.click()} disabled={actionLocked}>
        <span className="upload-icon"><UploadSimple size={21} weight="bold" /></span>
        <span className="upload-copy">
          <strong id="upload-title">{file ? file.name : "Drop your short video"}</strong>
          <small>{file ? `${(file.size / 1_048_576).toFixed(1)} MB · ready` : "MP4, MOV, WebM or another browser-readable video"}</small>
        </span>
      </button>
      <div className="upload-actions">
        <button className="primary-action" type="button" disabled={!file || actionLocked} onClick={onAnalyze}>
          {actionLocked ? <span className="button-spinner" /> : <Sparkle size={17} weight="fill" />}
          <span>{actionLabel}</span>
        </button>
        {file && !actionLocked && <button className="icon-action" type="button" onClick={onReset} aria-label="Remove selected video"><X size={17} /></button>}
      </div>
      <input ref={inputRef} className="visually-hidden" type="file" disabled={actionLocked} accept="video/*,video/mp4,video/webm,video/ogg,video/quicktime,.mov,.mkv" onChange={(event) => onPick(event.target.files?.[0])} />
      {error && <p className="inline-error" role="alert">{error}</p>}
    </section>
  );
}

function ProxyCard({ result }) {
  const score = engagementScore(result);
  const coverage = result?.scores?.coverage;
  return (
    <section className="proxy-card engagement-card" aria-label="Modeled engagement score">
      <div className="engagement-card-head">
        <span>Modeled engagement</span>
        <small>{Number.isFinite(coverage) ? `${coverage}% coverage` : "Pre-publish"}</small>
      </div>
      <div className="engagement-score-row">
        <strong>{score === null ? "—" : Math.round(score)}<small>/100</small></strong>
        <WaveSine className="proxy-wave" size={38} weight="thin" aria-hidden="true" />
      </div>
      <p className="engagement-summary">{result ? engagementDescriptor(score) : "Analyze a clip to measure content signals"}</p>
      <div className="engagement-components" aria-label="Engagement model components">
        <span>Early <b>{result?.engagement?.components?.earlyAttention ?? result?.scores?.hookScore ?? "—"}</b></span>
        <span>Sustain <b>{result?.engagement?.components?.sustainedAttention ?? result?.scores?.holdRate ?? "—"}</b></span>
        <span>Pace <b>{result?.engagement?.components?.pacing ?? result?.scores?.editDynamics ?? "—"}</b></span>
      </div>
      <div className="proxy-disclaimer">Directional · not a probability or guarantee</div>
    </section>
  );
}

function ViralityOutlookCard({ result }) {
  const outlook = viralityOutlook(result);
  return (
    <section className="virality-outlook-card" aria-label="Local Virality outlook">
      <div className="virality-outlook-head">
        <div>
          <span>Virality outlook</span>
          <small>LOCAL CONTENT MODEL</small>
        </div>
        <ChartLineUp size={22} weight="bold" aria-hidden="true" />
      </div>
      <div className="virality-outlook-score">
        <strong>{outlook.score === null ? "—" : Math.round(outlook.score)}<small>/100</small></strong>
        <p>{outlook.interpretation}</p>
      </div>
      <div className="virality-outlook-components" aria-label="Virality outlook content-signal components">
        {outlook.components.map((component) => (
          <span key={component.key}>{component.label}<b>{component.score === null ? "—" : Math.round(component.score)}</b></span>
        ))}
      </div>
      <div className="virality-outlook-evidence">
        <span>{outlook.evidenceCoverage === null ? "Awaiting evidence" : `${Math.round(outlook.evidenceCoverage)}% evidence coverage`}</span>
        <small>Input coverage, not model confidence</small>
      </div>
      <p className="virality-outlook-boundary">Content-signal proxy · not TRIBE output · not a calibrated probability</p>
    </section>
  );
}

function ClipPanel({ videoUrl, result, videoRef, playing, onPlayingChange, onMediaTime, onMetadata, onOpenPreview }) {
  const metrics = result ? PROXY_METRICS : [];
  const score = engagementScore(result);

  return (
    <section className="clip-panel" aria-label="Video and analysis details">
      <button className="side-arrow side-arrow--left" type="button" aria-label="Previous result" disabled><ArrowRight size={17} /></button>
      <div className="clip-preview">
        {videoUrl ? (
          <>
            <video
              ref={videoRef}
              src={videoUrl}
              muted
              playsInline
              preload="metadata"
              onLoadedMetadata={(event) => {
                onMetadata(event.currentTarget.duration);
                onMediaTime(event.currentTarget.currentTime);
              }}
              onTimeUpdate={(event) => onMediaTime(event.currentTarget.currentTime)}
              onSeeked={(event) => onMediaTime(event.currentTarget.currentTime)}
              onPlay={() => onPlayingChange(true)}
              onPause={() => onPlayingChange(false)}
              onEnded={() => onPlayingChange(false)}
            />
            <button className="video-play" type="button" onClick={() => (playing ? videoRef.current?.pause() : videoRef.current?.play())} aria-label={playing ? "Pause video" : "Play video"}>
              {playing ? <Pause size={18} weight="fill" /> : <Play size={18} weight="fill" />}
            </button>
            <button className="video-expand" type="button" onClick={onOpenPreview} aria-label="Open video preview"><ArrowsOutSimple size={17} /></button>
          </>
        ) : (
          <div className="empty-preview"><VideoCamera size={32} /><span>Your clip appears here</span></div>
        )}
      </div>
      <div className="clip-insights">
        <div className="clip-heading">
          <span>{result ? "Your engagement read" : "Engagement report"}</span>
          <h2>{result ? engagementDescriptor(score) : "See what the clip is doing."}</h2>
          <p>{result ? `Peak modeled signal at ${formatTime(result.peakTime)}. Evidence coverage ${result.scores.coverage}%.` : "Local visual and audio measurements appear here. They stay separate from TRIBE v2."}</p>
        </div>
        <div className="metric-list">
          {metrics.length ? metrics.map((metric) => (
            <div className="metric-line" key={metric.key} title={metric.description}>
              <span>{metric.label}</span><div><i style={{ width: `${clamp(result.scores[metric.key])}%` }} /></div><strong>{result.scores[metric.key]}</strong>
            </div>
          )) : <div className="metric-empty"><FileVideo size={18} /><span>No proxy data yet</span></div>}
        </div>
      </div>
      <button className="side-arrow side-arrow--right" type="button" aria-label="Next result" disabled><ArrowRight size={17} /></button>
    </section>
  );
}

function rankTribeResponseMoments(report, limit = 4) {
  if (!Array.isArray(report?.frames) || !report.frames.length) return [];
  const framesByIndex = new Map(report.frames.map((frame) => [frame.index, frame]));
  const byMagnitude = [...report.frames].sort((left, right) => right.rms - left.rms || left.index - right.index);
  const byChange = report.frames
    .filter((frame) => Number.isFinite(frame.changeRms))
    .sort((left, right) => right.changeRms - left.changeRms || left.index - right.index);
  const moments = [];
  const seen = new Set();
  const add = (frame, label) => {
    if (!frame || seen.has(frame.index) || moments.length >= limit) return;
    seen.add(frame.index);
    moments.push({ frame, label });
  };

  const peakFrame = framesByIndex.get(report.clip?.peakMagnitude?.index);
  const changeFrame = framesByIndex.get(report.clip?.largestChange?.index);
  add(peakFrame, peakFrame?.index === changeFrame?.index ? "Peak magnitude + largest change" : "Peak response magnitude");
  add(changeFrame, "Largest pattern change");
  for (let index = 0; moments.length < limit && (index < byMagnitude.length || index < byChange.length); index += 1) {
    add(byMagnitude[index], "High response magnitude");
    add(byChange[index], "High pattern change");
  }
  return moments;
}

function TribeCreatorReport({ report, reportMessage, manifest, revision, onSeek, onOpenAbLab }) {
  if (!report) {
    return (
      <section className="creator-report creator-report--empty" role="tabpanel" aria-label="TRIBE Creator report">
        <Brain size={32} weight="thin" />
        <strong>No cortical descriptor report</strong>
        <span>{reportMessage}</span>
      </section>
    );
  }

  const moments = rankTribeResponseMoments(report);
  const largestChange = report.clip.largestChange;
  const phases = Array.isArray(report.phases) ? report.phases : [];
  const parcels = Array.isArray(report.parcels) ? report.parcels.slice(0, 5) : [];
  const usableVertices = report.clip.usableVertexCount || report.source?.usableVertexCount || 0;
  const hemodynamicOffset = manifest?.frames?.hemodynamicOffsetSeconds;
  const peakMagnitude = report.clip.peakMagnitude;
  const previewChangeTest = largestChange && largestChange.time > 1;

  return (
    <section className="creator-report" role="tabpanel" aria-label="TRIBE Creator report">
      <header className="creator-report-head">
        <div><span>CORTICAL DESCRIPTOR REPORT</span><strong>What the predicted tensor changes over time.</strong></div>
        <small>TRIBE OUTPUT ONLY</small>
      </header>

      <div className="creator-report-primary" aria-label="Primary cortical descriptors">
        <article>
          <span>Predicted response magnitude</span>
          <strong>{formatDecimal(report.clip.responseMagnitude)}</strong>
          <small>usable-cortex RMS · target z units</small>
        </article>
        <article>
          <span>Largest pattern change</span>
          <strong>{largestChange ? formatPreciseTime(largestChange.time) : "—"}</strong>
          <small>{largestChange ? `RMS Δ ${formatDecimal(largestChange.value)}` : "one frame required"}</small>
        </article>
        <article>
          <span>Pattern continuity</span>
          <strong>{formatDecimal(report.clip.continuity, 2, true)}</strong>
          <small>mean adjacent-frame cosine</small>
        </article>
        <article>
          <span>Spatial distribution</span>
          <strong>{Number.isFinite(report.clip.spatialDistribution) ? `${Math.round(report.clip.spatialDistribution)}%` : "—"}</strong>
          <small>surface spread index</small>
        </article>
      </div>

      <section className="creator-report-section" aria-labelledby="response-moments-title">
        <div className="creator-report-section-head">
          <div><span id="response-moments-title">Response moments</span><small>USABLE-CORTEX RMS / FRAME-TO-FRAME RMS Δ</small></div>
          <strong>{String(moments.length).padStart(2, "0")}</strong>
        </div>
        <div className="creator-moment-list">
          {moments.map(({ frame, label }, index) => (
            <button type="button" key={frame.index} onClick={() => onSeek(frame.time)} aria-label={`Seek to ${formatPreciseTime(frame.time)}, ${label}`}>
              <span className="creator-row-rank">{String(index + 1).padStart(2, "0")}</span>
              <span className="creator-row-copy"><strong>{label}</strong><small>{frame.topParcel ? `${frame.topParcel.hemisphere === "left" ? "L" : "R"} · ${displayAtlasLabel(frame.topParcel.name)}` : "Whole cortex"}</small></span>
              <span className="creator-row-values"><strong>{formatPreciseTime(frame.time)}</strong><small>RMS {formatDecimal(frame.rms)} · Δ {formatDecimal(frame.changeRms)}</small></span>
              <ArrowRight size={12} />
            </button>
          ))}
        </div>
      </section>

      <section className="creator-report-section" aria-labelledby="phase-descriptors-title">
        <div className="creator-report-section-head"><div><span id="phase-descriptors-title">Clip phases</span><small>EARLY / MIDDLE / LATE</small></div></div>
        <div className="creator-phase-grid">
          {phases.map((phase) => (
            <article key={phase.id}>
              <span>{String(phase.id).toUpperCase()}</span>
              <strong>{formatDecimal(phase.responseMagnitude)}</strong>
              <small>{formatPreciseTime(phase.startTime)}–{formatPreciseTime(phase.endTime)} · mean {formatDecimal(phase.signedMean, 3, true)}</small>
            </article>
          ))}
        </div>
      </section>

      <section className="creator-report-section" aria-labelledby="parcel-moments-title">
        <div className="creator-report-section-head">
          <div><span id="parcel-moments-title">Parcel moments</span><small>TOP DESCRIPTIVE RMS · 148 USABLE PARCELS</small></div>
        </div>
        <div className="creator-parcel-list">
          {parcels.map((parcel, index) => (
            <button type="button" key={parcel.key} onClick={() => onSeek(parcel.peakMagnitude.time)} aria-label={`Seek to ${displayAtlasLabel(parcel.name)} peak at ${formatPreciseTime(parcel.peakMagnitude.time)}`}>
              <span className="creator-row-rank">{String(index + 1).padStart(2, "0")}</span>
              <span className="creator-row-copy"><strong>{displayAtlasLabel(parcel.name)}</strong><small>{parcel.hemisphere.toUpperCase()} · {parcel.vertexCount} VERTICES</small></span>
              <span className="creator-row-values"><strong>RMS {formatDecimal(parcel.rms)}</strong><small>peak {formatPreciseTime(parcel.peakMagnitude.time)} · mean {formatDecimal(parcel.signedMean, 3, true)}</small></span>
              <ArrowRight size={12} />
            </button>
          ))}
        </div>
      </section>

      <section className="creator-report-section" aria-labelledby="creator-tests-title">
        <div className="creator-report-section-head">
          <div><span id="creator-tests-title">A/B tests to run</span><small>MOMENT-BASED EDIT EXPERIMENTS</small></div>
          <strong>{previewChangeTest ? "02" : "01"}</strong>
        </div>
        <div className="creator-test-list">
          {previewChangeTest && (
            <article>
              <span>01 · OPENING PREVIEW</span>
              <p>Original vs Variant B: preview the exact visual event at {formatPreciseTime(largestChange.time)} in the first ~1.5s.</p>
              <div><small>Measure actual stayed-to-watch / 3s hold.</small><button type="button" onClick={() => onSeek(largestChange.time)}>Seek moment <ArrowRight size={11} /></button></div>
            </article>
          )}
          {peakMagnitude && (
            <article>
              <span>{previewChangeTest ? "02" : "01"} · ENDING CUT</span>
              <p>Compare the current cut with a cut ending just after the response-magnitude peak at {formatPreciseTime(peakMagnitude.time)}.</p>
              <div><small>Measure actual completion / rewatch.</small><button type="button" onClick={() => onSeek(peakMagnitude.time)}>Seek moment <ArrowRight size={11} /></button></div>
            </article>
          )}
        </div>
        <p className="creator-test-boundary">TRIBE selects moments to test; it does not predict the winner or uplift.</p>
        <button className="creator-test-launch" type="button" onClick={onOpenAbLab}>
          <Flask size={14} weight="fill" /> Open A/B Cut Lab <ArrowRight size={12} />
        </button>
      </section>

      <div className="creator-report-source" aria-label="TRIBE descriptor evidence">
        <ShieldCheck size={16} weight="fill" />
        <p><strong>Evidence boundary</strong><span>Computed only from {report.frames.length} verified prediction frames across {usableVertices.toLocaleString()} usable cortical vertices. Describes model-target z patterns; it does not infer viewer behavior or publishing outcomes.</span></p>
      </div>
      <div className="creator-report-provenance">
        <span>148 usable Destrieux hemisphere parcels</span>
        <span>{Number.isFinite(hemodynamicOffset) ? `${hemodynamicOffset}s hemodynamic offset` : "published frame timing"}</span>
        <span>{revision ? `rev ${revision}` : report.schemaVersion}</span>
      </div>
    </section>
  );
}

function TribeFrameIntervals({ report, reportMessage, activeFrameIndex, isVisionOnly, onSeek, sourceVideoUrl, verifiedThumbnails }) {
  const [sortMode, setSortMode] = useState("magnitude");
  const intervalState = useMemo(() => {
    if (!report) return { intervals: [], error: "" };
    try {
      return { intervals: rankTribeFrameIntervals(report, sortMode), error: "" };
    } catch (intervalError) {
      return { intervals: [], error: intervalError?.message || "The verified intervals could not be ranked." };
    }
  }, [report, sortMode]);
  const thumbnailState = useVideoIntervalThumbnails(sourceVideoUrl, intervalState.intervals, verifiedThumbnails);

  if (!report || intervalState.error || !intervalState.intervals.length) {
    return (
      <section className="frame-intervals frame-intervals--empty" role="tabpanel" aria-label="TRIBE frame intervals">
        <WaveSine size={32} weight="thin" />
        <strong>{intervalState.error ? "Interval report unavailable" : "No frame intervals"}</strong>
        <span>{intervalState.error || reportMessage || "Run or restore a verified TRIBE prediction to rank its intervals."}</span>
      </section>
    );
  }

  const intervals = intervalState.intervals;
  return (
    <section className="frame-intervals" role="tabpanel" aria-label="TRIBE frame intervals">
      <header className="frame-intervals-head">
        <div><span>TRIBE FRAME INTERVALS</span><strong>Every predicted interval, ranked.</strong></div>
        <small>{String(intervals.length).padStart(2, "0")} INTERVALS</small>
      </header>

      <div className="frame-interval-sort" role="group" aria-label="Rank frame intervals by">
        <button type="button" aria-pressed={sortMode === "magnitude"} onClick={() => setSortMode("magnitude")}>Response magnitude</button>
        <button type="button" aria-pressed={sortMode === "change"} onClick={() => setSortMode("change")}>Pattern change</button>
      </div>

      <div className="frame-interval-list" aria-label={`All intervals ranked by ${sortMode === "magnitude" ? "response magnitude" : "pattern change"}`}>
        {intervals.map((interval) => {
          const isActive = interval.index === activeFrameIndex;
          const percentile = Number.isFinite(interval.withinClipPercentile) ? `P${Math.round(interval.withinClipPercentile)}` : "—";
          const thumbnail = thumbnailState.byIndex[interval.index];
          const thumbnailLabel = thumbnailState.sourceKind === "verified-server" ? "VERIFIED SOURCE STILL" : "SESSION SOURCE STILL";
          return (
            <button
              type="button"
              className={isActive ? "is-active" : ""}
              key={interval.index}
              aria-current={isActive ? "true" : undefined}
              aria-label={`Seek to TRIBE interval ${interval.index + 1}, ${formatPreciseTime(interval.startTime)} to ${formatPreciseTime(interval.endTime)}`}
              onClick={() => onSeek(interval.startTime)}
            >
              <span className="frame-interval-rank">{interval.rank === null ? "—" : `#${String(interval.rank).padStart(2, "0")}`}</span>
              <span className={`frame-interval-thumbnail frame-interval-thumbnail--${thumbnail?.status || "unavailable"}`}>
                {thumbnail?.status === "ready" ? (
                  <>
                    <img src={thumbnail.url} alt={`Source-video still requested at ${formatPreciseTime(interval.startTime)} for TRIBE interval ${interval.index + 1}`} />
                    <small>{thumbnailLabel}</small>
                  </>
                ) : (
                  <>
                    <FileVideo size={15} weight="thin" aria-hidden="true" />
                    <small>{thumbnail?.status === "loading" ? "LOADING SOURCE" : thumbnail?.status === "error" ? "VERIFY FAILED" : thumbnail?.reason === "outside-source-video" || thumbnail?.reason === "outside-source-duration" ? "NO SOURCE FRAME AT THIS INTERVAL" : "SOURCE STILL UNAVAILABLE"}</small>
                  </>
                )}
              </span>
              <span className="frame-interval-main">
                <span className="frame-interval-time"><strong>{formatPreciseTime(interval.startTime)}–{formatPreciseTime(interval.endTime)}</strong><small>{percentile} · relative to this result</small></span>
                <span className="frame-interval-values"><b>RMS {formatDecimal(interval.responseMagnitude)}</b><b>Δ {formatDecimal(interval.changeRms)}</b></span>
                <span className="frame-interval-parcel"><small>Largest parcel RMS</small><strong>{interval.topParcel ? `${interval.topParcel.hemisphere === "left" ? "L" : "R"} · ${displayAtlasLabel(interval.topParcel.name)}` : "—"}</strong></span>
              </span>
              <span className="frame-interval-badges" aria-label={interval.labels.join(", ") || undefined}>
                {interval.badges.peakMagnitude && <i>PEAK RESPONSE</i>}
                {interval.badges.largestChange && <i>LARGEST CHANGE</i>}
              </span>
              <ArrowRight size={12} aria-hidden="true" />
            </button>
          );
        })}
      </div>

      <div className="frame-interval-boundary">
        <Info size={15} weight="fill" />
        <p><strong>Ranked TRIBE-predicted cortical response—not measured attention, watch time, interaction, or virality.</strong><span>Stills are real source-video frames requested at each published interval start and are visual context, not brain evidence. Source-video frames are not TRIBE intervals. Percentiles are relative to this TRIBE result only. “Largest parcel RMS” is descriptive, not region importance. Timing is shown without subtracting 5s.{isVisionOnly ? " This run excludes audio and text." : ""}</span></p>
      </div>
    </section>
  );
}

function TribePlainLanguageFinalReport({ report, reportMessage }) {
  const plainState = useMemo(() => {
    if (!report) return { report: null, error: "" };
    try {
      return { report: createTribePlainLanguageReport(report), error: "" };
    } catch (plainError) {
      return { report: null, error: plainError?.message || "The creator-language summary could not be built." };
    }
  }, [report]);

  if (!plainState.report) {
    return (
      <section className="plain-final-report plain-final-report--empty" role="tabpanel" aria-label="Plain-language final report">
        <List size={32} weight="thin" />
        <strong>{plainState.error ? "Final report unavailable" : "No final report yet"}</strong>
        <span>{plainState.error || reportMessage || "Run or restore a verified TRIBE prediction to build the creator summary."}</span>
      </section>
    );
  }

  const plain = plainState.report;
  return (
    <section className="plain-final-report" role="tabpanel" aria-label="Plain-language final report">
      <header className="plain-final-report-head">
        <div><span>FINAL CREATOR READOUT</span><strong>{plain.title}</strong><p>{plain.introduction}</p></div>
        <small>PLAIN LANGUAGE</small>
      </header>

      <div className="plain-final-report-grid">
        <article><span>CLIP FLOW</span><strong>{plain.timing.title}</strong><p>{plain.timing.body}</p></article>
        <article><span>PATTERN OVER TIME</span><strong>{plain.dynamics.title}</strong><p>{plain.dynamics.body}</p></article>
        <article><span>SPATIAL SPREAD</span><strong>{plain.spread.title}</strong><p>{plain.spread.body}</p></article>
      </div>

      <section className="plain-final-report-test" aria-labelledby="plain-report-test-title">
        <span>CREATOR EXPERIMENT</span>
        <strong id="plain-report-test-title">{plain.experiment.title}</strong>
        <p>{plain.experiment.body}</p>
      </section>

      <section className="plain-final-report-modality" aria-labelledby="plain-report-modality-title">
        <VideoCamera size={17} weight="fill" aria-hidden="true" />
        <p><strong id="plain-report-modality-title">{plain.modality.title}</strong><span>{plain.modality.body}</span></p>
      </section>

      <div className="plain-final-report-boundary">
        <ShieldCheck size={17} weight="fill" aria-hidden="true" />
        <p><strong>What this report can claim</strong><span>{plain.boundary}</span></p>
      </div>
    </section>
  );
}

function TribeAbCutLab({ open, onClose, primaryReport, primaryPrediction, atlas, sourceVideoUrl, busyOwner, onPredictVariant }) {
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const dialogRef = useRef(null);
  const dragRef = useRef(null);
  const primaryResultRef = useRef(primaryReport?.source?.resultId ?? null);
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [state, setState] = useState({ status: "idle", stage: "idle", message: "Choose a real Variant B video.", startedAt: null, comparison: null });
  const [elapsed, setElapsed] = useState(0);
  const [maximized, setMaximized] = useState(false);
  const [dialogOffset, setDialogOffset] = useState({ x: 0, y: 0 });

  useEffect(() => () => {
    abortRef.current?.abort();
    if (videoUrl) URL.revokeObjectURL(videoUrl);
  }, [videoUrl]);

  useEffect(() => {
    if (state.status !== "running" || !Number.isFinite(state.startedAt)) {
      setElapsed(0);
      return undefined;
    }
    const tick = () => setElapsed(Math.max(0, Math.floor((Date.now() - state.startedAt) / 1000)));
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [state.startedAt, state.status]);

  useEffect(() => {
    if (state.status !== "running") return undefined;
    const controller = new AbortController();
    let stopped = false;
    let timer;
    const workerStages = new Set(["requesting", "not_loaded", "loading", "predicting", "ready"]);
    const poll = async () => {
      try {
        const worker = await getTribeWorkerStatus({ signal: controller.signal });
        if (stopped) return;
        const update = describeWorkerStage(worker.state);
        setState((current) => current.status === "running" && workerStages.has(current.stage)
          ? { ...current, stage: update.stage, message: update.message }
          : current);
      } catch (pollError) {
        if (pollError?.name === "AbortError") return;
        // A status-poll failure never replaces or aborts the authoritative POST.
      } finally {
        if (!stopped) timer = window.setTimeout(poll, 2000);
      }
    };
    timer = window.setTimeout(poll, 1200);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [state.status]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl("");
    setFile(null);
    setElapsed(0);
    setState({ status: "idle", stage: "idle", message: "Choose a real Variant B video.", startedAt: null, comparison: null });
  }, [videoUrl]);

  useEffect(() => {
    const nextResultId = primaryReport?.source?.resultId ?? null;
    if (primaryResultRef.current !== nextResultId) {
      primaryResultRef.current = nextResultId;
      reset();
    }
  }, [primaryReport?.source?.resultId, reset]);

  const closeLab = useCallback(() => {
    onClose();
  }, [onClose]);

  useEffect(() => {
    if (!open) return undefined;
    const closeOnEscape = (event) => event.key === "Escape" && closeLab();
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [closeLab, open]);

  useEffect(() => {
    if (open) return;
    dragRef.current = null;
    setMaximized(false);
    setDialogOffset({ x: 0, y: 0 });
  }, [open]);

  const toggleMaximized = useCallback(() => {
    dragRef.current = null;
    setDialogOffset({ x: 0, y: 0 });
    setMaximized((current) => !current);
  }, []);

  const beginDialogDrag = useCallback((event) => {
    if (maximized || window.matchMedia("(max-width: 760px)").matches || event.button !== 0 || event.target.closest("button, a, input, video")) return;
    const dialog = dialogRef.current;
    if (!dialog) return;
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startOffset: dialogOffset,
      bounds: dialog.getBoundingClientRect(),
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [dialogOffset, maximized]);

  const moveDialog = useCallback((event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const gutter = 8;
    const deltaX = Math.min(window.innerWidth - gutter - drag.bounds.right, Math.max(gutter - drag.bounds.left, event.clientX - drag.startX));
    const deltaY = Math.min(window.innerHeight - gutter - drag.bounds.bottom, Math.max(gutter - drag.bounds.top, event.clientY - drag.startY));
    setDialogOffset({ x: drag.startOffset.x + deltaX, y: drag.startOffset.y + deltaY });
  }, []);

  const endDialogDrag = useCallback((event) => {
    if (dragRef.current?.pointerId !== event.pointerId) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }, []);

  const chooseFile = (nextFile) => {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("video/") && !/\.(mp4|mov|mkv|webm|ogg|avi)$/i.test(nextFile.name)) {
      setState({ status: "error", stage: "error", message: "Choose a browser-readable video file.", startedAt: null, comparison: null });
      return;
    }
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setFile(nextFile);
    setVideoUrl(URL.createObjectURL(nextFile));
    setState({ status: "ready", stage: "ready", message: "Variant B is ready for the same pinned TRIBE pipeline.", startedAt: null, comparison: null });
  };

  const run = async () => {
    if (!file || !primaryReport || !primaryPrediction || !atlas || state.status === "running" || busyOwner) return;
    const controller = new AbortController();
    abortRef.current = controller;
    const startedAt = Date.now();
    setState({ status: "running", stage: "requesting", message: "Submitting Variant B to the pinned worker.", startedAt, comparison: null });
    try {
      const variantPrediction = await onPredictVariant(file, {
        signal: controller.signal,
        onProgress: (update) => setState((current) => current.status === "running" ? { ...current, stage: update.stage, message: update.message } : current),
      });
      const variantReport = deriveTribeDescriptors(variantPrediction, atlas);
      const comparison = compareTribeVariantReports(primaryReport, variantReport);
      setState({ status: "complete", stage: "complete", message: "Both tensors passed the same compatibility checks.", startedAt: null, comparison });
    } catch (variantError) {
      if (variantError?.name === "AbortError") {
        return;
      } else {
        setState({ status: "error", stage: "error", message: variantError?.message || "Variant B could not be compared.", startedAt: null, comparison: null });
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  };

  if (!open) return null;
  const lockedByMain = Boolean(busyOwner && busyOwner !== "variant");
  const canRun = Boolean(file && primaryReport && primaryPrediction && atlas && state.status !== "running" && !busyOwner);
  const stageLabel = state.status === "running" ? labelForTribeStage(state.stage) : state.status === "complete" ? "Comparison ready" : "Variant B";

  return (
    <div className="modal-backdrop ab-lab-backdrop" onMouseDown={(event) => event.target === event.currentTarget && closeLab()}>
      <section
        ref={dialogRef}
        className={`ab-cut-lab${maximized ? " ab-cut-lab--maximized" : ""}`}
        style={maximized ? undefined : { transform: `translate3d(${dialogOffset.x}px, ${dialogOffset.y}px, 0)` }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="ab-lab-title"
        aria-describedby="ab-lab-description"
      >
        <header
          className="ab-lab-head"
          title={maximized ? undefined : "Drag to move · resize from the lower-right corner"}
          onDoubleClick={(event) => !event.target.closest("button") && toggleMaximized()}
          onPointerDown={beginDialogDrag}
          onPointerMove={moveDialog}
          onPointerUp={endDialogDrag}
          onPointerCancel={endDialogDrag}
        >
          <div className="ab-lab-title"><span><Flask size={16} weight="fill" /> DESCRIPTIVE CUT COMPARISON</span><h2 id="ab-lab-title">A/B Cut Lab</h2><p id="ab-lab-description">Run Variant B through the same pinned TRIBE model and compare its cortical descriptors with the original.</p></div>
          <div className="ab-lab-window-actions">
            <button className="ab-lab-maximize" type="button" onClick={toggleMaximized} aria-pressed={maximized} aria-label={maximized ? "Restore A/B Cut Lab window" : "Maximize A/B Cut Lab window"}><ArrowsOutSimple size={18} /><span>{maximized ? "Restore" : "Maximize"}</span></button>
            <button className="ab-lab-close" type="button" onClick={closeLab} aria-label={state.status === "running" ? "Close A/B Cut Lab; analysis keeps running" : "Close A/B Cut Lab"}><X size={20} /></button>
          </div>
        </header>

        <div className="ab-lab-scroll">
          <div className="ab-lab-videos">
            <article><span>ORIGINAL · A</span>{sourceVideoUrl ? <video src={sourceVideoUrl} controls muted playsInline preload="metadata" /> : <div className="ab-video-empty"><VideoCamera size={28} /><small>Stored tensor loaded; the original source video is not available in this browser session.</small></div>}<strong>{primaryReport ? "Verified baseline tensor" : "Run the original first"}</strong></article>
            <article><span>VARIANT · B</span>{videoUrl ? <video src={videoUrl} controls muted playsInline preload="metadata" /> : <button className="ab-video-picker" type="button" onClick={() => inputRef.current?.click()} disabled={state.status === "running"}><UploadSimple size={28} /><small>Choose the edited video cut</small></button>}<strong>{file?.name || "No variant selected"}</strong></article>
          </div>
          <input ref={inputRef} className="visually-hidden" type="file" accept="video/*,video/mp4,video/webm,video/ogg,video/quicktime,.mov,.mkv" onChange={(event) => chooseFile(event.target.files?.[0])} />

          <div className={`ab-lab-status ab-lab-status--${state.status}`} role={state.status === "error" ? "alert" : "status"} aria-live="polite">
            {state.status === "running" ? <span className="button-spinner" /> : <ShieldCheck size={18} weight="fill" />}
            <p><strong>{stageLabel}{state.status === "running" ? ` · ${formatInferenceElapsed(elapsed)}` : ""}</strong><span>{lockedByMain ? "The original prediction request is still using the single worker submission lane." : state.message}</span></p>
          </div>

          {state.comparison && (
            <section className="ab-comparison" aria-labelledby="ab-comparison-title">
              <div className="ab-comparison-head"><div><span>LIKE-FOR-LIKE TRIBE OUTPUT</span><strong id="ab-comparison-title">Descriptor differences</strong></div><small>{state.comparison.compatibility.inferenceMode.toUpperCase()} · SAME REVISION</small></div>
              <div className="ab-comparison-table" role="table" aria-label="TRIBE descriptor comparison">
                <div role="row" className="ab-comparison-labels"><span role="columnheader">Descriptor</span><span role="columnheader">Original</span><span role="columnheader">Variant B</span><span role="columnheader">Direction</span></div>
                {state.comparison.rows.map((item) => (
                  <div role="row" key={item.id}><span role="cell"><strong>{item.label}</strong><small>{item.description}</small></span><b role="cell">{formatDecimal(item.primary)}</b><b role="cell">{formatDecimal(item.variant)}</b><em role="cell" className={`ab-direction ab-direction--${item.direction}`}>{item.directionLabel}</em></div>
                ))}
              </div>
              <p className="ab-comparison-boundary"><Info size={17} weight="fill" /><span>{state.comparison.boundary}</span></p>
            </section>
          )}

          {!state.comparison && <p className="ab-lab-boundary">This lab compares predicted cortical response patterns only. It does not score either cut as better.</p>}
        </div>
        <div className="ab-lab-actions">
          <button type="button" className="ab-secondary" onClick={() => inputRef.current?.click()} disabled={state.status === "running"}>{file ? "Replace Variant B" : "Choose Variant B"}</button>
          {state.status === "running" ? <button type="button" className="ab-secondary" onClick={closeLab}>Close · analysis keeps running</button> : <button type="button" className="ab-primary" disabled={!canRun} onClick={run}>Run same TRIBE analysis <ArrowRight size={12} /></button>}
          {(file || state.comparison) && state.status !== "running" && <button type="button" className="ab-reset" onClick={reset}>Reset</button>}
        </div>
      </section>
    </div>
  );
}

function TribePanel({ tribeState, tribeElapsedSeconds, mediaTime, mediaDuration, activeFrame, playing, sourceVideoUrl, busyOwner, onPredictVariant, onFrameChange, onSeek, onTogglePlayback, onOpenMethod }) {
  const [view, setView] = useState("map");
  const [isExpanded, setIsExpanded] = useState(false);
  const [abLabOpen, setAbLabOpen] = useState(false);
  const panelRef = useRef(null);
  const [atlasState, setAtlasState] = useState({ status: "loading", atlas: null, message: "Loading Destrieux fsaverage5 atlas" });
  const [focusedParcel, setFocusedParcel] = useState(null);
  const prediction = tribeState.prediction;
  const manifest = prediction?.manifest;
  const frameCount = manifest?.frames.shape[0];
  const revision = manifest?.model.revision?.slice(0, 8);
  const isVisionOnly = manifest?.inferenceMode === "vision-only";
  const modeLabel = isVisionOnly ? "TRIBE V2 · VIDEO-ONLY CORTICAL BOLD" : "TRIBE V2 · PREDICTED CORTICAL BOLD";
  const inputLabel = isVisionOnly ? "VIDEO" : manifest ? "A + V + T" : "—";
  const stageLabel = labelForTribeStage(tribeState.stage);
  const elapsedLabel = formatInferenceElapsed(tribeElapsedSeconds);
  const statusTitle = tribeState.status === "ready" ? "Predicted cortical BOLD" : tribeState.status === "predicting" ? stageLabel : "No prediction loaded";
  const stateLabel = tribeState.status === "ready"
    ? (isVisionOnly ? "CONTRACT OK · VIDEO" : "CONTRACT OK")
    : tribeState.status === "predicting"
      ? `RUNNING · ${elapsedLabel}`
      : tribeState.status.toUpperCase();
  const statusMessage = tribeState.status === "predicting"
    ? `${stageLabel} · ${elapsedLabel} elapsed. ${tribeState.message}`
    : tribeState.message;
  const timelineEnd = useMemo(() => {
    if (!manifest) return Number.isFinite(mediaDuration) ? mediaDuration : 0;
    const times = manifest.frames.timesSeconds;
    const durations = manifest.frames.durationsSeconds;
    return Math.max(Number.isFinite(mediaDuration) ? mediaDuration : 0, times.at(-1) + durations.at(-1));
  }, [manifest, mediaDuration]);
  const currentPrediction = useMemo(() => (
    tribeState.status === "ready" && prediction
      ? predictionFrameAtMediaTime(prediction.values, prediction.manifest, mediaTime)
      : { index: null, frame: null }
  ), [mediaTime, prediction, tribeState.status]);
  const liveParcels = useMemo(() => (
    currentPrediction.frame && atlasState.atlas
      ? summarizeFrameByParcel(currentPrediction.frame, atlasState.atlas, 7)
      : []
  ), [atlasState.atlas, currentPrediction.frame]);
  const descriptorState = useMemo(() => {
    if (tribeState.status !== "ready" || !prediction) {
      return { report: null, message: "Run or restore a verified TRIBE prediction to build cortical descriptors." };
    }
    if (atlasState.status === "loading") return { report: null, message: "Loading the verified Destrieux atlas." };
    if (atlasState.status === "error" || !atlasState.atlas) return { report: null, message: atlasState.message || "The verified cortical atlas is unavailable." };
    try {
      return { report: deriveTribeDescriptors(prediction, atlasState.atlas), message: "" };
    } catch (descriptorError) {
      return { report: null, message: descriptorError?.message || "The verified tensor could not be summarized." };
    }
  }, [atlasState, prediction, tribeState.status]);

  useEffect(() => {
    const controller = new AbortController();
    loadDestrieuxAtlas(controller.signal)
      .then((atlas) => setAtlasState({ status: "ready", atlas, message: "Verified Destrieux atlas · 148 usable hemispheric parcels" }))
      .catch((atlasError) => {
        if (atlasError?.name !== "AbortError") setAtlasState({ status: "error", atlas: null, message: atlasError?.message || "Cortical atlas unavailable" });
      });
    return () => controller.abort();
  }, []);

  useEffect(() => setFocusedParcel(null), [prediction]);

  useEffect(() => {
    if (!isExpanded) return undefined;
    const previousBodyOverflow = document.body.style.overflow;
    const panelHost = panelRef.current?.closest(".reveal--four");
    document.body.style.overflow = "hidden";
    panelHost?.classList.add("tribe-panel-host--expanded");
    return () => {
      document.body.style.overflow = previousBodyOverflow;
      panelHost?.classList.remove("tribe-panel-host--expanded");
    };
  }, [isExpanded]);

  useEffect(() => {
    if (!isExpanded) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape" && !abLabOpen) setIsExpanded(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [abLabOpen, isExpanded]);

  useEffect(() => {
    if (!isExpanded || !panelRef.current) return;
    panelRef.current.scrollTop = 0;
    panelRef.current.scrollLeft = 0;
  }, [isExpanded, view]);

  const chooseParcel = (parcel) => {
    setFocusedParcel(parcel);
    setView("map");
  };
  const seekReportMoment = useCallback((requestedTime) => {
    if (!manifest || !prediction) return;
    const times = manifest.frames.timesSeconds;
    let frameIndex = 0;
    let distance = Infinity;
    times.forEach((time, index) => {
      const nextDistance = Math.abs(time - requestedTime);
      if (nextDistance < distance) {
        frameIndex = index;
        distance = nextDistance;
      }
    });
    onSeek(times[frameIndex]);
    setView("map");
    try {
      window.history.replaceState(window.history.state, "", urlWithTribeFrame(window.location.href, frameIndex));
    } catch {
      // Frame persistence is advisory when the current result is not URL-restorable.
    }
  }, [manifest, onSeek, prediction]);
  const isExpandedDataView = view === "report" || view === "intervals" || view === "final";

  return (
    <aside ref={panelRef} className={`tribe-panel tribe-panel--${view}${isExpanded ? " tribe-panel--expanded" : ""}`} aria-labelledby="tribe-heading">
      <div className="tribe-panel-top">
        <div><span>{modeLabel}</span><h2 id="tribe-heading">{statusTitle}</h2></div>
        <div className="tribe-panel-actions">
          <span className={`tribe-state tribe-state--${tribeState.status}`}>{stateLabel}</span>
          <button
            className="tribe-expand-button"
            type="button"
            aria-label={isExpanded ? "Collapse TRIBE results" : "Expand TRIBE results to full screen"}
            aria-pressed={isExpanded}
            title={isExpanded ? "Collapse results (Esc)" : "Expand results"}
            onClick={() => setIsExpanded((current) => !current)}
          >
            {isExpanded ? <ArrowsInSimple size={15} weight="bold" /> : <ArrowsOutSimple size={15} weight="bold" />}
            <span>{isExpanded ? "Collapse" : "Expand"}</span>
          </button>
        </div>
      </div>
      <div className="prediction-tabs" role="tablist" aria-label="TRIBE result views">
        <button type="button" role="tab" aria-selected={view === "map"} onClick={() => setView("map")}>Prediction map</button>
        <button type="button" role="tab" aria-selected={view === "regions"} onClick={() => setView("regions")}>Atlas regions <span>{liveParcels.length || "—"}</span></button>
        <button type="button" role="tab" aria-selected={view === "report"} onClick={() => setView("report")}>Creator report <span>{descriptorState.report?.frames?.length || "—"}</span></button>
        <button type="button" role="tab" aria-selected={view === "intervals"} onClick={() => setView("intervals")}>Frame intervals <span>{descriptorState.report?.frames?.length || "—"}</span></button>
        <button type="button" role="tab" aria-selected={view === "final"} onClick={() => setView("final")}>Final report</button>
      </div>
      <div className={`prediction-stage ${isExpandedDataView ? `prediction-stage--${view}` : ""}`}>
        {view === "map" ? (
          <>
            <TribeBrainViewer prediction={prediction} mediaTime={mediaTime} status={tribeState.status} atlas={atlasState.atlas} focus={focusedParcel} onFrameChange={onFrameChange} />
            {focusedParcel && (
              <button className="parcel-focus" type="button" onClick={() => setFocusedParcel(null)} title={focusedParcel.name}>
                <span>{focusedParcel.hemisphere === "left" ? "L" : "R"}</span>{displayAtlasLabel(focusedParcel.name)}<X size={12} />
              </button>
            )}
          </>
        ) : view === "regions" ? (
          <section className="atlas-regions" role="tabpanel" aria-label="Highest predicted Destrieux parcels">
            <div className="atlas-regions-head"><span>HIGHEST PREDICTED PARCELS</span><small>{currentPrediction.frame ? `FRAME ${currentPrediction.index + 1}` : "NO FRAME"}</small></div>
            {liveParcels.length ? liveParcels.map((parcel, index) => (
              <button type="button" key={parcel.key} onClick={() => chooseParcel(parcel)} title={`${parcel.name} · ${parcel.vertexCount} vertices`}>
                <span className="parcel-rank">{String(index + 1).padStart(2, "0")}</span>
                <span className="parcel-name"><strong>{displayAtlasLabel(parcel.name)}</strong><small>{parcel.hemisphere.toUpperCase()} · {parcel.vertexCount} VERTICES</small></span>
                <span className="parcel-value">{parcel.mean >= 0 ? "+" : ""}{parcel.mean.toFixed(3)}</span>
              </button>
            )) : (
              <div className="atlas-empty">
                <Brain size={30} weight="thin" />
                <strong>{tribeState.status === "ready" ? "No frame at this time" : "No verified region data"}</strong>
                <span>{atlasState.status === "error" ? atlasState.message : "Region values appear only after a real TRIBE tensor is loaded."}</span>
              </div>
            )}
            {liveParcels.length > 0 && <p>Destrieux 2010 parcels · mean model-target z-score · select a row to focus its exact vertices.</p>}
          </section>
        ) : view === "report" ? (
          <TribeCreatorReport report={descriptorState.report} reportMessage={descriptorState.message} manifest={manifest} revision={revision} onSeek={seekReportMoment} onOpenAbLab={() => setAbLabOpen(true)} />
        ) : view === "intervals" ? (
          <TribeFrameIntervals report={descriptorState.report} reportMessage={descriptorState.message} activeFrameIndex={currentPrediction.index} isVisionOnly={isVisionOnly} onSeek={seekReportMoment} sourceVideoUrl={sourceVideoUrl} verifiedThumbnails={manifest?.thumbnails} />
        ) : (
          <TribePlainLanguageFinalReport report={descriptorState.report} reportMessage={descriptorState.message} />
        )}
      </div>
      {!isExpandedDataView && (
        <>
          <div className="neural-readout">
            <div><span>VIDEO TIME</span><strong>{formatTime(mediaTime)}</strong></div>
            <div><span>PREDICTION FRAME</span><strong>{activeFrame === null ? "—" : `${activeFrame + 1}/${frameCount}`}</strong></div>
          </div>
          <div className="prediction-transport">
            <button type="button" onClick={onTogglePlayback} disabled={!prediction} aria-label={playing ? "Pause prediction replay" : "Play prediction replay"}>
              {playing ? <Pause size={13} weight="fill" /> : <Play size={13} weight="fill" />}
            </button>
            <input
              type="range"
              aria-label="Prediction timeline"
              min="0"
              max={Math.max(0.01, timelineEnd)}
              step="0.01"
              value={Math.min(Math.max(mediaTime, 0), Math.max(0.01, timelineEnd))}
              disabled={!prediction}
              onChange={(event) => onSeek(Number(event.target.value))}
            />
            <span>{prediction ? "MODEL TIMESTAMPS" : "NO FRAMES"}</span>
          </div>
          <div className="brain-legend" aria-label="TRIBE prediction color scale"><span>LOWER</span><i /><span>MODEL-TARGET Z</span><i /><span>HIGHER</span></div>
          <div className="tribe-facts">
            <div><span>Vertices</span><strong>{manifest ? "20,484" : "—"}<small>fsaverage5</small></strong></div>
            <div><span>Inputs</span><strong>{inputLabel}<small>{manifest ? "model modalities" : "awaiting result"}</small></strong></div>
            <div><span>Lag offset</span><strong>{manifest ? `${manifest.frames.hemodynamicOffsetSeconds}s` : "—"}<small>hemodynamic</small></strong></div>
            <div><span>Revision</span><strong>{revision || "—"}<small>immutable</small></strong></div>
          </div>
        </>
      )}
      <div className={`tribe-message tribe-message--${tribeState.status}`}><Info size={17} weight="fill" /><span>{statusMessage}</span>{tribeState.status === "unconfigured" && <button type="button" onClick={() => { setIsExpanded(false); onOpenMethod(); }}>Connect</button>}</div>
      <TribeAbCutLab open={abLabOpen} onClose={() => setAbLabOpen(false)} primaryReport={descriptorState.report} primaryPrediction={prediction} atlas={atlasState.atlas} sourceVideoUrl={sourceVideoUrl} busyOwner={busyOwner} onPredictVariant={onPredictVariant} />
    </aside>
  );
}

function MethodModal({ open, onClose, configuration, workerState }) {
  useEffect(() => {
    if (!open) return undefined;
    const closeOnEscape = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose, open]);
  if (!open) return null;
  const videoPolicy = workerState.model?.extractors?.video;
  const executionLabel = videoPolicy
    ? ` · ${String(videoPolicy.device).toUpperCase()} / ${videoPolicy.precision === "autocast-fp16" ? "FP16 autocast" : "FP32"}`
    : "";

  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="method-modal" role="dialog" aria-modal="true" aria-labelledby="method-title">
        <button className="modal-close" type="button" onClick={onClose} aria-label="Close method"><X size={20} /></button>
        <span className="modal-kicker"><Flask size={17} weight="fill" /> METHOD / NO GIMMICKS</span>
        <h2 id="method-title">Three evidence lanes.<br />No blurred claims.</h2>
        <div className="method-grid">
          <article><span>01</span><h3>Local content signals</h3><p>Modeled engagement and Virality Outlook run in your browser from measured opening, sustain, visual, edit, and audio signals. They are directional content-signal proxies—not observed outcomes or calibrated probabilities. They never color the brain.</p></article>
          <article><span>02</span><h3>TRIBE v2</h3><p>Runs on a configured inference worker and returns predicted fMRI BOLD values for 20,484 exact fsaverage5 cortical vertices. Only those values color the brain.</p></article>
          <article><span>03</span><h3>Forecast Lab</h3><p>Combines local descriptive evidence with creator-declared context. Optional models and retrieval providers are capability-gated; behavioral values stay locked until a production-calibrated outcome head is verified. TRIBE remains separate.</p></article>
        </div>
        <div className="method-status"><ShieldCheck size={20} weight="fill" /><span>{workerState.status === "connected" ? `Official worker reachable · ${workerState.model.inferenceMode === "vision-only" ? "video-only" : "audio + video + text"}${executionLabel}.` : configuration.configured ? "A TRIBE endpoint is configured but is not currently reachable." : "No TRIBE worker is configured, so the cortical surface remains neutral."}</span></div>
        <p className="license-note">TRIBE v2 predicts average-subject cortical BOLD, not virality. Video-only mode is a deliberate modality ablation; its public code and weights use CC BY-NC 4.0.</p>
      </section>
    </div>
  );
}

function PreviewModal({ open, videoUrl, fileName, onClose }) {
  if (!open || !videoUrl) return null;
  return (
    <div className="modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="preview-modal" role="dialog" aria-modal="true" aria-label="Video preview">
        <div><span>{fileName}</span><button type="button" onClick={onClose} aria-label="Close video preview"><X size={20} /></button></div>
        <video src={videoUrl} controls autoPlay playsInline />
      </section>
    </div>
  );
}

export function App() {
  const configuration = useMemo(() => tribeBackendConfiguration(), []);
  const forecastConfiguration = useMemo(() => forecastBackendConfiguration(), []);
  const videoRef = useRef(null);
  const tribeAbortRef = useRef(null);
  const tribeRunTokenRef = useRef(0);
  const tribeSubmitInFlightRef = useRef(false);
  const forecastJobInFlightRef = useRef(false);
  const forecastJobAbortRef = useRef(null);
  const restoredResultRef = useRef("");
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState({ value: 0, label: "Ready" });
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [mediaTime, setMediaTime] = useState(0);
  const [mediaDuration, setMediaDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [activeFrame, setActiveFrame] = useState(null);
  const [methodOpen, setMethodOpen] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [forecastOpen, setForecastOpen] = useState(false);
  const [forecastCapabilityRequestId, setForecastCapabilityRequestId] = useState(0);
  const [creatorInputs, setCreatorInputs] = useState(() => ({ ...INITIAL_CREATOR_INPUTS }));
  const [forecastEvaluatedAt, setForecastEvaluatedAt] = useState(() => new Date().toISOString());
  const [forecastCapabilityState, setForecastCapabilityState] = useState({
    status: forecastConfiguration.configured ? "idle" : "not-configured",
    report: null,
    message: forecastConfiguration.configured
      ? "Open Forecast Lab to verify the backend capability registry."
      : "No backend base URL is configured for the forecast capability registry.",
  });
  const [forecastJobState, setForecastJobState] = useState({
    status: "idle",
    job: null,
    result: null,
    message: forecastConfiguration.configured
      ? "The selected clip can be submitted to the bounded evidence-job queue."
      : "No backend base URL is configured for forecast jobs.",
    sourceFileName: "",
    retrySafe: false,
    idempotencyKey: "",
  });
  const [tribeElapsedSeconds, setTribeElapsedSeconds] = useState(0);
  const [tribeBusyOwner, setTribeBusyOwner] = useState(null);
  const [workerState, setWorkerState] = useState({ status: configuration.configured ? "checking" : "unconfigured", model: null });
  const [tribeState, setTribeState] = useState({
    status: configuration.configured ? "idle" : "unconfigured",
    stage: configuration.configured ? "idle" : "unconfigured",
    operation: null,
    runToken: 0,
    startedAt: null,
    prediction: null,
    message: configuration.configured ? "Upload a video to run the official model on your configured worker." : "Connect a Linux/NVIDIA TRIBE v2 worker to enable real cortical predictions.",
  });

  useEffect(() => {
    if (!configuration.configured) return undefined;
    const controller = new AbortController();
    getTribeWorkerStatus({ signal: controller.signal })
      .then((worker) => {
        setWorkerState({ status: "connected", model: worker.model });
        setTribeState((current) => current.status === "idle"
          ? { ...current, message: `Official TRIBE worker ready · ${worker.model.inferenceMode === "vision-only" ? "video-only" : "audio + video + text"}. Upload a video to predict cortical response.` }
          : current);
      })
      .catch((workerError) => {
        if (workerError?.name === "AbortError") return;
        setWorkerState({ status: "offline", model: null });
        setTribeState((current) => current.status === "idle"
          ? { ...current, status: "unconfigured", stage: "unconfigured", operation: null, startedAt: null, prediction: null, message: workerError?.message || "The configured TRIBE worker is offline." }
          : current);
      });
    return () => controller.abort();
  }, [configuration.configured]);

  useEffect(() => {
    if (!forecastOpen) return undefined;
    if (!forecastConfiguration.configured) {
      setForecastCapabilityState({
        status: "not-configured",
        report: null,
        message: "No backend base URL is configured for the forecast capability registry.",
      });
      return undefined;
    }
    const controller = new AbortController();
    setForecastCapabilityState({ status: "checking", report: null, message: "Checking the backend capability registry." });
    getForecastCapabilityStatus({ signal: controller.signal })
      .then((report) => setForecastCapabilityState({
        status: "ready",
        report,
        message: report.scoreGenerationAvailable
          ? "The evidence-job route and one or more executable calibrated heads are configured."
          : "The evidence-job route is available; no executable calibrated behavioral head is configured.",
      }))
      .catch((capabilityError) => {
        if (capabilityError?.name === "AbortError") return;
        setForecastCapabilityState({
          status: capabilityError?.code === "FORECAST_NOT_CONFIGURED" ? "not-configured" : "offline",
          report: null,
          message: capabilityError?.message || "The forecast capability registry is unavailable.",
        });
      });
    return () => controller.abort();
  }, [forecastCapabilityRequestId, forecastConfiguration.configured, forecastOpen]);

  const retryForecastCapabilities = useCallback(() => {
    if (!forecastConfiguration.configured) return;
    setForecastCapabilityRequestId((current) => current + 1);
  }, [forecastConfiguration.configured]);

  const followForecastJob = useCallback(async (initialJob, sourceFileName, controller, idempotencyKey = initialJob.jobId) => {
    const outcome = await monitorForecastJob(initialJob, {
      signal: controller.signal,
      baseUrl: forecastConfiguration.baseUrl,
      onJob: (job) => setForecastJobState((current) => ({
        ...current,
        status: job.state,
        job,
        result: null,
        message: job.state === "complete"
          ? "The server completed the evidence job; validating its result contract."
          : job.state === "failed"
            ? job.error.message
            : `Authoritative server stage: ${job.stage}. No completion percentage is estimated.`,
        sourceFileName,
        retrySafe: job.state === "failed",
        idempotencyKey,
      })),
    });
    if (outcome.result) {
      setForecastJobState({
        status: "complete",
        job: outcome.job,
        result: outcome.result,
        message: "The job and result passed strict client-side contract validation.",
        sourceFileName,
        retrySafe: true,
        idempotencyKey,
      });
    } else {
      setForecastJobState({
        status: "failed",
        job: outcome.job,
        result: null,
        message: outcome.job.error?.message || "The authoritative forecast job failed.",
        sourceFileName,
        retrySafe: true,
        idempotencyKey,
      });
    }
  }, [forecastConfiguration.baseUrl]);

  const executeForecastJob = useCallback(async (priorIdempotencyKey = "") => {
    if (!file || forecastJobInFlightRef.current || !forecastConfiguration.configured) return;
    let idempotencyKey = priorIdempotencyKey;
    if (!idempotencyKey) {
      try {
        idempotencyKey = createForecastIdempotencyKey();
      } catch (jobError) {
        setForecastJobState({
          status: "submit-unknown",
          job: null,
          result: null,
          message: jobError?.message || "A secure forecast submission identity could not be created.",
          sourceFileName: file.name,
          retrySafe: false,
          idempotencyKey: "",
        });
        return;
      }
    }
    forecastJobInFlightRef.current = true;
    const controller = new AbortController();
    forecastJobAbortRef.current = controller;
    const sourceFileName = file.name;
    const intendedPostAt = explicitPostInstant(creatorInputs.intendedPostAt, creatorInputs.utcOffset);
    const context = buildForecastCreatorContext(creatorInputs, { intendedPostAt });
    setForecastJobState({
      status: "submitting",
      job: null,
      result: null,
      message: "Streaming the selected File and strict creator-context JSON to the bounded queue.",
      sourceFileName,
      retrySafe: false,
      idempotencyKey,
    });
    try {
      const job = await submitForecastJob(file, context, {
        signal: controller.signal,
        baseUrl: forecastConfiguration.baseUrl,
        idempotencyKey,
      });
      await followForecastJob(job, sourceFileName, controller, idempotencyKey);
    } catch (jobError) {
      if (jobError?.name === "AbortError") return;
      const admittedJob = jobError?.job ?? null;
      setForecastJobState({
        status: admittedJob ? "monitoring-error" : jobError?.retrySafe ? "submit-error" : "submit-unknown",
        job: admittedJob,
        result: null,
        message: jobError?.message || "The forecast evidence job could not be completed.",
        sourceFileName,
        retrySafe: Boolean(jobError?.retrySafe),
        idempotencyKey: jobError?.idempotencyKey ?? idempotencyKey,
      });
    } finally {
      forecastJobInFlightRef.current = false;
      if (forecastJobAbortRef.current === controller) forecastJobAbortRef.current = null;
    }
  }, [creatorInputs, file, followForecastJob, forecastConfiguration.baseUrl, forecastConfiguration.configured]);

  const runForecastJob = useCallback(() => executeForecastJob(), [executeForecastJob]);

  const retryForecastJob = useCallback(() => {
    // A transport/admission retry must reuse the same identity. A completed
    // authoritative failure is a new attempt and therefore receives a new ID.
    const reusableKey = forecastJobState.status === "submit-error"
      ? forecastJobState.idempotencyKey
      : "";
    return executeForecastJob(reusableKey);
  }, [executeForecastJob, forecastJobState.idempotencyKey, forecastJobState.status]);

  const resumeForecastJob = useCallback(async () => {
    const job = forecastJobState.job;
    if (!job || forecastJobInFlightRef.current) return;
    forecastJobInFlightRef.current = true;
    const controller = new AbortController();
    forecastJobAbortRef.current = controller;
    setForecastJobState((current) => ({
      ...current,
      status: job.state,
      message: "Resuming observation of the same server job; no new upload was created.",
      retrySafe: false,
    }));
    try {
      await followForecastJob(job, forecastJobState.sourceFileName, controller, forecastJobState.idempotencyKey || job.jobId);
    } catch (jobError) {
      if (jobError?.name === "AbortError") return;
      setForecastJobState((current) => ({
        ...current,
        status: "monitoring-error",
        job: jobError?.job ?? job,
        result: null,
        message: jobError?.message || "Monitoring of the existing forecast job was interrupted.",
        retrySafe: Boolean(jobError?.retrySafe),
      }));
    } finally {
      forecastJobInFlightRef.current = false;
      if (forecastJobAbortRef.current === controller) forecastJobAbortRef.current = null;
    }
  }, [followForecastJob, forecastJobState.idempotencyKey, forecastJobState.job, forecastJobState.sourceFileName]);

  useEffect(() => () => {
    forecastJobAbortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (workerState.status !== "connected") return undefined;
    const resultUrl = new URL(window.location.href);
    const resultId = resultUrl.searchParams.get("tribeResult");
    if (!resultId || restoredResultRef.current === resultId) return undefined;
    restoredResultRef.current = resultId;
    const runToken = ++tribeRunTokenRef.current;
    const startedAt = Date.now();
    const controller = new AbortController();
    tribeAbortRef.current = controller;
    setTribeElapsedSeconds(0);
    setTribeState({ status: "predicting", stage: "restoring", operation: "restore", runToken, startedAt, prediction: null, message: "Loading a stored, genuine TRIBE prediction artifact." });
    loadTribeResult(resultId, {
      signal: controller.signal,
      onProgress: (update) => {
        if (tribeRunTokenRef.current !== runToken) return;
        setTribeState((current) => current.runToken === runToken ? { ...current, stage: update.stage, message: update.message } : current);
      },
    }).then((prediction) => {
      if (tribeRunTokenRef.current !== runToken) return;
      const modalityNote = prediction.manifest.inferenceMode === "vision-only" ? "Official video-only mode; audio and text were not modeled." : "Official audio, video, and text mode.";
      setTribeState({ status: "ready", stage: "complete", operation: null, runToken, startedAt: null, prediction, message: `Restored contract-validated, hash-checked TRIBE tensor. ${modalityNote}` });
      const requestedFrame = Number(resultUrl.searchParams.get("tribeFrame"));
      const frameIndex = Number.isInteger(requestedFrame) && requestedFrame >= 1
        ? Math.min(requestedFrame - 1, prediction.manifest.frames.shape[0] - 1)
        : 0;
      setMediaTime(prediction.manifest.frames.timesSeconds[frameIndex]);
    }).catch((restoreError) => {
      if (tribeRunTokenRef.current !== runToken || restoreError?.name === "AbortError") return;
      setTribeState({ status: "error", stage: "error", operation: null, runToken, startedAt: null, prediction: null, message: restoreError?.message || "The stored TRIBE result could not be loaded." });
    }).finally(() => {
      if (tribeRunTokenRef.current === runToken && tribeAbortRef.current === controller) tribeAbortRef.current = null;
    });
    return () => {
      controller.abort();
      if (restoredResultRef.current === resultId && tribeRunTokenRef.current === runToken) restoredResultRef.current = "";
    };
  }, [workerState.status]);

  useEffect(() => {
    if (tribeState.status !== "predicting" || !Number.isFinite(tribeState.startedAt)) {
      setTribeElapsedSeconds(0);
      return undefined;
    }
    const startedAt = tribeState.startedAt;
    const updateElapsed = () => setTribeElapsedSeconds(Math.max(0, Math.floor((Date.now() - startedAt) / 1000)));
    updateElapsed();
    const timer = window.setInterval(updateElapsed, 1000);
    return () => window.clearInterval(timer);
  }, [tribeState.startedAt, tribeState.status]);

  useEffect(() => {
    if (!configuration.configured || tribeState.status !== "predicting" || tribeState.operation !== "predict") return undefined;
    const runToken = tribeState.runToken;
    const controller = new AbortController();
    let stopped = false;
    let timer;

    const pollWorker = async () => {
      try {
        const worker = await getTribeWorkerStatus({ signal: controller.signal });
        if (stopped || tribeRunTokenRef.current !== runToken) return;
        setWorkerState({ status: "connected", model: worker.model });
        const update = describeWorkerStage(worker.state);
        setTribeState((current) => (
          current.status === "predicting" && current.operation === "predict" && current.runToken === runToken
            ? { ...current, stage: update.stage, message: update.message }
            : current
        ));
      } catch (pollError) {
        if (pollError?.name === "AbortError") return;
        // A status-poll failure never aborts or replaces the authoritative POST.
      } finally {
        if (!stopped) timer = window.setTimeout(pollWorker, 2000);
      }
    };

    timer = window.setTimeout(pollWorker, 1200);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [configuration.configured, tribeState.operation, tribeState.runToken, tribeState.status]);

  useEffect(() => () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    tribeAbortRef.current?.abort();
  }, [videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl || typeof video.requestVideoFrameCallback !== "function") return undefined;
    let callbackId;
    const readFrameTime = (_now, metadata) => {
      setMediaTime(metadata.mediaTime);
      callbackId = video.requestVideoFrameCallback(readFrameTime);
    };
    callbackId = video.requestVideoFrameCallback(readFrameTime);
    return () => video.cancelVideoFrameCallback?.(callbackId);
  }, [videoUrl]);

  useEffect(() => {
    const prediction = tribeState.prediction;
    if (!playing || videoUrl || tribeState.status !== "ready" || !prediction) return undefined;
    const times = prediction.manifest.frames.timesSeconds;
    const durations = prediction.manifest.frames.durationsSeconds;
    const firstTime = times[0];
    const replayEnd = times[times.length - 1] + durations[durations.length - 1];
    const replayDuration = replayEnd - firstTime;
    if (!(replayDuration > 0)) return undefined;

    const initialOffset = ((mediaTime - firstTime) % replayDuration + replayDuration) % replayDuration;
    const startedAt = performance.now();
    let animationFrame;
    const replay = (now) => {
      const elapsedSeconds = (now - startedAt) / 1000;
      setMediaTime(firstTime + ((initialOffset + elapsedSeconds) % replayDuration));
      animationFrame = requestAnimationFrame(replay);
    };
    animationFrame = requestAnimationFrame(replay);
    return () => cancelAnimationFrame(animationFrame);
  }, [playing, tribeState.prediction, tribeState.status, videoUrl]);

  const resetTribe = useCallback(() => {
    const runToken = ++tribeRunTokenRef.current;
    tribeSubmitInFlightRef.current = false;
    tribeAbortRef.current?.abort();
    tribeAbortRef.current = null;
    restoredResultRef.current = "";
    setTribeElapsedSeconds(0);
    setActiveFrame(null);
    setTribeState({
      status: configuration.configured ? "idle" : "unconfigured",
      stage: configuration.configured ? "idle" : "unconfigured",
      operation: null,
      runToken,
      startedAt: null,
      prediction: null,
      message: configuration.configured ? "Upload a video to run the official model on your configured worker." : "Connect a Linux/NVIDIA TRIBE v2 worker to enable real cortical predictions.",
    });
    const nextUrl = urlWithoutTribeResult(window.location.href);
    if (nextUrl !== window.location.href) window.history.replaceState(window.history.state, "", nextUrl);
  }, [configuration.configured]);

  const selectFile = (nextFile) => {
    if (!nextFile) return;
    if (tribeBusyOwner) {
      setError("The active TRIBE request must finish before you replace the original video.");
      return;
    }
    if (forecastJobInFlightRef.current) {
      setError("The active forecast evidence job must finish before you replace the selected video.");
      return;
    }
    if (!nextFile.type.startsWith("video/") && !/\.(mp4|mov|mkv|webm|ogg|avi)$/i.test(nextFile.name)) {
      setError("Choose a video file your browser can decode.");
      return;
    }
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setFile(nextFile);
    setVideoUrl(URL.createObjectURL(nextFile));
    setStatus("ready");
    setProgress({ value: 0, label: "Ready" });
    setResult(null);
    setMediaTime(0);
    setMediaDuration(0);
    setPlaying(false);
    setError("");
    setForecastJobState({
      status: "idle",
      job: null,
      result: null,
      message: forecastConfiguration.configured
        ? "The selected clip can be submitted to the bounded evidence-job queue."
        : "No backend base URL is configured for forecast jobs.",
      sourceFileName: "",
      retrySafe: false,
      idempotencyKey: "",
    });
    resetTribe();
  };

  const reset = () => {
    if (tribeBusyOwner || forecastJobInFlightRef.current) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setFile(null);
    setVideoUrl("");
    setStatus("idle");
    setResult(null);
    setMediaTime(0);
    setMediaDuration(0);
    setPlaying(false);
    setError("");
    setPreviewOpen(false);
    setForecastJobState({
      status: "idle",
      job: null,
      result: null,
      message: forecastConfiguration.configured
        ? "Choose a clip before starting a forecast evidence job."
        : "No backend base URL is configured for forecast jobs.",
      sourceFileName: "",
      retrySafe: false,
      idempotencyKey: "",
    });
    resetTribe();
  };

  const analyze = async () => {
    if (!file || status === "analyzing" || tribeBusyOwner || tribeSubmitInFlightRef.current || tribeState.status === "predicting") return;
    setStatus("analyzing");
    setError("");
    setProgress(ANALYSIS_STEPS[0]);

    if (configuration.configured) {
      tribeAbortRef.current?.abort();
      const runToken = ++tribeRunTokenRef.current;
      tribeSubmitInFlightRef.current = true;
      setTribeBusyOwner("main");
      restoredResultRef.current = "";
      const startedAt = Date.now();
      const controller = new AbortController();
      tribeAbortRef.current = controller;
      setTribeElapsedSeconds(0);
      setTribeState({ status: "predicting", stage: "requesting", operation: "predict", runToken, startedAt, prediction: null, message: "The request is active. Worker stages come from the separate status endpoint; no completion percentage is estimated." });
      predictWithTribe(file, {
        signal: controller.signal,
        onProgress: (update) => {
          if (tribeRunTokenRef.current !== runToken) return;
          setTribeState((current) => current.runToken === runToken ? { ...current, stage: update.stage, message: update.message } : current);
        },
      }).then((prediction) => {
        if (tribeRunTokenRef.current !== runToken) return;
        const modalityNote = prediction.manifest.inferenceMode === "vision-only" ? "Official video-only mode; audio and text were not modeled." : "Official audio, video, and text mode.";
        const resultId = prediction.manifest.resultId;
        restoredResultRef.current = resultId;
        try {
          window.history.replaceState(window.history.state, "", urlWithTribeResult(window.location.href, resultId));
        } catch {
          // URL persistence is advisory; it must never invalidate a verified tensor.
        }
        setTribeState({ status: "ready", stage: "complete", operation: null, runToken, startedAt: null, prediction, message: `Vertex colors are driven only by this contract-validated, hash-checked TRIBE v2 tensor. ${modalityNote}` });
        const firstPredictionTime = prediction.manifest.frames.timesSeconds[0];
        const video = videoRef.current;
        if (video && video.currentTime < firstPredictionTime) video.currentTime = firstPredictionTime;
        setMediaTime((current) => Math.max(current, firstPredictionTime));
      }).catch((tribeError) => {
        if (tribeRunTokenRef.current !== runToken || tribeError?.name === "AbortError") return;
        setTribeState({ status: "error", stage: "error", operation: null, runToken, startedAt: null, prediction: null, message: tribeError?.message || "TRIBE v2 inference failed. The surface remains neutral." });
      }).finally(() => {
        if (tribeRunTokenRef.current !== runToken) return;
        tribeSubmitInFlightRef.current = false;
        setTribeBusyOwner((current) => current === "main" ? null : current);
        if (tribeAbortRef.current === controller) tribeAbortRef.current = null;
      });
    }

    try {
      const nextResult = await analyzeVideoFile(file, (update) => {
        if (typeof update === "number") {
          const nearest = [...ANALYSIS_STEPS].reverse().find((step) => update >= step.value) || ANALYSIS_STEPS[0];
          setProgress({ value: clamp(update), label: nearest.label });
        } else if (update && typeof update === "object") {
          setProgress({ value: clamp(update.value ?? update.progress ?? 0), label: update.label || update.message || "Analyzing" });
        }
      });
      setResult(nextResult);
      setForecastEvaluatedAt(new Date().toISOString());
      setProgress({ value: 100, label: "Engagement ready" });
      setStatus("complete");
    } catch (analysisError) {
      setStatus("ready");
      setError(analysisError?.message || "This video could not be analyzed. Try MP4 or WebM.");
    }
  };

  const seekVideo = useCallback((time) => {
    const nextTime = Math.max(0, Number.isFinite(time) ? time : 0);
    if (videoRef.current) videoRef.current.currentTime = nextTime;
    setMediaTime(nextTime);
  }, []);

  const predictVariantWithTribe = useCallback(async (variantFile, options) => {
    if (tribeSubmitInFlightRef.current || tribeState.status === "predicting") {
      throw new Error("Another TRIBE request is already using the worker submission lane.");
    }
    tribeSubmitInFlightRef.current = true;
    setTribeBusyOwner("variant");
    try {
      return await predictWithTribe(variantFile, options);
    } finally {
      tribeSubmitInFlightRef.current = false;
      setTribeBusyOwner((current) => current === "variant" ? null : current);
    }
  }, [tribeState.status]);

  const togglePlayback = useCallback(() => {
    const video = videoRef.current;
    if (!video) {
      if (tribeState.prediction) setPlaying((current) => !current);
      return;
    }
    if (video.paused || video.ended) {
      if (video.ended) video.currentTime = 0;
      video.play().catch((playError) => setError(playError?.message || "The video could not be played."));
    } else {
      video.pause();
    }
  }, [tribeState.prediction]);

  const creatorForecast = useMemo(() => {
    const capabilityReport = forecastCapabilityState.status === "ready"
      ? forecastCapabilityState.report
      : null;
    const intendedPostAt = explicitPostInstant(creatorInputs.intendedPostAt, creatorInputs.utcOffset);
    const execution = {
      browserAudio: { available: result?.metadata?.audioAvailable === true },
      vjepa21: { configured: capabilityBranchExecutable(capabilityReport, "vjepa21") },
      videoSemantics: { configured: capabilityBranchExecutable(capabilityReport, "videoLlama21Av") },
      dedicatedAudioModel: { configured: capabilityBranchExecutable(capabilityReport, "audioModel") },
      accountContext: { configured: capabilityBranchExecutable(capabilityReport, "account") },
    };
    try {
      return deriveCreatorForecast({
        durationSeconds: result?.metadata?.duration ?? (mediaDuration > 0 ? mediaDuration : null),
        analysis: result,
        creatorMetadata: {
          caption: creatorInputs.caption,
          topic: creatorInputs.topic,
          genre: creatorInputs.genre,
        },
        creatorContext: {
          platform: creatorInputs.platform,
          locale: creatorInputs.locale,
          language: creatorInputs.language,
          intendedPostAt,
          accountBand: creatorInputs.accountBand,
          accountHistorySummary: creatorInputs.accountHistorySummary,
        },
        domain: {
          platform: creatorInputs.platform,
          locale: creatorInputs.locale,
        },
        execution,
        backendCapabilities: capabilityReport,
        evaluatedAt: forecastEvaluatedAt,
        trendSnapshot: null,
        tribePrediction: tribeState.status === "ready" ? tribeState.prediction : null,
      });
    } catch (forecastError) {
      return Object.freeze({
        schemaVersion: "creator-performance-forecast/1",
        executionProfile: "available-now",
        reason: forecastError?.message || "The local forecast contract could not be derived.",
        metrics: Object.freeze({}),
        capabilities: Object.freeze({ branches: Object.freeze({}) }),
        evidence: Object.freeze({
          contentSignals: Object.freeze({
            status: "unavailable",
            kind: "descriptive-content-feature-index",
            calibrated: false,
            values: Object.freeze({}),
          }),
          tribe: Object.freeze({ present: false, forecastContribution: false }),
        }),
        confidence: Object.freeze({ status: "withheld", value: null, validation: Object.freeze({ status: "not-calibrated" }) }),
      });
    }
  }, [creatorInputs, forecastCapabilityState, forecastEvaluatedAt, mediaDuration, result, tribeState.prediction, tribeState.status]);

  return (
    <div className="studio-page" id="predictor">
      <main className="console-shell">
        <header className="console-topbar">
          <Logo />
          <div className="topbar-tools">
            <span className="model-mode"><ShieldCheck size={14} weight="fill" /> {workerState.status === "connected" ? "TRIBE WORKER CONNECTED" : workerState.status === "checking" ? "CHECKING TRIBE WORKER" : workerState.status === "offline" ? "TRIBE WORKER OFFLINE" : "LOCAL PROXY MODE"}</span>
            <button className="forecast-launch-button" type="button" onClick={() => setForecastOpen(true)}><ChartLineUp size={15} weight="bold" /><span>Forecast Lab</span></button>
            <button className="method-button" type="button" onClick={() => setMethodOpen(true)}>Method</button>
            <button className="menu-button" type="button" onClick={() => setMethodOpen(true)} aria-label="Open method"><List size={22} /></button>
          </div>
        </header>

        <div className="console-layout">
          <section className="workspace-panel">
            <div className="hero-copy reveal reveal--one">
              <span>PRE-PUBLISH CREATIVE LAB</span>
              <h1>See what your<br />clip signals<br /><em>before you post.</em></h1>
            </div>
            <div className="summary-grid reveal reveal--two">
              <UploadCard file={file} status={status} progress={progress} error={error} tribeRunning={tribeState.status === "predicting" || Boolean(tribeBusyOwner)} tribeBusyOwner={tribeBusyOwner} tribeElapsedSeconds={tribeElapsedSeconds} tribeStage={tribeState.stage} onPick={selectFile} onDrop={(event) => { event.preventDefault(); selectFile(event.dataTransfer.files?.[0]); }} onAnalyze={analyze} onReset={reset} />
              <ProxyCard result={result} />
              <ViralityOutlookCard result={result} />
            </div>
            <div className="reveal reveal--three">
              <ClipPanel videoUrl={videoUrl} result={result} videoRef={videoRef} playing={playing} onPlayingChange={setPlaying} onMediaTime={setMediaTime} onMetadata={setMediaDuration} onOpenPreview={() => setPreviewOpen(true)} />
            </div>
          </section>
          <div className="reveal reveal--four"><TribePanel tribeState={tribeState} tribeElapsedSeconds={tribeElapsedSeconds} mediaTime={mediaTime} mediaDuration={mediaDuration} activeFrame={activeFrame} playing={playing} sourceVideoUrl={videoUrl} busyOwner={tribeBusyOwner} onPredictVariant={predictVariantWithTribe} onFrameChange={setActiveFrame} onSeek={seekVideo} onTogglePlayback={togglePlayback} onOpenMethod={() => setMethodOpen(true)} /></div>
        </div>
      </main>

      <section className="truth-strip" aria-label="Model truth and privacy">
        <div><Check size={17} weight="bold" /><span>Local pixel and audio evidence stays descriptive</span></div>
        <div><ChartLineUp size={17} weight="fill" /><span>Forecast providers are gated; behavior requires production calibration</span></div>
        <div><Brain size={17} weight="fill" /><span>TRIBE colors only verified cortical tensors and never forecasts performance</span></div>
      </section>

      <footer><Logo /><p>Local evidence and creator-declared context do not become audience predictions by themselves. Optional Forecast Lab providers are capability-gated, behavioral values require production calibration, and TRIBE-predicted cortical BOLD remains separate from performance forecasting.</p><button type="button" onClick={() => setMethodOpen(true)}>Read the method <ArrowRight size={15} /></button></footer>
      <MethodModal open={methodOpen} onClose={() => setMethodOpen(false)} configuration={configuration} workerState={workerState} />
      <PreviewModal open={previewOpen} videoUrl={videoUrl} fileName={file?.name || "Selected video"} onClose={() => setPreviewOpen(false)} />
      <ForecastLab
        open={forecastOpen}
        onClose={() => setForecastOpen(false)}
        forecast={creatorForecast}
        capabilityReport={forecastCapabilityState.report}
        capabilityState={forecastCapabilityState}
        creatorInputs={creatorInputs}
        onCreatorInputsChange={setCreatorInputs}
        localResult={result}
        fileName={file?.name || ""}
        tribeState={tribeState}
        onRetryCapabilities={retryForecastCapabilities}
        forecastJobState={forecastJobState}
        onRunForecastJob={runForecastJob}
        onRetryForecastJob={retryForecastJob}
        onResumeForecastJob={resumeForecastJob}
      />
    </div>
  );
}
