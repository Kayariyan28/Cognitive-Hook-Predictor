import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  Brain,
  CalendarDots,
  ChartLineUp,
  CheckCircle,
  Database,
  FileVideo,
  GlobeHemisphereWest,
  Info,
  LockKey,
  ShieldCheck,
  Sparkle,
  UserCircle,
  WarningCircle,
  Waveform,
} from "@phosphor-icons/react";
import {
  FORECAST_METRIC_SPECS,
  presentForecastMetric,
  resolveForecastMetric,
} from "../forecast/forecastPresentation.js";
import { forecastFromJobResult, isActiveForecastJob } from "../forecast/jobClient.js";

const INITIAL_CREATOR_INPUTS = Object.freeze({
  platform: "youtube-shorts",
  caption: "",
  topic: "",
  genre: "",
  locale: "",
  language: "",
  intendedPostAt: "",
  utcOffset: "",
  accountBand: "unspecified",
  accountHistorySummary: "",
});

const CAPABILITY_SPECS = Object.freeze([
  {
    key: "vjepa21",
    aliases: ["vjepa2_1", "vjepa", "visualEncoder"],
    label: "V-JEPA 2.1",
    fallback: "Local pixel, motion, cut, clarity and stability measurements are used now. They are not equivalent to V-JEPA 2.1 and do not support behavioral probabilities.",
  },
  {
    key: "videoLlama21",
    aliases: ["videoLlama21Av", "video_llama_2_1", "videoLlama", "videoSemantics", "semanticModel"],
    label: "VideoLLaMA2.1",
    fallback: "Creator-entered caption, topic and genre are used as declared context. No model-derived semantic interpretation is claimed.",
  },
  {
    key: "audioModel",
    aliases: ["audio", "beats", "whisper"],
    label: "Audio semantics",
    fallback: "Browser audio energy, dynamics, silence and onset measurements are available when decoding succeeds. No BEATs or Whisper semantics are inferred.",
  },
  {
    key: "semanticModel",
    aliases: ["nanoLlava", "nanollava"],
    label: "NanoLLaVA keyframes",
    fallback: "The pinned NanoLLaVA fallback is not installed. When available, it describes sampled still frames and is never relabeled as VideoLLaMA.",
  },
  {
    key: "measuredAudio",
    aliases: ["serverMeasuredAudio"],
    label: "Measured audio",
    fallback: "Server-side waveform descriptors are not available for this result. They are measured signals, never learned speech/music semantics.",
  },
  {
    key: "account",
    aliases: ["accountContext"],
    label: "Account context",
    fallback: "Creator-declared account context is retained as evidence, but no authenticated account-history provider is configured.",
  },
  {
    key: "trends",
    aliases: ["liveTrends", "trendRetrieval"],
    label: "Live trends",
    fallback: "No timestamped live-web retrieval is configured, so Trend Alignment stays withheld.",
  },
  {
    key: "competitors",
    aliases: ["competitorRetrieval", "saturationRetrieval"],
    label: "Competitors",
    fallback: "No source-backed competitor corpus is configured, so Content Saturation stays withheld.",
  },
]);

const CONTENT_SIGNAL_SPECS = Object.freeze([
  { key: "openingSignalIndex", aliases: ["openingSignal", "opening", "hookScore"], label: "Opening signal" },
  { key: "continuitySignalIndex", aliases: ["postOpeningContinuity", "sustain", "holdRate", "sustainedAttention"], label: "Post-opening continuity" },
  { key: "editDynamicsIndex", aliases: ["pacing", "editDynamics"], label: "Pacing" },
  { key: "endingSignalIndex", aliases: ["ending", "endingStrength"], label: "Ending" },
  { key: "visualSignalIndex", aliases: ["visualClarity", "visualSignal"], label: "Visual clarity" },
  { key: "visualStabilityIndex", aliases: ["visualStability"], label: "Visual stability" },
  { key: "audioSignalIndex", aliases: ["audioSupport", "audioSignal", "audioDynamics"], label: "Audio support" },
]);

const FORECAST_WORKSPACE_VIEWS = Object.freeze([
  { id: "context", label: "Post context" },
  { id: "scores", label: "Forecast" },
  { id: "evidence", label: "Evidence" },
  { id: "all", label: "All panels" },
]);

function normalizeToken(value) {
  return typeof value === "string" ? value.trim().toLowerCase().replace(/[\s_]+/g, "-") : "";
}

function firstText(...values) {
  return values.find((value) => typeof value === "string" && value.trim())?.trim() || "";
}

function firstFiniteSignal(values, keys) {
  for (const key of keys) {
    const candidate = values?.[key];
    if (candidate === null || candidate === undefined || candidate === "") continue;
    const number = Number(candidate);
    if (Number.isFinite(number)) return number;
  }
  return null;
}

function resolveContentSignals(forecast) {
  const contentSignals = forecast?.contentSignals ?? forecast?.evidence?.contentSignals;
  const trustedContract = contentSignals?.kind === "descriptive-content-feature-index"
    && contentSignals?.calibrated === false
    && contentSignals?.values
    && typeof contentSignals.values === "object";
  return CONTENT_SIGNAL_SPECS.map((spec) => ({
    ...spec,
    value: trustedContract
      ? firstFiniteSignal(contentSignals.values, [spec.key, ...spec.aliases])
      : null,
  }));
}

function capabilityMap(forecast, capabilityReport) {
  const source = capabilityReport?.branches
    ?? forecast?.capabilities?.branches
    ?? forecast?.capabilities
    ?? forecast?.evidenceLedger
    ?? forecast?.provenance?.capabilities;
  if (Array.isArray(source)) {
    return Object.fromEntries(source
      .filter((entry) => entry && typeof entry === "object")
      .map((entry) => [entry.key ?? entry.id ?? entry.name, entry]));
  }
  return source && typeof source === "object" ? source : {};
}

function resolveCapability(forecast, capabilityReport, spec) {
  const capabilities = capabilityMap(forecast, capabilityReport);
  const alias = [spec.key, ...spec.aliases].find((key) => Object.hasOwn(capabilities, key));
  const raw = alias ? capabilities[alias] : null;
  const status = normalizeToken(
    typeof raw?.status === "string" ? raw.status : raw?.status?.code ?? raw?.state ?? raw?.availability,
  );
  const isReady = status === "available"
    || status === "ready"
    || status === "configured"
    || status === "implemented-client-runtime";
  return {
    ...spec,
    status: isReady ? "available" : status || "not-configured",
    statusLabel: isReady
      ? "AVAILABLE"
      : status === "loading"
        ? "CHECKING"
        : status === "configured-readiness-deferred"
          ? "READINESS PER JOB"
        : status.startsWith("registered-")
          ? "RUNTIME UNAVAILABLE"
          : status === "configured-runtime-unavailable"
            ? "RUNTIME UNAVAILABLE"
          : "NOT CONFIGURED",
    detail: firstText(
      raw?.detail,
      raw?.description,
      raw?.reason,
      typeof raw?.substitution === "string" ? raw.substitution : raw?.substitution?.description,
      raw?.status?.reason,
      spec.fallback,
    ),
  };
}

function providerForSpec(result, spec) {
  const providers = result?.evidence?.optionalProviders;
  if (!providers || typeof providers !== "object") return null;
  const key = [spec.key, ...spec.aliases].find((candidate) => Object.hasOwn(providers, candidate));
  return key ? providers[key] : null;
}

function resolveProviderCapability(result, spec) {
  const provider = providerForSpec(result, spec);
  if (!provider) return null;
  const output = provider.result;
  const provenance = output?.provenance;
  const observations = Array.isArray(output?.observations) ? output.observations : [];
  const featureCount = output?.features && typeof output.features === "object" ? Object.keys(output.features).length : 0;
  return {
    ...spec,
    status: provider.status,
    statusLabel: provider.status === "available" ? "EVIDENCE READY" : provider.configured ? "RUNTIME UNAVAILABLE" : "NOT CONFIGURED",
    detail: provider.status === "available"
      ? `${provider.evidenceKind} · ${featureCount} verified scalar features · ${observations.length} interval observations.`
      : provider.reason,
    evidenceKind: provider.evidenceKind ?? null,
    featureCount,
    observations,
    warnings: Array.isArray(output?.warnings) ? output.warnings : [],
    provenance: provenance ?? provider.provider?.provenance ?? null,
  };
}

function humanizeStage(stage) {
  if (typeof stage !== "string" || !stage) return "Waiting";
  return stage.split("-").map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
}

function formatSeconds(value) {
  const seconds = Math.max(0, Number.isFinite(Number(value)) ? Math.floor(Number(value)) : 0);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function readableObservationText(value) {
  if (typeof value !== "string") return "Verified interval observation";
  try {
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return value;
    const visibleText = Array.isArray(parsed.visibleText) && parsed.visibleText.length
      ? ` Visible text: ${parsed.visibleText.join(", ")}.`
      : "";
    return [parsed.scene, parsed.action, parsed.shot].filter((item) => typeof item === "string" && item.trim()).join(" ") + visibleText;
  } catch {
    return value;
  }
}

function ProviderEvidence({ capability }) {
  if (!capability.provenance && !capability.observations?.length && !capability.warnings?.length) return null;
  return (
    <div className="forecast-provider-evidence">
      {capability.provenance && (
        <p>
          <span>{capability.provenance.usesLearnedModel === false ? "MEASUREMENT PIPELINE" : "PINNED MODEL"}</span>
          <b>{capability.provenance.modelId || capability.provenance.adapterId || "Versioned provider"}</b>
          <code>{capability.provenance.modelRevision ? `${capability.provenance.modelRevision.slice(0, 10)} · weights ${capability.provenance.modelWeightsSha256?.slice(0, 10) || "withheld"}` : capability.provenance.preprocessingId || "Provenance supplied by provider"}</code>
        </p>
      )}
      <div className="forecast-provider-observations">
        <span>INTERVAL OBSERVATIONS</span>
        {capability.observations?.length ? capability.observations.map((observation, index) => (
          <p key={`${observation.startTime}-${observation.endTime}-${index}`}>
            <time>{formatSeconds(observation.startTime)}–{formatSeconds(observation.endTime)}</time>
            <b>{readableObservationText(observation.text)}</b>
          </p>
        )) : <small>No interval observation was emitted by this worker.</small>}
      </div>
      {capability.warnings?.map((warning) => <small className="forecast-provider-warning" key={warning}>{warning}</small>)}
    </div>
  );
}

function ForecastJobControl({ state, fileName, configured, onRun, onRetry, onResume }) {
  const job = state?.job;
  const active = isActiveForecastJob(job) || state?.status === "submitting";
  const complete = state?.status === "complete" && state?.result;
  const failed = state?.status === "failed";
  const monitoringError = state?.status === "monitoring-error";
  const resumableMonitoringError = monitoringError && state?.retrySafe;
  const submitError = state?.status === "submit-error" || state?.status === "submit-unknown";
  const statusClass = complete ? "complete" : failed || submitError || monitoringError ? "error" : active ? "running" : "idle";
  const title = state?.status === "submitting" ? "Uploading clip to the bounded queue"
    : active ? humanizeStage(job?.stage)
      : complete ? "Evidence result contract verified"
        : failed ? "Authoritative job failed"
          : monitoringError ? resumableMonitoringError ? "Job retained; retrieval paused" : "Result contract rejected"
            : submitError ? "Job was not started"
              : "Ready for server evidence";
  const message = state?.message || (fileName
    ? "Submit the selected clip to the server-side 10–60 second probe and configured evidence workers."
    : "Choose a 10–60 second video in the creative lab first.");
  return (
    <section className={`forecast-job-control forecast-job-control--${statusClass}`} aria-label="Forecast evidence job" aria-live="polite">
      <div className="forecast-job-control-head">
        <span>{active ? <span className="forecast-job-spinner" aria-hidden="true" /> : <ShieldCheck size={15} weight="fill" />}</span>
        <p><strong>{title}</strong><small>{message}</small></p>
        {job && <b>{formatSeconds(job.elapsedSeconds)}</b>}
      </div>
      {job && (
        <div className="forecast-job-facts">
          <span>JOB <code>{job.jobId.slice(0, 10)}</code></span>
          <span>STAGE <code>{job.stage}</code></span>
          <span>HEARTBEAT <time dateTime={job.heartbeatAt}>{new Date(job.heartbeatAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time></span>
        </div>
      )}
      <div className="forecast-job-actions">
        {resumableMonitoringError ? (
          <button type="button" onClick={onResume}>Resume this job</button>
        ) : failed || (state?.status === "submit-error" && state?.retrySafe) ? (
          <button type="button" onClick={onRetry}>Retry</button>
        ) : (
          <button type="button" onClick={onRun} disabled={!configured || !fileName || active || monitoringError || state?.status === "submit-unknown"}>
            {active ? "Evidence job running" : monitoringError ? "Rejected result withheld" : complete ? "Run again with current context" : "Run server evidence"}
          </button>
        )}
        <span>{state?.sourceFileName || fileName || "No video selected"}</span>
      </div>
      {state?.status === "submit-unknown" && <p className="forecast-job-unknown">Submission was not confirmed, so this client will not create a possible duplicate. Restore the service connection before taking another action.</p>}
    </section>
  );
}

function MetricRow({ forecast, spec, durationState, scoringDisabled, scoringLockReason }) {
  const rawMetric = resolveForecastMetric(forecast, spec);
  const presentation = presentForecastMetric(rawMetric, spec);
  const domainBlocked = durationState.status === "out-of-domain";
  const canShowValue = presentation.canShowValue && !domainBlocked && !scoringDisabled;
  const reason = domainBlocked
    ? durationState.reason
    : scoringDisabled
      ? scoringLockReason
      : presentation.reason;
  return (
    <div className={`forecast-metric-row ${canShowValue ? "is-available" : "is-withheld"}`}>
      <div className="forecast-metric-copy">
        <strong>{spec.label}</strong>
        <span>{canShowValue ? presentation.evidenceLabel : reason}</span>
      </div>
      <div className="forecast-metric-value" aria-label={canShowValue ? `${spec.label}: ${presentation.valueText}${presentation.unitText}` : `${spec.label} unavailable: ${reason}`}>
        <strong>{canShowValue ? presentation.valueText : "—"}</strong>
        {canShowValue && presentation.unitText && <small>{presentation.unitText}</small>}
        {!canShowValue && <LockKey size={13} weight="fill" aria-hidden="true" />}
      </div>
    </div>
  );
}

function InputField({ label, hint, children }) {
  return (
    <label className="forecast-field">
      <span>{label}{hint && <small>{hint}</small>}</span>
      {children}
    </label>
  );
}

export function ForecastLab({
  open,
  onClose,
  forecast,
  capabilityReport,
  capabilityState,
  creatorInputs: controlledCreatorInputs,
  onCreatorInputsChange,
  localResult,
  fileName,
  tribeState,
  onRetryCapabilities,
  forecastJobState,
  onRunForecastJob,
  onRetryForecastJob,
  onResumeForecastJob,
}) {
  const [uncontrolledCreatorInputs, setUncontrolledCreatorInputs] = useState(INITIAL_CREATOR_INPUTS);
  const [workspaceView, setWorkspaceView] = useState("scores");
  const [textScale, setTextScale] = useState(115);
  const labRef = useRef(null);
  const creatorInputs = controlledCreatorInputs ?? uncontrolledCreatorInputs;
  const setCreatorInputs = onCreatorInputsChange ?? setUncontrolledCreatorInputs;

  useEffect(() => {
    if (!open) return undefined;
    const closeOnEscape = (event) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose, open]);

  useEffect(() => {
    if (open) labRef.current?.scrollTo({ top: 0, behavior: "auto" });
  }, [open, workspaceView]);

  const jobForecast = useMemo(() => {
    if (!forecastJobState?.result) return null;
    try {
      return forecastFromJobResult(forecastJobState.result);
    } catch {
      return null;
    }
  }, [forecastJobState?.result]);
  const displayedForecast = jobForecast ?? forecast;
  const metricViews = useMemo(() => FORECAST_METRIC_SPECS.map((spec) => ({
    spec,
    presentation: presentForecastMetric(resolveForecastMetric(displayedForecast, spec), spec),
  })), [displayedForecast]);
  // Status can truthfully advertise executable heads, but it is not a score
  // payload. Only a strictly validated completed job result may unlock values.
  const scoringDisabled = !jobForecast;
  const scoringLockReason = jobForecast
    ? "The completed job returned no production-calibrated value for this metric."
    : capabilityState?.status === "ready"
    ? capabilityReport?.scoreGenerationAvailable
      ? "Executable calibrated heads are configured, but values stay locked until this clip returns a validated completed job result."
      : "Locked because no executable production-calibrated head is configured."
    : capabilityState?.status === "checking"
      ? "Locked while backend capability verification is pending."
      : capabilityState?.status === "offline"
        ? "Locked because the backend capability registry is unreachable."
        : "Locked because no backend capability registry is configured.";
  const calibratedMetricCount = scoringDisabled ? 0 : metricViews.filter((metric) => metric.presentation.canShowValue).length;
  const intrinsicMetrics = FORECAST_METRIC_SPECS.filter((metric) => metric.lane === "intrinsic");
  const contextMetrics = FORECAST_METRIC_SPECS.filter((metric) => metric.lane === "context");
  const outcomeMetrics = FORECAST_METRIC_SPECS.filter((metric) => metric.lane === "outcome");
  const capabilities = CAPABILITY_SPECS.map((spec) => (
    resolveProviderCapability(forecastJobState?.result, spec)
      ?? resolveCapability(forecast, capabilityReport, spec)
  ));
  const contentSignals = resolveContentSignals(forecast);
  const contentSignalsReady = contentSignals.some((signal) => signal.value !== null);
  const localVisualReady = Boolean(localResult?.metadata?.frameSampleCount);
  const localAudioReady = localResult?.metadata?.audioAvailable === true;
  const briefReady = Boolean(
    creatorInputs.caption.trim()
      || creatorInputs.topic.trim()
      || creatorInputs.genre.trim()
      || creatorInputs.locale.trim()
      || creatorInputs.language.trim()
      || creatorInputs.intendedPostAt,
  );
  const tribeRuntimeState = capabilityReport?.branches?.tribeCortical?.state ?? tribeState?.status ?? "unconfigured";
  const tribeReady = tribeRuntimeState === "ready";
  const durationSeconds = Number(forecastJobState?.result?.input?.durationSeconds ?? localResult?.metadata?.duration);
  const durationState = Number.isFinite(durationSeconds) && durationSeconds > 0
    ? durationSeconds >= 10 && durationSeconds <= 60
      ? { status: "eligible", label: `${durationSeconds.toFixed(1)}s · 10–60s inclusive`, reason: "" }
      : { status: "out-of-domain", label: `${durationSeconds.toFixed(1)}s · outside 10–60s`, reason: "Withheld because the clip duration is outside the supported 10–60 second domain." }
    : { status: "pending", label: "Upload a 10–60s clip", reason: "Clip duration has not been measured." };

  if (!open) return null;

  const updateInput = (key) => (event) => setCreatorInputs((current) => ({
    ...current,
    [key]: event.target.value,
  }));
  const updateTextScale = (event) => setTextScale(Number(event.currentTarget.value));

  return (
    <section
      className={`forecast-lab forecast-lab--view-${workspaceView}`}
      ref={labRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby="forecast-lab-title"
      style={{ "--forecast-text-scale": textScale / 100 }}
    >
      <header className="forecast-lab-topbar">
        <div className="forecast-lab-brand">
          <span className="forecast-lab-brandmark"><ChartLineUp size={15} weight="bold" /></span>
          <span>SignalFrame.</span><i>/</i><b>Forecast Lab</b>
        </div>
        <div className="forecast-workspace-tools">
          <div className="forecast-view-switcher" role="tablist" aria-label="Forecast Lab workspace views">
            {FORECAST_WORKSPACE_VIEWS.map((view) => (
              <button
                key={view.id}
                type="button"
                role="tab"
                aria-selected={workspaceView === view.id}
                aria-controls={view.id === "all"
                  ? "forecast-context-view forecast-scores-view forecast-evidence-view"
                  : `forecast-${view.id}-view`}
                onClick={() => setWorkspaceView(view.id)}
              >
                {view.label}
              </button>
            ))}
          </div>
          <label className="forecast-text-scale-control">
            <span>Text size</span>
            <input
              type="range"
              min="100"
              max="140"
              step="5"
              value={textScale}
              onInput={updateTextScale}
              onChange={updateTextScale}
              aria-label="Forecast Lab text size"
            />
            <output>{textScale}%</output>
          </label>
        </div>
        <div className="forecast-lab-top-actions">
          <span className="forecast-execution-pill"><CheckCircle size={14} weight="fill" /> {isActiveForecastJob(forecastJobState?.job) ? "JOB RUNNING" : forecastJobState?.result ? "RESULT VERIFIED" : "AVAILABLE NOW"}</span>
          <span className={`forecast-calibration-pill ${calibratedMetricCount ? "has-calibration" : ""}`}><ShieldCheck size={14} weight="fill" /> {calibratedMetricCount ? `${calibratedMetricCount} CALIBRATED HEADS` : "NO CALIBRATED HEADS"}</span>
          <button type="button" onClick={onClose} aria-label={isActiveForecastJob(forecastJobState?.job) ? "Close Forecast Lab; evidence job keeps running" : "Back to creative lab"}><ArrowLeft size={16} weight="bold" /><span>{isActiveForecastJob(forecastJobState?.job) ? "Close · job continues" : "Back to creative lab"}</span></button>
        </div>
      </header>

      <div className={`forecast-lab-layout forecast-lab-layout--${workspaceView}`}>
        <aside
          id="forecast-context-view"
          className="forecast-context-panel"
          aria-label="Creator and publishing context"
          hidden={workspaceView !== "all" && workspaceView !== "context"}
        >
          <div className="forecast-panel-kicker"><span>01</span> ACCOUNT / DISTRIBUTION</div>
          <div className="forecast-context-heading">
            <h2 id="forecast-lab-title">Describe the post,<br /><em>not a fantasy result.</em></h2>
            <p>These are declared covariates. They become predictive only after a matching platform model is trained and validated.</p>
          </div>

          <form className="forecast-inputs" onSubmit={(event) => event.preventDefault()}>
            <div className="forecast-input-grid">
              <InputField label="Platform">
                <select value={creatorInputs.platform} onChange={updateInput("platform")}>
                  <option value="youtube-shorts">YouTube Shorts</option>
                  <option value="instagram-reels">Instagram Reels</option>
                  <option value="tiktok">TikTok</option>
                  <option value="snapchat-spotlight">Snapchat Spotlight</option>
                </select>
              </InputField>
              <InputField label="Account band">
                <select value={creatorInputs.accountBand} onChange={updateInput("accountBand")}>
                  <option value="unspecified">Not specified</option>
                  <option value="new">New / under 1K</option>
                  <option value="emerging">1K–10K</option>
                  <option value="established">10K–100K</option>
                  <option value="large">100K+</option>
                </select>
              </InputField>
            </div>
            <InputField label="Caption" hint="DECLARED">
              <textarea rows="2" value={creatorInputs.caption} onChange={updateInput("caption")} placeholder="Paste the caption you plan to publish" />
            </InputField>
            <div className="forecast-input-grid">
              <InputField label="Topic">
                <input value={creatorInputs.topic} onChange={updateInput("topic")} placeholder="e.g. budget travel" />
              </InputField>
              <InputField label="Genre">
                <input value={creatorInputs.genre} onChange={updateInput("genre")} placeholder="e.g. tutorial" />
              </InputField>
            </div>
            <div className="forecast-input-grid">
              <InputField label="Locale">
                <input value={creatorInputs.locale} onChange={updateInput("locale")} placeholder="e.g. en-IN" />
              </InputField>
              <InputField label="Language">
                <input value={creatorInputs.language} onChange={updateInput("language")} placeholder="e.g. English" />
              </InputField>
            </div>
            <div className="forecast-input-grid">
              <InputField label="Intended post date / time">
                <input type="datetime-local" value={creatorInputs.intendedPostAt} onChange={updateInput("intendedPostAt")} />
              </InputField>
              <InputField label="UTC offset" hint="REQUIRED FOR CUTOFF">
                <input value={creatorInputs.utcOffset} onChange={updateInput("utcOffset")} placeholder="e.g. +05:30" inputMode="text" />
              </InputField>
            </div>
            <InputField label="Account history summary" hint="OPTIONAL · FACTS ONLY">
              <textarea rows="3" value={creatorInputs.accountHistorySummary} onChange={updateInput("accountHistorySummary")} placeholder="Example: median 7-day views and completion across your last ten comparable posts" />
            </InputField>
          </form>

          <ForecastJobControl
            state={forecastJobState}
            fileName={fileName}
            configured={capabilityState?.status !== "not-configured" && capabilityReport?.uploadsAccepted !== false}
            onRun={onRunForecastJob}
            onRetry={onRetryForecastJob}
            onResume={onResumeForecastJob}
          />

          <div className="forecast-input-boundary"><LockKey size={15} weight="fill" /><span>The submitted job receives this declared context as strict JSON evidence. It is not converted into an outcome score unless a matching authenticated provider and production-calibrated head explicitly accept it.</span></div>
        </aside>

        <main
          id="forecast-scores-view"
          className="forecast-score-panel"
          aria-label="Behavioral forecast metrics"
          hidden={workspaceView !== "all" && workspaceView !== "scores"}
        >
          <div className={`forecast-domain-band forecast-domain-band--${durationState.status}`}>
            <CalendarDots size={16} weight="fill" />
            <p><strong>{durationState.status === "eligible" ? "BROWSER PREFLIGHT ELIGIBLE · SERVER PROBE REQUIRED" : durationState.status === "out-of-domain" ? "BROWSER PREFLIGHT OUTSIDE FORECAST DOMAIN" : "FORECAST ELIGIBILITY PENDING"}</strong><span>{durationState.label}</span></p>
            <small>10–60S INCLUSIVE</small>
          </div>
          {scoringDisabled && (
            <div className="forecast-backend-lock">
              <LockKey size={15} weight="fill" />
              <p><strong>FORECAST NUMBERS LOCKED</strong><span>{scoringLockReason}{capabilityState?.status === "ready" && <> <code>{capabilityReport?.scoreGenerationAvailable ? "completedJobResult:required" : "scoreGenerationAvailable:false"}</code></>}</span></p>
            </div>
          )}
          <div className="forecast-score-head">
            <div>
              <span>02 / BEHAVIORAL FORECAST</span>
              <h3>Scores that earn the right to appear.</h3>
            </div>
            <p>A value renders only when its head reports <b>available</b> and its validation is <b>production-calibrated</b>. Everything else stays withheld with a reason.</p>
          </div>

          <section className="forecast-metric-group" aria-labelledby="intrinsic-metrics-title">
            <div className="forecast-group-title"><span><FileVideo size={14} weight="fill" /> INTRINSIC CLIP EVIDENCE</span><small>VIDEO / AUDIO / SEMANTICS</small></div>
            <h4 id="intrinsic-metrics-title" className="visually-hidden">Intrinsic clip evidence metrics</h4>
            {intrinsicMetrics.map((spec) => <MetricRow key={spec.key} forecast={displayedForecast} spec={spec} durationState={durationState} scoringDisabled={scoringDisabled} scoringLockReason={scoringLockReason} />)}
          </section>

          <section className="forecast-metric-group" aria-labelledby="context-metrics-title">
            <div className="forecast-group-title"><span><GlobeHemisphereWest size={14} weight="fill" /> CONTEXT OPPORTUNITY</span><small>TIMESTAMPED EXTERNAL EVIDENCE</small></div>
            <h4 id="context-metrics-title" className="visually-hidden">Context opportunity metrics</h4>
            {contextMetrics.map((spec) => <MetricRow key={spec.key} forecast={displayedForecast} spec={spec} durationState={durationState} scoringDisabled={scoringDisabled} scoringLockReason={scoringLockReason} />)}
          </section>

          <section className="forecast-outcome-card" aria-labelledby="outcome-metrics-title">
            <div><span>COMBINED OUTCOME</span><small>PLATFORM + HORIZON MUST BE DEFINED</small></div>
            <h4 id="outcome-metrics-title" className="visually-hidden">Combined outcome metrics</h4>
            {outcomeMetrics.map((spec) => <MetricRow key={spec.key} forecast={displayedForecast} spec={spec} durationState={durationState} scoringDisabled={scoringDisabled} scoringLockReason={scoringLockReason} />)}
          </section>
        </main>

        <aside
          id="forecast-evidence-view"
          className="forecast-evidence-panel"
          aria-label="Available evidence and capability ledger"
          hidden={workspaceView !== "all" && workspaceView !== "evidence"}
        >
          <div className="forecast-panel-kicker forecast-panel-kicker--dark"><span>03</span> EVIDENCE LEDGER</div>
          <div className={`forecast-registry-state forecast-registry-state--${capabilityState?.status || "not-configured"}`}>
            <ShieldCheck size={13} weight="fill" />
            <span>{capabilityState?.status === "ready" ? "BACKEND CAPABILITY REGISTRY VERIFIED" : capabilityState?.status === "checking" ? "CHECKING BACKEND CAPABILITY REGISTRY" : capabilityState?.status === "offline" ? "BACKEND CAPABILITY REGISTRY OFFLINE" : "BACKEND CAPABILITY REGISTRY NOT CONFIGURED"}</span>
            {capabilityState?.status === "offline" && onRetryCapabilities && <button type="button" onClick={onRetryCapabilities}>Retry</button>}
          </div>
          <div className="available-now-head">
            <div><Sparkle size={17} weight="fill" /><span>AVAILABLE NOW</span></div>
            <p>Useful, descriptive evidence runs without pretending it predicts audience behavior.</p>
          </div>

          <div className="available-evidence-grid">
            <article className={localVisualReady ? "is-ready" : ""}>
              <FileVideo size={18} weight="fill" />
              <div><strong>Local visual</strong><span>{localVisualReady ? `${localResult.metadata.frameSampleCount} source-frame samples measured` : fileName ? "Ready when local analysis finishes" : "Upload a clip to measure"}</span></div>
              <b>{localVisualReady ? "MEASURED" : "WAITING"}</b>
            </article>
            <article className={localAudioReady ? "is-ready" : ""}>
              <Waveform size={18} weight="fill" />
              <div><strong>Local audio</strong><span>{localAudioReady ? "Energy, dynamics, silence and onsets measured" : localResult ? "No decodable audio evidence" : "Checked during local analysis"}</span></div>
              <b>{localAudioReady ? "MEASURED" : localResult ? "ABSENT" : "WAITING"}</b>
            </article>
            <article className={briefReady ? "is-ready" : ""}>
              <UserCircle size={18} weight="fill" />
              <div><strong>Creator brief</strong><span>{briefReady ? "Declared publishing context is present" : "Add factual post context"}</span></div>
              <b>{briefReady ? "DECLARED" : "OPTIONAL"}</b>
            </article>
          </div>

          {(localResult?.strengths?.[0] || localResult?.improvements?.[0]) && (
            <div className="forecast-local-readout">
              {localResult?.strengths?.[0] && <p><CheckCircle size={14} weight="fill" /><span><b>Measured strength</b>{localResult.strengths[0]}</span></p>}
              {localResult?.improvements?.[0] && <p><Info size={14} weight="fill" /><span><b>Creative check</b>{localResult.improvements[0]}</span></p>}
            </div>
          )}

          <section className="content-signal-indices" aria-labelledby="content-signal-indices-title">
            <div className="content-signal-title">
              <div><ChartLineUp size={14} weight="fill" /><span id="content-signal-indices-title">Available content-signal indices</span></div>
              <small>DESCRIPTIVE · UNCALIBRATED</small>
            </div>
            <div className="content-signal-grid">
              {contentSignals.map((signal) => (
                <article key={signal.key}>
                  <span>{signal.label}</span>
                  <strong>{signal.value === null ? "—" : Math.round(signal.value)}{signal.value !== null && <small>/100</small>}</strong>
                </article>
              ))}
            </div>
            <p>{contentSignalsReady ? "Directional content evidence from the local measurement branch." : "Run local analysis to populate this audited descriptive-index contract."} Not predicted attention, retention, engagement or virality.</p>
          </section>

          <div className="capability-ledger" aria-label="Forecast model capabilities">
            <div className="capability-ledger-title"><span>MODEL / RETRIEVAL STATUS</span><small>SUBSTITUTIONS ARE NOT EQUIVALENT</small></div>
            {capabilities.map((capability) => (
              <article key={capability.key} className={`${capability.status === "available" ? "is-ready" : ""} ${capability.provenance || capability.observations?.length ? "has-provider-evidence" : ""}`}>
                {capability.status === "available" ? <CheckCircle size={14} weight="fill" /> : <WarningCircle size={14} weight="fill" />}
                <div><strong>{capability.label}</strong><p>{capability.detail}</p><ProviderEvidence capability={capability} /></div>
                <b>{capability.statusLabel}</b>
              </article>
            ))}
            <article className={`is-separate ${tribeReady ? "is-ready" : ""}`}>
              <Brain size={14} weight="fill" />
              <div><strong>TRIBE v2</strong><p>{tribeReady ? "Verified cortical tensor is available in its separate brain workspace." : `Cortical runtime state: ${tribeRuntimeState}.`} It never contributes to creator-performance probabilities.</p></div>
              <b>FORECAST: NONE</b>
            </article>
          </div>

          <div className="forecast-next-valid-step">
            <Database size={18} weight="fill" />
            <p><strong>What unlocks real scores</strong><span>Legally licensed, platform-specific outcomes; a defined prediction horizon; chronological holdout tests; probability calibration; and drift monitoring.</span></p>
          </div>
          <div className="forecast-truth-note"><WarningCircle size={15} weight="fill" /><span>TRIBE is kept separate with <code>forecastContribution:false</code>. Local substitutes are descriptive and never relabeled as V-JEPA, VideoLLaMA, audio semantics, trends or competitors.</span></div>
        </aside>
      </div>
    </section>
  );
}

export { INITIAL_CREATOR_INPUTS };
