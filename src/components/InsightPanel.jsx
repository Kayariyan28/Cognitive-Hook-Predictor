import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CaretDown, Flask, Quotes, ShieldWarning, Sparkle } from "@phosphor-icons/react";

import { fetchInsightEvidence, fetchInsightStatus, generateInsight } from "../insight/client.js";
import { HYPOTHESIS_LABEL } from "../insight/contract.js";
import {
  HOOK_REPORT_SECTION_LABELS,
  TRIBE_STANDING_CAPTION,
  buildAbCompareLink,
  experimentsByEffort,
  phaseCommentaryInClipOrder,
  presentCitation,
  presentProviderState,
  presentUnavailable,
} from "../insight/presentation.js";

const IDLE = Object.freeze({ status: "idle", artifact: null, unavailable: null, message: "" });

function CitationChips({ citations, bundle, onSeek }) {
  const [openCitation, setOpenCitation] = useState(null);
  const revealed = useMemo(
    () => (openCitation ? presentCitation(bundle, openCitation) : null),
    [openCitation, bundle],
  );
  return (
    <div className="insight-citations">
      {citations.map((citation) => (
        <button
          className={`insight-citation-chip${openCitation === citation ? " insight-citation-chip--open" : ""}`}
          key={citation}
          type="button"
          aria-expanded={openCitation === citation}
          onClick={() => {
            const next = openCitation === citation ? null : citation;
            setOpenCitation(next);
            if (!next) return;
            const presented = presentCitation(bundle, citation);
            if (presented.status === "resolved" && presented.seekSeconds !== null) {
              onSeek?.(presented.seekSeconds);
            }
          }}
        >
          <Quotes size={11} weight="fill" aria-hidden="true" />
          <span>{citation}</span>
        </button>
      ))}
      {revealed && (
        <div className="insight-citation-value" role="note">
          {revealed.status === "resolved" ? (
            <>
              <strong>{revealed.label}</strong>
              <code>{revealed.valueText}</code>
              {revealed.caption && <small>{revealed.caption}</small>}
            </>
          ) : (
            <small className="insight-citation-missing">{revealed.reason}</small>
          )}
        </div>
      )}
    </div>
  );
}

function TextItems({ items, bundle, onSeek }) {
  return (
    <ul className="insight-item-list">
      {items.map((item, index) => (
        <li key={`${item.text}-${index}`}>
          <p>{item.text}</p>
          <CitationChips citations={item.citations} bundle={bundle} onSeek={onSeek} />
        </li>
      ))}
    </ul>
  );
}

function Hypotheses({ items, bundle, onSeek }) {
  return (
    <ul className="insight-item-list">
      {items.map((item) => (
        <li key={item.id}>
          <p>
            <span className="insight-heuristic-tag">{HYPOTHESIS_LABEL}</span>
            {item.text}
          </p>
          <CitationChips citations={item.citations} bundle={bundle} onSeek={onSeek} />
        </li>
      ))}
    </ul>
  );
}

function Experiments({ items, bundle, forecastResultId, insightId, onSeek, onOpenAbCompare }) {
  return (
    <ol className="insight-experiment-list">
      {experimentsByEffort(items).map((experiment) => (
        <li key={experiment.id}>
          <div className="insight-experiment-head">
            <span className={`insight-effort insight-effort--${experiment.effort}`}>
              {experiment.effort} effort
            </span>
            <span className="insight-heuristic-tag">{HYPOTHESIS_LABEL}</span>
          </div>
          <p>{experiment.edit}</p>
          <ul className="insight-signal-shift">
            {experiment.expectedSignalShift.map((shift) => (
              <li key={`${shift.metricPath}-${shift.direction}`}>
                <code>{shift.metricPath}</code>
                <em>{shift.direction}</em>
              </li>
            ))}
          </ul>
          <CitationChips citations={experiment.citations} bundle={bundle} onSeek={onSeek} />
          <button
            className="insight-ab-link"
            type="button"
            onClick={() =>
              onOpenAbCompare?.(buildAbCompareLink(experiment, { forecastResultId, insightId }))
            }
            disabled={!onOpenAbCompare}
          >
            <Flask size={13} weight="fill" aria-hidden="true" /> Test via A/B compare
          </button>
        </li>
      ))}
    </ol>
  );
}

export function InsightPanel({
  forecastResultId,
  tribeResultId = null,
  tribeDescriptors = null,
  baseUrl,
  onSeek,
  onOpenAbCompare,
}) {
  const [statusState, setStatusState] = useState({ status: "loading", provider: null, message: "" });
  const [generation, setGeneration] = useState(IDLE);
  const [bundle, setBundle] = useState(null);
  const [provenanceOpen, setProvenanceOpen] = useState(false);
  const abortRef = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    fetchInsightStatus({ signal: controller.signal, baseUrl })
      .then((payload) =>
        setStatusState({ status: "ready", provider: presentProviderState(payload), message: "" }),
      )
      .catch((error) => {
        if (error?.name === "AbortError") return;
        setStatusState({
          status: "unavailable",
          provider: null,
          message: error?.message || "The insight service is unreachable.",
        });
      });
    return () => controller.abort();
  }, [baseUrl]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const run = useCallback(
    async (hookOnly) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setBundle(null);
      setGeneration({
        status: "running",
        artifact: null,
        unavailable: null,
        message: hookOnly ? "Reading the first 3 seconds of evidence" : "Reading this clip's evidence",
      });
      try {
        const outcome = await generateInsight(
          { forecastResultId, tribeResultId, tribeDescriptors, hookOnly },
          { signal: controller.signal, baseUrl },
        );
        if (outcome.status === "unavailable") {
          setGeneration({
            status: "unavailable",
            artifact: null,
            unavailable: presentUnavailable(outcome.document),
            message: "",
          });
          return;
        }
        setGeneration({ status: "available", artifact: outcome.artifact, unavailable: null, message: "" });
        try {
          setBundle(
            await fetchInsightEvidence(outcome.artifact.insightId, {
              signal: controller.signal,
              baseUrl,
            }),
          );
        } catch {
          setBundle(null);
        }
      } catch (error) {
        if (error?.name === "AbortError") return;
        setGeneration({
          status: "unavailable",
          artifact: null,
          unavailable: {
            status: "unavailable",
            reasonCode: error?.code ?? null,
            message: error?.message || "The insight request failed.",
            detailText: null,
            rejectionId: null,
          },
          message: "",
        });
      }
    },
    [baseUrl, forecastResultId, tribeDescriptors, tribeResultId],
  );

  const provider = statusState.provider;
  const canGenerate = statusState.status === "ready" && provider?.available && Boolean(forecastResultId);
  const artifact = generation.artifact;

  return (
    <section className="insight-panel" aria-labelledby="insight-panel-title">
      <header className="insight-panel-head">
        <div>
          <span className="insight-panel-kicker">
            <Sparkle size={13} weight="fill" aria-hidden="true" /> DESCRIPTIVE INSIGHT LANE
          </span>
          <h3 id="insight-panel-title">Insight</h3>
          <p>
            Short notes generated from this clip's validated evidence. Every sentence cites the
            evidence it came from. Nothing here is a prediction about an audience.
          </p>
        </div>
        <div className="insight-panel-actions">
          <button type="button" onClick={() => run(false)} disabled={!canGenerate || generation.status === "running"}>
            Generate insight
          </button>
          <button type="button" onClick={() => run(true)} disabled={!canGenerate || generation.status === "running"}>
            Hook Doctor (first 3s)
          </button>
        </div>
      </header>

      {statusState.status === "loading" && (
        <p className="insight-state">Checking which insight model is available.</p>
      )}

      {statusState.status === "unavailable" && (
        <p className="insight-state insight-state--unavailable">
          <ShieldWarning size={14} weight="fill" aria-hidden="true" /> {statusState.message}
        </p>
      )}

      {statusState.status === "ready" && !provider?.available && (
        <p className="insight-state insight-state--unavailable">
          <ShieldWarning size={14} weight="fill" aria-hidden="true" /> Insight generation is
          unavailable: {provider?.message}
        </p>
      )}

      {!forecastResultId && statusState.status === "ready" && provider?.available && (
        <p className="insight-state insight-state--unavailable">
          Run an evidence job first. There is nothing measured to describe yet.
        </p>
      )}

      {generation.status === "running" && <p className="insight-state">{generation.message}</p>}

      {generation.status === "unavailable" && generation.unavailable && (
        <div className="insight-state insight-state--unavailable" role="status">
          <p>
            <ShieldWarning size={14} weight="fill" aria-hidden="true" /> {generation.unavailable.message}
          </p>
          {generation.unavailable.detailText && <small>{generation.unavailable.detailText}</small>}
          {generation.unavailable.reasonCode && (
            <small className="insight-reason-code">reasonCode: {generation.unavailable.reasonCode}</small>
          )}
        </div>
      )}

      {generation.status === "available" && artifact && (
        <div className="insight-artifact">
          <section className="insight-section">
            <h4>
              {HOOK_REPORT_SECTION_LABELS.whatTheHookContains} ({artifact.hookReport.windowSeconds[0]}
              –{artifact.hookReport.windowSeconds[1]}s)
            </h4>
            <TextItems items={artifact.hookReport.whatTheHookContains} bundle={bundle} onSeek={onSeek} />
          </section>

          <section className="insight-section">
            <h4>{HOOK_REPORT_SECTION_LABELS.observations}</h4>
            <TextItems items={artifact.hookReport.observations} bundle={bundle} onSeek={onSeek} />
          </section>

          <section className="insight-section">
            <h4>{HOOK_REPORT_SECTION_LABELS.hypotheses}</h4>
            <Hypotheses items={artifact.hookReport.hypotheses} bundle={bundle} onSeek={onSeek} />
          </section>

          <section className="insight-section">
            <h4>{HOOK_REPORT_SECTION_LABELS.experiments}</h4>
            <Experiments
              items={artifact.hookReport.experiments}
              bundle={bundle}
              forecastResultId={forecastResultId}
              insightId={artifact.insightId}
              onSeek={onSeek}
              onOpenAbCompare={onOpenAbCompare}
            />
          </section>

          {artifact.phaseCommentary.length > 0 && (
            <section className="insight-section insight-section--separated">
              <h4>Phase commentary</h4>
              <ul className="insight-item-list">
                {phaseCommentaryInClipOrder(artifact.phaseCommentary).map((note) => (
                  <li key={`${note.phase}-${note.text}`}>
                    <p>
                      <span className="insight-phase-tag">{note.phase}</span>
                      {note.text}
                    </p>
                    <CitationChips citations={note.citations} bundle={bundle} onSeek={onSeek} />
                  </li>
                ))}
              </ul>
            </section>
          )}

          {artifact.tribeNotes.length > 0 && (
            <section className="insight-section insight-section--separated insight-section--tribe">
              <h4>TRIBE notes</h4>
              <p className="insight-tribe-caption">{TRIBE_STANDING_CAPTION}</p>
              <TextItems items={artifact.tribeNotes} bundle={bundle} onSeek={onSeek} />
            </section>
          )}

          <p className="insight-limits">{artifact.limits}</p>

          <details
            className="insight-provenance"
            open={provenanceOpen}
            onToggle={(event) => setProvenanceOpen(event.currentTarget.open)}
          >
            <summary>
              <CaretDown size={12} weight="bold" aria-hidden="true" /> Provenance
            </summary>
            <dl>
              <div>
                <dt>Provider</dt>
                <dd>{artifact.provenance.provider ?? "—"}</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{artifact.provenance.modelId ?? "—"}</dd>
              </div>
              <div>
                <dt>Revision</dt>
                <dd>
                  <code>{artifact.provenance.modelRevision ?? "—"}</code>
                </dd>
              </div>
              <div>
                <dt>Prompt</dt>
                <dd>{artifact.provenance.promptTemplateId ?? "—"}</dd>
              </div>
              <div>
                <dt>Evidence hash</dt>
                <dd>
                  <code>{artifact.provenance.inputEvidenceHash ?? "—"}</code>
                </dd>
              </div>
              <div>
                <dt>Output hash</dt>
                <dd>
                  <code>{artifact.provenance.outputHash ?? "—"}</code>
                </dd>
              </div>
            </dl>
          </details>
        </div>
      )}
    </section>
  );
}

export default InsightPanel;
