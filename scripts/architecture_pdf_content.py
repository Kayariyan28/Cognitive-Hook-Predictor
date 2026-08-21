"""The document body. Every figure here was read from the repository."""

from reportlab.platypus import KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

from build_architecture_pdf import (  # noqa
    ACCENT, ACCENT_WASH, ArchitectureDiagram, CONTENT_W, INK, MUTED, RISK, RISK_WASH,
    RULE, S_BODY, S_BULLET, S_CELL, S_CELL_MUTED, S_H1, S_H2, S_KICKER, S_LEAD, S_MONO,
    S_SMALL, S_SUBTITLE, S_TITLE, TRIBE, TRIBE_WASH, Rule, bullet_list, callout, table,
)

W = CONTENT_W


def mono(text):
    return Paragraph(f"<font name='Courier' size='7.3'>{text}</font>", S_CELL)


STORY = []
A = STORY.append

# ---------------------------------------------------------------- cover
A(Spacer(1, 30))
A(Paragraph("SYSTEM DOCUMENTATION &nbsp;&middot;&nbsp; VERIFIED AGAINST THE RUNNING BUILD", S_KICKER))
A(Paragraph("SignalFrame", S_TITLE))
A(Paragraph("Architecture, capabilities, and what this system deliberately will not do", S_SUBTITLE))
A(Spacer(1, 10))
A(Rule(W, 1.4, INK))
A(Spacer(1, 12))
A(Paragraph(
    "SignalFrame is a pre-publish analysis lab for short-form video. It measures a clip, describes what it "
    "measured, and helps a creator turn that into an experiment they can run. It does not predict how an "
    "audience will respond, and it is built so that it cannot quietly start to.",
    S_LEAD))
A(Paragraph(
    "This document is the complete inventory: every evidence lane, every route, every enforcement layer, and "
    "every limit. It is written to be checkable. Where a capability is unavailable, that is stated. Where a "
    "number could not be verified, it is not printed.",
    S_BODY))
A(Spacer(1, 8))

A(table(
    [["VERIFIED FACT", "VALUE", "WHERE IT COMES FROM"],
     ["Evidence branches defined", "10", "backend/forecast/schema.py"],
     ["Evidence lanes in the insight bundle", "8", "backend/insight/bundle.py"],
     ["HTTP routes", "21", "forecast and insight routers"],
     ["Behavioral targets defined", "7", "backend/forecast/schema.py"],
     ["Behavioral heads approved and installed", "0", "APPROVED_TARGET_CONTRACTS is empty"],
     ["Fail-closed reason codes", "13", "backend/insight/validation.py"],
     ["Red-team fixtures running in CI", "53", "backend/tests/insight_fixtures/"],
     ["Backend tests", "403", "python -m unittest discover -s backend/tests"],
     ["Frontend tests", "150", "npm test"]],
    [None, 60, 200]))
A(Spacer(1, 10))
A(callout(
    "THE ONE SENTENCE THAT GOVERNS EVERYTHING ELSE",
    "A model that describes a clip is not a model that predicts an audience. SignalFrame keeps those two "
    "things in separate lanes, and refuses to emit the second one at all until a separately trained, "
    "target-specific, production-calibrated head passes a gate that no bundled artifact currently passes.",
    RISK, RISK_WASH))

A(PageBreak())

# ---------------------------------------------------------------- architecture
A(Paragraph("1.&nbsp;&nbsp;How the system is put together", S_H1))
A(Paragraph(
    "One clip enters. Three independent lanes measure it. Their outputs are assembled into a citable evidence "
    "bundle, and four surfaces read that bundle. Nothing crosses from a lane into a behavioral claim, because "
    "the only component that could make one does not exist in this build.",
    S_BODY))
A(Spacer(1, 6))
A(ArchitectureDiagram(W))
A(Spacer(1, 8))

A(Paragraph("Why the lanes are separate", S_H2))
A(Paragraph(
    "The separation is not stylistic. Each lane answers a different question, and blending them would produce "
    "a number that answers none of them. Content signals describe measurable structure. Encoder evidence "
    "describes what a pinned model extracted. TRIBE v2 describes a predicted average-subject cortical BOLD "
    "response. A single blended score would be defensible as none of the three.",
    S_BODY))
A(Spacer(1, 4))
A(table(
    [["LANE", "WHAT IT IS", "WHAT IT IS NOT", "CONTRIBUTES TO A FORECAST?"],
     ["Content signals",
      "Deterministic measurements of the file: pixel and motion statistics in the browser, PCM and STFT "
      "descriptors on the server, authoritative duration from ffprobe.",
      "Not semantics, not quality, not behaviour.",
      Paragraph("<font color='#8C3A16'><b>No head exists</b></font>", S_CELL)],
     ["Encoder evidence",
      "What pinned, hash-verified models extract: visual-temporal representations, keyframe descriptions, "
      "sound labels, a transcript, on-screen glyphs.",
      "Not comprehension, not attention, not retention.",
      Paragraph("<font color='#8C3A16'><b>No head exists</b></font>", S_CELL)],
     ["TRIBE v2 cortical",
      "A verified T x 20,484 fsaverage5 tensor of predicted average-subject BOLD, summarised into interval, "
      "phase and parcel descriptors.",
      "Not a scan, not an individual, not a mental state, not virality.",
      Paragraph("<font color='#2E5C8A'><b>forecastContribution: false</b></font>", S_CELL)]],
    [72, 168, 130, 92]))
A(Spacer(1, 8))
A(callout(
    "TRIBE IS THE CREDIBILITY, NOT THE FORECAST ENGINE",
    "TRIBE v2 predicts an average subject's fMRI BOLD response with a five-second hemodynamic offset, and on "
    "Apple silicon it runs vision-only, so it does not even hear the clip. It is a genuine, independently "
    "verified signal for describing and for comparing two cuts. It is never used as a proxy for attention, "
    "engagement, or performance, and text that tries is rejected automatically.",
    TRIBE, TRIBE_WASH))

A(PageBreak())

# ---------------------------------------------------------------- capabilities
A(Paragraph("2.&nbsp;&nbsp;Complete capability inventory", S_H1))
A(Paragraph(
    "Every measurement surface in the system, what each one supports, and what it cannot establish. Optional "
    "branches are unavailable until their exact pinned artifact and runtime are configured; when they are "
    "unavailable they say so with a reason, and the job still completes.",
    S_BODY))
A(Spacer(1, 5))

A(Paragraph("2.1&nbsp;&nbsp;Measurement and evidence branches", S_H2))
A(table(
    [["BRANCH", "WHAT IT MEASURES", "WHAT IT CANNOT ESTABLISH", "STATUS"],
     ["Media metadata", "Duration, size, content type, from an authoritative ffprobe probe that runs before any model.",
      "Nothing about content or audience.", "Always on"],
     ["Browser content signals", "Opening, continuity, pacing, ending, visual clarity and stability, audio support, from decoded frames and Web Audio.",
      "Watch time, retention, views, engagement, probability.", "Always on (in a capable browser)"],
     ["Measured audio", "PCM and STFT descriptors: RMS, peak, silence fraction, dynamic range, spectral centroid, flatness, flux, and short-window energy peaks.",
      "Speech content, music quality, sentiment, audience response.", "Needs ffmpeg"],
     ["V-JEPA 2.1", "Learned visual-temporal representations over deterministic windows; consistency and change summaries.",
      "Attention, story quality, comprehension, retention.", "Optional, artifact-gated"],
     ["NanoLLaVA keyframes", "Scene, action and shot descriptions over six proportional keyframes plus six fixed hook-window frames.",
      "Full-video semantics, motion, soundtrack, audience behaviour.", "Optional, artifact-gated"],
     ["AST AudioSet", "Uncalibrated sound-event label scores over decoded windows.",
      "A transcript, music appeal, quality, audience response.", "Optional, artifact-gated"],
     ["Transcript (Whisper)", "Which words were spoken and when, via a pinned mlx-whisper revision over the shared 16 kHz mono decode.",
      "Speaker identity, sentiment, tone, meaning, delivery quality.", "Optional, macOS"],
     ["On-screen text", "Recognised glyphs, confidences and bounding boxes over both keyframe passes, via Apple Vision or Tesseract.",
      "Meaning, emphasis, reading order, whether anyone read it.", "Optional, macOS"],
     ["TRIBE v2 cortical", "A hash-verified predicted BOLD tensor and its interval, phase and top-8 parcel descriptors.",
      "Measured viewer activity, an individual brain, any mental state.", "Optional, gated model access"],
     ["Account / trends / competitors", "Nothing yet: no authenticated, timestamped provider is configured.",
      "Any distribution or saturation context.", Paragraph("<font color='#8C3A16'>Not configured</font>", S_CELL)]],
    [78, 168, 122, 74]))
A(Spacer(1, 8))

A(Paragraph("2.2&nbsp;&nbsp;What the system does with that evidence", S_H2))
A(table(
    [["SURFACE", "WHAT IT PRODUCES", "NEEDS A LANGUAGE MODEL?"],
     ["Hook readout", "A timeline of every timed item in the first three seconds and a five-point checklist against declared conventions. Every marker carries the citation it came from.",
      Paragraph("<b>No</b>", S_CELL)],
     ["Insight / Hook Doctor", "Cited, descriptive notes: what the hook contains, observations, hypotheses labelled 'untested heuristic', experiments, proposed rewrite lines, phase commentary, TRIBE notes.",
      "Yes, a pinned local model"],
     ["Variant comparison", "Measured signals across two to six analysed cuts, naming which sits highest and lowest only where values actually differ.",
      Paragraph("<b>No</b>", S_CELL)],
     ["Recut assistant", "Three mechanical edits (trim start, trim end, keep window), rendered and handed straight back for resubmission.",
      Paragraph("<b>No</b>", S_CELL)],
     ["Experiment tracker", "A proposed edit through to measured signal deltas, classified matched / opposite / unmatched / unmeasured.",
      Paragraph("<b>No</b>", S_CELL)],
     ["Comparative context", "Where this clip's measurements rank among the operator's own recent clips, with the corpus declared.",
      Paragraph("<b>No</b>", S_CELL)],
     ["Calibration candidate harness", "A chronologically evaluated single-creator candidate with Brier, log loss, ECE and calibration slope - and an explicit blocker list.",
      Paragraph("<b>No</b>", S_CELL)]],
    [96, 258, 88]))
A(Spacer(1, 6))
A(Paragraph(
    "Six of the seven surfaces work with no language model installed at all. That is deliberate: the model's "
    "job is to explain and to propose, never to be the reason the product is useful.",
    S_BODY))

A(PageBreak())

# ---------------------------------------------------------------- insight lane
A(Paragraph("3.&nbsp;&nbsp;The insight lane, and why its output can be trusted", S_H1))
A(Paragraph(
    "The insight lane is the only part of SignalFrame that produces natural language, and it is the part with "
    "the least authority. It receives derived JSON evidence, never media. It may restate that evidence and "
    "propose experiments. It may do nothing else.",
    S_BODY))

A(Paragraph("3.1&nbsp;&nbsp;Everything is cited, and every citation resolves", S_H2))
A(Paragraph(
    "Each emitted sentence names the evidence it came from, using a grammar of "
    "<font name='Courier' size='8'>lane:/json/pointer</font> with an optional time-window assertion. A "
    "citation into a lane that carries no evidence is unresolvable, and one unresolvable citation rejects the "
    "entire artifact. There is no partial acceptance.",
    S_BODY))
A(Spacer(1, 3))
A(table(
    [["LANE", "CARRIES", "EXAMPLE CITATION"],
     ["measured", "Media metadata, measured audio descriptors, energy peaks, comparative rank", mono("measured:/audio/onset/prePeakSilenceSec")],
     ["nanollava", "Keyframe scene, action and shot descriptions", mono("nanollava:/keyframes/0/parsed/scene")],
     ["ast", "AudioSet label windows and their model scores", mono("ast:/windows/0/labels/0/modelScore")],
     ["vjepa", "Decoded visual windows and summary descriptors", mono("vjepa:/descriptors/temporal_change_mean")],
     ["asr", "Transcript segments with timings", mono("asr:/segments/0/text")],
     ["ocr", "Recognised text blocks per keyframe", mono("ocr:/frames/0/blocks/0/text")],
     ["context", "Creator-declared publishing context", mono("context:/declared/platform")],
     ["tribe", "Cortical intervals, phases and top-8 parcels", mono("tribe:/intervals/0/magnitude")]],
    [58, 210, 174]))
A(Spacer(1, 8))

A(Paragraph("3.2&nbsp;&nbsp;Three enforcement layers, in order of trust", S_H2))
A(table(
    [["LAYER", "WHAT IT DOES", "WHY IT IS NOT ENOUGH ON ITS OWN"],
     ["1. The prompt", "Embeds the forbidden vocabulary and a distilled limits summary, so a compliant model rarely offends.",
      "A model can ignore any instruction, and nothing in the prompt verifies that it did not."],
     ["2. The validator", "Decides, deterministically, whether anything may be published: closed schema, resolvable citations, the numeric-copy rule, sentence-scoped claim lint.",
      "It can only catch what its vocabulary and rules describe."],
     ["3. CI and the judge", "53 red-team fixtures run on every test run, including the valid twins that must keep passing. An env-gated LLM judge is a tripwire only.",
      "The judge is itself a model, and it never gates a request."]],
    [70, 210, 162]))
A(Spacer(1, 6))
A(Paragraph(
    "Only layer 2 publishes or refuses. When a model violates a boundary, layer 1 is what changes - loosening "
    "the validator to make output pass is treated as a defect. When the validator misses something, layer 3 "
    "found it, and the fix is to extend the shared vocabulary and add the fixture that keeps the rule alive. "
    "That has already happened once in this build: a rank restated as 'best-performing' slipped past the lint, "
    "and the vocabulary was extended rather than the fixture softened.",
    S_BODY))
A(Spacer(1, 6))

A(Paragraph("3.3&nbsp;&nbsp;The numeric-copy rule", S_H2))
A(Paragraph(
    "No numeral may appear in insight text unless the evidence that sentence cites actually contains it. A "
    "value may be copied exactly or rounded to two significant figures, and nothing else. Times equal to the "
    "requested window are exempt. One narrow further exemption exists: a numeral inside a proposed rewrite "
    "line, because a suggested script is not an assertion about the clip that exists. The identical numeral in "
    "an observation is still rejected.",
    S_BODY))

A(Paragraph("3.4&nbsp;&nbsp;How a refusal is reported", S_H2))
A(Paragraph(
    "Thirteen reason codes cover every way the lane can decline to publish. Each one is specific enough to act "
    "on, and rejections are persisted so an operator can read the offending sentence.",
    S_BODY))
A(Spacer(1, 3))
A(table(
    [["REASON CODE", "MEANING", "REASON CODE", "MEANING"],
     [mono("bundle_unavailable"), "Upstream evidence missing or incomplete", mono("missing_citation"), "An item carries no citation"],
     [mono("provider_unavailable"), "Model disabled, unverified or absent", mono("citation_malformed"), "The citation does not parse"],
     [mono("provider_error"), "The model ran and returned nothing usable", mono("citation_unresolvable"), "It parses but resolves to nothing"],
     [mono("output_not_json"), "Not a single strict JSON object", mono("numeric_not_in_evidence"), "A numeral the evidence lacks"],
     [mono("output_too_large"), "Beyond the configured byte limit", mono("claim_boundary_violation"), "An outcome or mental-state claim"],
     [mono("schema_invalid"), "Types, enums, bounds or references failed", mono("unknown_field"), "A key the schema does not define"],
     [mono("server_owned_field"), "The model tried to set a server-owned field", "", ""]],
    [92, 130, 92, 128]))

A(PageBreak())

# ---------------------------------------------------------------- no gimmick
A(Paragraph("4.&nbsp;&nbsp;No gimmick: what this system deliberately will not do", S_H1))
A(Paragraph(
    "Most tools in this category are built the other way round: produce a confident number, then decorate it. "
    "SignalFrame is built to make that impossible. This section is the list of things a reasonable person "
    "might expect to find, and the reason each one is absent.",
    S_BODY))
A(Spacer(1, 5))
A(table(
    [["WHAT IS ABSENT", "WHY"],
     ["A hook score out of 100",
      "The moment a single number exists, every cited sentence beside it becomes decoration and the number becomes a claim that cannot be defended. No such score exists anywhere in the product."],
     ["Retention, watch-time or virality predictions",
      "Each would need a separately trained, target-specific head with a pinned event, denominator, platform, horizon, population and locale; creator-disjoint chronological evaluation; and locked calibration evidence. Zero heads currently pass that gate."],
     ["TRIBE presented as attention or engagement",
      "It predicts average-subject cortical BOLD, offset five seconds, vision-only on Apple silicon, never evaluated against a creator outcome. Text that reverse-infers from it is rejected automatically."],
     ["A confidence percentage",
      "What the interface calls coverage means input coverage - which measurements were available. It is never displayed as statistical confidence in an outcome."],
     ["Fallback values when a model is missing",
      "A missing model, an unverified artifact, a failed hash or a WebGL failure produces an explicit unavailable state with a reason. No branch is silently substituted and no weight is redistributed."],
     ["Invented numbers in generated text",
      "Every numeral must be a copy of cited evidence. A model that writes a plausible figure has its whole response rejected."],
     ["Cloud processing of your media",
      "Raw video, audio, frames and tensors never leave the machine. Only derived JSON evidence may go to a remote model, only when an operator sets two separate switches, and a test walks the import graph to prove that code path cannot reach an upload directory."],
     ["Outcome data influencing the advice",
      "Imported post-publish numbers are stored as labels for a future calibration head. There is no outcomes lane, a citation naming one fails to parse, and the module is unreachable from the assembler, validator, prompt and providers."]],
    [128, 314]))
A(Spacer(1, 8))
A(callout(
    "THE HONEST TEST OF ANY TOOL LIKE THIS",
    "Ask what it does when a model is missing, when a file will not decode, or when it simply does not know. "
    "SignalFrame answers with a named unavailable state and a reason, in every one of those cases. That is "
    "the whole design, and everything else in this document follows from it.",
    ACCENT, ACCENT_WASH))

A(Spacer(1, 10))
A(Paragraph("4.1&nbsp;&nbsp;Two things that are honest but easy to misread", S_H2))
A(Paragraph(
    "<b>Modeled Engagement and Virality Outlook.</b> The interface contains two transparent, hand-weighted "
    "indices assembled from measured content features. They are directional product heuristics for comparing "
    "your own clips inside this tool. They are not calibrated probabilities, expected views, or platform "
    "guarantees, and they are computationally separate from TRIBE. They are the closest thing in the product "
    "to a score, and they are labelled as heuristics wherever they appear.",
    S_BODY))
A(Paragraph(
    "<b>Checklist thresholds.</b> The hook checklist compares measurements against round numbers this project "
    "chose - 0.8 seconds of opening silence, speech by 1.5 seconds. Every threshold is labelled a declared "
    "convention. None has been evaluated against an audience outcome. A flag is a prompt to look, not a defect.",
    S_BODY))

A(PageBreak())

# ---------------------------------------------------------------- creators
A(Paragraph("5.&nbsp;&nbsp;How a content creator uses this", S_H1))
A(Paragraph(
    "The loop is: measure the clip, read what is actually there, form one hypothesis, change one thing, "
    "measure again. SignalFrame's job is to make each of those steps concrete and to stop you fooling "
    "yourself at every one of them.",
    S_LEAD))
A(Spacer(1, 4))
A(table(
    [["STEP", "WHAT YOU DO", "WHAT THE SYSTEM GIVES BACK"],
     ["1. Measure", "Upload a 10-60 second cut and add your publishing context: platform, caption, topic, intended post time.",
      "A durable, resumable evidence job. Every branch that ran, and every branch that did not, with its reason."],
     ["2. See the opening", "Open the hook readout. No model needed.",
      "A three-second timeline: when you first speak, when sound first peaks, when text appears, what is on screen, and how much changes visually. Plus five checks against declared conventions."],
     ["3. Read the notes", "Run Hook Doctor if a local model is installed.",
      "What the hook contains, observations, hypotheses labelled 'untested heuristic', experiments ranked by effort, and up to four proposed replacement opening lines."],
     ["4. Check the claim", "Click any citation chip.",
      "The exact measured value that sentence came from, and the player seeks to that moment. If it does not resolve, the chip says so."],
     ["5. Change one thing", "Pick an experiment. Use the recut assistant for a mechanical trim, or re-record the opening.",
      "A recut clip handed straight back to you. Nothing is retained."],
     ["6. Measure again", "Submit the new cut and attach it to the experiment.",
      "Measured deltas on the signals the hypothesis named, each one matched, opposite, unmatched, or unmeasured."],
     ["7. Compare cuts", "Record two or three openings and compare them.",
      "Every shared measured signal side by side, with which cut sits highest and lowest - only where they actually differ."]],
    [56, 176, 210]))
A(Spacer(1, 8))
A(Paragraph("What a creator should take from it, and what they should not", S_H2))
A(Paragraph(
    "<b>Take:</b> a precise account of your own opening. You have 1.4 seconds of dead air before you speak. "
    "There is no on-screen text on the first frame. Nothing changes visually for two seconds. Your opening is "
    "the quietest of your last twenty clips. These are facts about your file, and every one of them is "
    "actionable tonight.",
    S_BODY))
A(Paragraph(
    "<b>Do not take:</b> that fixing any of them will improve your results. Nothing in this system has ever "
    "seen an audience. A hypothesis here is a starting point for a real test on a real platform with a metric "
    "you decided in advance. The tool is honest about this everywhere, and you should hold it to that.",
    S_BODY))
A(Spacer(1, 6))
A(callout(
    "THE HONEST SCOPE OF THE ADVICE",
    "The model reads derived JSON, never your video, and its only visual input is still frames. It can tell "
    "you your hook is silent for 1.4 seconds. It cannot tell you the framing is unflattering or the lighting "
    "is flat. Advice here is structurally about timing, speech, on-screen text and audio - and the product is "
    "stronger for saying so than for being quietly bad at the rest.",
    ACCENT, ACCENT_WASH))

A(PageBreak())

# ---------------------------------------------------------------- agencies
A(Paragraph("6.&nbsp;&nbsp;How a marketing agency uses this", S_H1))
A(Paragraph(
    "An agency's problem is different from a creator's. The creator wants to improve one clip. The agency "
    "needs consistency across many, a defensible reason for each edit, and something to show a client that is "
    "not an opinion. SignalFrame is useful precisely because its outputs are checkable.",
    S_LEAD))
A(Spacer(1, 4))
A(table(
    [["USE", "HOW IT WORKS", "WHY IT HOLDS UP"],
     ["Pre-publish QC",
      "Run the deterministic hook readout on every deliverable before it ships. Five checks, no model, seconds per clip.",
      "The same declared conventions applied identically to every asset. Nothing depends on who reviewed it or how tired they were."],
     ["Defensible creative notes",
      "Hook Doctor output where a local model is installed: every sentence carries a citation to the measurement behind it.",
      "A note you can defend line by line. 'Because the first sound peak is at 1.4 s' beats 'because it feels slow'."],
     ["A/B evidence for edits",
      "Two or three cuts of the same idea, measured side by side on the shared metric list.",
      "You can show a client exactly what the edit changed in the file - not what you hope it will do to the numbers."],
     ["House-style consistency",
      "Comparative context ranks a clip's measurements against the studio's own recent output on that machine.",
      "'Quietest opening of the last twenty' is a measured statement about your own library, with the corpus size declared."],
     ["Client-safe language",
      "The claim-boundary validator refuses outcome and mental-state claims in generated text, on every lane.",
      "Generated copy that reaches a client cannot promise views, retention, or engagement - the system will not produce it."],
     ["Privacy-sensitive accounts",
      "Everything runs locally by default. The remote model is off unless two separate switches are set.",
      "Client footage never leaves the machine. Only derived JSON can, and only on an explicit opt-in."],
     ["Evidence retention",
      "Results and derived evidence persist under a private directory until an operator removes them.",
      "You can re-open the exact evidence bundle a note was generated from, months later, and check it."]],
    [92, 176, 174]))
A(Spacer(1, 8))

A(Paragraph("What an agency must not sell with it", S_H2))
A(Paragraph(
    "This is worth stating plainly, because it is the one way an agency could misuse the tool badly. "
    "SignalFrame produces no performance prediction, and presenting any of its output as one would be a claim "
    "the system itself refuses to make. There is no forecast of views, retention, completion, shares, or "
    "virality available in this build - not withheld pending a licence, but absent because no head has passed "
    "the evaluation gate. Cortical output in particular must be described as predicted average-subject "
    "cortical BOLD; describing it as attention or emotional response is exactly the claim the validator exists "
    "to block.",
    S_BODY))
A(Spacer(1, 4))
A(callout(
    "WHAT AN AGENCY CAN HONESTLY CLAIM",
    "That every creative note is traceable to a measurement of the client's own file; that the same checks "
    "were applied to every deliverable; that edits were chosen from measured evidence rather than taste alone; "
    "and that the client's footage never left the machine. Those are real, unusual, and defensible. A "
    "performance promise is none of those things.",
    ACCENT, ACCENT_WASH))

A(PageBreak())

# ---------------------------------------------------------------- api + ops
A(Paragraph("7.&nbsp;&nbsp;Interface surface", S_H1))
A(Paragraph("Twenty-one routes across three services. All insight and forecast responses are private and no-store.", S_BODY))
A(Spacer(1, 4))
A(table(
    [["METHOD", "ROUTE", "PURPOSE"],
     ["GET", mono("/api/tribe/v1/status"), "Live cortical worker state and pinned provenance"],
     ["POST", mono("/api/tribe/v1/predict"), "Run validated cortical inference for one video"],
     ["GET", mono("/api/tribe/v1/results/{id}/manifest.json"), "Retrieve a result manifest"],
     ["GET", mono("/api/tribe/v1/results/{id}/frames.f32"), "Retrieve the verified tensor"],
     ["GET", mono("/api/forecast/v1/status"), "Score-free capability and provider contract"],
     ["POST", mono("/api/forecast/v1/jobs"), "Submit a private 10-60 second evidence job"],
     ["GET", mono("/api/forecast/v1/jobs/{id}"), "Poll durable job state and real stages"],
     ["GET", mono("/api/forecast/v1/results/{id}"), "Retrieve an atomically published result"],
     ["GET", mono("/api/insight/v1/status"), "Provider readiness, pinned model, prompt version"],
     ["POST", mono("/api/insight/v1/hook-readout"), "Deterministic timeline and checklist (no model)"],
     ["POST", mono("/api/insight/v1/generate"), "Generate one cited, validated insight artifact"],
     ["GET", mono("/api/insight/v1/results/{id}"), "Retrieve a published artifact"],
     ["GET", mono("/api/insight/v1/results/{id}/evidence"), "The exact evidence an artifact cites"],
     ["GET", mono("/api/insight/v1/rejections/{id}"), "A persisted rejection, with the offending sentence"],
     ["POST", mono("/api/insight/v1/variants"), "Measured signals across several analysed cuts"],
     ["GET", mono("/api/insight/v1/recut/operations"), "Which mechanical recuts are available"],
     ["POST", mono("/api/insight/v1/recut"), "Render one recut and hand the clip back"],
     ["POST", mono("/api/insight/v1/experiments"), "Track one proposed experiment"],
     ["POST", mono("/api/insight/v1/experiments/{id}/variant"), "Attach a variant and measure the deltas"],
     ["GET", mono("/api/insight/v1/experiments"), "Tracked experiments, newest first"],
     ["POST", mono("/api/insight/v1/outcomes"), "Import creator-declared post-publish labels"]],
    [42, 208, 192]))
A(Spacer(1, 10))

A(Paragraph("8.&nbsp;&nbsp;Operating requirements and limits", S_H1))
A(table(
    [["AREA", "WHAT YOU NEED TO KNOW"],
     ["Platform", "macOS on Apple silicon is the primary target: the MLX and MPS paths are Mac-only. There is no CUDA-only code path anywhere. Optional acceleration is guarded and falls back explicitly."],
     ["Base install", "Node 20+, Python 3.11, ffmpeg and ffprobe, and a hardware-accelerated browser for the 3D viewer. The interface, measured evidence and job service run on this alone."],
     ["Optional models", "V-JEPA 2.1, NanoLLaVA, AST, Whisper, Vision or Tesseract, and the insight model are each separately installed and separately pinned. None is bundled. Each is unavailable until its exact revision verifies."],
     ["Model licences", "TRIBE v2 code and weights are CC BY-NC 4.0: that path is research or non-commercial unless you obtain separate permission. Every other model keeps its own upstream licence and dataset terms."],
     ["Privacy", "Uploads are bounded, private, no-store, and deleted after the job. Cortical tensors, thumbnails and result JSON persist under a private directory until removed. Treat them as sensitive."],
     ["Security", "The local server has no authentication layer. Bind it to localhost for personal use. Authentication, TLS, tenant isolation, quotas and retention controls are required before any network exposure - and are a hard prerequisite before storing outcome data for more than one person."],
     ["Not for", "Medical, diagnostic, neuroimaging, employment, insurance, educational, legal or law-enforcement use. This is not a medical device and not a mental-state classifier."]],
    [72, 370]))
A(Spacer(1, 10))

A(Paragraph("9.&nbsp;&nbsp;How to verify every claim in this document", S_H1))
A(Paragraph(
    "Nothing here needs to be taken on trust. The whole system is checkable from a clean checkout.",
    S_BODY))
A(Spacer(1, 3))
A(table(
    [["TO CHECK", "RUN"],
     ["Every contract, boundary and fail-closed path", mono("python -m unittest discover -s backend/tests -v")],
     ["Frontend contracts and presentation rules", mono("npm run build &amp;&amp; npm test")],
     ["That no component hard-codes a forbidden claim", mono("node --test tests/insight-claim-terms.test.mjs")],
     ["That the approval table is still empty", mono("grep -A3 APPROVED_TARGET_CONTRACTS backend/forecast/calibrated_heads.py")],
     ["What each branch reports right now", mono("curl --fail http://127.0.0.1:8000/api/forecast/v1/status")],
     ["Whether an insight model is available and pinned", mono("curl --fail http://127.0.0.1:8000/api/insight/v1/status")]],
    [190, 252]))
A(Spacer(1, 12))
A(Rule(W, 1.2, INK))
A(Spacer(1, 6))
A(Paragraph(
    "SignalFrame - sole project author Karan Chandra Dey. Original source released under the MIT Licence. "
    "Third-party model code, weights, data and assets remain under their own licences and are not relicensed. "
    "See NOTICE.md for required attributions, and docs/SCIENTIFIC_LIMITS.md for the full statement of what "
    "every output can and cannot support.",
    S_SMALL))
