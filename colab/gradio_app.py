"""A Gradio front-end for SignalFrame, for Colab and other Linux hosts.

This is an alternative interface, not a replacement for the React app. It talks
to the same FastAPI service over HTTP, so every boundary the backend enforces
applies here unchanged: unavailable lanes report their reason, generated text is
validated before it is shown, and nothing is substituted when a branch is
missing.

Run it with::

    python -m colab.gradio_app            # starts the API too, then the UI
    SIGNALFRAME_API=http://127.0.0.1:8000 python -m colab.gradio_app  # reuse one

On Colab, ``share=True`` gives you a public URL. Two lanes cannot run there at
all: the local insight model (mlx-lm) and the transcript branch (mlx-whisper)
are Apple-silicon only. The interface says so rather than hiding it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import gradio as gr
import httpx


API = os.environ.get("SIGNALFRAME_API", "http://127.0.0.1:8000").rstrip("/")
API_PORT = int(os.environ.get("SIGNALFRAME_API_PORT", API.rsplit(":", 1)[-1] or 8000))
TIMEOUT = httpx.Timeout(600.0, connect=10.0)

# Kept in module state so one tab can hand a result id to another without the
# creator copying identifiers around by hand.
RESULTS: list[dict[str, str]] = []


# --------------------------------------------------------------------------
# Backend plumbing
# --------------------------------------------------------------------------


def start_backend_in_background(port: int = API_PORT) -> None:
    """Start the real FastAPI app in this process, for a single-cell Colab run."""

    import uvicorn

    from backend.app import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    # A 200 is required, not merely a response: a server that binds the port but
    # answers 404 or 500 on its own status route is not ready, and reporting it as
    # started would surface as an interface where every tab fails separately.
    for _ in range(120):
        try:
            probe = httpx.get(
                f"http://127.0.0.1:{port}/api/insight/v1/status", timeout=2.0
            )
        except Exception:
            time.sleep(0.5)
            continue
        if probe.status_code == 200:
            return
        raise RuntimeError(
            f"the API answered {probe.status_code} on its own status route. "
            "Its routers did not register — check that fastapi matches the pin "
            "in backend/requirements-local.txt."
        )
    raise RuntimeError("the SignalFrame API did not start")


def _get(path: str) -> Any:
    with httpx.Client(timeout=TIMEOUT) as client:
        return client.get(f"{API}{path}").json()


def _post(path: str, payload: dict[str, Any]) -> Any:
    with httpx.Client(timeout=TIMEOUT) as client:
        return client.post(f"{API}{path}", json=payload).json()


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _result_choices() -> list[str]:
    return [f"{item['label']}  ·  {item['id']}" for item in RESULTS]


def _id_from_choice(choice: str | None) -> str | None:
    if not choice:
        return None
    return choice.rsplit("·", 1)[-1].strip() or None


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------


def read_status() -> tuple[str, str]:
    try:
        forecast = _get("/api/forecast/v1/status")
        insight = _get("/api/insight/v1/status")
        tribe = _get("/api/tribe/v1/status")
    except Exception as exc:
        return f"Could not reach the API at {API}: {exc}", ""

    rows = ["| Lane | State | Why |", "| --- | --- | --- |"]
    for key, branch in forecast["branches"].items():
        if key.startswith("browserLocal"):
            continue
        reason = (branch.get("reason") or "").replace("|", "/")
        rows.append(f"| `{key}` | **{branch['state']}** | {reason[:150]} |")
    rows.append(
        f"| `tribeCortical` | **{tribe['state']}** | "
        f"configured={tribe['configured']}, mode={tribe['model']['inferenceMode']} |"
    )
    provider = insight["provider"]
    rows.append(
        f"| `insight model` | **{insight['state']}** | {(provider.get('reason') or '')[:150]} |"
    )

    notes = [
        "### What can and cannot run here",
        "",
        "- **`measuredAudio`** needs only `ffmpeg`, so it works anywhere.",
        "- **`ocr`** works on Linux through the `pytesseract` fallback "
        "(`apt-get install tesseract-ocr`); Apple Vision is macOS-only.",
        "- **`tribeCortical`**, **`vjepa21`** and **`audioModel`** can run on a "
        "Colab GPU once their pinned, hash-verified artifacts are installed.",
        "- **`asr`** (mlx-whisper) and **`semanticModel`** (mlx_vlm) are "
        "Apple-silicon only and cannot run on Colab at all.",
        "- The **local insight model** (mlx-lm) is Apple-silicon only. On Colab, "
        "set `INSIGHT_PROVIDER=anthropic`, `INSIGHT_CLOUD_ENABLED=true` and "
        "`ANTHROPIC_API_KEY` to use the remote provider instead.",
        "",
        "A lane reporting unavailable is the design working: it names what it "
        "needs rather than substituting anything.",
    ]
    return "\n".join(rows), "\n".join(notes)


# --------------------------------------------------------------------------
# Analyse
# --------------------------------------------------------------------------


def analyse(video_path: str | None, platform: str, topic: str, caption: str):
    if not video_path:
        return "Choose a clip first.", "", gr.update(choices=_result_choices())

    context = {
        "schemaVersion": "creator-forecast-context/1",
        "platform": (platform or "unspecified").strip(),
        "topic": (topic or "").strip() or None,
        "caption": (caption or "").strip() or None,
    }
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with open(video_path, "rb") as handle:
                response = client.post(
                    f"{API}/api/forecast/v1/jobs",
                    files={"video": (os.path.basename(video_path), handle, "video/mp4")},
                    data={"context": json.dumps(context)},
                )
            if response.status_code >= 400:
                return f"The job was refused: {_pretty(response.json())}", "", gr.update()
            job = response.json()
            job_id = job["jobId"]

            state = job["state"]
            for _ in range(600):
                time.sleep(1.0)
                state = client.get(f"{API}/api/forecast/v1/jobs/{job_id}").json()["state"]
                if state in {"complete", "failed"}:
                    break
            if state != "complete":
                detail = client.get(f"{API}/api/forecast/v1/jobs/{job_id}").json()
                return f"The job ended `{state}`.\n\n```\n{_pretty(detail)}\n```", "", gr.update()

            result = client.get(f"{API}/api/forecast/v1/results/{job_id}").json()
    except Exception as exc:
        return f"The job could not run: {exc}", "", gr.update()

    RESULTS.insert(0, {"id": job_id, "label": os.path.basename(video_path)[:40]})
    del RESULTS[8:]

    lines = [
        f"**Result `{job_id}`** — duration {result['input']['durationSeconds']} s",
        "",
        "| Branch | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for key, provider in sorted(result["evidence"]["optionalProviders"].items()):
        if provider["status"] == "available":
            features = provider["result"].get("features", {})
            detail = ", ".join(f"{k.split('.')[-1]}={v:.4g}" for k, v in list(features.items())[:3])
        else:
            detail = (provider.get("reason") or "")[:110]
        lines.append(f"| `{key}` | **{provider['status']}** | {detail.replace('|', '/')} |")
    lines += [
        "",
        "Every behavioral head remains unavailable: no approved, calibrated head "
        "is installed, so no probability is produced anywhere.",
    ]
    return "\n".join(lines), job_id, gr.update(choices=_result_choices(), value=_result_choices()[0])


# --------------------------------------------------------------------------
# Hook readout
# --------------------------------------------------------------------------


def hook_readout(choice: str | None):
    result_id = _id_from_choice(choice)
    if not result_id:
        return "Analyse a clip first.", []
    payload = _post("/api/insight/v1/hook-readout", {"forecastResultId": result_id})
    if payload.get("unavailable"):
        return f"**{payload['reasonCode']}** — {payload['detail']}", []

    rows = [
        [f"{m['startSec']:.2f}s", m["kind"], m["label"][:70], m["citation"]]
        for m in payload["timeline"]
    ]
    lines = [
        f"### The first {payload['windowSeconds'][1]} seconds, measured",
        "",
        f"{payload['flaggedCount']} flagged · {payload['unmeasuredCount']} not measured",
        "",
        "| | Check | What was measured |",
        "| --- | --- | --- |",
    ]
    mark = {"flagged": "**worth a look**", "clear": "clear", "unmeasured": "_not measured_"}
    for check in payload["checklist"]:
        lines.append(
            f"| {mark.get(check['status'], check['status'])} | {check['label']} | "
            f"{check['detail'].replace('|', '/')} |"
        )
    lines += ["", f"_{payload['limits']}_"]
    return "\n".join(lines), rows


# --------------------------------------------------------------------------
# Hook Doctor
# --------------------------------------------------------------------------


def generate_insight(choice: str | None, hook_only: bool):
    result_id = _id_from_choice(choice)
    if not result_id:
        return "Analyse a clip first."
    payload = _post(
        "/api/insight/v1/generate",
        {"forecastResultId": result_id, "hookOnly": bool(hook_only)},
    )
    if payload.get("unavailable"):
        detail = payload["detail"]
        if isinstance(detail, dict):
            detail = (
                f"term “{detail.get('term')}” in “{detail.get('sentence')}” "
                f"at `{detail.get('itemPath')}`"
            )
        return (
            f"### Nothing was published\n\n**`{payload['reasonCode']}`** — {detail}\n\n"
            "That is the lane refusing to publish, not an error to work around. "
            "If the reason is `provider_unavailable`, configure a provider: on "
            "Colab that means the remote one."
        )

    report = payload["hookReport"]
    out = [f"### Hook report — {report['windowSeconds'][0]}–{report['windowSeconds'][1]}s", ""]

    def block(title: str, items: list[dict[str, Any]], key: str = "text") -> None:
        if not items:
            return
        out.append(f"**{title}**")
        out.append("")
        for item in items:
            citations = " ".join(f"`{c}`" for c in item.get("citations", []))
            out.append(f"- {item[key]}  \n  <sub>{citations}</sub>")
        out.append("")

    block("What the hook contains", report["whatTheHookContains"])
    block("Observations", report["observations"])
    if report["hypotheses"]:
        out += ["**Hypotheses** — every one an untested heuristic", ""]
        for item in report["hypotheses"]:
            citations = " ".join(f"`{c}`" for c in item["citations"])
            out.append(f"- _{item['label']}_ — {item['text']}  \n  <sub>{citations}</sub>")
        out.append("")
    if report["experiments"]:
        out += ["**Experiments**", ""]
        for item in report["experiments"]:
            shift = ", ".join(
                f"`{s['metricPath']}` {s['direction']}" for s in item["expectedSignalShift"]
            )
            out.append(f"- **{item['effort']} effort** — {item['edit']}  \n  <sub>{shift}</sub>")
        out.append("")
    block("Proposed opening lines", payload.get("hookRewrites", []), key="line")
    block("Phase commentary", payload.get("phaseCommentary", []))
    block("TRIBE notes", payload.get("tribeNotes", []))

    provenance = payload["provenance"]
    out += [
        "---",
        f"_{payload['limits']}_",
        "",
        f"<sub>provider `{provenance['provider']}` · model `{provenance['modelId']}` · "
        f"revision `{provenance['modelRevision']}` · prompt `{provenance['promptTemplateId']}` · "
        f"behavioralOutcome `{payload['behavioralOutcome']}`</sub>",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Compare cuts
# --------------------------------------------------------------------------


def compare_cuts(first: str | None, second: str | None):
    ids = [_id_from_choice(first), _id_from_choice(second)]
    if not all(ids):
        return "Pick two analysed cuts.", []
    if ids[0] == ids[1]:
        return "Pick two different cuts.", []
    payload = _post("/api/insight/v1/variants", {"resultIds": ids})
    if payload.get("unavailable"):
        return f"**{payload['reasonCode']}** — {payload['detail']}", []

    labels = {v["resultId"]: v["label"] for v in payload["variants"]}
    rows = []
    for metric in payload["metrics"]:
        values = {e["resultId"]: e["value"] for e in metric["values"]}
        rows.append([
            metric["metricPath"].split("/")[-1],
            f"{values.get(ids[0], float('nan')):.5g}",
            f"{values.get(ids[1], float('nan')):.5g}",
            "moved" if metric["differs"] else "no change",
            labels.get(metric["lowestResultId"], "—") if metric["differs"] else "—",
        ])
    summary = (
        f"### {payload['differingMetricCount']} measured signals moved\n\n"
        f"_{payload['limits']}_"
    )
    return summary, rows


# --------------------------------------------------------------------------
# Recut
# --------------------------------------------------------------------------


def recut(video_path: str | None, operation: str, duration: float, seconds: float):
    if not video_path:
        return None, "Choose the clip to recut."
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            with open(video_path, "rb") as handle:
                response = client.post(
                    f"{API}/api/insight/v1/recut",
                    files={"video": (os.path.basename(video_path), handle, "video/mp4")},
                    data={
                        "operation": operation,
                        "durationSeconds": str(duration),
                        "seconds": str(seconds),
                    },
                )
    except Exception as exc:
        return None, f"The recut failed: {exc}"
    if response.status_code >= 400:
        try:
            body = response.json()
            return None, f"**{body.get('reasonCode')}** — {body.get('detail')}"
        except Exception:
            return None, f"The recut was refused ({response.status_code})."

    out_path = "/tmp/signalframe-recut.mp4"
    with open(out_path, "wb") as handle:
        handle.write(response.content)
    plan = json.loads(response.headers.get("X-Recut-Plan", "{}"))
    note = (
        f"Kept {plan.get('startSec')}s to {plan.get('endSec')}s — "
        f"{plan.get('resultDurationSec')}s.\n\n_{plan.get('limits', '')}_\n\n"
        "Analyse this file in the first tab to measure what the edit changed."
    )
    return out_path, note


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


def build_interface() -> gr.Blocks:
    # No theme is set on purpose: Gradio moved that argument from Blocks to
    # launch() in 6.0, and Colab pins whatever version it pins. The default
    # theme renders on every version.
    with gr.Blocks(title="SignalFrame") as demo:
        gr.Markdown(
            "# SignalFrame\n"
            "Measures a short clip, describes what it measured, and helps you turn that "
            "into an experiment. **It does not predict how an audience will respond**, and "
            "no behavioral head is installed."
        )

        with gr.Tab("1 · Analyse"):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Clip (10–60 seconds)")
                    platform = gr.Textbox(label="Platform", value="reels")
                    topic = gr.Textbox(label="Topic")
                    caption = gr.Textbox(label="Caption")
                    run = gr.Button("Run evidence job", variant="primary")
                with gr.Column():
                    evidence_out = gr.Markdown()
                    result_id_out = gr.Textbox(label="Result id", interactive=False)

        with gr.Tab("2 · Hook readout (no model needed)"):
            gr.Markdown(
                "Computed entirely from measurements. Every marker carries the citation "
                "it came from, and a check that cannot be measured says so rather than "
                "passing."
            )
            readout_pick = gr.Dropdown(label="Analysed clip", choices=[], interactive=True)
            readout_go = gr.Button("Read the first 3 seconds", variant="primary")
            readout_md = gr.Markdown()
            readout_table = gr.Dataframe(
                headers=["At", "Kind", "What", "Citation"], label="Timeline", wrap=True
            )

        with gr.Tab("3 · Hook Doctor"):
            gr.Markdown(
                "Cited, descriptive notes from a language model. Requires a configured "
                "provider — on Colab that is the remote one, since mlx-lm is "
                "Apple-silicon only."
            )
            insight_pick = gr.Dropdown(label="Analysed clip", choices=[], interactive=True)
            hook_only = gr.Checkbox(label="Hook window only (first 3 seconds)", value=True)
            insight_go = gr.Button("Generate", variant="primary")
            insight_md = gr.Markdown()

        with gr.Tab("4 · Compare cuts"):
            gr.Markdown(
                "Measured signals across two analysed cuts. Higher is not better and "
                "lower is not worse: neither cut has been shown to an audience."
            )
            with gr.Row():
                cut_a = gr.Dropdown(label="Cut A", choices=[], interactive=True)
                cut_b = gr.Dropdown(label="Cut B", choices=[], interactive=True)
            compare_go = gr.Button("Compare", variant="primary")
            compare_md = gr.Markdown()
            compare_table = gr.Dataframe(
                headers=["Signal", "Cut A", "Cut B", "Change", "Lower in"], wrap=True
            )

        with gr.Tab("5 · Recut"):
            gr.Markdown(
                "Three mechanical edits. The clip comes straight back — nothing is kept "
                "— and a recut measures nothing until you analyse the result."
            )
            with gr.Row():
                with gr.Column():
                    recut_video = gr.Video(label="Clip")
                    operation = gr.Radio(
                        ["trim_start", "trim_end"], value="trim_start", label="Operation"
                    )
                    duration = gr.Number(label="Source duration (s)", value=14.0)
                    seconds = gr.Number(label="Seconds to trim", value=1.4)
                    recut_go = gr.Button("Render", variant="primary")
                with gr.Column():
                    recut_file = gr.File(label="Recut clip")
                    recut_md = gr.Markdown()

        with gr.Tab("Status"):
            status_go = gr.Button("Refresh", variant="primary")
            status_md = gr.Markdown()
            status_notes = gr.Markdown()

        run.click(
            analyse,
            [video, platform, topic, caption],
            [evidence_out, result_id_out, readout_pick],
        ).then(
            lambda: (gr.update(choices=_result_choices()),) * 3,
            None,
            [insight_pick, cut_a, cut_b],
        )
        readout_go.click(hook_readout, readout_pick, [readout_md, readout_table])
        insight_go.click(generate_insight, [insight_pick, hook_only], insight_md)
        compare_go.click(compare_cuts, [cut_a, cut_b], [compare_md, compare_table])
        recut_go.click(recut, [recut_video, operation, duration, seconds], [recut_file, recut_md])
        status_go.click(read_status, None, [status_md, status_notes])
        demo.load(read_status, None, [status_md, status_notes])

    return demo


def main() -> None:
    if os.environ.get("SIGNALFRAME_START_API", "true").lower() not in {"0", "false", "no"}:
        start_backend_in_background()
    share = os.environ.get("SIGNALFRAME_SHARE", "false").lower() in {"1", "true", "yes"}
    build_interface().launch(share=share, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
