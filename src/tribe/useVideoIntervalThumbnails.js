import { useEffect, useMemo, useState } from "react";
import { extractVideoIntervalThumbnails, fetchVerifiedVideoThumbnails } from "./videoThumbnails.js";

function stableIntervalKey(intervals) {
  return intervals
    .map((interval) => [interval.index, interval.startTime])
    .sort((left, right) => left[0] - right[0])
    .map(([index, startTime]) => `${index}:${startTime}`)
    .join("|");
}

export function useVideoIntervalThumbnails(sourceUrl, intervals, verifiedThumbnails) {
  const intervalKey = useMemo(() => stableIntervalKey(intervals), [intervals]);
  const verifiedKey = `${verifiedThumbnails?.status || ""}|${verifiedThumbnails?.items?.map((item) => `${item.index}:${item.sha256}`).join("|") || ""}|${verifiedThumbnails?.unavailableItems?.map((item) => `${item.index}:${item.reason}`).join("|") || ""}`;
  const hasVerifiedServerSource = verifiedThumbnails?.status === "available" || verifiedThumbnails?.status === "partial";
  const hasLocalSource = !hasVerifiedServerSource && Boolean(sourceUrl);
  const [state, setState] = useState({ sourceAvailable: hasVerifiedServerSource || hasLocalSource, sourceKind: null, byIndex: {} });

  useEffect(() => {
    const serverAvailable = verifiedThumbnails?.status === "available" || verifiedThumbnails?.status === "partial";
    const useLocalSource = !serverAvailable && Boolean(sourceUrl);
    if (!serverAvailable && !useLocalSource) {
      setState({ sourceAvailable: false, sourceKind: verifiedThumbnails?.status === "unavailable" ? "server-unavailable" : null, byIndex: {} });
      return undefined;
    }

    const controller = new AbortController();
    const objectUrls = [];
    const initial = Object.fromEntries(intervals.map((interval) => [interval.index, { status: "loading" }]));
    const sourceKind = serverAvailable ? "verified-server" : "local-session";
    setState({ sourceAvailable: true, sourceKind, byIndex: initial });

    const extraction = serverAvailable
      ? fetchVerifiedVideoThumbnails({ thumbnails: verifiedThumbnails, signal: controller.signal,
          onThumbnail: receiveThumbnail, onUnavailable: receiveUnavailable })
      : extractVideoIntervalThumbnails({ sourceUrl, intervals, signal: controller.signal,
          onThumbnail: receiveThumbnail, onUnavailable: receiveUnavailable });

    function receiveThumbnail(thumbnail) {
      if (controller.signal.aborted) return;
      const url = URL.createObjectURL(thumbnail.blob);
      objectUrls.push(url);
      setState((current) => ({
        ...current,
        byIndex: { ...current.byIndex, [thumbnail.index]: { ...thumbnail, blob: undefined, status: "ready", url } },
      }));
    }

    function receiveUnavailable(thumbnail) {
      if (controller.signal.aborted) return;
      setState((current) => ({
        ...current,
        byIndex: { ...current.byIndex, [thumbnail.index]: { ...thumbnail, status: "unavailable" } },
      }));
    }

    extraction.catch((error) => {
      if (controller.signal.aborted || error?.name === "AbortError") return;
      setState((current) => ({
        ...current,
        byIndex: Object.fromEntries(intervals.map((interval) => [
          interval.index,
          ["ready", "unavailable"].includes(current.byIndex[interval.index]?.status)
            ? current.byIndex[interval.index]
            : { status: "error", message: error?.message || "Thumbnail extraction failed." },
        ])),
      }));
    });

    return () => {
      controller.abort();
      objectUrls.forEach((url) => URL.revokeObjectURL(url));
    };
  // intervalKey deliberately stabilizes extraction when only the visible ranking order changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceUrl, intervalKey, verifiedKey]);

  return state;
}
