import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { loadFsaverage5Mesh } from "../tribe/meshLoader.js";
import {
  predictionFrameAtMediaTime,
  writeNeutralColors,
  writeScientificColors,
} from "../tribe/frame.js";
import { applyParcelFocus } from "../tribe/atlas.js";
import { TRIBE_VERTEX_COUNT } from "../tribe/contract.js";

const STATUS_LABELS = {
  unconfigured: "TRIBE endpoint not connected",
  idle: "Ready for a video",
  predicting: "Running TRIBE v2 inference",
  ready: "Contract-verified TRIBE cortical prediction",
  error: "TRIBE v2 unavailable",
};

const CAMERA_FIT_PADDING = 1.12;

function fitCameraToSurface(camera, controls, positionAttribute, worldMatrix, aspect) {
  if (!positionAttribute?.count) return;

  const surfaceBounds = new THREE.Box3().setFromBufferAttribute(positionAttribute).applyMatrix4(worldMatrix);
  const surfaceCenter = surfaceBounds.getCenter(new THREE.Vector3());
  const viewBack = camera.position.clone().sub(controls.target);
  if (viewBack.lengthSq() < Number.EPSILON) viewBack.set(0, -1, 0.5);
  viewBack.normalize();

  const viewRight = new THREE.Vector3().crossVectors(camera.up, viewBack).normalize();
  const viewUp = new THREE.Vector3().crossVectors(viewBack, viewRight).normalize();
  const verticalTangent = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const horizontalTangent = verticalTangent * Math.max(aspect, 0.01);
  const worldPoint = new THREE.Vector3();
  const relativePoint = new THREE.Vector3();
  let requiredDistance = 0;
  let surfaceRadius = 0;

  // Fit the actual rotated fsaverage5 vertices, rather than a decorative proxy
  // volume, so narrow portrait rails cannot clip either cortical hemisphere.
  for (let index = 0; index < positionAttribute.count; index += 1) {
    worldPoint.fromBufferAttribute(positionAttribute, index).applyMatrix4(worldMatrix);
    relativePoint.copy(worldPoint).sub(surfaceCenter);
    const depthOffset = relativePoint.dot(viewBack);
    requiredDistance = Math.max(
      requiredDistance,
      depthOffset + (Math.abs(relativePoint.dot(viewRight)) * CAMERA_FIT_PADDING) / horizontalTangent,
      depthOffset + (Math.abs(relativePoint.dot(viewUp)) * CAMERA_FIT_PADDING) / verticalTangent,
    );
    surfaceRadius = Math.max(surfaceRadius, relativePoint.length());
  }

  requiredDistance = Math.max(requiredDistance, surfaceRadius * CAMERA_FIT_PADDING);
  camera.position.copy(surfaceCenter).addScaledVector(viewBack, requiredDistance);
  camera.near = Math.max(0.1, requiredDistance - surfaceRadius * 1.5);
  camera.far = requiredDistance + surfaceRadius * 4;
  camera.aspect = aspect;
  camera.updateProjectionMatrix();
  controls.target.copy(surfaceCenter);
  controls.minDistance = Math.max(1, surfaceRadius * 1.15);
  controls.maxDistance = Math.max(requiredDistance * 2, surfaceRadius * 5);
  controls.update();
}

export function TribeBrainViewer({ prediction, mediaTime = 0, status = "unconfigured", atlas = null, focus = null, onFrameChange }) {
  const canvasRef = useRef(null);
  const colorAttributeRef = useRef(null);
  const lastFrameRef = useRef(Symbol("none"));
  const [surfaceState, setSurfaceState] = useState({ status: "loading", message: "Loading verified fsaverage5 surface" });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const controller = new AbortController();
    let renderer;
    let controls;
    let geometry;
    let material;
    let resizeObserver;
    let animationFrame;
    let corticalSurface;
    let disposed = false;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(27, 1, 1, 1_200);
    camera.position.set(238, -310, 196);

    const onContextLost = (event) => {
      event.preventDefault();
      setSurfaceState({ status: "error", message: "WebGL context lost — cortical data is not being shown" });
    };
    canvas.addEventListener("webglcontextlost", onContextLost);

    try {
      renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.05;
      controls = new OrbitControls(camera, canvas);
      controls.enableDamping = true;
      controls.dampingFactor = 0.07;
      controls.enablePan = false;
      controls.autoRotate = false;
      controls.minDistance = 250;
      controls.maxDistance = 610;
      controls.target.set(0, -7, 7);

      scene.add(new THREE.HemisphereLight(0xffffff, 0x17171a, 2.2));
      const key = new THREE.DirectionalLight(0xf8ffce, 3.4);
      key.position.set(150, -90, 210);
      scene.add(key);
      const rim = new THREE.DirectionalLight(0x8b63ff, 1.8);
      rim.position.set(-170, 90, -120);
      scene.add(rim);
    } catch {
      setSurfaceState({ status: "error", message: "WebGL is unavailable — cortical data is not being shown" });
      return () => canvas.removeEventListener("webglcontextlost", onContextLost);
    }

    loadFsaverage5Mesh(undefined, controller.signal)
      .then((mesh) => {
        if (disposed) return;
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(mesh.positions, 3));
        geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
        geometry.computeVertexNormals();
        const colors = writeNeutralColors(new Float32Array(TRIBE_VERTEX_COUNT * 3));
        const colorAttribute = new THREE.BufferAttribute(colors, 3);
        colorAttribute.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute("color", colorAttribute);
        colorAttributeRef.current = colorAttribute;

        material = new THREE.MeshStandardMaterial({
          color: 0xffffff,
          vertexColors: true,
          roughness: 0.72,
          metalness: 0.04,
          side: THREE.DoubleSide,
        });
        corticalSurface = new THREE.Mesh(geometry, material);
        corticalSurface.rotation.set(-0.03, 0, -0.02);
        scene.add(corticalSurface);
        corticalSurface.updateMatrixWorld(true);
        fitCameraToSurface(
          camera,
          controls,
          geometry.getAttribute("position"),
          corticalSurface.matrixWorld,
          Math.max(canvas.clientWidth, 1) / Math.max(canvas.clientHeight, 1),
        );
        setSurfaceState({ status: "ready", message: "Hash-verified fsaverage5 · 20,484 vertices" });
      })
      .catch((error) => {
        if (!disposed && error?.name !== "AbortError") {
          setSurfaceState({ status: "error", message: error?.message || "The cortical surface could not be verified" });
        }
      });

    const resize = () => {
      const width = Math.max(1, canvas.clientWidth);
      const height = Math.max(1, canvas.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      if (geometry && corticalSurface) {
        corticalSurface.updateMatrixWorld(true);
        fitCameraToSurface(camera, controls, geometry.getAttribute("position"), corticalSurface.matrixWorld, camera.aspect);
      } else {
        camera.updateProjectionMatrix();
      }
    };
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    resize();

    const render = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(render);
    };
    render();

    return () => {
      disposed = true;
      controller.abort();
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      resizeObserver?.disconnect();
      canvas.removeEventListener("webglcontextlost", onContextLost);
      controls?.dispose();
      geometry?.dispose();
      material?.dispose();
      renderer?.dispose();
      colorAttributeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const colorAttribute = colorAttributeRef.current;
    if (!colorAttribute) return;
    if (status !== "ready" || !prediction) {
      if (lastFrameRef.current !== null) {
        writeNeutralColors(colorAttribute.array);
        colorAttribute.needsUpdate = true;
        lastFrameRef.current = null;
        onFrameChange?.(null);
      }
      return;
    }

    const { index, frame } = predictionFrameAtMediaTime(prediction.values, prediction.manifest, mediaTime);
    const paintKey = `${index ?? "none"}:${focus?.key || "all"}`;
    if (lastFrameRef.current === paintKey) return;
    if (frame) {
      writeScientificColors(frame, prediction.manifest.frames.displayScale.domain, colorAttribute.array);
      applyParcelFocus(colorAttribute.array, atlas, focus);
    }
    else writeNeutralColors(colorAttribute.array);
    colorAttribute.needsUpdate = true;
    lastFrameRef.current = paintKey;
    onFrameChange?.(index);
  }, [atlas, focus, mediaTime, onFrameChange, prediction, status, surfaceState.status]);

  const label = surfaceState.status === "error"
    ? surfaceState.message
    : status === "ready"
      ? STATUS_LABELS.ready
      : STATUS_LABELS[status] || STATUS_LABELS.unconfigured;

  return (
    <div className={`brain-viewer brain-viewer--${surfaceState.status}`}>
      <canvas
        ref={canvasRef}
        className="brain-canvas"
        aria-label={`Interactive fsaverage5 cortical surface. Vertex colors appear only when a contract-validated, hash-checked TRIBE v2 prediction tensor is loaded.${focus ? ` Focused on ${focus.hemisphere} ${focus.name}.` : ""}`}
      />
      <div className="brain-viewer-status" role="status" aria-live="polite">
        <span className={`status-dot status-dot--${status}`} />
        <span>{surfaceState.status === "loading" ? surfaceState.message : label}</span>
      </div>
      <span className="orbit-note">Drag to inspect · scroll to zoom</span>
    </div>
  );
}
