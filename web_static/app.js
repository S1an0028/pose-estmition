const video = document.getElementById("camera");
const canvas = document.getElementById("capture");
const result = document.getElementById("result");
const emptyState = document.getElementById("emptyState");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const refreshCamerasBtn = document.getElementById("refreshCamerasBtn");
const detectMode = document.getElementById("detectMode");
const cameraSelect = document.getElementById("cameraSelect");
const statusEl = document.getElementById("status");
const detectionsEl = document.getElementById("detections");
const countLabel = document.getElementById("countLabel");
const latencyEl = document.getElementById("latency");
const fpsEl = document.getElementById("fps");

let stream = null;
let running = false;
let inFlight = false;
let lastFrameTime = 0;
let fps = 0;

function numberOrNull(id) {
  const value = document.getElementById(id).value;
  return value === "" ? null : Number(value);
}

function payloadFromControls(image) {
  return {
    image,
    mode: detectMode.value,
    profile: document.getElementById("profile").value,
    imgsz: numberOrNull("imgsz"),
    conf: numberOrNull("conf"),
    kptConf: numberOrNull("kptConf"),
    smooth: numberOrNull("smooth"),
    maxItems: Number(document.getElementById("maxItems").value || 2),
    labelAllPoints: document.getElementById("labels").checked,
    augment: document.getElementById("augment").checked,
  };
}

async function listCameras() {
  // Browser labels for cameras usually become visible only after the user has
  // granted camera permission once.
  if (!navigator.mediaDevices?.enumerateDevices) {
    statusEl.textContent = "浏览器不支持摄像头枚举";
    return;
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === "videoinput");
  const currentValue = cameraSelect.value;

  cameraSelect.innerHTML = "";
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "默认摄像头";
  cameraSelect.appendChild(defaultOption);

  cameras.forEach((camera, index) => {
    const option = document.createElement("option");
    option.value = camera.deviceId;
    option.textContent = camera.label || `摄像头 ${index + 1}`;
    cameraSelect.appendChild(option);
  });

  if ([...cameraSelect.options].some((option) => option.value === currentValue)) {
    cameraSelect.value = currentValue;
  }
}

function updateModeLabels() {
  const bodyMode = detectMode.value === "body";
  countLabel.textContent = bodyMode ? "人数" : "手数";
  document.querySelector(".brand p").textContent = bodyMode
    ? "YOLO11 body pose · local GPU"
    : "YOLO hand keypoints · local GPU";
}

function drawCameraFrame() {
  // The browser owns the webcam stream. We draw the current video frame to a
  // canvas, optionally mirror it, then send that JPEG to the local Python API.
  const width = video.videoWidth || 960;
  const height = video.videoHeight || 540;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  ctx.save();
  if (document.getElementById("mirror").checked) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, width, height);
  ctx.restore();
}

async function sendFrame() {
  // Keep at most one inference request in flight. This prevents slow model
  // frames from stacking up and making the preview lag farther behind reality.
  if (!running || inFlight || video.readyState < 2) {
    requestAnimationFrame(sendFrame);
    return;
  }

  inFlight = true;
  drawCameraFrame();
  const image = canvas.toDataURL("image/jpeg", 0.78);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromControls(image)),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    if (!data.ok) {
      throw new Error(data.error || "prediction failed");
    }
    result.src = data.image;
    detectionsEl.textContent = data.detections;
    latencyEl.textContent = `${data.elapsedMs} ms`;
    statusEl.textContent = "运行中";

    const now = performance.now();
    if (lastFrameTime) {
      const instant = 1000 / Math.max(now - lastFrameTime, 1);
      fps = fps ? fps * 0.85 + instant * 0.15 : instant;
      fpsEl.textContent = fps.toFixed(1);
    }
    lastFrameTime = now;
  } catch (error) {
    statusEl.textContent = `错误: ${error.message}`;
  } finally {
    inFlight = false;
    if (running) {
      setTimeout(sendFrame, 20);
    }
  }
}

async function start() {
  if (running) return;

  // If the user chose an external camera, request that exact device. Otherwise
  // let the browser pick the default front-facing camera.
  const selectedCamera = cameraSelect.value;
  const videoConstraints = selectedCamera
    ? { deviceId: { exact: selectedCamera }, width: { ideal: 1280 }, height: { ideal: 720 } }
    : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" };

  stream = await navigator.mediaDevices.getUserMedia({
    video: videoConstraints,
    audio: false,
  });

  video.srcObject = stream;
  await video.play();
  await listCameras();
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: detectMode.value }),
  });

  running = true;
  inFlight = false;
  lastFrameTime = 0;
  fps = 0;
  emptyState.style.display = "none";
  startBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = "运行中";
  sendFrame();
}

function stop() {
  running = false;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  statusEl.textContent = "已停止";
}

async function restartIfRunning() {
  if (!running) return;
  stop();
  await start();
}

startBtn.addEventListener("click", () => {
  start().catch((error) => {
    statusEl.textContent = `摄像头错误: ${error.message}`;
  });
});

stopBtn.addEventListener("click", stop);
refreshCamerasBtn.addEventListener("click", () => {
  listCameras().catch((error) => {
    statusEl.textContent = `摄像头列表错误: ${error.message}`;
  });
});

detectMode.addEventListener("change", () => {
  updateModeLabels();
  fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: detectMode.value }),
  }).catch(() => {});
});

cameraSelect.addEventListener("change", () => {
  restartIfRunning().catch((error) => {
    statusEl.textContent = `切换摄像头错误: ${error.message}`;
  });
});

window.addEventListener("beforeunload", stop);

updateModeLabels();
listCameras().catch(() => {});
