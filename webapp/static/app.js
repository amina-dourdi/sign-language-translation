// DOM Elements
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const translationText = document.getElementById('translation-result');
const statusText = document.getElementById('status-text');
const wsStatus = document.getElementById('ws-status');

// WebSocket Connection
let ws;
let isConnected = false;

function connectWebSocket() {
    // Determine the WS URL based on the current window location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/translate`;
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket Connected');
        isConnected = true;
        wsStatus.textContent = 'Connected 🟢';
        wsStatus.style.color = '#10b981';
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.translation && data.translation.trim() !== '') {
            translationText.textContent = data.translation;
            translationText.classList.remove('placeholder');
        }
    };

    ws.onclose = () => {
        console.log('WebSocket Disconnected');
        isConnected = false;
        wsStatus.textContent = 'Disconnected 🔴';
        wsStatus.style.color = '#ef4444';
        // Auto-reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };
}

connectWebSocket();

// Keypoint Extraction Helper Function
// Converts MediaPipe landmarks to the 411 flat array expected by the CSLT model
function extractKeypoints(results) {
    // The CSLT model expects exactly 411 floats:
    // Body (25*3=75), Face (70*3=210), Left Hand (21*3=63), Right Hand (21*3=63)
    // Note: MediaPipe returns slightly different numbers of landmarks. 
    // We pad or truncate to match the exact 411 dimension expected by the backend.
    
    let keypoints = [];

    // 1. Pose (Body)
    if (results.poseLandmarks) {
        // Take first 25 landmarks to match OpenPose
        for (let i = 0; i < 25; i++) {
            if (i < results.poseLandmarks.length) {
                const lm = results.poseLandmarks[i];
                keypoints.push(lm.x, lm.y, lm.visibility);
            } else {
                keypoints.push(0, 0, 0);
            }
        }
    } else {
        for(let i=0; i<75; i++) keypoints.push(0);
    }

    // 2. Face
    if (results.faceLandmarks) {
        // Take first 70 to match OpenPose
        for (let i = 0; i < 70; i++) {
            if (i < results.faceLandmarks.length) {
                const lm = results.faceLandmarks[i];
                // MediaPipe face doesn't have visibility, we default to 1
                keypoints.push(lm.x, lm.y, 1.0); 
            } else {
                keypoints.push(0, 0, 0);
            }
        }
    } else {
        for(let i=0; i<210; i++) keypoints.push(0);
    }

    // 3. Left Hand
    if (results.leftHandLandmarks) {
        for (let i = 0; i < 21; i++) {
            const lm = results.leftHandLandmarks[i];
            keypoints.push(lm.x, lm.y, 1.0);
        }
    } else {
        for(let i=0; i<63; i++) keypoints.push(0);
    }

    // 4. Right Hand
    if (results.rightHandLandmarks) {
        for (let i = 0; i < 21; i++) {
            const lm = results.rightHandLandmarks[i];
            keypoints.push(lm.x, lm.y, 1.0);
        }
    } else {
        for(let i=0; i<63; i++) keypoints.push(0);
    }

    return keypoints;
}

// MediaPipe Holistic Setup
function onResults(results) {
    // 1. Draw Skeleton on Canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw video frame
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    // Draw Hands (Purple/Mauve theme)
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, {color: '#c084fc', lineWidth: 2});
        drawLandmarks(canvasCtx, results.leftHandLandmarks, {color: '#d8b4fe', lineWidth: 1, radius: 2});
    }
    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, {color: '#c084fc', lineWidth: 2});
        drawLandmarks(canvasCtx, results.rightHandLandmarks, {color: '#d8b4fe', lineWidth: 1, radius: 2});
    }
    
    // Draw Body
    if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#8b5cf6', lineWidth: 3});
        drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#a78bfa', lineWidth: 1, radius: 3});
    }
    
    canvasCtx.restore();

    // 2. Extract and Send Keypoints to Backend
    if (isConnected && ws.readyState === WebSocket.OPEN) {
        const flatKeypoints = extractKeypoints(results);
        // Only send if a person is actually detected (at least some body parts are non-zero)
        const isPersonVisible = flatKeypoints.some(val => val !== 0);
        
        if (isPersonVisible) {
            ws.send(JSON.stringify(flatKeypoints));
        }
    }
}

const holistic = new Holistic({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
}});

holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: false,
  smoothSegmentation: false,
  refineFaceLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

holistic.onResults(onResults);

// Setup Camera
const camera = new Camera(videoElement, {
  onFrame: async () => {
    // Match canvas size to video frame
    if(canvasElement.width !== videoElement.videoWidth) {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        statusText.textContent = 'Camera active, tracking movements...';
    }
    await holistic.send({image: videoElement});
  },
  width: 1280,
  height: 720
});

camera.start();
