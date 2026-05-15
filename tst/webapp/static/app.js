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
    let keypoints = [];

    // 1. Pose (OpenPose 25 joints)
    // MediaPipe to OpenPose mapping (approximate for upper body)
    // OpenPose: 0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist, 5:LShoulder, 6:LElbow, 7:LWrist, 8:MidHip
    if (results.poseLandmarks) {
        const p = results.poseLandmarks;
        const pushLm = (idx) => {
            if (idx >= 0 && idx < p.length) {
                keypoints.push(p[idx].x, p[idx].y, p[idx].visibility);
            } else {
                keypoints.push(0, 0, 0);
            }
        };
        const pushMid = (idx1, idx2) => {
            if (idx1 < p.length && idx2 < p.length) {
                keypoints.push((p[idx1].x + p[idx2].x)/2, (p[idx1].y + p[idx2].y)/2, Math.min(p[idx1].visibility, p[idx2].visibility));
            } else {
                keypoints.push(0, 0, 0);
            }
        };

        pushLm(0); // 0: Nose
        pushMid(11, 12); // 1: Neck (mid-shoulder)
        pushLm(12); // 2: RShoulder
        pushLm(14); // 3: RElbow
        pushLm(16); // 4: RWrist
        pushLm(11); // 5: LShoulder
        pushLm(13); // 6: LElbow
        pushLm(15); // 7: LWrist
        pushMid(23, 24); // 8: MidHip
        pushLm(24); // 9: RHip
        pushLm(26); // 10: RKnee
        pushLm(28); // 11: RAnkle
        pushLm(23); // 12: LHip
        pushLm(25); // 13: LKnee
        pushLm(27); // 14: LAnkle
        pushLm(5); // 15: REye
        pushLm(2); // 16: LEye
        pushLm(8); // 17: REar
        pushLm(7); // 18: LEar
        // OpenPose body in this specific model uses 21 joints (63 floats) instead of 25.
        // So we stop here and DO NOT push joints 19-24.
        // Total so far: 19 joints * 3 = 57 floats. We need 2 more joints to reach 21 (63 floats).
        // Let's push LBigToe and RBigToe as 19 and 20 to make it 21 joints exactly.
        pushLm(29); // 19: LBigToe (approx)
        pushLm(30); // 20: RBigToe (approx)
    } else {
        for(let i=0; i<63; i++) keypoints.push(0);
    }

    // 2. Face (OpenPose 70 joints)
    // Mapping 468 MediaPipe face landmarks to 70 OpenPose is very complex.
    // We will zero them out to prevent them from injecting massive noise.
    for(let i=0; i<210; i++) keypoints.push(0);

    // 3. Left Hand (21 joints - topology matches perfectly)
    if (results.leftHandLandmarks) {
        for (let i = 0; i < 21; i++) {
            const lm = results.leftHandLandmarks[i];
            keypoints.push(lm.x, lm.y, 1.0);
        }
    } else {
        for(let i=0; i<63; i++) keypoints.push(0);
    }

    // 4. Right Hand (21 joints - topology matches perfectly)
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
