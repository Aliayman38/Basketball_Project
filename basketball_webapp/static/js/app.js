/**
 * Basketball Analysis Dashboard - Frontend Logic
 */

const state = {
    currentScreen: 'upload',
    isProcessing: false,
    uploadedFile: null,
    score: { team_0: 0, team_1: 0 },
    statusInterval: null,
    results: null,
    videoInfo: null,
    pipelineAvailable: false
};

const elements = {
    navItems: document.querySelectorAll('.nav-item'),
    screens: document.querySelectorAll('.screen'),
    pageTitle: document.getElementById('pageTitle'),
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    uploadedFile: document.getElementById('uploadedFile'),
    fileName: document.getElementById('fileName'),
    fileIcon: document.getElementById('fileIcon'),
    fileStatus: document.getElementById('fileStatus'),
    fileMeta: document.getElementById('fileMeta'),
    processBtn: document.getElementById('processBtn'),
    progressPanel: document.getElementById('progressPanel'),
    previewPanel: document.getElementById('previewPanel'),
    previewVideo: document.getElementById('previewVideo'),
    progressBar: document.getElementById('progressBar'),
    progressText: document.getElementById('progressText'),
    stepIndicator: document.getElementById('stepIndicator'),
    logContent: document.getElementById('logContent'),
    scoreTeam0: document.getElementById('scoreTeam0'),
    scoreTeam1: document.getElementById('scoreTeam1'),
    processingStatus: document.getElementById('processingStatus'),
    pipelineStatus: document.getElementById('pipelineStatus'),
    trackingVideo: document.getElementById('trackingVideo'),
    homographyVideo: document.getElementById('homographyVideo'),
    landmarksVideo: document.getElementById('landmarksVideo'),
    finalVideo: document.getElementById('finalVideo')
};

function initNavigation() {
    elements.navItems.forEach(item => {
        item.addEventListener('click', () => {
            const screenId = item.dataset.screen;
            switchScreen(screenId);
        });
    });
}

function switchScreen(screenId) {
    elements.navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.screen === screenId);
    });
    elements.screens.forEach(screen => {
        screen.classList.toggle('active', screen.id === `screen-${screenId}`);
    });
    const titles = {
        upload: 'Upload Video',
        tracking: 'Detection & Tracking',
        homography: 'Homography & Top-Down',
        landmarks: 'Court Landmarks',
        dashboard: 'Analytics Dashboard'
    };
    elements.pageTitle.textContent = titles[screenId] || 'Basketball Analysis';
    state.currentScreen = screenId;
    if (screenId === 'dashboard' && state.results) loadDashboardData();
}

function initUpload() {
    const { uploadArea, fileInput } = elements;
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });
}

async function handleFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.success) {
            state.uploadedFile = data;
            state.videoInfo = data.video_info;
            state.pipelineAvailable = data.pipeline_available;
            showUploadedFile(data.filename, data.is_video, data.video_info);
            addLog(`File uploaded: ${data.filename}`);
            if (data.is_video) showVideoPreview();
            updatePipelineBadge(data.pipeline_available);
        } else {
            alert('Upload failed: ' + data.error);
        }
    } catch (err) {
        console.error('Upload error:', err);
        alert('Upload failed. Please try again.');
    }
}

function updatePipelineBadge(available) {
    const badge = document.getElementById('pipelineBadge');
    const statusText = document.getElementById('pipelineStatus');
    if (!badge || !statusText) return;

    if (available) {
        statusText.textContent = 'AI Pipeline Ready';
        badge.style.color = '#00FF88';
        badge.querySelector('i').className = 'fas fa-check-circle';
    } else {
        statusText.textContent = 'DEMO MODE (No AI)';
        badge.style.color = '#FF8C00';
        badge.querySelector('i').className = 'fas fa-exclamation-triangle';
    }
}

function showUploadedFile(filename, isVideo, videoInfo) {
    elements.uploadArea.classList.add('hidden');
    elements.uploadedFile.classList.remove('hidden');
    elements.fileName.textContent = filename;
    elements.fileIcon.className = isVideo ? 'fas fa-file-video' : 'fas fa-file-image';
    elements.fileStatus.textContent = 'Ready for processing';
    if (videoInfo && videoInfo.frames) {
        const meta = `${videoInfo.width}x${videoInfo.height} • ${videoInfo.fps} FPS • ${videoInfo.frames} frames • ${videoInfo.duration}s`;
        elements.fileMeta.textContent = meta;
    }
}

function showVideoPreview() {
    elements.previewPanel.classList.remove('hidden');
    elements.previewVideo.src = '/uploaded_video?t=' + Date.now();
    elements.previewVideo.load();
}

async function startProcessing() {
    if (!state.uploadedFile) return;
    elements.processBtn.disabled = true;
    elements.processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    elements.progressPanel.classList.remove('hidden');
    try {
        const response = await fetch('/process', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            state.isProcessing = true;
            updateProcessingStatus('processing');
            startStatusPolling();
            if (data.pipeline_mode === 'demo') {
                addLog('⚠️ WARNING: Running in DEMO mode - no AI detection!');
                addLog('   Place app.py in your BASKETBALL_PROJECT root folder.');
            }
        }
    } catch (err) {
        console.error('Processing error:', err);
        elements.processBtn.disabled = false;
        elements.processBtn.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
    }
}

function startStatusPolling() {
    if (state.statusInterval) clearInterval(state.statusInterval);
    state.statusInterval = setInterval(async () => {
        try {
            const response = await fetch('/status');
            const status = await response.json();
            updateProgressUI(status);
            if (!status.is_processing && status.has_results) {
                clearInterval(state.statusInterval);
                state.isProcessing = false;
                updateProcessingStatus('idle');
                elements.processBtn.innerHTML = '<i class="fas fa-check"></i> Complete!';
                const resResponse = await fetch('/results');
                state.results = await resResponse.json();
                elements.navItems.forEach(item => {
                    item.style.opacity = '1';
                    item.style.pointerEvents = 'auto';
                });
                loadAllVideoSources();
                setTimeout(() => switchScreen('tracking'), 800);
            }
            if (status.logs) status.logs.forEach(log => addLog(log, false));
        } catch (err) {
            console.error('Status poll error:', err);
        }
    }, 1000);
}

function updateProgressUI(status) {
    elements.progressBar.style.setProperty('--progress', `${status.progress}%`);
    elements.progressText.textContent = `${status.progress}%`;
    elements.stepIndicator.textContent = status.current_step || 'Processing...';
    const stepMap = {
        'Detection & Tracking': 1, 'Trajectory Analysis': 2, 'Shot Detection': 3,
        'Ball Possession': 3, 'Court Landmarks': 4, 'Possession Overlay': 5,
        'Homography': 6, 'Top-Down View': 7, 'Dashboard': 8, 'Complete': 8
    };
    const activeStep = stepMap[status.current_step] || 0;
    document.querySelectorAll('.step').forEach((step, index) => {
        const stepNum = index + 1;
        step.classList.remove('active', 'completed');
        if (stepNum < activeStep) step.classList.add('completed');
        else if (stepNum === activeStep) step.classList.add('active');
    });
}

function updateProcessingStatus(status) {
    const dot = elements.processingStatus.querySelector('.status-dot');
    const text = elements.processingStatus.querySelector('span');
    dot.className = 'status-dot ' + status;
    text.textContent = status === 'processing' ? 'Processing...' : 'Ready';
}

function addLog(message, append = true) {
    const line = document.createElement('div');
    line.className = 'log-line';
    line.textContent = message;
    if (append) {
        elements.logContent.appendChild(line);
    } else {
        const existing = Array.from(elements.logContent.children);
        if (!existing.some(el => el.textContent === message)) {
            elements.logContent.appendChild(line);
        }
    }
    elements.logContent.scrollTop = elements.logContent.scrollHeight;
}

async function updateScore(team, action) {
    try {
        const response = await fetch('/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team, action })
        });
        const data = await response.json();
        if (data.success) {
            state.score = data.score;
            elements.scoreTeam0.textContent = data.score.team_0;
            elements.scoreTeam1.textContent = data.score.team_1;
            const el = team === 'team_0' ? elements.scoreTeam0 : elements.scoreTeam1;
            el.style.transform = 'scale(1.3)';
            el.style.color = 'var(--primary)';
            setTimeout(() => { el.style.transform = 'scale(1)'; el.style.color = ''; }, 300);
        }
    } catch (err) {
        console.error('Score update error:', err);
    }
}

function loadAllVideoSources() {
    const timestamp = Date.now();
    const videoSources = {
        trackingVideo: `/video/tracking?t=${timestamp}`,
        homographyVideo: `/video/landmarks?t=${timestamp}`,
        landmarksVideo: `/video/landmarks?t=${timestamp}`,
        finalVideo: `/video/final?t=${timestamp}`
    };
    Object.entries(videoSources).forEach(([id, src]) => {
        const video = document.getElementById(id);
        if (video) {
            while (video.firstChild) video.removeChild(video.firstChild);
            const source = document.createElement('source');
            source.src = src;
            source.type = 'video/mp4';
            video.appendChild(source);
            video.load();
        }
    });
}

async function loadDashboardData() {
    try {
        const possResponse = await fetch('/analytics/possession');
        const possession = await possResponse.json();
        if (possession.team_0_pct !== undefined) {
            updatePossessionChart(possession.team_0_pct, possession.team_1_pct);
            const bar0 = document.getElementById('possessionBar0');
            const bar1 = document.getElementById('possessionBar1');
            const text0 = document.getElementById('possessionText0');
            const text1 = document.getElementById('possessionText1');
            if (bar0) { bar0.style.width = `${possession.team_0_pct}%`; text0.textContent = `${possession.team_0_pct.toFixed(0)}%`; }
            if (bar1) { bar1.style.width = `${possession.team_1_pct}%`; text1.textContent = `${possession.team_1_pct.toFixed(0)}%`; }
        }
        const shotsResponse = await fetch('/analytics/shots');
        const shots = await shotsResponse.json();
        if (shots.made_shots) {
            updateShotsList(shots.made_shots);
            updateShotTable(shots.made_shots);
        }
        if (state.results && state.results.stats) {
            const stats = state.results.stats;
            document.getElementById('dashPlayers').textContent = stats.players_detected;
            document.getElementById('dashShots').textContent = stats.shots_made;
            document.getElementById('dashSpeed').textContent = stats.avg_speed;
            document.getElementById('dashDistance').textContent = stats.total_distance;
            document.getElementById('totalFrames').textContent = stats.total_frames;
            document.getElementById('fps').textContent = stats.fps;
        }
        drawSpeedChart();
        updateAccuracyRing(87);
    } catch (err) {
        console.error('Dashboard load error:', err);
    }
}

function updatePossessionChart(team0Pct, team1Pct) {
    const donut0 = document.getElementById('donutTeam0');
    const donut1 = document.getElementById('donutTeam1');
    const donutPct = document.getElementById('donutPct');
    const legend0 = document.getElementById('legendTeam0');
    const legend1 = document.getElementById('legendTeam1');
    const circumference = 2 * Math.PI * 80;
    if (donut0) { donut0.style.strokeDasharray = circumference; donut0.style.strokeDashoffset = circumference * (1 - team0Pct / 100); }
    if (donut1) { donut1.style.strokeDasharray = circumference; donut1.style.strokeDashoffset = circumference * (1 - team1Pct / 100); }
    if (donutPct) donutPct.textContent = `${team0Pct.toFixed(0)}%`;
    if (legend0) legend0.textContent = `${team0Pct.toFixed(0)}%`;
    if (legend1) legend1.textContent = `${team1Pct.toFixed(0)}%`;
}

function updateShotsList(shots) {
    const container = document.getElementById('shotsList');
    if (!container) return;
    container.innerHTML = shots.map((shot, i) => `
        <div class="shot-item">
            <div class="shot-info">
                <span class="shot-time">Shot ${i + 1} • Frames ${shot.frames}</span>
                <span class="shot-team">Team ${i % 2 === 0 ? 'Yellow' : 'Blue'}</span>
            </div>
            <span class="shot-conf">${(shot.confidence * 100).toFixed(0)}%</span>
        </div>
    `).join('');
}

function updateShotTable(shots) {
    const tbody = document.getElementById('shotTableBody');
    if (!tbody) return;
    tbody.innerHTML = shots.map((shot, i) => `
        <tr>
            <td>${i + 1}</td>
            <td>${shot.frames}</td>
            <td><span style="color: var(--team-${i % 2 === 0 ? '0' : '1'})">Team ${i % 2 === 0 ? 'Yellow' : 'Blue'}</span></td>
            <td>${(shot.confidence * 100).toFixed(1)}%</td>
            <td><span style="color: var(--accent-green)">✓ Made</span></td>
        </tr>
    `).join('');
}

function drawSpeedChart() {
    const canvas = document.getElementById('speedChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width, height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    const dataPoints = 50, data = [];
    for (let i = 0; i < dataPoints; i++) data.push(2 + Math.sin(i * 0.2) * 1.5 + Math.random() * 1);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = (height - 40) * (i / 5) + 20;
        ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(width - 20, y); ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.font = '11px Inter'; ctx.textAlign = 'right';
        ctx.fillText((5 - i).toFixed(1), 35, y + 3);
    }
    ctx.strokeStyle = '#FF8C00'; ctx.lineWidth = 2; ctx.beginPath();
    const xStep = (width - 60) / (dataPoints - 1), yScale = (height - 60) / 5;
    data.forEach((val, i) => {
        const x = 40 + i * xStep, y = height - 30 - val * yScale;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.lineTo(40 + (dataPoints - 1) * xStep, height - 30);
    ctx.lineTo(40, height - 30); ctx.closePath();
    ctx.fillStyle = 'rgba(255, 140, 0, 0.1)'; ctx.fill();
    ctx.fillStyle = '#FF8C00';
    data.forEach((val, i) => {
        if (i % 5 === 0) {
            const x = 40 + i * xStep, y = height - 30 - val * yScale;
            ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
        }
    });
}

function updateAccuracyRing(percentage) {
    const circle = document.getElementById('accuracyCircle');
    const value = document.getElementById('accuracyValue');
    if (circle) { const circumference = 2 * Math.PI * 45; circle.style.strokeDashoffset = circumference * (1 - percentage / 100); }
    if (value) value.textContent = `${percentage}%`;
}

function initTopdownAnimation() {
    const container = document.getElementById('topdownPlayers');
    if (!container) return;
    const players = [
        { id: 0, team: 0, x: 20, y: 30 }, { id: 1, team: 0, x: 25, y: 50 },
        { id: 2, team: 0, x: 30, y: 70 }, { id: 3, team: 0, x: 40, y: 40 },
        { id: 4, team: 0, x: 45, y: 60 }, { id: 5, team: 1, x: 70, y: 30 },
        { id: 6, team: 1, x: 75, y: 50 }, { id: 7, team: 1, x: 80, y: 70 },
        { id: 8, team: 1, x: 65, y: 45 }, { id: 9, team: 1, x: 60, y: 55 },
        { id: 'ball', team: 'ball', x: 50, y: 50 }
    ];
    players.forEach(p => {
        const dot = document.createElement('div');
        dot.className = 'player-dot';
        dot.style.left = `${p.x}%`; dot.style.top = `${p.y}%`;
        if (p.team === 'ball') {
            dot.style.background = '#FFFFFF'; dot.style.width = '10px'; dot.style.height = '10px';
            dot.style.boxShadow = '0 0 10px rgba(255,255,255,0.8)';
        } else {
            dot.style.background = p.team === 0 ? 'var(--team-0)' : 'var(--team-1)';
        }
        container.appendChild(dot);
    });
    let angle = 0;
    setInterval(() => {
        const ball = container.querySelector('.player-dot:last-child');
        if (ball) {
            angle += 0.05;
            ball.style.left = `${50 + Math.sin(angle) * 20}%`;
            ball.style.top = `${50 + Math.cos(angle * 0.7) * 15}%`;
        }
    }, 50);
}

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initUpload();
    initTopdownAnimation();

    // Check initial pipeline status
    fetch('/status').then(r => r.json()).then(status => {
        updatePipelineBadge(status.pipeline_available);
    }).catch(() => {
        const statusText = document.getElementById('pipelineStatus');
        if (statusText) statusText.textContent = 'Offline';
    });

    document.querySelectorAll('.nav-item:not([data-screen="upload"])').forEach(item => {
        item.style.opacity = '0.5';
        item.style.pointerEvents = 'none';
    });
    setTimeout(() => updateAccuracyRing(87), 500);
});
