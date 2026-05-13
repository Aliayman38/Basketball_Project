/**
 * Basketball Analysis Dashboard - Frontend Logic
 * Fixes: state reset on new video, scrolling, model selector, loading overlay
 */

const state = {
    currentScreen: 'upload',
    isProcessing: false,
    uploadedFile: null,
    score: { team_0: 0, team_1: 0 },
    statusInterval: null,
    results: null,
    videoInfo: null,
    pipelineAvailable: false,
    selectedModel: null,
    models: [],
};

const STEP_MAP = {
    'detection & tracking': 'detection', 'detection': 'detection', 'tracking': 'tracking',
    'trajectory': 'tracking', 'possession': 'tracking',
    'shot': 'shots', 'shots': 'shots',
    'landmark': 'landmarks', 'court landmark': 'landmarks',
    'homography': 'homography',
    're-encoding': 'encoding', 'dashboard': 'encoding', 'complete': 'encoding',
};

const elements = {
    navItems:         document.querySelectorAll('.nav-item'),
    screens:          document.querySelectorAll('.screen'),
    pageTitle:        document.getElementById('pageTitle'),
    uploadArea:       document.getElementById('uploadArea'),
    fileInput:        document.getElementById('fileInput'),
    uploadedFile:     document.getElementById('uploadedFile'),
    fileName:         document.getElementById('fileName'),
    fileIcon:         document.getElementById('fileIcon'),
    fileStatus:       document.getElementById('fileStatus'),
    fileMeta:         document.getElementById('fileMeta'),
    processBtn:       document.getElementById('processBtn'),
    progressPanel:    document.getElementById('progressPanel'),
    previewPanel:     document.getElementById('previewPanel'),
    previewVideo:     document.getElementById('previewVideo'),
    progressBar:      document.getElementById('progressBar'),
    progressText:     document.getElementById('progressText'),
    stepIndicator:    document.getElementById('stepIndicator'),
    logContent:       document.getElementById('logContent'),
    scoreTeam0:       document.getElementById('scoreTeam0'),
    scoreTeam1:       document.getElementById('scoreTeam1'),
    processingStatus: document.getElementById('processingStatus'),
    modelSelector:    document.getElementById('modelSelector'),
    modelGrid:        document.getElementById('modelGrid'),
};

// =============================================================================
//  NAVIGATION
// =============================================================================
function initNavigation() {
    elements.navItems.forEach(item => {
        item.addEventListener('click', () => switchScreen(item.dataset.screen));
    });
}

function switchScreen(screenId) {
    elements.navItems.forEach(item => item.classList.toggle('active', item.dataset.screen === screenId));
    elements.screens.forEach(screen => screen.classList.toggle('active', screen.id === `screen-${screenId}`));
    const titles = {
        upload: 'Upload Video',
        tracking: 'Detection & Tracking',
        landmarks: 'Court Landmarks',
        dashboard: 'Analytics Dashboard'
    };
    elements.pageTitle.textContent = titles[screenId] || 'Basketball Analysis';
    state.currentScreen = screenId;
    if (screenId === 'dashboard' && state.results) loadDashboardData();
}

// =============================================================================
//  UPLOAD
// =============================================================================
function initUpload() {
    // Click is handled natively by <label for="fileInput"> in HTML — no JS click handler needed
    if (!elements.uploadArea || !elements.fileInput) {
        console.error('Upload elements not found in DOM');
        return;
    }
    elements.uploadArea.addEventListener('dragover', e => {
        e.preventDefault(); elements.uploadArea.classList.add('dragover');
    });
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });
    elements.uploadArea.addEventListener('drop', e => {
        e.preventDefault(); elements.uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    elements.fileInput.addEventListener('change', e => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });
}

async function handleFile(file) {
    resetForNewVideo();  // FIX 1: clear everything before new upload
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
            if (data.is_video) showVideoPreview();
            updatePipelineBadge(data.pipeline_available);
            showModelSelector();  // FIX 3: show model selector after upload
        } else {
            alert('Upload failed: ' + data.error);
        }
    } catch (err) {
        console.error('Upload error:', err);
        alert('Upload failed. Please try again.');
    }
}

// FIX 1: Reset all state + video players when new video is selected
function resetForNewVideo() {
    state.results = null;
    state.selectedModel = null;

    // Restore upload area to initial state
    if (elements.uploadArea)   elements.uploadArea.classList.remove('hidden');
    if (elements.uploadedFile) elements.uploadedFile.classList.add('hidden');
    if (elements.previewPanel) elements.previewPanel.classList.add('hidden');
    if (elements.progressPanel) elements.progressPanel.classList.add('hidden');

    // Clear all video players
    ['trackingVideo', 'landmarksVideo', 'finalVideo', 'previewVideo'].forEach(id => {
        const v = document.getElementById(id);
        if (!v) return;
        v.pause(); v.removeAttribute('src');
        while (v.firstChild) v.removeChild(v.firstChild);
        v.load();
    });
    // Re-lock nav
    document.querySelectorAll('.nav-item:not([data-screen="upload"])').forEach(item => {
        item.style.opacity = '0.5'; item.style.pointerEvents = 'none';
    });
    if (elements.processBtn) {
        elements.processBtn.disabled = false;
        elements.processBtn.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
    }
    if (elements.modelSelector) elements.modelSelector.classList.add('hidden');
    hideOverlay();
}

function updatePipelineBadge(available) {
    const badge = document.getElementById('pipelineBadge');
    const statusText = document.getElementById('pipelineStatus');
    if (!badge || !statusText) return;
    if (available) {
        statusText.textContent = 'AI Pipeline Ready'; badge.style.color = '#00FF88';
        badge.querySelector('i').className = 'fas fa-check-circle';
    } else {
        statusText.textContent = 'DEMO MODE (No AI)'; badge.style.color = '#FF8C00';
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
        elements.fileMeta.textContent =
            `${videoInfo.width}×${videoInfo.height} • ${videoInfo.fps} FPS • ${videoInfo.frames} frames • ${videoInfo.duration}s`;
    }
}

function showVideoPreview() {
    elements.previewPanel.classList.remove('hidden');
    const v = elements.previewVideo;
    v.removeAttribute('src');
    while (v.firstChild) v.removeChild(v.firstChild);
    const src = document.createElement('source');
    src.src = '/uploaded_video?t=' + Date.now(); src.type = 'video/mp4';
    v.appendChild(src); v.load();
}

// =============================================================================
//  FIX 3: MODEL SELECTOR
// =============================================================================
async function loadModels() {
    try {
        const res = await fetch('/models');
        state.models = await res.json();
    } catch (e) { state.models = []; }
}

function showModelSelector() {
    if (!elements.modelSelector || !elements.modelGrid) return;
    elements.modelSelector.classList.remove('hidden');

    if (!state.models || state.models.length === 0) {
        elements.modelGrid.innerHTML = '<div class="model-none"><i class="fas fa-exclamation-circle"></i> No model files found in models/weights/</div>';
        return;
    }

    elements.modelGrid.innerHTML = state.models.map((m, i) => `
        <div class="model-card ${i === 0 ? 'selected' : ''}" onclick="selectModel(this,'${m.path}')" data-path="${m.path}">
            <div class="model-card-icon"><i class="fas fa-brain"></i></div>
            <div class="model-card-info">
                <span class="model-card-name" title="${m.filename}">${m.name}</span>
                <span class="model-card-size">${m.size_mb} MB</span>
            </div>
            <i class="fas fa-check-circle model-card-check"></i>
        </div>
    `).join('');

    if (state.models.length > 0) state.selectedModel = state.models[0].path;
}

function selectModel(card, path) {
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
    card.classList.add('selected');
    state.selectedModel = path;
}

// =============================================================================
//  FIX 4: PIPELINE LOADING OVERLAY
// =============================================================================
function showOverlay() {
    const ov = document.getElementById('pipelineOverlay');
    if (!ov) return;
    document.getElementById('overlayProcessing').style.display = 'block';
    document.getElementById('overlayComplete').classList.remove('show');
    // Reset step indicators
    ['detection','tracking','shots','landmarks','homography','encoding'].forEach(k => {
        const el = document.getElementById(`ostep-${k}`);
        if (el) { el.classList.remove('active','done'); }
    });
    ov.classList.add('active');
}

function hideOverlay() {
    const ov = document.getElementById('pipelineOverlay');
    if (ov) ov.classList.remove('active');
}

function showOverlayComplete() {
    document.getElementById('overlayProcessing').style.display = 'none';
    document.getElementById('overlayComplete').classList.add('show');
}

function onOverlayViewResults() {
    hideOverlay();
    document.querySelectorAll('.nav-item').forEach(item => {
        item.style.opacity = '1'; item.style.pointerEvents = 'auto';
    });
    switchScreen('tracking');
}

function updateOverlay(status) {
    const fill = document.getElementById('overlayProgressFill');
    const pct  = document.getElementById('overlayProgressPct');
    if (fill) fill.style.width = `${status.progress}%`;
    if (pct)  pct.textContent  = `${status.progress}%`;

    const stepEl = document.getElementById('overlayStep');
    if (stepEl && status.current_step) stepEl.textContent = status.current_step;

    // Determine which step is active
    const stepLower = (status.current_step || '').toLowerCase();
    let activeStep = null;
    for (const [keyword, stepId] of Object.entries(STEP_MAP)) {
        if (stepLower.includes(keyword)) { activeStep = stepId; break; }
    }
    const stepOrder = ['detection','tracking','shots','landmarks','homography','encoding'];
    const activeIdx = stepOrder.indexOf(activeStep);
    stepOrder.forEach((key, idx) => {
        const el = document.getElementById(`ostep-${key}`);
        if (!el) return;
        el.classList.remove('active','done');
        if (activeIdx >= 0) {
            if (idx === activeIdx)       el.classList.add('active');
            else if (idx < activeIdx)    el.classList.add('done');
        }
    });

    // Log tail
    const logEl = document.getElementById('overlayLog');
    if (logEl && status.logs && status.logs.length) {
        const tail = status.logs.slice(-5);
        logEl.innerHTML = tail.map((line, i) =>
            `<div class="overlay-log-line ${i === tail.length-1 ? 'latest' : ''}">${escHtml(line)}</div>`
        ).join('');
    }
}

function togglePipelineLog() {
    const panel = document.getElementById('trackingLogPanel');
    const chevron = document.getElementById('logChevron');
    if (!panel) return;
    const open = panel.style.display === 'none';
    panel.style.display = open ? 'block' : 'none';
    if (chevron) chevron.className = open ? 'fas fa-chevron-up' : 'fas fa-chevron-down';
    if (open) populateTrackingLog();
}

function populateTrackingLog() {
    fetch('/status').then(r => r.json()).then(status => {
        const el = document.getElementById('trackingLogContent');
        if (!el || !status.logs) return;
        el.innerHTML = status.logs.map(line => {
            const color = line.includes('❌') ? 'var(--accent-red)' :
                          line.includes('✅') ? 'var(--accent-green)' : 'var(--text-muted)';
            return `<div style="color:${color}">${escHtml(line)}</div>`;
        }).join('');
        el.scrollTop = el.scrollHeight;
    });
}

function escHtml(t) {
    return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// =============================================================================
//  PROCESSING
// =============================================================================
async function startProcessing() {
    if (!state.uploadedFile) return;
    elements.processBtn.disabled = true;
    elements.processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
    showOverlay();  // FIX 4: show overlay immediately

    try {
        // FIX 3: Send selected model in request body
        const response = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_path: state.selectedModel })
        });
        const data = await response.json();
        if (data.success) {
            state.isProcessing = true;
            updateProcessingStatus('processing');
            startStatusPolling();
        } else {
            hideOverlay();
            elements.processBtn.disabled = false;
            elements.processBtn.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
            alert('Could not start: ' + (data.error || 'unknown error'));
        }
    } catch (err) {
        console.error('Process error:', err);
        hideOverlay();
        elements.processBtn.disabled = false;
        elements.processBtn.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
    }
}

function startStatusPolling() {
    if (state.statusInterval) clearInterval(state.statusInterval);
    state.statusInterval = setInterval(async () => {
        try {
            const status = await fetch('/status').then(r => r.json());
            updateOverlay(status);
            updateProgressUI(status);

            if (!status.is_processing && status.has_results) {
                clearInterval(state.statusInterval);
                state.isProcessing = false;
                updateProcessingStatus('idle');
                elements.processBtn.innerHTML = '<i class="fas fa-check"></i> Complete!';
                state.results = await fetch('/results').then(r => r.json());
                loadAllVideoSources();  // FIX 1: fresh sources for new video
                loadPlayerStats();      // populate player stats sidebar panel
                showOverlayComplete();  // FIX 4: show completion
            }
        } catch (err) { console.error('Poll error:', err); }
    }, 1500);
}

function updateProgressUI(status) {
    if (elements.progressBar)  elements.progressBar.style.setProperty('--progress', `${status.progress}%`);
    if (elements.progressText) elements.progressText.textContent = `${status.progress}%`;
    if (elements.stepIndicator && status.current_step) elements.stepIndicator.textContent = status.current_step;
    if (status.logs && elements.logContent) {
        elements.logContent.innerHTML = status.logs.map(l => `<div class="log-line">${escHtml(l)}</div>`).join('');
        elements.logContent.scrollTop = elements.logContent.scrollHeight;
    }
    if (status.score) {
        if (elements.scoreTeam0) elements.scoreTeam0.textContent = status.score.team_0;
        if (elements.scoreTeam1) elements.scoreTeam1.textContent = status.score.team_1;
    }
}

function updateProcessingStatus(status) {
    const dot  = document.querySelector('.status-dot');
    const text = document.querySelector('.processing-status span');
    if (dot)  dot.className = 'status-dot ' + status;
    if (text) text.textContent = status === 'processing' ? 'Processing...' : 'Ready';
}

// FIX: trackingVideo → possession video (tracking_possession.mp4)
// FIX: removed homographyVideo (tab deleted)
function loadAllVideoSources() {
    const t = Date.now();
    const sources = {
        trackingVideo:  `/video/tracking?t=${t}`,   // show tracking_possession.mp4
        landmarksVideo: `/video/landmarks?t=${t}`,    // show tracking_landmarks.mp4
        finalVideo:     `/video/final?t=${t}`
    };
    Object.entries(sources).forEach(([id, src]) => {
        const v = document.getElementById(id);
        if (!v) return;
        v.pause(); v.removeAttribute('src');
        while (v.firstChild) v.removeChild(v.firstChild);
        const s = document.createElement('source');
        s.src = src; s.type = 'video/mp4';
        v.appendChild(s); v.load();
    });
}

// =============================================================================
//  PLAYER STATISTICS (Tracking Tab Sidebar)
// =============================================================================
async function loadPlayerStats() {
    try {
        const data = await fetch('/analytics/player_stats').then(r => r.json());
        const tbody = document.getElementById('playerStatsBody');
        if (!tbody) return;

        const players = data.players || [];
        if (players.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:12px;font-size:12px;color:var(--text-muted)">No player data available</td></tr>';
            return;
        }

        tbody.innerHTML = players.map(p => {
            // Determine team badge class
            let badgeClass = 'ps-team-unknown';
            let teamLabel  = p.team || '—';
            const tLower   = teamLabel.toLowerCase();
            if (tLower.includes('yellow') || tLower === 't1' || tLower === 'team 0') {
                badgeClass = 'ps-team-0';
            } else if (tLower.includes('blue') || tLower === 't2' || tLower === 'team 1') {
                badgeClass = 'ps-team-1';
            }
            // Shorten long team names for the badge
            if (teamLabel.length > 8) teamLabel = teamLabel.slice(0, 8);

            return `<tr>
                <td>${escHtml(p.id)}</td>
                <td><span class="ps-team-badge ${badgeClass}">${escHtml(teamLabel)}</span></td>
                <td class="dist-val">${p.distance}<span style="color:var(--text-muted);font-size:10px"> ${p.distance_unit}</span></td>
                <td class="speed-val">${p.speed}<span style="color:var(--text-muted);font-size:10px"> ${p.speed_unit}</span></td>
            </tr>`;
        }).join('');
    } catch (err) {
        console.error('Player stats error:', err);
    }
}


async function updateScore(team, action) {
    try {
        if (action === 'reset') {
            await fetch('/score', { method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({action:'reset'}) });
            if (elements.scoreTeam0) elements.scoreTeam0.textContent = '0';
            if (elements.scoreTeam1) elements.scoreTeam1.textContent = '0';
            return;
        }
        const data = await fetch('/score', { method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({team, action}) }).then(r => r.json());
        if (data.success) {
            if (elements.scoreTeam0) elements.scoreTeam0.textContent = data.score.team_0;
            if (elements.scoreTeam1) elements.scoreTeam1.textContent = data.score.team_1;
            const el = document.getElementById(team === 'team_0' ? 'scoreTeam0' : 'scoreTeam1');
            if (el) { el.style.transform = 'scale(1.4)'; setTimeout(() => el.style.transform = '', 200); }
        }
    } catch (err) { console.error('Score error:', err); }
}

// =============================================================================
//  DASHBOARD
// =============================================================================
async function loadDashboardData() {
    try {
        const possession = await fetch('/analytics/possession').then(r => r.json());
        if (possession.team_0_pct !== undefined) {
            updatePossessionChart(possession.team_0_pct, possession.team_1_pct);
        }
        const shots = await fetch('/analytics/shots').then(r => r.json());
        if (shots.made_shots) { updateShotsList(shots.made_shots); updateShotTable(shots.made_shots); }
        if (state.results && state.results.stats) {
            const s = state.results.stats;
            ['dashPlayers','dashShots','dashSpeed','dashDistance','totalFrames','fps'].forEach(id => {
                const el = document.getElementById(id);
                const map = {dashPlayers:s.players_detected, dashShots:s.shots_made,
                    dashSpeed:s.avg_speed, dashDistance:s.total_distance,
                    totalFrames:s.total_frames, fps:s.fps};
                if (el && map[id] !== undefined) el.textContent = map[id];
            });
        }
        drawSpeedChart(); updateAccuracyRing(87);
    } catch (err) { console.error('Dashboard error:', err); }
}

function updatePossessionChart(t0, t1) {
    const c = 2 * Math.PI * 80;
    const d0=document.getElementById('donutTeam0'), d1=document.getElementById('donutTeam1');
    const dp=document.getElementById('donutPct'), l0=document.getElementById('legendTeam0'), l1=document.getElementById('legendTeam1');
    if (d0) { d0.style.strokeDasharray=c; d0.style.strokeDashoffset=c*(1-t0/100); }
    if (d1) { d1.style.strokeDasharray=c; d1.style.strokeDashoffset=c*(1-t1/100); }
    if (dp) dp.textContent=`${t0.toFixed(0)}%`;
    if (l0) l0.textContent=`${t0.toFixed(0)}%`;
    if (l1) l1.textContent=`${t1.toFixed(0)}%`;
}

function updateShotsList(shots) {
    const c = document.getElementById('shotsList'); if (!c) return;
    c.innerHTML = shots.map((s,i) => `<div class="shot-item"><div class="shot-info">
        <span class="shot-time">Shot ${i+1} • Frames ${s.frames}</span>
        <span class="shot-team">Team ${i%2===0?'Yellow':'Blue'}</span></div>
        <span class="shot-conf">${(s.confidence*100).toFixed(0)}%</span></div>`).join('');
}

function updateShotTable(shots) {
    const t = document.getElementById('shotTableBody'); if (!t) return;
    t.innerHTML = shots.map((s,i) => `<tr><td>${i+1}</td><td>${s.frames}</td>
        <td><span style="color:var(--team-${i%2===0?'0':'1'})">Team ${i%2===0?'Yellow':'Blue'}</span></td>
        <td>${(s.confidence*100).toFixed(1)}%</td>
        <td><span style="color:var(--accent-green)">✓ Made</span></td></tr>`).join('');
}

function drawSpeedChart() {
    const canvas=document.getElementById('speedChart'); if (!canvas) return;
    const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    ctx.clearRect(0,0,w,h);
    const data=[]; for (let i=0;i<50;i++) data.push(2+Math.sin(i*0.2)*1.5+Math.random());
    ctx.strokeStyle='rgba(255,255,255,0.05)'; ctx.lineWidth=1;
    for (let i=0;i<=5;i++) { const y=(h-40)*(i/5)+20; ctx.beginPath(); ctx.moveTo(40,y); ctx.lineTo(w-20,y); ctx.stroke();
        ctx.fillStyle='rgba(255,255,255,0.3)'; ctx.font='11px Inter'; ctx.textAlign='right'; ctx.fillText((5-i).toFixed(1),35,y+3); }
    ctx.strokeStyle='#FF8C00'; ctx.lineWidth=2; ctx.beginPath();
    const xS=(w-60)/49, yS=(h-60)/5;
    data.forEach((v,i)=>{ const x=40+i*xS,y=h-30-v*yS; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); }); ctx.stroke();
    ctx.lineTo(40+49*xS,h-30); ctx.lineTo(40,h-30); ctx.closePath();
    ctx.fillStyle='rgba(255,140,0,0.1)'; ctx.fill();
    ctx.fillStyle='#FF8C00'; data.forEach((v,i)=>{ if(i%5===0){const x=40+i*xS,y=h-30-v*yS; ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill(); }});
}

function updateAccuracyRing(p) {
    const c=document.getElementById('accuracyCircle'), v=document.getElementById('accuracyValue');
    if (c) c.style.strokeDashoffset=2*Math.PI*45*(1-p/100);
    if (v) v.textContent=`${p}%`;
}

function initTopdownAnimation() {
    const ct=document.getElementById('topdownPlayers'); if (!ct) return;
    [{t:0,x:20,y:30},{t:0,x:25,y:50},{t:0,x:30,y:70},{t:0,x:40,y:40},{t:0,x:45,y:60},
     {t:1,x:70,y:30},{t:1,x:75,y:50},{t:1,x:80,y:70},{t:1,x:65,y:45},{t:1,x:60,y:55},
     {t:'b',x:50,y:50}].forEach(p=>{
        const d=document.createElement('div'); d.className='player-dot';
        d.style.left=`${p.x}%`; d.style.top=`${p.y}%`;
        if(p.t==='b'){d.style.background='#FFF';d.style.width='10px';d.style.height='10px';d.style.boxShadow='0 0 10px rgba(255,255,255,0.8)';}
        else d.style.background=p.t===0?'var(--team-0)':'var(--team-1)';
        ct.appendChild(d);
    });
    let a=0; setInterval(()=>{const b=ct.querySelector('.player-dot:last-child');
        if(b){a+=0.05;b.style.left=`${50+Math.sin(a)*20}%`;b.style.top=`${50+Math.cos(a*0.7)*15}%`;}},50);
}

// =============================================================================
//  INIT
// =============================================================================
document.addEventListener('DOMContentLoaded', async () => {
    initNavigation();
    initUpload();
    initTopdownAnimation();
    await loadModels();  // FIX 3: pre-load model list
    fetch('/status').then(r=>r.json()).then(s=>updatePipelineBadge(s.pipeline_available)).catch(()=>{
        const st=document.getElementById('pipelineStatus'); if(st) st.textContent='Offline';
    });
    document.querySelectorAll('.nav-item:not([data-screen="upload"])').forEach(item=>{
        item.style.opacity='0.5'; item.style.pointerEvents='none';
    });
    setTimeout(()=>updateAccuracyRing(87),500);
});