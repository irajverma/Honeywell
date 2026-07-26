/**
 * Honeywell Cyber SOC // Dashboard Logic & Analytics Visualization Engine
 * Connects to Phase 4 generated data feed (data/dashboard_feed.json) and renders interactive SOC widgets.
 */

// Global State
let dashboardFeed = null;
let filteredAlerts = [];
let currentPage = 1;
const pageSize = 25;
let currentSortCol = 'hybrid_risk';
let currentSortAsc = false;

// Chart Instances
let chartModelComp = null;
let chartAttackRecall = null;
let chartTimeline = null;
let currentChartMetric = 'prauc'; // 'prauc' or 'recall'

// Initialize Dashboard on DOM Load
document.addEventListener('DOMContentLoaded', async () => {
    initClock();
    await loadDashboardData();
});

/**
 * Live UTC Clock display
 */
function initClock() {
    const updateTime = () => {
        const now = new Date();
        const timeStr = now.toISOString().split('T')[1].split('.')[0] + ' UTC';
        const clockEl = document.getElementById('utc-clock');
        if (clockEl) clockEl.textContent = timeStr;
    };
    updateTime();
    setInterval(updateTime, 1000);
}

/**
 * Load JSON data feed from ML pipeline
 */
async function loadDashboardData() {
    const paths = [
        'data/dashboard_feed.json',
        './data/dashboard_feed.json',
        '../data/dashboard_feed.json'
    ];
    let loadedData = null;
    let fetchError = null;

    for (const p of paths) {
        try {
            const response = await fetch(p);
            if (response.ok) {
                loadedData = await response.json();
                console.log(`Successfully fetched telemetry feed from: ${p}`);
                break;
            }
        } catch (e) {
            fetchError = e;
        }
    }

    if (loadedData) {
        dashboardFeed = loadedData;
        renderKPIs(dashboardFeed.kpis);
        initCharts(dashboardFeed);
        
        // Initialize Alert Triage Table
        filteredAlerts = [...dashboardFeed.alerts];
        sortAlerts('hybrid_risk', false); // Initial sort by highest risk
    } else {
        console.error("Failed to load dashboard feed:", fetchError);
        document.getElementById('alerts-tbody').innerHTML = `
            <tr>
                <td colspan="7" style="text-align:center; padding:30px; color:#ff3366;">
                    ⚠️ Error Loading Telemetry Feed: Ensure local web server is running from repository root (e.g., 'python -m http.server 8000') and access via http://localhost:8000/
                </td>
            </tr>
        `;
    }
}

/**
 * Populate Top KPI Summary Cards
 */
function renderKPIs(kpis) {
    document.getElementById('kpi-total-events').textContent = kpis.total_events.toLocaleString();
    document.getElementById('kpi-total-alerts').textContent = kpis.total_alerts_top1pct.toLocaleString();
    document.getElementById('kpi-critical-threats').textContent = kpis.critical_threats.toLocaleString();
    document.getElementById('kpi-prauc').textContent = kpis.pr_auc.toFixed(4);
}

/**
 * Initialize Chart.js Analytics Visualizations
 */
function initCharts(feed) {
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";

    renderModelComparisonChart(feed.model_comparison, currentChartMetric);
    renderAttackRecallChart(feed.model_comparison);
    renderTimelineChart(feed.timeline);
}

/**
 * Chart 1: 4-Way Model Comparison Bar Chart
 */
function renderModelComparisonChart(compData, metric) {
    const ctx = document.getElementById('modelComparisonChart').getContext('2d');
    if (chartModelComp) chartModelComp.destroy();

    const modelKeys = [
        'Baseline Profiler (Static Rules)',
        'Isolation Forest (Tabular ML)',
        'LSTM Autoencoder (Sequence ML)',
        'Hybrid Ensemble (Monotonic Blend)'
    ];

    let dataValues = [];
    let chartLabel = '';
    let bgColor = '';
    let borderColor = '';

    if (metric === 'prauc') {
        chartLabel = 'Core Anomaly PR-AUC (Primary Metric)';
        dataValues = modelKeys.map(k => compData[k]?.pr_auc || 0);
        bgColor = ['rgba(0, 240, 255, 0.25)', 'rgba(255, 170, 0, 0.25)', 'rgba(255, 0, 127, 0.25)', 'rgba(0, 255, 102, 0.4)'];
        borderColor = ['#00f0ff', '#ffaa00', '#ff007f', '#00ff66'];
    } else {
        chartLabel = 'Recall @ Top 1% Alert Budget';
        dataValues = modelKeys.map(k => compData[k]?.recall || 0);
        bgColor = ['rgba(0, 240, 255, 0.25)', 'rgba(255, 170, 0, 0.25)', 'rgba(255, 0, 127, 0.25)', 'rgba(0, 255, 102, 0.4)'];
        borderColor = ['#00f0ff', '#ffaa00', '#ff007f', '#00ff66'];
    }

    chartModelComp = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [
                'Static Baseline\n(Rules)',
                'Isolation Forest\n(Tabular ML)',
                'LSTM Autoencoder\n(Sequence ML)',
                'Hybrid Ensemble\n(Monotonic Blend)'
            ],
            datasets: [{
                label: chartLabel,
                data: dataValues,
                backgroundColor: bgColor,
                borderColor: borderColor,
                borderWidth: 2,
                borderRadius: 8,
                barThickness: 50
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${chartLabel}: ${ctx.raw.toFixed(4)}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', font: { weight: 'bold' } }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#f0f4ff', font: { weight: '600', size: 11 } }
                }
            }
        }
    });
}

function switchChartMetric(metric) {
    currentChartMetric = metric;
    document.getElementById('btn-chart-prauc').classList.toggle('active', metric === 'prauc');
    document.getElementById('btn-chart-recall').classList.toggle('active', metric === 'recall');
    if (dashboardFeed) {
        renderModelComparisonChart(dashboardFeed.model_comparison, metric);
    }
}

/**
 * Chart 2: Per-Attack-Type Recall Grouped Bar Chart
 */
function renderAttackRecallChart(compData) {
    const ctx = document.getElementById('attackRecallChart').getContext('2d');
    if (chartAttackRecall) chartAttackRecall.destroy();

    const attackTypes = ['credential_stuffing', 'brute_force', 'lateral_movement', 'exfiltration', 'impossible_travel'];
    const formattedLabels = ['Cred Stuffing', 'Brute Force', 'Lateral Move', 'Exfiltration', 'Imposs Travel'];

    const getRecallArray = (modelName) => {
        return attackTypes.map(t => (compData[modelName]?.per_attack_recall[t] || 0) * 100);
    };

    chartAttackRecall = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: formattedLabels,
            datasets: [
                {
                    label: 'Static Baseline',
                    data: getRecallArray('Baseline Profiler (Static Rules)'),
                    backgroundColor: 'rgba(0, 240, 255, 0.6)',
                    borderColor: '#00f0ff',
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: 'LSTM Sequence',
                    data: getRecallArray('LSTM Autoencoder (Sequence ML)'),
                    backgroundColor: 'rgba(255, 0, 127, 0.6)',
                    borderColor: '#ff007f',
                    borderWidth: 1,
                    borderRadius: 4
                },
                {
                    label: 'Hybrid Ensemble',
                    data: getRecallArray('Hybrid Ensemble (Monotonic Blend)'),
                    backgroundColor: 'rgba(0, 255, 102, 0.8)',
                    borderColor: '#00ff66',
                    borderWidth: 1,
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#f0f4ff', boxWidth: 12, font: { size: 11, weight: '600' } }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}% Recall`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: 'Recall % (@ Top 1% Budget)', color: '#64748b', font: { size: 11 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#f0f4ff', font: { weight: '600', size: 11 } }
                }
            }
        }
    });
}

/**
 * Chart 3: 60-Day Temporal Threat Timeline Chart
 */
function renderTimelineChart(timeline) {
    const ctx = document.getElementById('timelineChart').getContext('2d');
    if (chartTimeline) chartTimeline.destroy();

    chartTimeline = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: timeline.days.map(d => d.slice(5)), // MM-DD
            datasets: [
                {
                    label: 'Total Telemetry Volume',
                    data: timeline.total_events,
                    type: 'bar',
                    backgroundColor: 'rgba(0, 240, 255, 0.2)',
                    borderColor: '#00f0ff',
                    borderWidth: 1,
                    borderRadius: 4,
                    yAxisID: 'y'
                },
                {
                    label: 'Top 1% SOC Escalations',
                    data: timeline.alerts,
                    type: 'line',
                    borderColor: '#ff3366',
                    backgroundColor: 'rgba(255, 51, 102, 0.25)',
                    borderWidth: 3,
                    pointBackgroundColor: '#ff3366',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Total Daily Events', color: '#00f0ff' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Top Alert Volume', color: '#ff3366' },
                    grid: { display: false },
                    min: 0
                },
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 15, color: '#94a3b8' }
                }
            }
        }
    });
}

/**
 * Triage Console Filtering & Search Logic
 */
function filterAlerts() {
    if (!dashboardFeed) return;
    const query = document.getElementById('alert-search').value.toLowerCase().trim();
    const severityVal = document.getElementById('severity-filter').value;
    const attackVal = document.getElementById('attack-filter').value;

    filteredAlerts = dashboardFeed.alerts.filter(alert => {
        const matchesQuery = !query || 
            alert.entity_id.toLowerCase().includes(query) ||
            alert.resource_accessed.toLowerCase().includes(query) ||
            alert.explanation.toLowerCase().includes(query);
        
        const matchesSev = severityVal === 'ALL' || alert.severity === severityVal;
        const matchesAttack = attackVal === 'ALL' || alert.attack_subtype === attackVal;

        return matchesQuery && matchesSev && matchesAttack;
    });

    currentPage = 1;
    renderTable();
}

function resetFilters() {
    document.getElementById('alert-search').value = '';
    document.getElementById('severity-filter').value = 'ALL';
    document.getElementById('attack-filter').value = 'ALL';
    filterAlerts();
}

/**
 * Table Sorting
 */
function sortAlerts(col, forceAsc = null) {
    if (forceAsc !== null) {
        currentSortAsc = forceAsc;
    } else if (currentSortCol === col) {
        currentSortAsc = !currentSortAsc;
    } else {
        currentSortCol = col;
        currentSortAsc = false;
    }

    filteredAlerts.sort((a, b) => {
        let valA = a[col];
        let valB = b[col];

        if (typeof valA === 'number') {
            return currentSortAsc ? valA - valB : valB - valA;
        } else {
            return currentSortAsc ? strCompare(valA, valB) : strCompare(valB, valA);
        }
    });

    renderTable();
}

function strCompare(a, b) {
    return String(a).localeCompare(String(b));
}

/**
 * Render Table Page
 */
function renderTable() {
    const tbody = document.getElementById('alerts-tbody');
    tbody.innerHTML = '';

    const totalAlerts = filteredAlerts.length;
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = Math.min(startIndex + pageSize, totalAlerts);
    const pageAlerts = filteredAlerts.slice(startIndex, endIndex);

    if (totalAlerts === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:30px; color:#64748b;">No security escalations match current filter criteria.</td></tr>`;
        updatePagination(0, 0, 0);
        return;
    }

    pageAlerts.forEach((alert, idx) => {
        const tr = document.createElement('tr');
        tr.onclick = () => openModal(alert);

        let sevBadgeClass = 'badge-cyan';
        if (alert.severity === 'Critical') sevBadgeClass = 'badge-red pulsing-badge';
        if (alert.severity === 'High') sevBadgeClass = 'badge-amber';

        let subBadgeClass = 'badge-cyan';
        if (alert.attack_subtype === 'credential_stuffing') subBadgeClass = 'badge-amber';
        if (alert.attack_subtype === 'lateral_movement') subBadgeClass = 'badge-magenta';
        if (alert.attack_subtype === 'brute_force') subBadgeClass = 'badge-red';

        let barFillClass = 'fill-medium';
        if (alert.hybrid_risk >= 99.0) barFillClass = 'fill-critical';
        else if (alert.hybrid_risk >= 95.0) barFillClass = 'fill-high';

        tr.innerHTML = `
            <td class="font-mono">${alert.timestamp.split(' ')[1] || alert.timestamp}</td>
            <td class="font-mono" style="font-weight:700; color:#fff;">${alert.entity_id}</td>
            <td><span class="badge ${sevBadgeClass}">${alert.severity}</span></td>
            <td>
                <div class="risk-cell">
                    <span class="font-mono" style="font-weight:700;">${alert.hybrid_risk.toFixed(1)}</span>
                    <div class="mini-bar-bg">
                        <div class="mini-bar-fill ${barFillClass}" style="width: ${alert.hybrid_risk}%"></div>
                    </div>
                </div>
            </td>
            <td><span class="badge ${subBadgeClass}">${alert.attack_subtype}</span></td>
            <td style="max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #cbd5e1;">${alert.explanation}</td>
            <td><button class="btn-view" onclick="event.stopPropagation(); openModalDirect(${startIndex + idx})">Triage</button></td>
        `;
        tbody.appendChild(tr);
    });

    updatePagination(startIndex + 1, endIndex, totalAlerts);
}

function updatePagination(start, end, total) {
    document.getElementById('pagination-info').textContent = total > 0 ? `Showing ${start} to ${end} of ${total.toLocaleString()} alerts` : `Showing 0 alerts`;
    document.getElementById('btn-prev').disabled = currentPage <= 1;
    document.getElementById('btn-next').disabled = currentPage * pageSize >= total;
}

function changePage(delta) {
    currentPage += delta;
    renderTable();
}

/**
 * Modal Detail Drawer Logic
 */
function openModalDirect(idx) {
    const alert = filteredAlerts[idx];
    if (alert) openModal(alert);
}

function openModal(alert) {
    document.getElementById('modal-severity').textContent = alert.severity;
    document.getElementById('modal-severity').className = `badge ${alert.severity === 'Critical' ? 'badge-red pulsing-badge' : alert.severity === 'High' ? 'badge-amber' : 'badge-cyan'}`;
    document.getElementById('modal-title').textContent = `Incident Details // Entity ${alert.entity_id}`;
    
    document.getElementById('modal-ts').textContent = alert.timestamp;
    document.getElementById('modal-entity-type').textContent = alert.entity_type;
    document.getElementById('modal-attack-type').textContent = alert.attack_subtype.toUpperCase();
    document.getElementById('modal-label').textContent = alert.label.toUpperCase();
    
    document.getElementById('modal-geo').textContent = alert.geo_location;
    document.getElementById('modal-resource').textContent = alert.resource_accessed;
    document.getElementById('modal-device').textContent = alert.device_fingerprint;

    // Risk bars
    document.getElementById('modal-hybrid-val').textContent = `${alert.hybrid_risk.toFixed(1)} / 100`;
    document.getElementById('modal-hybrid-bar').style.width = `${alert.hybrid_risk}%`;

    document.getElementById('modal-base-val').textContent = `${alert.baseline_risk.toFixed(1)} / 100`;
    document.getElementById('modal-base-bar').style.width = `${alert.baseline_risk}%`;

    document.getElementById('modal-lstm-val').textContent = `${alert.lstm_risk.toFixed(1)} / 100`;
    document.getElementById('modal-lstm-bar').style.width = `${alert.lstm_risk}%`;

    document.getElementById('modal-explanation').textContent = alert.explanation;

    const modal = document.getElementById('alert-modal');
    modal.style.display = 'flex';
    setTimeout(() => modal.classList.add('active'), 10);
}

function closeModalDirect() {
    const modal = document.getElementById('alert-modal');
    modal.classList.remove('active');
    setTimeout(() => modal.style.display = 'none', 300);
}

function closeModal(event) {
    if (event.target === document.getElementById('alert-modal')) {
        closeModalDirect();
    }
}

// --- LIVE ATTACK REPLAY ANIMATION ENGINE ---
let isReplaying = false;
let replayInterval = null;

function toggleAttackReplay() {
    if (isReplaying) {
        stopAttackReplay();
    } else {
        startAttackReplay();
    }
}

function stopAttackReplay() {
    isReplaying = false;
    if (replayInterval) clearInterval(replayInterval);
    const btn = document.getElementById('btn-replay-attack');
    const banner = document.getElementById('replay-status-banner');
    if (btn) {
        btn.classList.remove('active-replay');
        btn.innerHTML = '⚡ Replay Live Attack';
    }
    if (banner) banner.classList.add('hidden');
    
    if (dashboardFeed) {
        filteredAlerts = [...dashboardFeed.alerts];
        renderAlertsTable();
    }
}

function startAttackReplay() {
    if (!dashboardFeed || !dashboardFeed.alerts) return;
    
    isReplaying = true;
    const btn = document.getElementById('btn-replay-attack');
    const banner = document.getElementById('replay-status-banner');
    const statusText = document.getElementById('replay-status-text');
    const counterEl = document.getElementById('replay-counter');
    
    if (btn) {
        btn.classList.add('active-replay');
        btn.innerHTML = '⏹ Stop Replay';
    }
    if (banner) banner.classList.remove('hidden');

    const targetEntity = 'user_018';
    const candidateAlert = dashboardFeed.alerts.find(a => a.entity_id === targetEntity && a.timestamp.includes('2026-06-29')) || dashboardFeed.alerts.find(a => a.entity_id === targetEntity) || dashboardFeed.alerts[0];
    
    const sampleEvents = dashboardFeed.alerts.filter(a => a.entity_id !== targetEntity).slice(0, 11);
    sampleEvents.splice(6, 0, candidateAlert);

    const tbody = document.getElementById('alerts-tbody');
    tbody.innerHTML = '';
    
    let currentIndex = 0;

    replayInterval = setInterval(() => {
        if (currentIndex >= sampleEvents.length) {
            clearInterval(replayInterval);
            if (statusText) statusText.innerHTML = `✅ <strong>REPLAY COMPLETE:</strong> Lateral Movement attack for <code>user_018</code> successfully flagged into Top SOC Queue by Monotonic Hybrid Ensemble!`;
            if (counterEl) counterEl.textContent = `12 / 12 Streamed`;
            return;
        }

        const alert = sampleEvents[currentIndex];
        currentIndex++;

        if (counterEl) counterEl.textContent = `${currentIndex} / ${sampleEvents.length} Streamed`;
        if (statusText) statusText.innerHTML = `LIVE TELEMETRY INGESTION: Processing event <code>${alert.entity_id}</code> [${alert.attack_subtype.toUpperCase()}]`;

        const tr = document.createElement('tr');
        tr.className = 'row-replay-highlight';
        if (alert.entity_id === targetEntity) {
            tr.style.background = 'rgba(255, 0, 127, 0.25)';
            tr.style.borderLeft = '4px solid #ff007f';
        }

        const sevBadge = alert.severity === 'CRITICAL' ? 'badge-red' : (alert.severity === 'HIGH' ? 'badge-orange' : 'badge-yellow');
        
        tr.innerHTML = `
            <td class="mono-font">${alert.timestamp.split(' ')[1] || alert.timestamp}</td>
            <td><strong>${alert.entity_id}</strong></td>
            <td><span class="badge ${sevBadge}">${alert.severity}</span></td>
            <td>
                <div class="risk-bar-container">
                    <div class="risk-bar-fill" style="width: ${alert.hybrid_risk}%"></div>
                    <span class="risk-score-text">${alert.hybrid_risk.toFixed(1)}</span>
                </div>
            </td>
            <td><span class="subtype-tag">${alert.attack_subtype.toUpperCase()}</span></td>
            <td class="narrative-cell">${alert.explanation}</td>
            <td><button class="btn-triage" onclick="openAlertModal('${alert.entity_id}', '${alert.timestamp}')">Triage</button></td>
        `;

        tbody.insertBefore(tr, tbody.firstChild);
    }, 1200);
}
