// ==========================================
// AQMSS TradingView-Style Dashboard
// AI Market Quality Scoring System
// ==========================================

const API_URL = window.location.origin + '/api';

// Global State
let markets = [];
let filteredMarkets = [];
let aqmssData = null;
let charts = {};
let autoRefreshInterval = null;
let currentSortField = 'quality_score';
let currentSortAsc = false;

// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    console.log('🚀 Initializing AQMSS Dashboard...');

    setupNavigation();
    setupTableSorting();

    document.getElementById('search-market')?.addEventListener('input', filterMarkets);
    document.getElementById('quality-filter')?.addEventListener('change', filterMarkets);

    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }

    window.onclick = (event) => {
        const modal = document.getElementById('marketModal');
        if (event.target === modal) closeModal();
    };

    updateTime();
    setInterval(updateTime, 1000);

    // Sequential data loading
    try {
        console.log('📊 Step 1: Loading markets...');
        await loadMarkets();

        console.log('📊 Step 2: Loading AQMSS data...');
        await loadAQMSSData();

        console.log('📊 Step 3: Loading stats...');
        await loadDashboardStats();

        console.log('📊 Step 4: Loading insights...');
        await loadInsights();

        console.log('📊 Step 5: Loading AI pulse...');
        await loadAIPulse();

        console.log('✅ All data loaded successfully');
    } catch (err) {
        console.error('❌ Initialization error:', err);
    }
}

function setupNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchTab(link.dataset.tab);
        });
    });
}

function setupTableSorting() {
    document.querySelectorAll('.sortable').forEach(header => {
        header.addEventListener('click', () => {
            sortTable(header.dataset.sort);
        });
    });
}

// ==========================================
// DATA LOADING
// ==========================================

async function loadMarkets() {
    try {
        const response = await axios.get(`${API_URL}/markets`);
        if (response.data.success) {
            markets = response.data.data;
            filteredMarkets = [...markets];
            console.log(`✅ Loaded ${markets.length} markets`);
            renderMarkets();
            updateDataPointsCount();
            return true;
        } else {
            console.error('Markets API returned success: false');
            return false;
        }
    } catch (error) {
        console.error('❌ Error loading markets:', error);
        const tbody = document.getElementById('table-body');
        if (tbody) {
            tbody.innerHTML = `
                <tr><td colspan="10" style="text-align:center;padding:2rem;color:var(--tv-red);">
                    ❌ Failed to load market data. Check if server is running.
                </td></tr>
            `;
        }
        return false;
    }
}

async function loadAQMSSData() {
    try {
        const response = await axios.get(`${API_URL}/aqmss`);
        if (response.data.success) {
            aqmssData = response.data.data;
            console.log('✅ AQMSS data loaded:', aqmssData);
            updateAQMSSDisplay();
            return true;
        }
    } catch (error) {
        console.error('❌ Error loading AQMSS data:', error);
    }
    return false;
}

async function loadDashboardStats() {
    try {
        const response = await axios.get(`${API_URL}/dashboard/stats`);
        if (response.data.success) {
            const stats = response.data.data;
            console.log('✅ Stats loaded:', stats);
            updateStatsDisplay(stats);
            return true;
        }
    } catch (error) {
        console.error('❌ Error loading stats:', error);
    }
    return false;
}

async function loadInsights() {
    try {
        const response = await axios.get(`${API_URL}/ai/insights`);
        if (response.data.success) {
            renderInsights(response.data.data);
            return true;
        } else {
            renderInsights([]);
        }
    } catch (error) {
        console.error('❌ Error loading insights:', error);
        const container = document.getElementById('insights-container');
        if (container) {
            container.innerHTML = `
                <div class="insight-card warning">
                    <div class="insight-type">⚠️ ERROR</div>
                    <div class="insight-title">Unable to Load Insights</div>
                    <div class="insight-description">${error.message || 'Please refresh'}</div>
                </div>
            `;
        }
    }
    return false;
}

async function loadAIPulse() {
    const pulseEl = document.getElementById('ai-pulse');
    if (!pulseEl) return false;

    try {
        if (markets.length === 0) {
            pulseEl.textContent = 'No market data available';
            return false;
        }

        const avgQuality = markets.reduce((sum, m) => sum + (m.quality_score || 0), 0) / markets.length;
        const bullish = markets.filter(m => String(m.momentum).toUpperCase() === 'BULLISH').length;
        const bearish = markets.filter(m => String(m.momentum).toUpperCase() === 'BEARISH').length;
        const highQ = markets.filter(m => (m.quality_score || 0) > 75).length;

        let emoji, text;
        if (avgQuality > 70) {
            emoji = '🟢';
            text = `Strong conditions! ${highQ} high-quality markets. ${bullish} bullish trends detected.`;
        } else if (avgQuality > 55) {
            emoji = '🟡';
            text = `Moderate conditions. ${bullish} bullish vs ${bearish} bearish. ${highQ} high-quality picks.`;
        } else {
            emoji = '🔴';
            text = `Weak conditions. Exercise caution. Focus on highest quality picks only.`;
        }

        // Add AQMSS info if available
        if (aqmssData && aqmssData.current) {
            const score = aqmssData.current.total_score;
            const cond = aqmssData.current.market_condition;
            text += ` AQMSS Core: ${score.toFixed(2)}/10 (${cond})`;
        }

        pulseEl.innerHTML = `${emoji} ${text}`;
        console.log('✅ AI Pulse rendered');
        return true;
    } catch (error) {
        console.error('❌ Error in AI pulse:', error);
        pulseEl.textContent = '⚠️ Unable to generate pulse';
        return false;
    }
}

// ==========================================
// AQMSS DISPLAY
// ==========================================

function updateAQMSSDisplay() {
    if (!aqmssData) return;

    const accuracyEl = document.getElementById('model-accuracy');
    if (accuracyEl && aqmssData.model_metrics) {
        const acc = (aqmssData.model_metrics.roc_auc * 100).toFixed(1);
        accuracyEl.textContent = acc + '%';
    }

    const dataPointsEl = document.getElementById('data-points');
    if (dataPointsEl) {
        dataPointsEl.textContent = aqmssData.history_count || markets.length;
    }

    const aqmssScoreEl = document.getElementById('aqmss-score');
    if (aqmssScoreEl && aqmssData.current) {
        aqmssScoreEl.textContent = aqmssData.current.total_score.toFixed(2) + '/10';
    }

    const aqmssCondEl = document.getElementById('aqmss-condition');
    if (aqmssCondEl && aqmssData.current) {
        aqmssCondEl.textContent = aqmssData.current.market_condition;
    }

    const aiProbEl = document.getElementById('ai-probability');
    if (aiProbEl && aqmssData.current && aqmssData.current.ai_high_quality_prob != null) {
        aiProbEl.textContent = (aqmssData.current.ai_high_quality_prob * 100).toFixed(1) + '%';
    }
}

// ==========================================
// UI RENDERING
// ==========================================

function updateStatsDisplay(stats) {
    const mapping = {
        'stat-total': stats.total_markets,
        'stat-avg': stats.avg_quality.toFixed(1) + '%',
        'stat-high': stats.high_quality_count,
        'stat-bullish': stats.bullish_count,
        'stat-bearish': stats.bearish_count,
        'stat-volatile': stats.volatile_count
    };

    Object.entries(mapping).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = value;
            el.style.transform = 'scale(1.1)';
            setTimeout(() => { el.style.transform = 'scale(1)'; }, 200);
        }
    });
}

function renderMarkets() {
    const tbody = document.getElementById('table-body');
    if (!tbody) return;

    tbody.innerHTML = '';

    if (filteredMarkets.length === 0) {
        tbody.innerHTML = `
            <tr><td colspan="10" style="text-align:center;padding:2rem;">
                📭 No markets found matching your criteria
            </td></tr>
        `;
        return;
    }

    filteredMarkets.forEach(market => {
        const row = document.createElement('tr');

        const change = parseFloat(market.change || 0);
        const changeClass = change > 0 ? 'positive' : change < 0 ? 'negative' : '';
        const changeSign = change > 0 ? '+' : '';

        const quality = parseFloat(market.quality_score || 0);
        let qClass = 'quality-poor';
        if (quality >= 75) qClass = 'quality-excellent';
        else if (quality >= 60) qClass = 'quality-good';
        else if (quality >= 45) qClass = 'quality-medium';

        const mom = String(market.momentum || 'NEUTRAL').toUpperCase();
        const momClass = mom.includes('BULL') ? 'bullish' : mom.includes('BEAR') ? 'bearish' : 'neutral';

        row.innerHTML = `
            <td class="symbol-cell" onclick="showMarketDetail('${escapeHtml(market.symbol)}')" title="Click for details">
                ${escapeHtml(market.symbol)}
            </td>
            <td class="price">${formatPrice(market.price)}</td>
            <td><span class="change ${changeClass}">${changeSign}${change.toFixed(2)}%</span></td>
            <td>${formatVolume(market.volume)}</td>
            <td>
                <div style="display:flex;align-items:center;gap:0.5rem;">
                    <div style="flex:1;background:var(--border-default);height:4px;border-radius:2px;overflow:hidden;">
                        <div style="width:${Math.min(market.liquidity || 0, 100)}%;height:100%;background:var(--tv-blue);"></div>
                    </div>
                    <span>${(market.liquidity || 0).toFixed(1)}</span>
                </div>
            </td>
            <td>${(market.volatility || 0).toFixed(2)}%</td>
            <td><span class="momentum ${momClass}">${mom}</span></td>
            <td><span class="quality-score ${qClass}">${quality.toFixed(1)}</span></td>
            <td style="color:var(--text-secondary);font-size:0.8rem;">${escapeHtml(market.market_condition || '')}</td>
            <td><button class="btn-detail" onclick="showMarketDetail('${escapeHtml(market.symbol)}')">View</button></td>
        `;

        tbody.appendChild(row);
    });
}

function renderInsights(insights) {
    const container = document.getElementById('insights-container');
    if (!container) return;

    container.innerHTML = '';

    if (!insights || insights.length === 0) {
        container.innerHTML = '<div class="loading-state"><p>No insights available</p></div>';
        return;
    }

    insights.forEach(insight => {
        const card = document.createElement('div');
        card.className = `insight-card ${insight.type || 'info'}`;

        const icons = { opportunity: '✨', warning: '⚠️', bullish: '📈', bearish: '📉', info: 'ℹ️', aqmss: '🎯' };
        const icon = icons[insight.type] || '📊';

        const colors = { HIGH: '#EF5350', MEDIUM: '#FF9800', LOW: '#8B949E' };
        const pColor = colors[insight.priority] || colors.LOW;

        card.innerHTML = `
            <div class="insight-type">${icon} ${(insight.type || 'info').toUpperCase()}
                <span style="color:${pColor};font-weight:700;margin-left:0.5rem;">[${insight.priority || 'LOW'}]</span>
            </div>
            <div class="insight-title">${insight.title}</div>
            <div class="insight-description">${insight.description}</div>
        `;
        container.appendChild(card);
    });
}

// ==========================================
// FILTERING & SORTING
// ==========================================

function filterMarkets() {
    const search = (document.getElementById('search-market')?.value || '').toLowerCase();
    const qualityFilter = document.getElementById('quality-filter')?.value || '';

    filteredMarkets = markets.filter(m => {
        const matchSearch = !search ||
            m.symbol.toLowerCase().includes(search) ||
            (m.name && m.name.toLowerCase().includes(search));

        let matchQuality = true;
        if (qualityFilter) {
            const s = m.quality_score || 0;
            if (qualityFilter === 'high' && s <= 75) matchQuality = false;
            if (qualityFilter === 'medium' && (s > 75 || s < 50)) matchQuality = false;
            if (qualityFilter === 'low' && s >= 50) matchQuality = false;
        }
        return matchSearch && matchQuality;
    });

    renderMarkets();
}

function sortTable(field) {
    if (currentSortField === field) {
        currentSortAsc = !currentSortAsc;
    } else {
        currentSortField = field;
        currentSortAsc = false;
    }

    filteredMarkets.sort((a, b) => {
        let aVal = a[field];
        let bVal = b[field];
        if (typeof aVal === 'string') {
            aVal = aVal.toLowerCase();
            bVal = (bVal || '').toLowerCase();
        }
        const cmp = aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
        return currentSortAsc ? cmp : -cmp;
    });

    document.querySelectorAll('.sortable').forEach(header => {
        const arrow = header.querySelector('.sort-arrow');
        if (arrow) {
            if (header.dataset.sort === field) {
                arrow.textContent = currentSortAsc ? '↑' : '↓';
                arrow.style.opacity = '1';
            } else {
                arrow.textContent = '↕';
                arrow.style.opacity = '0.5';
            }
        }
    });

    renderMarkets();
}

// ==========================================
// MARKET DETAILS MODAL
// ==========================================

async function showMarketDetail(symbol) {
    try {
        const response = await axios.get(`${API_URL}/market/${symbol}`);
        if (response.data.success) {
            displayMarketModal(response.data.data);
        }
    } catch (error) {
        console.error('Error loading market detail:', error);
        showNotification('Failed to load market details', 'error');
    }
}

function displayMarketModal(market) {
    const modal = document.getElementById('marketModal');
    const body = document.getElementById('modal-body');
    if (!modal || !body) return;

    const change = parseFloat(market.change || 0);
    const changeClass = change > 0 ? 'positive' : change < 0 ? 'negative' : '';
    const changeSign = change > 0 ? '+' : '';

    body.innerHTML = `
        <h2 style="margin-bottom:0.5rem;">${escapeHtml(market.symbol)}</h2>
        <p style="color:var(--text-secondary);margin-bottom:2rem;">${escapeHtml(market.name || 'Market')}</p>

        <div class="modal-detail-grid">
            <div class="detail-item">
                <div class="detail-label">Current Price</div>
                <div class="detail-value">${formatPrice(market.price)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">24H Change</div>
                <div class="detail-value ${changeClass}">${changeSign}${change.toFixed(2)}%</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Volume (24H)</div>
                <div class="detail-value">${formatVolume(market.volume)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Quality Score</div>
                <div class="detail-value" style="color:${getQualityColor(market.quality_score)};">
                    ${(market.quality_score || 0).toFixed(1)}/100
                </div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Liquidity</div>
                <div class="detail-value">${(market.liquidity || 0).toFixed(1)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Volatility</div>
                <div class="detail-value">${(market.volatility || 0).toFixed(2)}%</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Momentum</div>
                <div class="detail-value">${market.momentum || 'N/A'}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Condition</div>
                <div class="detail-value">${market.market_condition || 'N/A'}</div>
            </div>
        </div>

        <div style="margin-top:2rem;padding-top:1.5rem;border-top:1px solid var(--border-default);">
            <button class="btn-refresh" onclick="analyzeWithAI('${escapeHtml(market.symbol)}')" style="width:100%;">
                🤖 Get AI Analysis
            </button>
        </div>
    `;

    modal.style.display = 'block';
}

function getQualityColor(score) {
    score = score || 0;
    if (score >= 75) return '#26A69A';
    if (score >= 60) return '#7CB342';
    if (score >= 45) return '#FF9800';
    return '#EF5350';
}

function closeModal() {
    const modal = document.getElementById('marketModal');
    if (modal) modal.style.display = 'none';
}

// ==========================================
// CHARTS
// ==========================================

function initializeCharts() {
    if (!markets.length) {
        console.warn('No market data for charts');
        return;
    }
    console.log('Initializing charts with', markets.length, 'markets');
    initQualityChart();
    initMomentumChart();
    initVolatilityChart();
    initLiquidityChart();

    if (aqmssData && aqmssData.current) {
        initAQMSSFactorChart();
    }
}

function initQualityChart() {
    const ctx = document.getElementById('qualityChart');
    if (!ctx) return;

    const excellent = markets.filter(m => (m.quality_score || 0) >= 75).length;
    const good = markets.filter(m => (m.quality_score || 0) >= 60 && m.quality_score < 75).length;
    const medium = markets.filter(m => (m.quality_score || 0) >= 45 && m.quality_score < 60).length;
    const poor = markets.filter(m => (m.quality_score || 0) < 45).length;

    if (charts.quality) charts.quality.destroy();

    charts.quality = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Excellent (75+)', 'Good (60-75)', 'Medium (45-60)', 'Poor (<45)'],
            datasets: [{
                data: [excellent, good, medium, poor],
                backgroundColor: ['#26A69A', '#7CB342', '#FF9800', '#EF5350'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { position: 'bottom', labels: { color: '#E6EDF3', padding: 15 } } }
        }
    });
}

function initMomentumChart() {
    const ctx = document.getElementById('momentumChart');
    if (!ctx) return;

    const bullish = markets.filter(m => String(m.momentum).toUpperCase().includes('BULLISH')).length;
    const bearish = markets.filter(m => String(m.momentum).toUpperCase().includes('BEARISH')).length;
    const neutral = markets.filter(m => String(m.momentum).toUpperCase().includes('NEUTRAL')).length;

    if (charts.momentum) charts.momentum.destroy();

    charts.momentum = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Bullish', 'Bearish', 'Neutral'],
            datasets: [{
                label: 'Market Count',
                data: [bullish, bearish, neutral],
                backgroundColor: ['#26A69A', '#EF5350', '#8B949E'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { ticks: { color: '#E6EDF3' }, grid: { color: '#30363D' } },
                x: { ticks: { color: '#E6EDF3' }, grid: { display: false } }
            }
        }
    });
}

function initVolatilityChart() {
    const ctx = document.getElementById('volatilityChart');
    if (!ctx) return;

    const low = markets.filter(m => (m.volatility || 0) < 20).length;
    const mod = markets.filter(m => (m.volatility || 0) >= 20 && m.volatility < 40).length;
    const high = markets.filter(m => (m.volatility || 0) >= 40).length;

    if (charts.volatility) charts.volatility.destroy();

    charts.volatility = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Low (<20)', 'Moderate (20-40)', 'High (40+)'],
            datasets: [{
                data: [low, mod, high],
                backgroundColor: ['#26A69A', '#FF9800', '#EF5350'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { position: 'bottom', labels: { color: '#E6EDF3', padding: 15 } } }
        }
    });
}

function initLiquidityChart() {
    const ctx = document.getElementById('liquidityChart');
    if (!ctx) return;

    const sorted = [...markets].sort((a, b) => (b.liquidity || 0) - (a.liquidity || 0)).slice(0, 10);
    const labels = sorted.map(m => m.symbol);
    const data = sorted.map(m => m.liquidity || 0);

    if (charts.liquidity) charts.liquidity.destroy();

    charts.liquidity = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Liquidity Score',
                data: data,
                borderColor: '#2962FF',
                backgroundColor: 'rgba(41, 98, 255, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { labels: { color: '#E6EDF3' } } },
            scales: {
                y: { ticks: { color: '#E6EDF3' }, grid: { color: '#30363D' } },
                x: { ticks: { color: '#E6EDF3' }, grid: { display: false } }
            }
        }
    });
}

function initAQMSSFactorChart() {
    const ctx = document.getElementById('aqmssFactorChart');
    if (!ctx || !aqmssData || !aqmssData.current) return;

    const c = aqmssData.current;

    if (charts.aqmssFactor) charts.aqmssFactor.destroy();

    charts.aqmssFactor = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Volatility Regime', 'Liquidity', 'Structure', 'Momentum', 'Volatility Level'],
            datasets: [{
                label: 'AQMSS Factors',
                data: [
                    c.volatility_regime || 0,
                    c.true_liquidity || 0,
                    c.structure || 0,
                    c.momentum || 0,
                    c.volatility_level || 0
                ],
                backgroundColor: 'rgba(41, 98, 255, 0.2)',
                borderColor: '#2962FF',
                pointBackgroundColor: '#2962FF',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#2962FF'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 10,
                    ticks: { color: '#8B949E', backdropColor: 'transparent' },
                    grid: { color: '#30363D' },
                    pointLabels: { color: '#E6EDF3', font: { size: 11 } }
                }
            },
            plugins: { legend: { labels: { color: '#E6EDF3' } } }
        }
    });
}

// ==========================================
// AI ASSISTANT
// ==========================================

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input?.value.trim();
    if (!message) return;

    addUserMessage(message);
    input.value = '';

    try {
        const response = await axios.post(`${API_URL}/ai/chat`, { message });
        if (response.data.success) {
            addAIMessage(response.data.data.ai_response);
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addAIMessage('Sorry, I encountered an error. Please try again.');
    }
}

function quickQuestion(question) {
    const input = document.getElementById('chat-input');
    if (input) {
        input.value = question;
        sendMessage();
    }
}

function addUserMessage(text) {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    const div = document.createElement('div');
    div.className = 'chat-message user-message';
    div.innerHTML = `
        <div class="message-avatar">👤</div>
        <div class="message-bubble">
            <div class="message-sender">You</div>
            <div class="message-content">${escapeHtml(text)}</div>
            <div class="message-time">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function addAIMessage(text) {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    const div = document.createElement('div');
    div.className = 'chat-message ai-message';
    div.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-bubble">
            <div class="message-sender">AI Assistant</div>
            <div class="message-content">${escapeHtml(text)}</div>
            <div class="message-time">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function clearChat() {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    container.innerHTML = `
        <div class="chat-message ai-message">
            <div class="message-avatar">🤖</div>
            <div class="message-bubble">
                <div class="message-sender">AI Assistant</div>
                <div class="message-content">Chat cleared. How can I help you analyze markets?</div>
                <div class="message-time">Now</div>
            </div>
        </div>
    `;
}

async function analyzeWithAI(symbol) {
    closeModal();
    switchTab('assistant');
    addAIMessage(`Analyzing ${symbol} with AI models...`);

    const market = markets.find(m => m.symbol === symbol);
    if (!market) return;

    try {
        const response = await axios.post(`${API_URL}/ai/predict`, {
            symbol: symbol,
            volatility_regime: market.volatility_regime || 'NORMAL',
            liquidity: market.liquidity,
            structure: 50,
            momentum: market.momentum === 'BULLISH' ? 70 : market.momentum === 'BEARISH' ? 30 : 50,
            volatility: market.volatility,
            atr: market.atr || 1,
            volume: market.volume
        });

        if (response.data.success) {
            const pred = response.data.data;
            addAIMessage(
                `📊 AI Analysis for ${symbol}:\n\n` +
                `Quality Probability: ${(pred.probability * 100).toFixed(1)}%\n` +
                `Confidence: ${pred.confidence}\n` +
                `Recommendation: ${pred.recommendation}\n\n` +
                (pred.recommendation === 'BUY' ? '✅ Conditions favor entry' :
                 pred.recommendation === 'SELL' ? '❌ Consider reducing exposure' :
                 '⚠️ Monitor for clearer signals')
            );
        }
    } catch (error) {
        console.error('Error in AI analysis:', error);
        addAIMessage('Unable to complete AI analysis at this time.');
    }
}

// ==========================================
// TAB SWITCHING
// ==========================================

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    const selectedTab = document.getElementById(`${tabName}-tab`);
    if (selectedTab) selectedTab.classList.add('active');

    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    const activeLink = document.querySelector(`[data-tab="${tabName}"]`);
    if (activeLink) activeLink.classList.add('active');

    if (tabName === 'charts') {
        if (markets.length > 0) {
            setTimeout(() => initializeCharts(), 100);
        } else {
            setTimeout(() => {
                if (markets.length > 0) initializeCharts();
            }, 1000);
        }
    }

    if (tabName === 'insights') {
        loadInsights();
    }
}

// ==========================================
// AUTO REFRESH
// ==========================================

function toggleAutoRefresh() {
    const btn = document.getElementById('auto-refresh-btn');

    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
        if (btn) {
            btn.classList.remove('active');
            btn.innerHTML = '<span class="btn-icon">⏱️</span> Auto';
        }
        showNotification('Auto-refresh disabled', 'info');
    } else {
        autoRefreshInterval = setInterval(async () => {
            await loadMarkets();
            await loadDashboardStats();
            await loadAIPulse();
        }, 30000);
        if (btn) {
            btn.classList.add('active');
            btn.innerHTML = '<span class="btn-icon">⏱️</span> Auto (On)';
        }
        showNotification('Auto-refresh enabled (30s)', 'success');
    }
}

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

function updateTime() {
    const el = document.getElementById('current-time');
    if (el) {
        el.textContent = new Date().toLocaleTimeString('en-US', {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    }
}

function updateDataPointsCount() {
    const el = document.getElementById('data-points');
    if (el) el.textContent = markets.length;
}

function formatPrice(price) {
    price = parseFloat(price) || 0;
    if (price >= 1000000) return '$' + (price / 1000000).toFixed(2) + 'M';
    if (price >= 1000) return '$' + price.toFixed(2);
    if (price >= 1) return '$' + price.toFixed(2);
    return '$' + price.toFixed(6);
}

function formatVolume(volume) {
    volume = parseFloat(volume) || 0;
    if (volume >= 1000000000) return (volume / 1000000000).toFixed(2) + 'B';
    if (volume >= 1000000) return (volume / 1000000).toFixed(2) + 'M';
    if (volume >= 1000) return (volume / 1000).toFixed(2) + 'K';
    return volume.toFixed(0);
}

function escapeHtml(text) {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}

function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    const colors = { success: '#26A69A', error: '#EF5350', warning: '#FF9800', info: '#2962FF' };

    const el = document.createElement('div');
    el.style.cssText = `
        position:fixed;top:80px;right:20px;background:${colors[type] || colors.info};
        color:white;padding:1rem 1.5rem;border-radius:8px;
        box-shadow:0 4px 12px rgba(0,0,0,0.4);z-index:10000;font-size:14px;
    `;
    el.textContent = message;
    document.body.appendChild(el);

    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translateX(100%)';
        el.style.transition = 'all 0.3s ease';
        setTimeout(() => el.remove(), 300);
    }, 3000);
}

// Global function exports
window.loadMarkets = loadMarkets;
window.loadDashboardStats = loadDashboardStats;
window.loadInsights = loadInsights;
window.showMarketDetail = showMarketDetail;
window.closeModal = closeModal;
window.sendMessage = sendMessage;
window.quickQuestion = quickQuestion;
window.clearChat = clearChat;
window.toggleAutoRefresh = toggleAutoRefresh;
window.analyzeWithAI = analyzeWithAI;
window.switchTab = switchTab;

console.log('✨ AQMSS Dashboard loaded');
