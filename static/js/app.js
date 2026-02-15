/* ═══════════════════════════════════════════════════
   DermaScan AI — Client-Side Application Logic
   ═══════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── DOM References ─────────────────────────────
    const uploadZone    = document.getElementById('uploadZone');
    const fileInput     = document.getElementById('fileInput');
    const uploadContent = document.getElementById('uploadContent');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage  = document.getElementById('previewImage');
    const removeBtn     = document.getElementById('removeBtn');
    const analyzeBtn    = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    const uploadSection = document.getElementById('uploadSection');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const resultHero    = document.getElementById('resultHero');
    const resultGrid    = document.getElementById('resultGrid');
    const newScanBtn    = document.getElementById('newScanBtn');
    const toast         = document.getElementById('toast');
    const toastMessage  = document.getElementById('toastMessage');
    const particlesEl   = document.getElementById('particles');

    let selectedFile = null;

    // ── Particles Background ───────────────────────
    function spawnParticles() {
        const colors = ['rgba(251,155,143,0.15)', 'rgba(245,119,153,0.12)', 'rgba(253,195,161,0.1)'];
        for (let i = 0; i < 25; i++) {
            const p = document.createElement('div');
            p.classList.add('particle');
            const size = Math.random() * 4 + 2;
            p.style.width = size + 'px';
            p.style.height = size + 'px';
            p.style.left = Math.random() * 100 + '%';
            p.style.background = colors[Math.floor(Math.random() * colors.length)];
            p.style.animationDuration = (Math.random() * 20 + 15) + 's';
            p.style.animationDelay = (Math.random() * 15) + 's';
            particlesEl.appendChild(p);
        }
    }
    spawnParticles();

    // ── File Handling ──────────────────────────────
    // Click to browse
    uploadZone.addEventListener('click', (e) => {
        if (e.target === removeBtn || removeBtn.contains(e.target)) return;
        fileInput.click();
    });

    uploadZone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    // Drag & Drop
    uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    // Remove image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function handleFile(file) {
        // Validate type
        const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            showToast('Unsupported file format. Use JPG, PNG, BMP, or WebP.');
            return;
        }
        // Validate size (10 MB)
        if (file.size > 10 * 1024 * 1024) {
            showToast('File is too large. Maximum size is 10 MB.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        analyzeBtn.disabled = true;
    }

    // ── Analysis ───────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Switch to loading state
        uploadSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Server error');
            }

            renderResults(data.predictions);
        } catch (err) {
            showToast(err.message || 'Something went wrong. Please try again.');
            // Return to upload
            loadingSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
        }
    });

    // ── Render Results ─────────────────────────────
    function renderResults(predictions) {
        loadingSection.classList.add('hidden');

        // Top prediction — hero card
        const top = predictions[0];
        const riskClass = riskToClass(top.risk);
        const riskBadge = riskToBadge(top.risk);
        const emoji = riskToEmoji(top.risk);

        resultHero.className = 'result-hero risk-' + riskClass;
        resultHero.innerHTML = `
            <div class="result-hero__header">
                <span class="result-hero__label">Top Prediction</span>
                <span class="risk-badge risk-badge--${riskClass}">${emoji} ${top.risk}</span>
            </div>
            <h2 class="result-hero__title">${top.label}</h2>
            <div class="confidence-bar-wrapper">
                <div class="confidence-bar-wrapper__label">
                    <span>Confidence</span>
                    <span class="confidence-bar-wrapper__value">${top.confidence.toFixed(1)}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-bar__fill confidence-bar__fill--${riskClass}" id="heroBar"></div>
                </div>
            </div>
            <div class="result-hero__info">
                ${top.description ? `<div class="clinical-box"><div class="clinical-box__title">Clinical Overview</div><div class="clinical-box__text">${top.description}</div></div>` : ''}
                ${top.recommendation ? `<div class="clinical-box"><div class="clinical-box__title">Recommendation</div><div class="clinical-box__text">${top.recommendation}</div></div>` : ''}
            </div>
        `;

        // Other predictions
        resultGrid.innerHTML = '';
        predictions.slice(1).forEach((pred, i) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.style.animationDelay = (i * 0.1) + 's';
            card.innerHTML = `
                <div class="result-card__rank">#${i + 2}</div>
                <div class="result-card__info">
                    <div class="result-card__name">${pred.label}</div>
                    <div class="result-card__bar"><div class="result-card__bar-fill" data-width="${pred.confidence}"></div></div>
                </div>
                <div class="result-card__conf">${pred.confidence.toFixed(1)}%</div>
            `;
            resultGrid.appendChild(card);
        });

        resultsSection.classList.remove('hidden');

        // Animate bars after paint
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const heroBar = document.getElementById('heroBar');
                if (heroBar) heroBar.style.width = top.confidence + '%';

                document.querySelectorAll('.result-card__bar-fill').forEach(bar => {
                    bar.style.width = bar.dataset.width + '%';
                });
            });
        });
    }

    function riskToClass(risk) {
        const map = { 'Benign': 'benign', 'Pre-cancerous': 'precancer', 'Malignant': 'malignant' };
        return map[risk] || 'benign';
    }

    function riskToBadge(risk) {
        const map = { 'Benign': '🟢', 'Pre-cancerous': '🟡', 'Malignant': '🔴' };
        return map[risk] || '⚪';
    }

    function riskToEmoji(risk) {
        return riskToBadge(risk);
    }

    // ── New Scan ───────────────────────────────────
    newScanBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        resetUpload();
        uploadSection.classList.remove('hidden');
    });

    // ── Toast ──────────────────────────────────────
    function showToast(message) {
        toastMessage.textContent = message;
        toast.classList.remove('hidden');
        clearTimeout(showToast._timer);
        showToast._timer = setTimeout(() => toast.classList.add('hidden'), 5000);
    }

})();
