const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const dropText = document.getElementById('drop-text');
const analyzeBtn = document.getElementById('analyze-btn');
const terminalContent = document.getElementById('terminal-content');
const resultPanel = document.getElementById('result-panel');

let selectedFile = null;

// Terminal Log Helper
function log(msg, type = 'info') {
    const p = document.createElement('div');
    p.className = `log ${type}`;
    p.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
    terminalContent.appendChild(p);
    terminalContent.parentElement.scrollTop = terminalContent.parentElement.scrollHeight;
}

// Drag & Drop Events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.name.endsWith('.apk')) {
        log('ERROR: Invalid file type. Must be .apk', 'error');
        dropText.innerHTML = `<span style="color:var(--error)">Invalid file.</span> Try again.`;
        analyzeBtn.disabled = true;
        selectedFile = null;
        return;
    }
    selectedFile = file;
    dropText.innerHTML = `TARGET ACQUIRED: <span class="highlight">${file.name}</span>`;
    log(`File loaded: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`, 'success');
    analyzeBtn.disabled = false;
    
    // Reset result panel if visible
    resultPanel.classList.add('hidden');
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const modelType = document.querySelector('input[name="model"]:checked').value;
    
    log(`Initiating scan sequence using [${modelType.toUpperCase()}] model...`, 'info');
    analyzeBtn.disabled = true;
    resultPanel.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_type', modelType);

    try {
        log('Extracting features from APK payload...', 'info');
        
        // Simüle edilmiş bir progress bar/loading efekti hissi için setTimeout kullanılabilir
        // ama fetch asenkron olduğu için direkt atıyoruz.
        const response = await fetch('https://infuse-intelligent-feature-fusion-for.onrender.com/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Server error');
        }

        const data = await response.json();
        log('Analysis complete. Processing results...', 'success');
        
        displayResult(data);

    } catch (error) {
        log(`CRITICAL FAILURE: ${error.message}`, 'error');
        analyzeBtn.disabled = false;
    }
});

function displayResult(data) {
    resultPanel.classList.remove('hidden');
    const statusEl = document.getElementById('result-status');
    const fillEl = document.getElementById('confidence-fill');
    const confText = document.querySelector('#confidence-text .value');
    const detailsText = document.getElementById('details-text');

    // Reset classes
    statusEl.className = 'status-text';
    fillEl.style.backgroundColor = '';
    fillEl.style.boxShadow = '';

    if (data.result === 'MALWARE') {
        statusEl.innerText = 'MALWARE DETECTED';
        statusEl.classList.add('malware');
        fillEl.style.backgroundColor = 'var(--error)';
        fillEl.style.boxShadow = '0 0 10px var(--error)';
    } else {
        statusEl.innerText = 'SYSTEM SECURE (BENIGN)';
        statusEl.classList.add('benign');
        fillEl.style.backgroundColor = 'var(--text-main)';
        fillEl.style.boxShadow = '0 0 10px var(--text-main)';
    }

    // Animate bar
    setTimeout(() => {
        fillEl.style.width = `${data.confidence}%`;
    }, 100);
    
    confText.innerText = `${data.confidence}%`;
    detailsText.innerText = `Model: ${data.model} | Details: ${data.details}`;

    analyzeBtn.disabled = false;
    log(`Results finalized: ${data.result} (${data.confidence}%)`, data.result === 'MALWARE' ? 'error' : 'success');
}
