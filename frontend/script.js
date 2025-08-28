// Global Variables
let selectedFiles = [];
const API_BASE_URL = 'http://localhost:8000';

// ECG Classification Data
const ECG_CLASSES = {
    'NORM': { name: 'Normal ECG', description: 'Healthy cardiac electrical activity with normal rhythm, rate, and waveform patterns.', type: 'normal', icon: 'fas fa-heart' },
    'CD': { name: 'Conduction Disturbance', description: 'Abnormal electrical conduction pathways affecting the heart\'s rhythm and coordination.', type: 'abnormal', icon: 'fas fa-exclamation-triangle' },
    'HYP': { name: 'Hypertrophy', description: 'Enlarged heart muscle, often due to high blood pressure or other cardiovascular conditions.', type: 'abnormal', icon: 'fas fa-expand-arrows-alt' },
    'MI': { name: 'Myocardial Infarction', description: 'Heart attack - damage to heart muscle due to blocked blood supply, requiring immediate medical attention.', type: 'critical', icon: 'fas fa-heart-broken' },
    'STTC': { name: 'ST/T Changes', description: 'Abnormal ST segment or T wave patterns indicating potential ischemia or other cardiac issues.', type: 'abnormal', icon: 'fas fa-wave-square' },
    'LVH': { name: 'Left Ventricular Hypertrophy', description: 'Thickening of the left ventricle wall, often associated with hypertension or valve disease.', type: 'abnormal', icon: 'fas fa-expand' },
    'LAFB': { name: 'Left Anterior Fascicular Block', description: 'Blocked electrical pathway in the left ventricle affecting the heart\'s conduction system.', type: 'abnormal', icon: 'fas fa-project-diagram' },
    'ISC_': { name: 'Ischemic', description: 'Reduced blood flow to heart muscle, potentially indicating coronary artery disease.', type: 'abnormal', icon: 'fas fa-bolt' },
    'IRBBB': { name: 'Incomplete Right Bundle Branch Block', description: 'Partial blockage in the right bundle branch affecting electrical conduction.', type: 'abnormal', icon: 'fas fa-ban' },
    'IVCD': { name: 'Intraventricular Conduction Disturbance', description: 'Abnormal electrical conduction within the ventricles affecting heart rhythm coordination.', type: 'abnormal', icon: 'fas fa-random' }
};

// DOM Elements
const themeToggle = document.getElementById('themeToggle');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const mobileMenu = document.getElementById('mobileMenu');
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('.section');

// Analyzer Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('ecgFile');
const filePreview = document.getElementById('filePreview');
const fileList = document.getElementById('fileList');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsContent = document.getElementById('resultsContent');
const statsContent = document.getElementById('statsContent');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

// Initialize Application
document.addEventListener('DOMContentLoaded', initializeApp);

function initializeApp() {
    initializeTheme();
    initializeNavigation();
    initializeAnalyzer();
    populateGuideSection();
    initializeFAQ();
    updateStatus('ready', 'Ready');
}

// Theme Management
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', savedTheme);
    
    themeToggle.addEventListener('click', toggleTheme);
}

function toggleTheme() {
    const currentTheme = document.body.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Add transition effect
    document.body.style.transition = 'all 0.3s ease';
    setTimeout(() => {
        document.body.style.transition = '';
    }, 300);
}

// Navigation Management
function initializeNavigation() {
    // Mobile menu toggle
    mobileMenuBtn.addEventListener('click', toggleMobileMenu);
    
    // Navigation link clicks
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetSection = link.getAttribute('href').substring(1);
            navigateToSection(targetSection);
        });
    });
    
    // Mobile navigation clicks
    document.querySelectorAll('.mobile-nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetSection = link.getAttribute('href').substring(1);
            navigateToSection(targetSection);
            toggleMobileMenu();
        });
    });
}

function toggleMobileMenu() {
    mobileMenu.classList.toggle('active');
    mobileMenuBtn.classList.toggle('active');
}

function navigateToSection(sectionId) {
    // Hide all sections
    sections.forEach(section => {
        section.classList.remove('active');
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
    }
    
    // Update active nav link
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${sectionId}`) {
            link.classList.add('active');
        }
    });
    
    // Close mobile menu if open
    mobileMenu.classList.remove('active');
    mobileMenuBtn.classList.remove('active');
}

// Analyzer Functionality
function initializeAnalyzer() {
    if (!uploadZone) return;
    
    // Upload zone events
    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    
    // File input change
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });
}

function handleDragOver(e) {
    uploadZone.classList.add('dragover');
}

function handleDragLeave(e) {
    uploadZone.classList.remove('dragover');
}

function handleDrop(e) {
    uploadZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
}

function handleFiles(files) {
    selectedFiles = Array.from(files);
    
    if (selectedFiles.length === 0) return;
    
    const validation = validateFiles(selectedFiles);
    if (!validation.valid) {
        showToast(validation.message, 'error');
        return;
    }
    
    displayFilePreview();
    updateUploadState();
}

function validateFiles(files) {
    const allowedExts = ['.dat', '.hea'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (files.length === 0) {
        return { valid: false, message: 'Please select files to upload.' };
    }
    
    const extensions = files.map(file => {
        const name = file.name.toLowerCase();
        return name.substring(name.lastIndexOf('.'));
    });
    
    const hasRequired = allowedExts.every(ext => extensions.includes(ext));
    
    if (!hasRequired) {
        return { 
            valid: false, 
            message: 'Please upload both .dat and .hea files for ECG analysis.' 
        };
    }
    
    for (let file of files) {
        if (file.size > maxSize) {
            return { 
                valid: false, 
                message: `File "${file.name}" exceeds the 10MB size limit.` 
            };
        }
    }
    
    return { valid: true };
}

function displayFilePreview() {
    fileList.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const ext = file.name.toLowerCase().split('.').pop();
        const size = formatFileSize(file.size);
        
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item fade-in';
        fileItem.innerHTML = `
            <div class="file-info">
                <div class="file-type ${ext}">${ext.toUpperCase()}</div>
                <div>
                    <div style="font-weight: 500; margin-bottom: 2px;">${file.name}</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">${size}</div>
                </div>
            </div>
            <button onclick="removeFile(${index})" style="background: none; border: none; color: var(--text-muted); cursor: pointer; padding: 0.5rem; border-radius: 4px; transition: var(--transition-fast);">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        fileList.appendChild(fileItem);
    });
    
    filePreview.style.display = 'block';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    
    if (selectedFiles.length === 0) {
        filePreview.style.display = 'none';
        clearResults();
    } else {
        displayFilePreview();
    }
    
    updateUploadState();
}

function clearFiles() {
    selectedFiles = [];
    fileInput.value = '';
    filePreview.style.display = 'none';
    clearResults();
    updateUploadState();
    updateStatus('ready', 'Ready');
}

function updateUploadState() {
    const hasFiles = selectedFiles.length > 0;
    
    analyzeBtn.disabled = !hasFiles;
    clearBtn.style.display = hasFiles ? 'flex' : 'none';
    
    const uploadIcon = document.getElementById('uploadIcon');
    const uploadTitle = document.getElementById('uploadTitle');
    const uploadSubtitle = document.getElementById('uploadSubtitle');
    
    if (hasFiles) {
        uploadIcon.className = 'fas fa-check-circle';
        uploadTitle.textContent = `${selectedFiles.length} file(s) selected`;
        uploadSubtitle.textContent = 'Ready for AI analysis';
    } else {
        uploadIcon.className = 'fas fa-cloud-upload-alt';
        uploadTitle.textContent = 'Drop ECG Files Here';
        uploadSubtitle.textContent = 'or click to browse files';
    }
}

async function uploadFiles() {
    if (selectedFiles.length === 0) return;
    
    setLoadingState(true);
    updateStatus('processing', 'Analyzing ECG...');
    
    try {
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });
        
        showToast('Processing ECG data with AI...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/predict_file`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
            displayStats(result.ecg_stats);
            updateStatus('ready', 'Analysis Complete');
            showToast('ECG analysis completed successfully!', 'success');
        } else {
            throw new Error(result.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        updateStatus('error', 'Analysis Failed');
        showToast(`Analysis failed: ${error.message}`, 'error');
        clearResults();
    } finally {
        setLoadingState(false);
    }
}

function setLoadingState(loading) {
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoading = analyzeBtn.querySelector('.btn-loading');
    
    if (loading) {
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
        analyzeBtn.disabled = true;
    } else {
        btnText.style.display = 'flex';
        btnLoading.style.display = 'none';
        analyzeBtn.disabled = selectedFiles.length === 0;
    }
}

function displayResults(result) {
    const classInfo = ECG_CLASSES[result.predicted_class] || {
        name: result.predicted_class,
        description: 'Unknown cardiac condition classification.',
        type: 'abnormal',
        icon: 'fas fa-question-circle'
    };
    
    const confidence = result.confidence ? (result.confidence * 100).toFixed(1) : 'N/A';
    
    resultsContent.innerHTML = `
        <div class="prediction-result ${classInfo.type} fade-in">
            <div class="result-icon ${classInfo.type}">
                <i class="${classInfo.icon}"></i>
            </div>
            <div class="result-label">${classInfo.name}</div>
            <div class="result-description">${classInfo.description}</div>
            <div class="confidence-badge">
                <i class="fas fa-chart-bar"></i>
                Confidence: ${confidence}%
            </div>
        </div>
    `;
}

function displayStats(stats) {
    statsContent.innerHTML = `
        <div class="stats-grid fade-in">
            <div class="stat-card">
                <div class="stat-value">${stats.num_leads}</div>
                <div class="stat-label">ECG Leads</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.signal_length.toLocaleString()}</div>
                <div class="stat-label">Signal Length</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.mean_amplitude.toFixed(3)}</div>
                <div class="stat-label">Mean Amplitude</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.std_amplitude.toFixed(3)}</div>
                <div class="stat-label">Std Deviation</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.max_amplitude.toFixed(3)}</div>
                <div class="stat-label">Max Amplitude</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.min_amplitude.toFixed(3)}</div>
                <div class="stat-label">Min Amplitude</div>
            </div>
        </div>
    `;
}

function clearResults() {
    if (resultsContent) {
        resultsContent.innerHTML = `
            <div class="no-results">
                <div class="empty-state-icon">
                    <i class="fas fa-chart-line"></i>
                </div>
                <h4>No Analysis Yet</h4>
                <p>Upload ECG files to see AI classification results</p>
            </div>
        `;
    }
    
    if (statsContent) {
        statsContent.innerHTML = `
            <div class="no-stats">
                <div class="empty-state-icon">
                    <i class="fas fa-calculator"></i>
                </div>
                <h4>No Data Available</h4>
                <p>Statistics will appear after ECG analysis</p>
            </div>
        `;
    }
}

function updateStatus(type, message) {
    if (statusDot && statusText) {
        statusDot.className = `status-dot ${type}`;
        statusText.textContent = message;
    }
}

// Guide Section Population
function populateGuideSection() {
    const guideSection = document.querySelector('.guide-content');
    if (!guideSection) return;
    
    let guideHTML = '<div class="conditions-grid">';
    
    Object.entries(ECG_CLASSES).forEach(([code, info]) => {
        guideHTML += `
            <div class="condition-card ${info.type} fade-in">
                <div class="condition-header">
                    <div class="condition-icon">
                        <i class="${info.icon}"></i>
                    </div>
                    <div class="condition-info">
                        <h4>${code}</h4>
                        <span class="condition-name">${info.name}</span>
                    </div>
                </div>
                <p>${info.description}</p>
            </div>
        `;
    });
    
    guideHTML += '</div>';
    guideSection.innerHTML = guideHTML;
}

// FAQ Functionality
function initializeFAQ() {
    const faqQuestions = document.querySelectorAll('.faq-question');
    
    faqQuestions.forEach(question => {
        question.addEventListener('click', () => {
            const faqItem = question.parentElement;
            const isActive = faqItem.classList.contains('active');
            
            // Close all FAQ items
            document.querySelectorAll('.faq-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // Toggle current item
            if (!isActive) {
                faqItem.classList.add('active');
            }
        });
    });
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    const icon = toast.querySelector('.toast-icon');
    const messageEl = toast.querySelector('.toast-message');
    
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };
    
    icon.className = `toast-icon ${icons[type]}`;
    messageEl.textContent = message;
    toast.className = `toast ${type}`;
    
    // Show toast
    setTimeout(() => {
        toast.classList.add('show');
    }, 100);
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        hideToast();
    }, 5000);
}

function hideToast() {
    const toast = document.getElementById('toast');
    toast.classList.remove('show');
}

// Keyboard Navigation
document.addEventListener('keydown', (e) => {
    // ESC key closes mobile menu
    if (e.key === 'Escape') {
        mobileMenu.classList.remove('active');
        mobileMenuBtn.classList.remove('active');
    }
    
    // Enter key on upload zone triggers file selection
    if (e.key === 'Enter' && e.target === uploadZone) {
        fileInput.click();
    }
});

// Smooth scrolling for hash links
window.addEventListener('hashchange', () => {
    const hash = window.location.hash.substring(1);
    if (hash) {
        navigateToSection(hash);
    }
});

// Handle initial hash on page load
window.addEventListener('load', () => {
    const hash = window.location.hash.substring(1);
    if (hash) {
        navigateToSection(hash);
    }
});

// Accessibility improvements
document.addEventListener('DOMContentLoaded', () => {
    // Add ARIA labels
    if (themeToggle) {
        themeToggle.setAttribute('aria-label', 'Toggle dark mode');
        themeToggle.setAttribute('role', 'button');
        themeToggle.setAttribute('tabindex', '0');
    }
    
    if (uploadZone) {
        uploadZone.setAttribute('aria-label', 'Upload ECG files');
        uploadZone.setAttribute('tabindex', '0');
    }
});

// Performance optimization: Lazy load animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', () => {
    const animateElements = document.querySelectorAll('.condition-card, .info-card, .stat-card');
    animateElements.forEach(el => observer.observe(el));
});
