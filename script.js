(function () {
  'use strict';

  var MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
  var ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];

  var dropZone = document.getElementById('dropZone');
  var fileInput = document.getElementById('fileInput');
  var previewArea = document.getElementById('previewArea');
  var previewImage = document.getElementById('previewImage');
  var fileInfo = document.getElementById('fileInfo');
  var clearBtn = document.getElementById('clearBtn');
  var uploadBtn = document.getElementById('uploadBtn');
  var status = document.getElementById('status');

  var selectedFile = null;

  // ── Drag & drop ─────────────────────────────────────────────────────────────

  dropZone.addEventListener('dragover', function (e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', function () {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', function (e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    var file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  // Keyboard accessibility: Enter / Space opens file picker
  dropZone.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInput.click();
    }
  });

  // ── File input ───────────────────────────────────────────────────────────────

  fileInput.addEventListener('change', function () {
    var file = fileInput.files[0];
    if (file) handleFile(file);
  });

  // ── Buttons ──────────────────────────────────────────────────────────────────

  clearBtn.addEventListener('click', resetState);

  uploadBtn.addEventListener('click', function () {
    if (!selectedFile) return;
    simulateUpload(selectedFile);
  });

  // ── Core helpers ─────────────────────────────────────────────────────────────

  function handleFile(file) {
    clearStatus();

    if (!ALLOWED_TYPES.includes(file.type)) {
      showStatus('Unsupported file type. Please select a JPG, PNG, GIF, or WebP image.', 'error');
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      showStatus('File is too large. Maximum size is 10 MB.', 'error');
      return;
    }

    selectedFile = file;

    var reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      fileInfo.textContent = file.name + ' — ' + formatSize(file.size);
      previewArea.hidden = false;
      uploadBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  function simulateUpload(file) {
    uploadBtn.disabled = true;
    clearBtn.disabled = true;

    // Inject a progress bar
    var wrap = document.createElement('div');
    wrap.className = 'progress-bar-wrap';
    var bar = document.createElement('div');
    bar.className = 'progress-bar';
    wrap.appendChild(bar);
    status.innerHTML = '';
    status.appendChild(wrap);
    status.className = 'status';

    // Animate the progress bar over ~1.5 s, then show success
    var progress = 0;
    var interval = setInterval(function () {
      progress = Math.min(progress + Math.random() * 20, 90);
      bar.style.width = progress + '%';
    }, 150);

    setTimeout(function () {
      clearInterval(interval);
      bar.style.width = '100%';

      setTimeout(function () {
        showStatus('✓ "' + file.name + '" uploaded successfully!', 'success');
        clearBtn.disabled = false;
      }, 300);
    }, 1500);
  }

  function resetState() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewArea.hidden = true;
    uploadBtn.disabled = true;
    clearStatus();
  }

  function showStatus(message, type) {
    status.textContent = message;
    status.className = 'status ' + (type || '');
  }

  function clearStatus() {
    status.textContent = '';
    status.className = 'status';
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }
})();
