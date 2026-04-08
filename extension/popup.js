document.addEventListener('DOMContentLoaded', () => {
  const snapBtn = document.getElementById('snap-btn');
  const statusText = document.getElementById('status-text');
  const loader = document.getElementById('loader');
  const resultsDiv = document.getElementById('results');
  const verdictEl = document.getElementById('verdict');
  const confidenceEl = document.getElementById('confidence');
  const heatmapImg = document.getElementById('heatmap-img');

  snapBtn.addEventListener('click', async () => {
    try {
      snapBtn.disabled = true;
      resultsDiv.style.display = 'none';
      loader.style.display = 'block';
      statusText.textContent = 'Capturing viewport...';

      // 1. Capture visible tab
      chrome.tabs.captureVisibleTab(null, { format: 'png' }, async (dataUrl) => {
        if (chrome.runtime.lastError || !dataUrl) {
          showError('Could not capture screen. ' + (chrome.runtime.lastError?.message || ''));
          return;
        }

        statusText.textContent = 'Analyzing Forensic Artifacts...';

        try {
          // 2. Convert DataURL to Blob
          const res = await fetch(dataUrl);
          const blob = await res.blob();

          // 3. Send to API (using /api/detect as defined in main.py)
          const formData = new FormData();
          formData.append('file', blob, 'screenshot.png');

          const detectRes = await fetch('http://127.0.0.1:8000/api/detect', {
            method: 'POST',
            body: formData
          });

          if (!detectRes.ok) {
            throw new Error(`API Error: ${detectRes.statusText}`);
          }

          const initData = await detectRes.json();
          const taskId = initData.task_id;

          // 4. Poll for result
          const pollResult = async () => {
            try {
              const pollRes = await fetch(`http://127.0.0.1:8000/api/detect/${taskId}`);
              if (!pollRes.ok) throw new Error('Polling failed');
              
              const pollData = await pollRes.json();
              if (pollData.status === 'completed') {
                displayResult(pollData.result);
              } else if (pollData.status === 'failed') {
                throw new Error(pollData.error || 'Analysis failed');
              } else {
                // Still processing, try again in 1s
                setTimeout(pollResult, 1000);
              }
            } catch (err) {
              showError(err.message);
            }
          };

          setTimeout(pollResult, 1000);
        } catch (err) {
          showError(err.message);
        }
      });
    } catch (err) {
      showError(err.message);
    }
  });

  function displayResult(result) {
    loader.style.display = 'none';
    statusText.textContent = 'Analysis complete.';
    snapBtn.disabled = false;
    
    resultsDiv.style.display = 'flex';
    
    if (result.is_fake) {
      verdictEl.textContent = 'DEEPFAKE';
      verdictEl.className = 'verdict fake';
    } else {
      verdictEl.textContent = 'AUTHENTIC';
      verdictEl.className = 'verdict real';
    }
    
    confidenceEl.textContent = result.confidence;
    heatmapImg.src = result.heatmap_url;
  }

  function showError(msg) {
    loader.style.display = 'none';
    snapBtn.disabled = false;
    statusText.textContent = 'Error: ' + msg;
  }
});
