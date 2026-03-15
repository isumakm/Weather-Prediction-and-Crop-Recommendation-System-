document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('recommendation-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = form.querySelector('.btn-text');
    const spinner = document.getElementById('spinner');
    const resultContainer = document.getElementById('result-container');

    const API_URL = 'http://localhost:8000/api/recommend';

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const location = document.getElementById('location').value;
        const date = document.getElementById('date').value;
        const crop = document.getElementById('crop').value;

        // UI Loading State
        submitBtn.disabled = true;
        btnText.textContent = 'Analyzing...';
        spinner.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        resultContainer.innerHTML = '';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    location,
                    date,
                    crop: crop.trim() ? crop.trim() : null
                })
            });

            if (!response.ok) {
                throw new Error('Failed to fetch data from the server.');
            }

            const result = await response.json();
            
            if (result.type === 'single_crop') {
                renderSingleCropReport(result.data);
            } else {
                renderMultiCropRanking(result.data);
            }
            
            resultContainer.classList.remove('hidden');

        } catch (error) {
            console.error(error);
            resultContainer.innerHTML = `
                <div class="glassmorphism report-card" style="border-color: var(--danger)">
                    <h3 style="color: var(--danger)">Error connecting to server</h3>
                    <p>Please ensure the backend server is running on localhost:8000.</p>
                </div>
            `;
            resultContainer.classList.remove('hidden');
        } finally {
            // Revert UI Loading State
            submitBtn.disabled = false;
            btnText.textContent = 'Analyze Suitability';
            spinner.classList.add('hidden');
        }
    });

    function renderMultiCropRanking(data) {
        let html = `<h2 style="margin-bottom: 1.5rem;">Top Recommended Crops</h2>`;
        html += `<div class="rank-list">`;
        
        data.forEach(item => {
            html += `
                <div class="rank-item">
                    <div class="rank-badge">${item.rank}</div>
                    <div class="rank-info">
                        <h3>${item.crop}</h3>
                        <div class="score-bar-container">
                            <div class="score-bar" style="width: ${item.suitability_score}%"></div>
                        </div>
                    </div>
                    <div class="score-text">${item.suitability_score}%</div>
                </div>
            `;
        });
        
        html += `</div>`;
        resultContainer.innerHTML = html;
    }

    function renderSingleCropReport(data) {
        const isSuitable = data.is_suitable;
        const statusClass = isSuitable ? 'status-suitable' : 'status-unsuitable';
        const statusText = isSuitable ? 'Highly Suitable' : 'Not Recommended';
        
        let html = `
            <div class="glassmorphism report-card">
                <div class="report-header">
                    <h2>Suitability Report: <span class="highlight">${data.crop}</span></h2>
                    <div class="status-badge ${statusClass}">${statusText}</div>
                </div>
                
                <div style="font-size: 1.5rem; font-weight: 700;">
                    Overall Score: <span class="${isSuitable ? 'highlight' : ''}" style="${!isSuitable ? 'color: var(--danger)' : ''}">${data.suitability_score}%</span>
                </div>

                <div class="report-section">
                    <h4>Analysis</h4>
                    <ul class="reasons-list">
                        ${data.analysis_report.map(reason => `<li>${reason}</li>`).join('')}
                    </ul>
                </div>

                <div class="report-section">
                    <h4>Weather Predictions</h4>
                    <div class="data-grid">
                        ${Object.entries(data.weather_context).map(([key, value]) => `
                            <div class="data-box">
                                <span class="data-label">${key}</span>
                                <span class="data-value">${value}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div class="report-section">
                    <h4>Soil Parameters</h4>
                    <div class="data-grid">
                        ${Object.entries(data.soil_context).map(([key, value]) => `
                            <div class="data-box">
                                <span class="data-label">${key}</span>
                                <span class="data-value">${value}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
        resultContainer.innerHTML = html;
    }
});
