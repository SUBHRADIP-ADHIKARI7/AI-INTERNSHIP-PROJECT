document.addEventListener('DOMContentLoaded', () => {
    // ─── 3D Tilt Effect ───────────────────────────────────────────────────────
    const card = document.querySelector('.glass-card');
    const heroVisual = document.querySelector('.hero-visual');

    if (heroVisual && card) {
        heroVisual.addEventListener('mousemove', (e) => {
            if(window.innerWidth > 968) {
                const rect = heroVisual.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const xAxis = ((rect.height / 2) - y) / 15;
                const yAxis = (x - (rect.width / 2)) / 15;
                card.style.transform = `rotateY(${yAxis}deg) rotateX(${xAxis}deg)`;
            }
        });

        heroVisual.addEventListener('mouseleave', () => {
            if(window.innerWidth > 968) {
                card.style.transform = `rotateY(-5deg) rotateX(5deg)`;
            }
        });
    }

    // ─── Helper: Create a match-item card from job data ───────────────────────
    const LOGO_COLORS = ['google', 'meta', 'apple', 'amazon'];
    function createJobCard(job, index) {
        const logoClass = LOGO_COLORS[index % LOGO_COLORS.length];
        const initial = (job.company || 'J')[0].toUpperCase();
        const source = job.source || '';

        const div = document.createElement('div');
        div.className = 'match-item sk-inset';
        div.style.opacity = '0';
        div.style.transform = 'translateY(-20px) scale(0.95)';
        div.style.transition = 'all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
        div.style.cursor = 'pointer';

        div.innerHTML = `
            <div class="company-logo ${logoClass}">${initial}</div>
            <div class="job-info">
                <h3>${job.title || 'Job Opportunity'}</h3>
                <p>${job.company || 'Company'} • ${job.location || 'India'}</p>
            </div>
            <div class="match-score sk-button" style="font-size: 0.7rem;">${source}</div>
        `;

        div.addEventListener('click', () => {
            if (job.apply_link) window.open(job.apply_link, '_blank');
        });

        return div;
    }

    // ─── "Start AI Scan" → calls /scan endpoint ──────────────────────────────
    const scanBtn = document.getElementById('startScanBtn');
    const aiFeed = document.getElementById('ai-feed');
    const statusText = document.querySelector('.status');

    if (scanBtn && aiFeed) {
        scanBtn.addEventListener('click', () => {
            // Redirect gracefully to the main Chatbot interface
            scanBtn.querySelector('.btn-content').innerHTML = '<span class="icon">🚀</span> Launching AI Agent...';
            scanBtn.style.pointerEvents = 'none';
            statusText.innerHTML = 'AI Status: <span class="pulse" style="color: #00f0ff; text-shadow: 0 0 10px #00f0ff;">Connecting...</span>';
            
            setTimeout(() => {
                window.location.href = '/bot';
            }, 600);
        });
    }

});
