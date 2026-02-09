
document.addEventListener('DOMContentLoaded', function() {

    const statusGrid = document.getElementById('status-grid');
    const modal = document.getElementById('details-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    const closeButton = document.querySelector('.close-button');

    function getStatusColor(congestion) {
        if (congestion > 0.7) return 'status-high';
        if (congestion > 0.4) return 'status-medium';
        return 'status-low';
    }

    function getTrendIndicator(trend) {
        if (trend > 0.01) return '<span class="trend trend-up">▲ Increasing</span>';
        if (trend < -0.01) return '<span class="trend trend-down">▼ Decreasing</span>';
        return '<span class="trend trend-stable">▬ Stable</span>';
    }
    
    function formatTimestamp(unix_timestamp) {
        const date = new Date(unix_timestamp * 1000);
        return date.toLocaleTimeString();
    }

    function createAirportCard(icao, node) {
        const statusColor = getStatusColor(node.congestion_index);
        const card = document.createElement('div');
        card.className = `airport-card ${statusColor}`;
        card.setAttribute('data-icao', icao); // Set ICAO code for click handling
        
        card.innerHTML = `
            <div class="card-header">
                <h3>${icao}</h3>
                <span class="timestamp">Updated: ${formatTimestamp(node.timestamp)}</span>
            </div>
            <div class="card-body">
                <p><strong>Congestion Index:</strong> ${node.congestion_index.toFixed(3)}</p>
                <p><strong>Short-Term Trend:</strong> ${getTrendIndicator(node.short_term_trend)}</p>
            </div>
        `;
        return card;
    }
    
    function renderObjectAsTable(obj) {
        if (!obj || Object.keys(obj).length === 0) return '<p>No data available.</p>';
        let table = '<table>';
        for (const [key, value] of Object.entries(obj)) {
            const formattedValue = typeof value === 'number' ? value.toFixed(3) : value;
            table += `<tr><td><strong>${key}</strong></td><td>${formattedValue}</td></tr>`;
        }
        table += '</table>';
        return table;
    }

    async function updateDashboard() {
        try {
            const response = await fetch('/api/status');
            if (!response.ok) {
                console.error("Failed to fetch API status");
                return;
            }
            const data = await response.json();

            // Clear existing grid
            statusGrid.innerHTML = '';

            const sortedIcaos = Object.keys(data).sort();

            if (sortedIcaos.length === 0) {
                statusGrid.innerHTML = '<p>Waiting for simulation data...</p>';
                return;
            }

            for (const icao of sortedIcaos) {
                const node = data[icao];
                const card = createAirportCard(icao, node);
                statusGrid.appendChild(card);
            }

        } catch (error) {
            console.error("Error updating dashboard:", error);
        }
    }

    async function openModal(icao) {
        try {
            const response = await fetch(`/api/details/${icao}`);
            if (!response.ok) {
                modalBody.innerHTML = '<p>Could not load details.</p>';
            } else {
                const details = await response.json();
                
                modalTitle.textContent = `Details for ${details.icao}`;
                
                let bodyContent = `
                    <h3>Prediction Output</h3>
                    ${renderObjectAsTable({
                        "Final Congestion Prediction": details.prediction,
                        "Short-term Trend": details.short_term_trend,
                    })}

                    <h3>Swarm Communication</h3>
                    ${renderObjectAsTable(details.swarm_communication)}

                    <h3>Extracted Features (Input to Model)</h3>
                    ${renderObjectAsTable(details.extracted_features)}
                    
                    <h3>Raw Data Summary</h3>
                    <pre>${JSON.stringify(details.raw_data_summary, null, 2)}</pre>
                `;
                
                modalBody.innerHTML = bodyContent;
            }
            
            modal.style.display = 'block';

        } catch (error) {
            console.error("Error fetching details:", error);
            modalBody.innerHTML = '<p>Error loading details.</p>';
            modal.style.display = 'block';
        }
    }

    // --- Event Listeners ---

    // Click on an airport card to open the modal
    statusGrid.addEventListener('click', function(event) {
        const card = event.target.closest('.airport-card');
        if (card) {
            const icao = card.dataset.icao;
            openModal(icao);
        }
    });

    // Close the modal
    closeButton.addEventListener('click', () => modal.style.display = 'none');
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });


    // --- Initial Load & Periodic Refresh ---
    updateDashboard(); // Initial call
    setInterval(updateDashboard, 5000); // Refresh every 5 seconds

});
