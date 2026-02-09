
document.addEventListener('DOMContentLoaded', function() {

    const sidebar = document.getElementById('sidebar');
    const sidebarContent = document.getElementById('details-body');
    const closeSidebarButton = document.getElementById('close-sidebar');

    const airportCoordinates = {
        "KJFK": { lat: 40.6413, lon: -73.7781 },
        "KLAX": { lat: 33.9416, lon: -118.4085 },
        "EGLL": { lat: 51.4700, lon: -0.4543 },
        "EDDF": { lat: 50.0379, lon: 8.5622 },
        "LFPG": { lat: 49.0097, lon: 2.5479 },
        "RJTT": { lat: 35.5494, lon: 139.7798 },
    };

    let map;
    const markers = {};

    function initializeMap() {
        map = L.map('map').setView([25, 0], 2); // Centered to show the world

        L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(map);
    }

    function getStatusClass(congestion) {
        if (congestion > 0.7) return 'status-high';
        if (congestion > 0.4) return 'status-medium';
        return 'status-low';
    }

    function createMarkerIcon(statusClass) {
        return L.divIcon({
            className: `leaflet-marker-icon ${statusClass}`,
            iconSize: [20, 20],
        });
    }
    
    function renderObjectAsTable(obj) {
        if (!obj || Object.keys(obj).length === 0) return '<p>No data available.</p>';
        let table = '<table>';
        for (const [key, value] of Object.entries(obj)) {
            const formattedValue = typeof value === 'number' ? value.toFixed(3) : (value === null ? 'N/A' : value);
            table += `<tr><td><strong>${key.replace(/_/g, ' ')}</strong></td><td>${formattedValue}</td></tr>`;
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

            for (const icao in data) {
                if (airportCoordinates[icao]) {
                    const node = data[icao];
                    const statusClass = getStatusClass(node.congestion_index);
                    const coords = airportCoordinates[icao];

                    if (markers[icao]) {
                        // Update existing marker
                        markers[icao].setIcon(createMarkerIcon(statusClass));
                        markers[icao].getTooltip().setContent(`<b>${icao}</b><br>Congestion: ${node.congestion_index.toFixed(3)}`);
                    } else {
                        // Create new marker
                        const icon = createMarkerIcon(statusClass);
                        const marker = L.marker([coords.lat, coords.lon], { icon: icon }).addTo(map);
                        marker.on('click', () => openSidebar(icao));
                        marker.bindTooltip(`<b>${icao}</b><br>Congestion: ${node.congestion_index.toFixed(3)}`).openTooltip();
                        markers[icao] = marker;
                    }
                }
            }
        } catch (error) {
            console.error("Error updating dashboard:", error);
        }
    }

    async function openSidebar(icao) {
        sidebar.classList.remove('collapsed');
        sidebarContent.innerHTML = '<p>Loading details...</p>';

        try {
            const response = await fetch(`/api/details/${icao}`);
            if (!response.ok) {
                const errorText = `Could not load details. Server responded with status: ${response.status}`;
                sidebarContent.innerHTML = `<p>${errorText}</p>`;
                console.error(errorText);
                return;
            }
            const details = await response.json();
            if (!details) {
                sidebarContent.innerHTML = `<p>Received empty details from server.</p>`;
                return;
            }
            
            let content = `
                <h2>${details.icao}</h2>
                <div class="section-explanation">
                    <p>This node uses a machine learning model to predict congestion. It enhances its prediction by communicating with neighboring airports in a "swarm," sharing data to make a more accurate, collective forecast.</p>
                </div>

                <h3>Prediction Output</h3>
                <div class="section-explanation">
                    <p>The final congestion score and the immediate trend, adjusted after swarm communication.</p>
                </div>
                ${renderObjectAsTable({
                    "Final Congestion": details.prediction,
                    "Short-term Trend": details.short_term_trend,
                })}

                <h3>Swarm Communication</h3>
                <div class="section-explanation">
                    <p>Data received from neighboring airports. The model uses this to adjust its local prediction.</p>
                </div>
                ${renderObjectAsTable(details.swarm_communication)}

                <h3>Model Features</h3>
                <div class="section-explanation">
                    <p>The key inputs fed into the local machine learning model to make an initial prediction.</p>
                </div>
                ${renderObjectAsTable(details.extracted_features)}
                
                <h3>Raw Data Summary</h3>
                <div class="section-explanation">
                    <p>A summary of the raw, unprocessed data used to generate the features above.</p>
                </div>
                <pre>${JSON.stringify(details.raw_data_summary, null, 2)}</pre>
            `;
            sidebarContent.innerHTML = content;

        } catch (error) {
            console.error("Error fetching details:", error);
            sidebarContent.innerHTML = '<p>Error loading details.</p>';
        }
    }

    function closeSidebar() {
        sidebar.classList.add('collapsed');
    }

    // --- Event Listeners ---
    closeSidebarButton.addEventListener('click', closeSidebar);


    // --- Initialization ---
    initializeMap();
    updateDashboard();
    setInterval(updateDashboard, 5000); // Refresh every 5 seconds
    
    // Initially the sidebar is closed
    closeSidebar();
});
