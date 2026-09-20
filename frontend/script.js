// =========================================
// Idukki Static Susceptibility Dashboard
// Frontend → FastAPI → Random Forest
// =========================================


// =========================================
// Configuration
// =========================================

const API_BASE = "http://127.0.0.1:8001";


// =========================================
// Update time
// =========================================

function updateTime() {

    const timeElement =
        document.getElementById("updateTime");

    if (!timeElement) return;

    const now = new Date();

    const hours =
        String(now.getHours()).padStart(2, "0");

    const minutes =
        String(now.getMinutes()).padStart(2, "0");

    timeElement.textContent =
        `${hours}:${minutes}`;
}

updateTime();

setInterval(updateTime, 30000);


// =========================================
// Reference location
// =========================================

const reference = {
    lat: 9.9189,
    lon: 77.1025
};


// =========================================
// Risk classification
// =========================================

function riskClass(value) {

    if (value < 0.25)
        return "LOW";

    if (value < 0.50)
        return "MODERATE";

    if (value < 0.75)
        return "HIGH";

    return "VERY HIGH";
}


// =========================================
// DOM helper
// =========================================

function setText(id, value) {

    const element =
        document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}


// =========================================
// Display prediction
// =========================================

function displayPrediction(data) {

    const susceptibility =
        Number(data.susceptibility);

    const percent =
        Number(data.risk_percent);

    // Main risk information
    setText(
        "riskScore",
        `${percent.toFixed(0)}%`
    );

    setText(
        "riskBadge",
        data.risk_class
    );

    setText(
        "selectedRegion",
        "Selected map point"
    );

    setText(
        "selectedCoords",
        `${Number(data.latitude).toFixed(5)}°, ${Number(data.longitude).toFixed(5)}°`
    );

    // Description
    setText(
        "scoreDescription",
        `${percent.toFixed(1)}% static susceptibility from the Random Forest model. This is not a rainfall-triggered warning.`
    );


    // =====================================
    // Conditioning factors
    // =====================================

    setText(
        "factorElevation",
        `${Number(data.elevation).toFixed(2)} m`
    );

    setText(
        "factorSlope",
        `${Number(data.slope).toFixed(2)}°`
    );

    setText(
        "factorAspect",
        `${Number(data.aspect).toFixed(2)}°`
    );

    setText(
        "factorCurvature",
        Number(data.curvature).toFixed(6)
    );

    setText(
        "factorTri",
        Number(data.tri).toFixed(2)
    );

    setText(
        "factorSoil",
        `${Number(data.clay).toFixed(2)}% / ${Number(data.sand).toFixed(2)}%`
    );

    setText(
        "factorLandcover",
        data.landcover
    );


    // =====================================
    // Risk badge styling
    // =====================================

    const badge =
        document.getElementById("riskBadge");

    if (badge) {

        badge.className = "risk-badge";

        if (susceptibility < 0.25) {

            badge.classList.add("risk-low");

        } else if (susceptibility < 0.50) {

            badge.classList.add("risk-moderate");

        } else if (susceptibility < 0.75) {

            badge.classList.add("risk-high");

        } else {

            badge.classList.add("risk-very-high");
        }
    }
}


// =========================================
// Loading state
// =========================================

function showLoading(lat, lon) {

    setText(
        "selectedRegion",
        "Analysing map point..."
    );

    setText(
        "selectedCoords",
        `${Number(lat).toFixed(5)}°, ${Number(lon).toFixed(5)}°`
    );

    setText(
        "riskScore",
        "..."
    );

    setText(
        "riskBadge",
        "LOADING"
    );

    setText(
        "scoreDescription",
        "Sending coordinates to the FastAPI backend and running the Random Forest model..."
    );
}


// =========================================
// Error state
// =========================================

function showPredictionError(error) {

    console.error(
        "Prediction error:",
        error
    );

    setText(
        "selectedRegion",
        "Prediction unavailable"
    );

    setText(
        "riskScore",
        "--"
    );

    setText(
        "riskBadge",
        "ERROR"
    );

    setText(
        "scoreDescription",
        `${error.message} Make sure FastAPI is running on port 8001.`
    );
}


// =========================================
// Send coordinates to FastAPI
// =========================================

async function predictLocation(lat, lon) {

    showLoading(lat, lon);

    try {

        const response =
            await fetch(
                `${API_BASE}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        lat: Number(lat),
                        lon: Number(lon)
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Backend prediction failed."
            );
        }


        console.log(
            "FastAPI prediction:",
            data
        );


        displayPrediction(data);


    } catch (error) {

        showPredictionError(error);
    }
}


// =========================================
// Leaflet map
// =========================================

const mapElement =
    document.getElementById("riskMap");

const loadingElement =
    document.getElementById("mapLoading");

const resetButton =
    document.getElementById("resetMap");


let riskMap = null;

let susceptibilityLayer = null;

let studyBounds = null;

let clickMarker = null;


// =========================================
// Fallback bounds
// =========================================

const fallbackBounds = [

    [9.2708, 76.6277],

    [10.3513, 77.4039]

];


// =========================================
// Map loading status
// =========================================

function setMapStatus(
    message,
    visible = true
) {

    if (!loadingElement) return;

    loadingElement.textContent =
        message;

    loadingElement.style.display =
        visible ? "flex" : "none";
}


// =========================================
// Load susceptibility map
// =========================================

async function loadSusceptibilityMap() {

    if (
        !mapElement ||
        typeof L === "undefined"
    ) {

        setMapStatus(
            "Leaflet could not be loaded.",
            true
        );

        return;
    }


    // =====================================
    // Create Leaflet map
    // =====================================

    riskMap =
        L.map(
            mapElement,
            {
                zoomControl: true,

                attributionControl: true,

                minZoom: 8,

                maxZoom: 14
            }
        );


    // =====================================
    // OpenStreetMap base layer
    // =====================================

    L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            maxZoom: 19,

            attribution:
                "© OpenStreetMap contributors"
        }
    ).addTo(riskMap);


    // =====================================
    // Load exact map bounds
    // =====================================

    try {

        const boundsResponse =
            await fetch(
                "map_data/idukki_map_bounds.json",
                {
                    cache: "no-store"
                }
            );


        if (boundsResponse.ok) {

            const b =
                await boundsResponse.json();


            studyBounds = [

                [b.south, b.west],

                [b.north, b.east]

            ];

        } else {

            studyBounds =
                fallbackBounds;
        }


    } catch (error) {

        console.warn(
            "Could not load map bounds. Using fallback bounds.",
            error
        );

        studyBounds =
            fallbackBounds;
    }


    // =====================================
    // Fit map to Idukki
    // =====================================

    riskMap.fitBounds(
        studyBounds,
        {
            padding: [10, 10]
        }
    );


    // =====================================
    // Susceptibility layer
    // =====================================

    try {

        susceptibilityLayer =
            L.imageOverlay(

                "map_data/idukki_static_susceptibility.png",

                studyBounds,

                {
                    opacity: 0.82,

                    interactive: false
                }

            ).addTo(riskMap);


        susceptibilityLayer.once(
            "load",
            () => {

                setMapStatus(
                    "",
                    false
                );
            }
        );


        susceptibilityLayer.once(
            "error",
            () => {

                setMapStatus(
                    "Susceptibility image not found. Run create_web_susceptibility_map.py first.",
                    true
                );
            }
        );


    } catch (error) {

        console.error(
            error
        );

        setMapStatus(
            "Could not load susceptibility layer.",
            true
        );
    }


    // =====================================
    // MAP CLICK
    // =====================================

    riskMap.on(
        "click",
        async (event) => {

            const lat =
                event.latlng.lat;

            const lon =
                event.latlng.lng;


            console.log(
                "Map clicked:",
                lat,
                lon
            );


            // Remove previous marker
            if (clickMarker) {

                clickMarker.remove();
            }


            // Add new marker
            clickMarker =
                L.circleMarker(
                    [lat, lon],
                    {
                        radius: 7,

                        weight: 2,

                        fillOpacity: 0.9
                    }
                ).addTo(riskMap);


            // Send coordinates
            // to FastAPI
            await predictLocation(
                lat,
                lon
            );
        }
    );


    // Fix Leaflet rendering
    setTimeout(
        () => {

            riskMap.invalidateSize();

        },
        150
    );


    // =====================================
    // Initial reference prediction
    // =====================================

    await predictLocation(
        reference.lat,
        reference.lon
    );
}


// =========================================
// Start map
// =========================================

loadSusceptibilityMap();


// =========================================
// Reset map
// =========================================

if (resetButton) {

    resetButton.addEventListener(
        "click",
        () => {

            if (
                riskMap &&
                studyBounds
            ) {

                riskMap.fitBounds(
                    studyBounds,
                    {
                        padding: [10, 10],

                        animate: true
                    }
                );
            }


            // Remove click marker
            if (clickMarker) {

                clickMarker.remove();

                clickMarker = null;
            }


            // Reset to reference location
            predictLocation(
                reference.lat,
                reference.lon
            );
        }
    );
}