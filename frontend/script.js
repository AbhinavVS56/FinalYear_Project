// =========================================
// Update time
// =========================================

function updateTime() {

    const timeElement =
        document.getElementById("updateTime");

    const now = new Date();

    const hours =
        String(now.getHours()).padStart(2, "0");

    const minutes =
        String(now.getMinutes()).padStart(2, "0");

    timeElement.textContent =
        `${hours}:${minutes}`;
}


updateTime();


// =========================================
// Reset map prototype
// =========================================

const resetButton =
    document.getElementById("resetMap");

resetButton.addEventListener(
    "click",
    () => {

        const map =
            document.querySelector(".map-container");

        map.style.transform =
            "scale(1)";

        setTimeout(() => {

            map.style.transform =
                "";

        }, 200);

    }
);


// =========================================
// Prototype location interaction
// =========================================

const locations =
    document.querySelectorAll(".map-location");

locations.forEach((location) => {

    location.addEventListener(
        "click",
        () => {

            locations.forEach((item) => {

                item.style.opacity =
                    "0.45";

            });

            location.style.opacity =
                "1";

            location.style.transform =
                "scale(1.08)";

            setTimeout(() => {

                location.style.transform =
                    "";

            }, 250);

        }
    );

});