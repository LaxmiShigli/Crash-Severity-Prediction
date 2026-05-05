const slides = Array.from(document.querySelectorAll(".slide"));
const dots = Array.from(document.querySelectorAll(".dot"));
const prevButton = document.querySelector(".slide-prev");
const nextButton = document.querySelector(".slide-next");
const speedInput = document.querySelector("#speed");
const speedValue = document.querySelector("#speed-value");

let activeSlide = 0;

function renderSlide(index) {
    if (!slides.length) return;
    activeSlide = (index + slides.length) % slides.length;

    slides.forEach((slide, currentIndex) => {
        slide.classList.toggle("active", currentIndex === activeSlide);
    });

    dots.forEach((dot, currentIndex) => {
        dot.classList.toggle("active", currentIndex === activeSlide);
    });
}

if (prevButton && nextButton) {
    prevButton.addEventListener("click", () => {
        renderSlide(activeSlide - 1);
    });

    nextButton.addEventListener("click", () => {
        renderSlide(activeSlide + 1);
    });
}

dots.forEach((dot) => {
    dot.addEventListener("click", () => {
        renderSlide(Number(dot.dataset.slideIndex || 0));
    });
});

if (slides.length) {
    renderSlide(0);
}

if (speedInput && speedValue) {
    const syncSpeedValue = () => {
        speedValue.textContent = speedInput.value;
    };

    syncSpeedValue();
    speedInput.addEventListener("input", syncSpeedValue);
}

const resultSection = document.querySelector("[data-has-result='true']");
if (resultSection) {
    window.requestAnimationFrame(() => {
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    });
}
