// your_app/static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    // These URLs would normally be hardcoded in script.js, but they contain Django template tags
    const weatherDescription = "{{ description|lower|default:'default' }}";
    const body = document.body;

    // Log the weather description for debugging
    console.log("Weather Description:", weatherDescription);

    // Function to set background image with error handling
    const setBackgroundImage = (imageUrl) => {
        const img = new Image();
        img.src = imageUrl;
        img.onload = () => {
            console.log(`Successfully loaded image: ${imageUrl}`);
            body.style.backgroundImage = `url('${imageUrl}')`;
        };
        img.onerror = () => {
            console.error(`Failed to load image: ${imageUrl}`);
            body.style.backgroundImage = 'none'; // Reset to allow CSS fallback gradient
        };
    };

    // Map weather descriptions to background images
    if (weatherDescription.includes('rain')) {
        setBackgroundImage("{% static 'images/rain.jpg' %}");
    } else if (weatherDescription.includes('cloud')) {
        setBackgroundImage("{% static 'images/cloudy.jpg' %}");
    } else if (weatherDescription.includes('sun') || weatherDescription.includes('clear')) {
        setBackgroundImage("{% static 'images/sunny.jpg' %}");
    } else if (weatherDescription.includes('snow')) {
        setBackgroundImage("{% static 'images/snow.jpg' %}");
    } else {
        const defaultImages = [
            "{% static 'images/default1.jpg' %}",
            "{% static 'images/default2.jpg' %}",
            "{% static 'images/default3.jpg' %}",
            "{% static 'images/default4.jpg' %}",
            "{% static 'images/default5.jpg' %}"
        ];
        const randomImage = defaultImages[Math.floor(Math.random() * defaultImages.length)];
        setBackgroundImage(randomImage);
    }

    // Add loading spinner on form submission
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', () => {
            // Remove any existing loading spinner
            const existingSpinner = document.querySelector('.loading');
            if (existingSpinner) {
                existingSpinner.remove();
            }
            // Add new loading spinner
            form.insertAdjacentHTML('afterend', '<p class="loading">Loading...</p>');
        });
    }
});