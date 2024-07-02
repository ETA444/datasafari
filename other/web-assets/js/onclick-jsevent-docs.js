// event code used on site (builder requires one liner)
(function(){var overlay=document.createElement('div');overlay.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;background-color:#303335;display:flex;justify-content:center;align-items:center;animation:fadein 0.5s;cursor:pointer';var img=document.createElement('img');img.src='https://datasafari.dev/docs/_static/logos/ds-branding-logo-big-darkmode.png';img.style.width='200px';overlay.appendChild(img);document.body.appendChild(overlay);setTimeout(function(){window.location.href='https://datasafari.dev/docs/';},500);var css=document.createElement('style');css.type='text/css';css.innerHTML='@keyframes fadein { from { opacity: 0; } to { opacity: 1; } }';document.body.appendChild(css);})();

// formatted for readability and benefit of this backup file
(function() {
    // Create the overlay element
    var overlay = document.createElement('div');
    overlay.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: #303335; display: flex; justify-content: center; align-items: center; animation: fadein 0.5s; cursor: pointer';

    // Create the image element
    var img = document.createElement('img');
    img.src = 'https://datasafari.dev/docs/_static/logos/ds-branding-logo-big-darkmode.png';
    img.style.width = '200px';

    // Append the image to the overlay
    overlay.appendChild(img);

    // Append the overlay to the body
    document.body.appendChild(overlay);

    // Redirect after a brief delay
    setTimeout(function() {
        window.location.href = 'https://datasafari.dev/docs/';
    }, 500);

    // Create a style element for the fade-in animation
    var css = document.createElement('style');
    css.type = 'text/css';
    css.innerHTML = '@keyframes fadein { from { opacity: 0; } to { opacity: 1; } }';
    document.body.appendChild(css);
})();
