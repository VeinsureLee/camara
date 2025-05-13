const socket = new WebSocket('ws://' + window.location.host + '/ws/video/');
const img = document.getElementById('video-frame');

socket.onmessage = function(e) {
    img.src = 'data:image/jpeg;base64,' + e.data;
};
