var mediaDevices = navigator.mediaDevices;
var constraints = window.constraints = {
  audio: false,
  video: true
};

var capture = document.querySelector('#captureCanvas');
var button = document.querySelector('#clickMe');
var output = document.querySelector('#output');

Webcam.set({
  width: 320,
  height: 240,
  image_format: 'jpeg',
  jpeg_quality: 90
});
Webcam.attach( '#captureCanvas' );

button.addEventListener('click', () => {
  console.log('test');
  Webcam.snap( data => {
    window.open(data);
  });
});

/*
//var videoObject = document.createElement('video');
navigator.mediaDevices.getUserMedia({video: true}).then(mediaStream => {

  console.log(mediaStream);
  context.srcObject = mediaStream;
  context.play();

  captureCanvas.width = context.scrollWidth;
  captureCanvas.height = context.scrollHeight;

  console.log(`${captureCanvas.width} ${captureCanvas.height}`);

  captureCanvas.clearRect(0, 0, captureCanvas.width, captureCanvas.height);
  captureCanvas.drawImage(context, 0, 0, captureCanvas.width, captureCanvas.height );
  console.log(capture.toDataURL('image/png'))

}).catch(e => {
  console.log(e);
});
*/
