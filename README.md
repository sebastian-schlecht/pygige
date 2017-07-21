# pygige
Highly experimental python wrapper for GigE Machine Vision Cameras

## Preface
This software is highly experimental and there is no guarantee that it'll work.

## Installation
Once you followed instructions from (here)[https://www.smartek.vision/media/downloads/SMARTEKVision_GigEVisionSDK_Linux_Readme.txt], clone this repository and run ```make```. You'll get a file ```pygige.so``` which you can import via ```import gige```.

## Usage
To setup the first camera, run
```
d = gige.setup()
```
This will return a device object, which can be used to grab frames. Run
```
frame = gige.getFrame(d, 3.)
```
to use grab a frame in numpy format using the previously obtained device and a timeout of 3s.
The connection to the camera will be closed automatically upon disposal of the device object.

## Todo
- get/set device parameters
- select device from a list of devices
- start/stop acquisition
- much more really
