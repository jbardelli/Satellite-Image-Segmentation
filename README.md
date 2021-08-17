# Satellite-Image-Segmentation
Satellite Image Segmentation using U-net

## Oil well pad segmentation model
The objective was to train a Unet model for segmenting oil well pads (patches of land where soil is flattened to drill a well) seen from satellite images.
A series of 25 images were taken from Google Earth from oil fields in Argentina where well pads were clearly visible, and these images annotated using apeer web interface.
Only two label categories were used:
- (1) for the well pad
- (0) for the background 
Images were cropped in a mosaic of 256x256 patches for training and testing, to avoind using a model for which too much memory would be needed. After training for 100 epochs, the model was saved.
Some of the predictions can be seen below for 256x256 patches from a test image.


