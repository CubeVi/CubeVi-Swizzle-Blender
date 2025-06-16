<h4 align="center">
  <img src="doc/doc/src/512x512.png" alt="openstageAI logo" style="width:15%; ">
  
<h1 align="center">CubeVi-Swizzle-Blender</h1>

</h3>

**Read this in other languages: [English](README.md), [中文](README_zh.md).**


[![OpenStageAI](https://img.shields.io/badge/OpenStageAI-web-blue)](https://cubevi.com/)
[![Blender](https://img.shields.io/badge/Blender-download-red)](https://www.blender.org/download/)
[![Chat](https://img.shields.io/badge/chat-discord-blue)](https://cubevi.com/pages/contact)
 <!-- this badge is too long, please place it in the last one to make it pretty --> 

<p align="center">
    👋 Join ours <a href="https://cubevi.com/pages/contact" target="_blank">Discord</a> 
</p>

## Project Introduction
This SDK is developed by **CubeVi**, aims at showing the preview result of Blender on our [**Light Field Display C1**](https://cubevi.com/products/cube-c1), to preview, render, save and upload the images and videos.


## Requirements

This project is designed for [**Light Field Display C1**](https://cubevi.com/products/cube-c1), please make sure your computer is connected to the light field display correctly [**Light Field Display C1**](https://cubevi.com/products/cube-c1), user manual and openstageai app downloading [click here](https://cubevi.com/pages/download-page)

**Supporting Blender version**

| Blender version | Download |
| :--- | :---: | 
| Blender3.4 | [3.4](https://download.blender.org/release/Blender3.4/) | 
| Blender3.5 | [3.5](https://download.blender.org/release/Blender3.5/) | 
| Blender3.6 | [3.6](https://download.blender.org/release/Blender3.6/) | 
| Blender4.0 | [4.0](https://download.blender.org/release/Blender4.0/) | 
| Blender4.1 | [4.1](https://download.blender.org/release/Blender4.1/) | 
| Blender4.2 | [4.2](https://download.blender.org/release/Blender4.2/) | 

**This SDK only support Windows**

## Installation guide

Installation and detailed guidance [Usage](doc/doc/usage.md)

## Usage

### Device connection

1. Please make sure your computer is connected to the light field display correctly [**Light Field Display C1**](https://cubevi.com/products/cube-c1), and open [**OpenstageAI**](https://cubevi.com/pages/download-page) platform（newest version）to detect the device.
    
2. Open blender, in edit->preference->plugin,import the plugin ZIP file installation package. If successful, you will see the LFD panel on the left.
    
3. Click the connect buttom, the SDK will detect the connected device and set the render resolution automatically.

### Camera Setting

4. By setting the front, back, and focal planes of the camera, you can achieve different effects in and out of the screen.

5. Near and Far Clipping Faces: Only objects within the near and far clipping face frustum will be rendered.

6. Focal plane: The focal plane of the camera. Objects in the focal plane will be the clearest. The side of the focal plane close to the camera will show the in-screen effect, the side of the focal plane away from the camera will show the out-screen effect, and objects far away from the focal plane will become blurry.



### Preview panel

7. After the device is successfully connected, click the real-time light field preview, and the light field preview picture of the current camera will be automatically displayed on C1.

8. Click save quilt image to save the grid preview image in the currently set file path.

9. Click save light field image to save the light field image in the currently set file path.

### Rendering

 10. After the device is successfully connected, click save preview image, and the camera will automatically take 40 single viewpoint images at the current location (press ESC to cancel),the name will be set from _000.png to _039.png.

 11. After the 40 single-view images are rendered, click synthesize quilt image to automatically combine the _000.png-_039.png into a quilt image.

 12. When the device is successfully connected and the platform is opened, click upload the quilt image, the SDK will upload the quilt image to the 3D gallery. Open 3D gallery -> Home -> My Creations to view the light field image generated from the quilt image. (The maximum size of the grid image is 70MB.)

### Rendering animation

 13. Set the start frame-end frame of the animation rendering, click Render Animation (ESC Cancel), the quilt image of each frame of the start frame to the end frame will be automatically rendered, named and saved to the quilt_frame_index under the current file path.png

 14. Set the start frame-end frame of the animation rendering, set the FPS of the output video, click to synthesize the quilt image sequence into a video, and the SDK will combine the quilt image from the start frame to the end frame into a output.mp4.

 15. Click Upload Video to 3D Gallery, the grid diagram video will be uploaded to the 3D Gallery, open the 3D Gallery - > Home - > My Creation to watch the light field video. (The maximum video size is 70MB)



## Limitation

- Due to the limitations of Blender's rendering engine, real-time light field previews can lag when performing real-time light field previews for scenes with textured details.
- Currently, the size of the grid diagram and the grid diagram video that can be uploaded to the platform is limited to 70MB.

## Discussion

You can report any problems in the issues.





