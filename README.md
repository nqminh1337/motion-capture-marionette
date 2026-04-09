# Intelligent Animation with OpenCV

A computer vision animation project built with **Python**, **OpenCV**, and **NumPy**.  
This project detects red motion markers from video footage, estimates a simple body pose, replaces the original blue background, draws a custom marionette, and adds intelligent moving objects with interactive behaviours.

## Overview

The animation pipeline works by processing a prerecorded motion video frame by frame.  
Red markers are segmented in HSV colour space and converted into a simplified set of body joints. These joints are then used to animate a marionette. The original blue background is removed and replaced with a generated scene, while additional intelligent objects move independently and interact with the character.

## Features

- Red marker detection using HSV thresholding
- Noise reduction with morphological operations
- Simple pose estimation from detected marker centroids
- Animated marionette rendering
- Blue-screen background replacement
- Intelligent object behaviours:
  - **Wanderer**: random movement with wall bouncing
  - **Seeker**: target-following movement with temporary avoidance behaviour
- Sprite-based rendering for objects
- Collision-based interaction effects

## How to Run

1. Install dependencies:

```bash
pip install opencv-python
pip install numpy
```
2. Run the program
```
python main.py
```
