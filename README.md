Footfall Counter Using YOLOv5 and Centroid Tracking

Project Overview

This project presents a practical computer vision solution for counting foot traffic using video input. It employs the highly accurate YOLOv5 model for human detection and a lightweight centroid-based tracker to persistently identify individuals across frames and count their crossings over a designated Region of Interest (ROI).

Methodology

    Detection: Utilized YOLOv5 pretrained on COCO for reliable person detection.

    Tracking: Implemented a custom centroid tracker to assign consistent IDs to individuals and maintain continuity despite occlusions and frame drops.

    Counting Logic: Tracks entry and exit counts by monitoring centroid movement relative to a user-defined ROI line.

    Adjustable Parameters: Key parameters such as detection confidence and ROI line position were fine-tuned to optimize performance for the test environment.

Results

The solution accurately detects and tracks individuals in video streams, delivering reliable counts of entries and exits. Visual overlays ensure results are interpretable and verifiable. Although performance may vary under extreme crowding, the system is robust in typical conditions and readily adaptable to other settings.


[Output Video on Google Drive](https://drive.google.com/file/d/1E54Rzp8JJ5x_TPM6o0_-yeiX5lOKvzXc/view?usp=sharing)
