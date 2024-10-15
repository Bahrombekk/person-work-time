Here's a `README.md` for your project:

---

# Person Work Time Tracker

This project tracks the presence of workers within predefined areas in a camera feed, recording their total work time using YOLOv8 for human detection. Results are saved in an Excel file for each camera.

## Features
- **Multiple Cameras**: Supports simultaneous tracking from multiple cameras.
- **Real-Time Detection**: Detects if a person is in a designated area and records time.
- **Excel Reports**: Generates daily reports for each camera in Excel format.

## Requirements
- Python 3.8+
- OpenCV
- YOLOv8 (`ultralytics`)
- `openpyxl` for Excel handling

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd person-work-time
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model:
   ```bash
   wget <yolo-model-url>
   ```

## Usage

1. Configure your camera settings in `camera_config.json`. Example:

```json
[
  {
    "id": 0,
    "rtsp_url": "your_camera_rtsp_link",
    "rectangles": [
      {
        "name": "Worker1",
        "coordinates": [861, 784, 332, 427]
      },
      {
        "name": "Worker2",
        "coordinates": [1284, 777, 366, 419]
      }
    ]
  }
]
```

2. Run the program:
   ```bash
   python oqim.py
   ```

3. Excel reports will be saved in the format `time_tracking_camera_<camera_id>.xlsx`.

## Camera Configuration
Each camera requires an RTSP URL and coordinates for the areas to track. Adjust the coordinates in `camera_config.json` for the specific camera views.

## Error Handling
In case of connection issues with the camera, the script will attempt to reconnect automatically.

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

--- 

This document will help users understand the project, set it up, and run it effectively. Let me know if you need any further modifications!
