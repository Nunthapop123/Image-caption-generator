# Image Caption Generator

A web application that uses AI to analyze your photos and describe what it sees.

## Features:
- **BLIP Model Integration:** Uses the accurate `Salesforce/blip-image-captioning-large` vision-language model.
- **File Upload:** Supports multiple static and animated image formats (PNG, JPG, JPEG, GIF, BMP, WEBP).
- **Real-time Processing:** Generates captions on-the-fly with blinking retro loading indicators.
- **Error Handling:** Comprehensive error handling and validation for file types and sizes.
- **Hardware Acceleration:** Automatically uses GPU via PyTorch/CUDA if available for faster image processing.

## To run the application:

1. **Install dependencies:**
   Make sure you have Python installed, then install the necessary packages using the `requirements.txt` file (or just `pip install flask torch transformers pillow`):
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   Make sure the code is saved in `app.py` in your current directory, then execute:
   ```bash
   python app.py
   ```

3. **Open the App:**
   Open your browser and navigate to exactly:
   [http://localhost:5050](http://localhost:5050)

> **Note:** The server port was set to `5050` in the application code.

## How it works:
1. Users can upload images by clicking the interactive `[+]` upload area or dragging and dropping the image source.
2. The image is locally processed by the BLIP model to generate a descriptive, matching caption without using external captioning APIs.
3. Results are returned with a resized preview of the original image and the ALL-CAPS text caption printed in 8-bit style.
4. The server actively enforces file limits — specifically capped at 16MB per upload — as well as rigid format validation checks.

> **Important:** The application will automatically download the BLIP model on its first run (this may take a few minutes depending on your internet connection). The model is quite large (~2GB), so make sure you have sufficient disk space and a stable internet connection for the initial download phase.

The interface is entirely user-friendly with direct visual feedback for all operations inside a simulated "CRT" monitor, and the backend is robustly supported with proper exception formatting and terminal logging.
