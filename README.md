

---

# Car Detection with YOLOv7 and PaddleOCR

This project involves detecting cars using YOLOv7 and reading their license plates using PaddleOCR. It also calculates and reports the speed of detected cars and their labels every 3 frames.

## Setup Instructions

To get started, follow these steps:

### 1. Set Up Your Environment

For the best experience, use a Miniconda environment:

1. **Create a new Conda environment:**
    ```bash
    conda create -n car_detection python=3.8
    ```

2. **Activate the environment:**
    ```bash
    conda activate car_detection
    ```

### 2. Install Dependencies

1. **Download `requirements.txt`:** This file lists all the required libraries.
2. **Install the libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Download and Place Files

- Ensure all files are in the same directory for ease of access.
- Update file paths in your code as needed.

### 4. Run the Code

1. **Execute the main script:**
    ```bash
    python roadson.py
    ```

### Troubleshooting
- **Missing Libraries:** I could not test from empty environment but if you are missing libraries just pip install them as the promptshell suggests if still missing you can get online help or ask gpt :)
- **File Locations:** Keep all necessary files in the same directory and update file paths in your code as needed. [Example Image](https://github.com/user-attachments/assets/2a1fb4c0-c56a-4f76-ba70-9d19c7589e24)
- **KMP DUPLICATE Error:** If you encounter this error, check `roadson.py`, which is the main Python file to run.
- **DLL Errors:** Create a new Conda environment and reinstall the requirements. Ensure that the version of PyTorch is compatible with your GPU and CUDA version.

### Test Run

For a demonstration of how to run the code properly, watch this video: [Test Run Video](https://youtu.be/FW_o0xQEuyo)

---

Feel free to adjust any specific details to better fit your project or personal preferences!
