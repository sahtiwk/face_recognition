# Custom Face Recognizer using OpenCV

This project demonstrates a complete, two-step process for building a custom face recognition system using Python and the OpenCV library. It first trains a model on a dataset of faces you provide and then uses that model to recognize those faces in a new image.

### Core Technology

It is very important to understand that this project uses built-in tools provided by OpenCV:
1.  **Face Detection**: We use the **Haar Cascade classifier** (`haar_face.xml`), a pre-trained model from OpenCV, to *detect the location* of faces in an image.
2.  **Face Recognition**: Once a face is detected, we use OpenCV's **Local Binary Patterns Histograms (LBPH) Face Recognizer** to *identify who the person is*. The recognizer is trained on our custom dataset.

***

## Required Directory Structure

For the scripts to work correctly, you **must** organize your files and folders exactly as shown below. The training script (`face_trainer.py`) is designed to automatically read from the `training_data` directory.

```
your_project_folder/
│
├── 📄 face_trainer.py           # The script to train the model
├── 📄 face_recognition.py       # The script to recognize faces
├── 📄 haar_face.xml             # The pre-trained detector from OpenCV
├── 🖼️ test_image.jpg            # The image you want to test for recognition
│
└── 📁 training_data/            # Folder containing all training images
    │
    ├── 📁 person_1_name/        # A folder for the first person
    │   ├── 🖼️ image1.jpg
    │   ├── 🖼️ image2.png
    │   └── 🖼️ ... (multiple pictures of this person)
    │
    ├── 📁 person_2_name/        # A folder for the second person
    │   ├── 🖼️ image_A.jpg
    │   ├── 🖼️ image_B.jpg
    │   └── 🖼️ ... (multiple pictures of this person)
    │
    └── 📁 ... (more folders for more people)
```

**Key Points:**
- The folder name for each person (e.g., `person_1_name`) will be used as their label.
- You need multiple images of each person for the model to train effectively.

***

## How to Use

Follow these steps to get the face recognizer up and running.

### Step 1: Setup and Dependencies

1.  **Clone or Download:** Get all the files from this repository.
2.  **Install Libraries:** Make sure you have the necessary Python libraries installed.
    ```bash
    pip install opencv-python numpy
    ```
3.  **Prepare Training Data:** Create the `training_data` directory and populate it with images according to the structure described above.

### Step 2: Train the Model

1.  **Run the Trainer Script:** Open your terminal or command prompt, navigate to the project folder, and run the following command:
    ```bash
    python face_trainer.py
    ```
2.  **Check for Output:** The script will scan your `training_data` folder. Once it's finished, you will see two new files in your project directory:
    - `face_trained.yml`: This is the trained LBPH model.
    - `people.npy`: This is a helper file that stores the names of the people (labels) the model was trained on.

### Step 3: Recognize Faces

1.  **Add a Test Image:** Place an image you want to test in the main project folder and make sure it is named `test_image.jpg`. This image should contain a face of one of the people from your training set.
2.  **Run the Recognition Script:** Execute the recognition script from your terminal:
    ```bash
    python face_recognition.py
    ```

A window will pop up displaying the `test_image.jpg` with a green rectangle around the detected face and a label identifying the person. The console will also print the predicted label and a confidence score.
