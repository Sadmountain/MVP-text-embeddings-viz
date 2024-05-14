
# Research MVP

This repository contains the code for the MVP of a research project built with Streamlit. The project includes clustering and visualization of research paper abstracts.

## Introduction
This project aims to provide an interactive tool for analyzing and visualizing research paper abstracts based on various clustering techniques and distance metrics.

## Folder Structure

- `preprocessing/`: Contains preprocessing scripts and modules.
- `data/`: Directory for storing dataset files (not included in the repository).
- `scripts/`: Contains the main application script and other relevant scripts.
- `output/`: Directory for storing output files.

## Setup and Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Sadmountain/paper-embedding-visualization
    cd research-mvp
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Place your dataset in the `data/` directory. For example this dataset:https://github.com/asreview/synergy-dataset/tree/master/datasets/van_de_Schoot_2018

5. Run the Streamlit app:
    ```sh
    streamlit run scripts/main_app.py
    ```

## Contact Information

For any questions or issues, please contact:
- Josh Bleijenberg
- Email: j.j.j.bleijenberg@students.uu.nl