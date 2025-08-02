# IMAGEO-Bench for Image Geolocalization

This repository contains a complete framework for benchmarking the performance of various Large Language Models (LLMs) on the task of image geolocalization.

# Overview
Image geolocalization, the task of identifying the geographic location depicted in an image, is important for applications in crisis response, digital forensics, and location-based intelligence. While recent advances in large language models (LLMs) offer new opportunities for visual reasoning, their ability to perform image geolocalization remains underexplored. In this study, we introduce a benchmark called IMAGEO-Bench that systematically evaluates accuracy, distance error, geospatial bias, and reasoning process. Our benchmark includes three diverse datasets covering global street scenes, points of interest (POIs) in the United States, and a private collection of unseen images. Through experiments on 10 state-of-the-art LLMs, including both open- and closed-source models, we reveal clear performance disparities, with closed-source models generally showing stronger reasoning. Importantly, we uncover geospatial biases as LLMs tend to perform better in high-resource regions (e.g., North America, Western Europe, and California) while exhibiting degraded performance in underrepresented areas. Regression diagnostics demonstrate that successful geolocalization is primarily dependent on recognizing urban settings, outdoor environments, street-level imagery, and identifiable landmarks. Overall, IMAGEO-Bench provides a rigorous lens into the spatial reasoning capabilities of LLMs and offers implications for building geolocation-aware AI systems.

The project is organized into two main parts:
1.  A data generation application (`src`) that queries LLMs to produce geocoding results from images.
2.  A suite of scripts (`analysis`) for in-depth analysis of the generated data.

## 📁 Project Structure

```bash
benchmark-reverse-geocoding/
├── analysis/
│   ├── result/
│   ├── analyse_factors.py
│   ├── dataset1_info.csv
│   ├── dataset2_info.csv
│   ├── dataset3_info.csv
│   ├── draw_feature_weights.py
│   ├── evaluation.ipynb
│   ├── heatmap.ipynb
│   ├── make_wordcloud.py
│   └── utils.py
├── data/
├── src/
│   └── images/
│       ├── config.py
│       ├── main.py
│       ├── model.py
│       ├── prompt.py
│       └── secrets.txt
├── README.md
└── requirements.txt
```

## 🚀 Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Lingyao1219/benchmark-reverse-geocoding.git](https://github.com/Lingyao1219/benchmark-reverse-geocoding.git)
    cd benchmark-reverse-geocoding
    ```

2.  **Create a Virtual Environment and Install Dependencies**
    A `requirements.txt` file is provided to install necessary packages.
    ```bash
    # Create and activate a virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Install required packages from requirements.txt
    pip install -r requirements.txt
    ```

3.  **Create and Configure `secrets.txt`**
    Create a file named `secrets.txt` in the root directory. This file stores your API keys. Add your keys in the format `key_name,key_value`, like so:
    ```ini
    # secrets.txt
    openai_key,sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    claude_key,sk-anthropic-xxxxxxxxxxxxxxxxxxx
    gemini_key,AIzaxxxxxxxxxxxxxxxxxxxxxxxxxxx
    together_key,xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

***

## ⚙️ Workflow and Usage

The project follows a two-stage workflow:

1.  **Stage 1: Run the Benchmark**: Use the application in the `src` folder to generate the raw `.jsonl` data files.
2.  **Stage 2: Analyze the Results**: Use the scripts in the `analysis` folder to process the generated data and create visualizations.

### How to Run the Benchmark

1.  **Configure the Run**: Open `src/config.py` and set the `MODEL`, `DEFAULT_PROVIDER`, and `DATASET` for your benchmark run.
2.  **Add Images**: Place the images you want to analyze into the corresponding dataset folder (e.g., `data/dataset2/`).
3.  **Execute the Script**: Run `main.py` from the root directory.
    ```bash
    python src/main.py
    ```
    Progress will be printed to the console, and results will be saved continuously to the `result/` directory.

***

## 📁 File Explanations

### `src` Directory

* **`main.py`**: This is the main entry point of the application. It manages the image processing loop, calls the LLM for each image, handles file I/O for results, and includes a cost analysis mode.
* **`config.py`**: A centralized configuration file where you can easily set parameters like the `MODEL` to use, the `DEFAULT_PROVIDER`, the `DATASET` folder, and the API `TEMPERATURE`.
* **`model.py`**: This script handles all direct interactions with the different LLM provider APIs (OpenAI, Anthropic, Google, etc.). It contains the logic for formatting requests, calculating API costs, and parsing the JSON responses.
* **`prompt.py`**: Contains the detailed system and user prompts that are sent to the LLM. It defines the required JSON output structure and provides examples to guide the model's response.

### `analysis` Directory

* **`evaluation.ipynb`**: Generates scatter plots comparing predicted vs. true latitude/longitude to visualize accuracy.
* **`heatmap.ipynb`**: Creates US state-level heatmaps of model accuracy to identify geographical performance biases.
* **`analyse_factors.py`**: Performs regression analysis (Logistic and Ridge) to determine which visual features (e.g., `environment`, `scene_type`) most influence prediction accuracy.
* **`draw_feature_weights.py`**: Visualizes the feature weights calculated by `analyse_factors.py` into bar plots.
* **`make_wordcloud.py`**: Creates a word cloud from the models' reasoning text to identify key terms used in localization.
* **`utils.py`**: A collection of helper functions used for post-processing the `.jsonl` result files. This includes functions to load data into a pandas DataFrame, calculate geographic distances, and analyze prediction accuracy.
