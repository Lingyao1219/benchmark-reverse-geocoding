# IMAGEO-Bench for Image Geolocalization

This repository contains a complete framework for benchmarking the performance of various Large Language Models (LLMs) on the task of image geolocalization.

# Overview
Image geolocalization, the task of identifying the geographic location depicted in an image, is important for applications in crisis response, digital forensics, and location-based intelligence. While recent advances in large language models (LLMs) offer new opportunities for visual reasoning, their ability to perform image geolocalization remains underexplored. In this study, we introduce a benchmark called IMAGEO-Bench that systematically evaluates accuracy, distance error, geospatial bias, and reasoning process. Our benchmark includes three diverse datasets covering global street scenes, points of interest (POIs) in the United States, and a private collection of unseen images. Through experiments on 10 state-of-the-art LLMs, including both open- and closed-source models, we reveal clear performance disparities, with closed-source models generally showing stronger reasoning. Importantly, we uncover geospatial biases as LLMs tend to perform better in high-resource regions (e.g., North America, Western Europe, and California) while exhibiting degraded performance in underrepresented areas. Regression diagnostics demonstrate that successful geolocalization is primarily dependent on recognizing urban settings, outdoor environments, street-level imagery, and identifiable landmarks. Overall, IMAGEO-Bench provides a rigorous lens into the spatial reasoning capabilities of LLMs and offers implications for building geolocation-aware AI systems.

The project is organized into two main parts:
1.  A data generation application (`src`) that queries LLMs to produce geocoding results from images.
2.  A suite of scripts (`analysis`) for in-depth analysis of the generated data.

## 📁 Project Structure



## 🚀 Stage 1: Running the Benchmark

This stage uses the application in the `src` directory to generate geocoding data.

### Features
* **Multi-Provider Support**: Out-of-the-box support for major LLM providers including OpenAI, Anthropic, Google, and TogetherAI.
* **Cost Tracking**: Automatically calculates and logs the cost in USD for each API call based on input/output tokens.
* **Resumable Processing**: The script automatically detects and skips images that have already been processed, allowing you to resume runs without duplicating work or cost.
* **Structured JSON Output**: Prompts are designed to enforce a specific JSON output, which is then parsed and saved into a clean `.jsonl` file.

### Setup and Installation

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

### How to Run

1.  **Configure the Run**: Open `src/config.py` and set the `MODEL`, `DEFAULT_PROVIDER`, and `DATASET` for your benchmark run.
2.  **Add Images**: Place the images you want to analyze into the corresponding dataset folder (e.g., `data/dataset2/`).
3.  **Execute the Script**: Run `main.py` from the root directory.
    ```bash
    python src/main.py
    ```
    Progress will be printed to the console, and results will be saved continuously to the `result/` directory.

---

## 🔬 Stage 2: Analyzing the Results

After generating data, use the scripts in the `analysis/` folder to interpret the results. These scripts consume the `.jsonl` files from the `result/` directory.

### Analysis Scripts

* **`evaluation.ipynb`**: Generates scatter plots comparing predicted vs. true latitude/longitude to visualize accuracy.
* **`heatmap.ipynb`**: Creates US state-level heatmaps of model accuracy to identify geographical performance biases.
* **`analyse_factors.py`**: Performs regression analysis (Logistic and Ridge) to determine which visual features (e.g., `environment`, `scene_type`) most influence prediction accuracy.
* **`draw_feature_weights.py`**: Visualizes the feature weights calculated by `analyse_factors.py` into bar plots.
* **`make_wordcloud.py`**: Creates a word cloud from the models' reasoning text to identify key terms used in localization.
