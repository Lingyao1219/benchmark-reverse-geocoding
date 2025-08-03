# IMAGEO-Bench for Image Geolocalization

Image geolocalization, the process of inferring geographic coordinates or addresses from visuals, presents a complex challenge at the intersection of computer vision and geographic information systems (GIS). The accurate identification of location from images has important implications for real-world scenarios, including digital forensics, urban analytics and crisis management. 

This repository contains a complete framework for benchmarking the performance of various Large Language Models (LLMs) on the task of image geolocalization. The framework for our study is illustrated below. 

<img width="3000" height="1519" alt="framework" src="https://github.com/user-attachments/assets/b4c8b80a-5311-4fc4-9c04-b69ce0099e7b" />


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
2.  **Add Images**: Place the images you want to analyze into the corresponding dataset folder (e.g., `data/dataset2/`). Datasets can be downloaded from this link: https://doi.org/10.5281/zenodo.16670471.
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
* 


## 📊 Datasets
This project compiles three benchmark datasets designed for the evaluation of LLMs' image geolocation tasks. In our code, we use dataset1, dataset2, and dataset3 to simplify the names. The benchmark datasets can be downloaded here: https://doi.org/10.5281/zenodo.16670471.

* **Dataset-GSS (Dataset1): Global Streetscape Set**
    * A set of 6,152 high-quality, street-level images from 123 countries, offering wide global diversity. It is derived from the NUS Global Streetscapes dataset https://ual.sg/project/global-streetscapes/.

* **Dataset-UPC (Dataset2): U.S. POIs Crowdsourced Set**
    * Contains 2,929 images of U.S. Points of Interest (POIs) compiled from a Google Maps dataset. The data is sampled to ensure balanced representation across all 50 states and 17 POI categories.

* **Dataset-PCW (Dataset3): Privately Collected Wild Set**
    * A private collection of 272 original images captured by the authors. It is designed for out-of-distribution evaluation to avoid data contamination from public web sources.

***


## 📈 Result Analysis

### Key Findings from Analysis

* The performance across ten LLMs on **Dataset-GSS** (global street-level images) on **Dataset-UPC** (crowdsourced POI images) is presented below. Overall, LLMs show better performance on Dataset-GSS that Dataset-UPC, possibly because most images in this dataset contain outdoor and street dtails that provide clearer geographic indicators than the often indoor, context-poor images in Dataset-UPC.

* **Closed-Source Models Lead**: Closed-sourced models (e.g., gpt-4.1, o3, gemini-2.5-flash, gemini-2.5-pro) from providers like Google and OpenAI consistently and substantially outperform open-source alternatives (e.g., Llama) across datasets and metrics.

* **Model Scaling Helps, But Isn't Everything**: While larger models generally perform better than their smaller counterparts, model size is not the only factor. 

* **Confidence Scores Can Be Misleading**: A model's self-reported confidence score is not a reliable indicator of its accuracy when compared against other models. While higher confidence can correlate with better accuracy *within a single model*, it should not be used alone to judge cross-model performance.

<img width="8932" height="4162" alt="dataset1_latitude_bench" src="https://github.com/user-attachments/assets/b87f7dd3-257c-42a7-885b-1af1668caefe" />
<img width="8906" height="4162" alt="dataset2_latitude_bench" src="https://github.com/user-attachments/assets/f62fbb0f-7110-4453-95b7-df35286396f7" />



