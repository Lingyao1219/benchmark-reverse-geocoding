import re
import os
import sys
import json
import config
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns


STATE_MAPPING = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR', 'california': 'CA',
    'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE', 'florida': 'FL', 'georgia': 'GA',
    'hawaii': 'HI', 'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA',
    'kansas': 'KS', 'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV', 'new hampshire': 'NH',
    'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY', 'north carolina': 'NC',
    'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK', 'oregon': 'OR', 'pennsylvania': 'PA',
    'rhode island': 'RI', 'south carolina': 'SC', 'south dakota': 'SD', 'tennessee': 'TN',
    'texas': 'TX', 'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
    'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY'
}


def simplify_model_name(name: str) -> str:
    """
    Converts a long model name into a simpler version using a direct
    mapping based on the provided list.
    """

    clean_name = name.lower()

    mapping = {
        'gpt-4.1-mini': 'gpt-4.1-mini',
        'gpt-4.1': 'gpt-4.1',
        'llama-4-maverick-17b': 'llama-4-17b',
        'llama-3.2-11b': 'llama-3.2-11b',
        'llama-3.2-90b': 'llama-3.2-90b',
        'qwen2.5-vl-72b-instruct': 'qwen-2.5-72b',
        'claude-3.5-sonnet': 'claude-3.5-sonnet',
        'claude-3-5-haiku-20241022': 'claude-3.5-haiku',
        'gemini-1.5-pro-latest': 'gemini-1.5-pro',
    }

    for key, simplified_name in mapping.items():
        if key in clean_name:
            return simplified_name
    return name.split('/')[-1]


def _get_model_order(all_data, model_order=None):
    """Helper to determine the order of models for plotting."""
    existing_models = list(all_data.keys())
    if model_order is None:
        return existing_models
    ordered_present_models = [m for m in model_order if m in existing_models]
    other_models = sorted([m for m in existing_models if m not in ordered_present_models])
    return ordered_present_models + other_models


def process_jsonl_to_dataframe(file_path: str):
    """
    Reads a JSONL file and creates a custom flattened pandas DataFrame
    with simplified column names as requested.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        sys.exit(1)

    print(f"Reading data from '{file_path}'...")
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line: {line.strip()}")
                continue
    
    if not records:
        print("No valid JSON records were found in the file.")
        return None

    processed_records = []
    for record in records:
        location_info = record.get('location_info', {})
        if not isinstance(location_info, dict):
            location_info = {}

        image_info = location_info.get('image_information', {})
        if not isinstance(image_info, dict):
            image_info = {}

        reasoning = location_info.get('reasoning', {})
        if not isinstance(reasoning, dict):
            reasoning = {}

        reverse_geocoding = location_info.get('reverse_geocoding', {})
        if not isinstance(reverse_geocoding, dict):
            reverse_geocoding = {}
            
        address = reverse_geocoding.get('address', {})
        if not isinstance(address, dict):
            address = {}
            
        coordinates = reverse_geocoding.get('coordinates', {})
        if not isinstance(coordinates, dict):
            coordinates = {}
            
        usage_info = record.get('usage_info', {})
        if not isinstance(usage_info, dict):
            usage_info = {}

        lat = coordinates.get('latitude')
        lon = coordinates.get('longitude')
        
        try:
            lat = float(lat) if lat is not None and str(lat).lower() not in ['null', 'none', ''] else None
            if lat == 0.0:
                lat = None
        except (ValueError, TypeError):
            lat = None
            
        try:
            lon = float(lon) if lon is not None and str(lon).lower() not in ['null', 'none', ''] else None
            if lon == 0.0:
                lon = None
        except (ValueError, TypeError):
            lon = None

        flat_record = {
            'image_file': record.get('image_file'),
            'model_used': record.get('model_used'),

            'environment': image_info.get('environment'),
            'scene_type': image_info.get('scene_type'),
            'setting': image_info.get('setting'),

            'landmark_recognition': reasoning.get('landmark_recognition'),
            'text_and_signage': reasoning.get('text_and_signage'),
            'cultural_indicators': reasoning.get('cultural_indicators'),
            'spatial_context': reasoning.get('spatial_context'),
            
            'confidence': reverse_geocoding.get('confidence'),
            'street': address.get('street'),
            'city': address.get('city'),
            'state': address.get('state'),
            'country': address.get('country'),
            'latitude': lat,
            'longitude': lon,
            'usage_info': usage_info 
        }
        processed_records.append(flat_record)

    df = pd.DataFrame(processed_records)
    core_cols = [
        'image_file', 'model_used', 'confidence', 'street', 'city', 'state', 
        'country', 'latitude', 'longitude', 'usage_info'
    ]
    
    existing_core_cols = [col for col in core_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in existing_core_cols]
    df = df[existing_core_cols + other_cols]

    return df



def parse_address(address_str):
    """
    Parse address string into street, city, and state components.
    This version works backward from the end of the string for better accuracy.
    """
    if pd.isna(address_str) or not isinstance(address_str, str) or address_str.strip() == '':
        return {'street': '', 'city': '', 'state': ''}

    # Clean and split the address into parts based on commas
    address_str = address_str.strip()
    parts = [part.strip() for part in address_str.split(',')]

    street = ''
    city = ''
    state = ''

    if len(parts) >= 3:
        state_zip_part = parts[-1]
        city = parts[-2]
        street = ", ".join(parts[:-2])
        state_match = re.search(r'\b([A-Z]{2})\b', state_zip_part)
        state = state_match.group(1) if state_match else ''

    elif len(parts) == 2:
        street = parts[0]
        city_state_part = parts[1]
        state_match = re.search(r'\b([A-Z]{2})\b', city_state_part)
        if state_match:
            state = state_match.group(1)
            city = city_state_part[:state_match.start()].strip()
        else:
            city = city_state_part
            state = ''
            
    # If there's only one part
    else:
        street = parts[0]
        state_match = re.search(r'\b([A-Z]{2})\b', street)
        if state_match:
            state = state_match.group(1)

    return {
        'street': street,
        'city': city,
        'state': state
    }


def standardize_state(state_val):
    if pd.isna(state_val) or state_val == '':
        return np.nan
    state_str = str(state_val).strip().lower()
    if len(state_str) == 2 and state_str.isalpha():
        return state_str.upper()
    return STATE_MAPPING.get(state_str, state_str.upper())


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in kilometers
    """
    try:
        # Convert to float first, then to radians
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        return c * r
    except (ValueError, TypeError):
        return np.nan


def calculate_distances(df):
    """
    Calculate distances between coordinates
    """
    df_with_dist = df.copy()
    
    # Only calculate distance where both coordinate pairs are available
    valid_coords = ~(df_with_dist[['latitude', 'longitude', 'true_latitude', 'true_longitude']].isna().any(axis=1))
    
    distances = []
    for idx, row in df_with_dist.iterrows():
        if valid_coords.iloc[idx]:
            dist = haversine_distance(
                row['latitude'], row['longitude'],
                row['true_latitude'], row['true_longitude']
            )
            distances.append(dist)
        else:
            distances.append(np.nan)
    
    df_with_dist['distance_km'] = distances
    return df_with_dist


def analyze_accuracy(df):
    """
    Analyze accuracy between city/state pairs
    """
    print("=== ACCURACY ANALYSIS ===\n")
    
    # City accuracy
    if 'city' in df.columns and 'true_city' in df.columns:
        print("1. CITY ACCURACY:")
        valid_cities = ~(df['city'].isna() | df['true_city'].isna())
        total_valid_cities = valid_cities.sum()
        
        if total_valid_cities > 0:
            city_matches = (df.loc[valid_cities, 'city'].str.lower() == df.loc[valid_cities, 'true_city'].str.lower())
            city_accuracy = city_matches.mean()
            print(f"   Valid city pairs: {total_valid_cities}")
            print(f"   Exact matches: {city_matches.sum()}")
            print(f"   City accuracy: {city_accuracy:.2%}")
        else:
            print("   No valid city pairs found")
    
    # State accuracy
    if 'state' in df.columns and 'true_state' in df.columns:
        print("\n2. STATE ACCURACY:")
        valid_states = ~(df['state'].isna() | df['true_state'].isna())
        total_valid_states = valid_states.sum()
        
        if total_valid_states > 0:
            state_matches = (df.loc[valid_states, 'state'] == df.loc[valid_states, 'true_state'])
            state_accuracy = state_matches.mean()
            print(f"   Valid state pairs: {total_valid_states}")
            print(f"   Exact matches: {state_matches.sum()}")
            print(f"   State accuracy: {state_accuracy:.2%}")
        else:
            print("   No valid state pairs found")
    
    # Country accuracy
    if 'country' in df.columns and 'true_country' in df.columns:
        print("\n3. COUNTRY ACCURACY:")
        valid_countries = ~(df['country'].isna() | df['true_country'].isna())
        total_valid_countries = valid_countries.sum()
        
        if total_valid_countries > 0:
            country_matches = (df.loc[valid_countries, 'country'].str.lower() == df.loc[valid_countries, 'true_country'].str.lower())
            country_accuracy = country_matches.mean()
            print(f"   Valid country pairs: {total_valid_countries}")
            print(f"   Exact matches: {country_matches.sum()}")
            print(f"   Country accuracy: {country_accuracy:.2%}")
        else:
            print("   No valid country pairs found")
    
    # Distance analysis
    if 'distance_km' in df.columns:
        print("\n4. COORDINATE DISTANCE ANALYSIS:")
        valid_distances = ~df['distance_km'].isna()
        if valid_distances.sum() > 0:
            distances = df.loc[valid_distances, 'distance_km']
            print(f"   Valid coordinate pairs: {valid_distances.sum()}")
            print(f"   Mean distance: {distances.mean():.2f} km")
            print(f"   Median distance: {distances.median():.2f} km")
            print(f"   Max distance: {distances.max():.2f} km")
            print(f"   Min distance: {distances.min():.2f} km")
            print(f"   Std deviation: {distances.std():.2f} km")
        else:
            print("   No valid coordinate pairs found")
    
    return df


def get_performance_df(all_data):
    """
    Calculates performance metrics and usage statistics for a dictionary of dataframes.
    """
    performance_results = []
    for model_name, df in all_data.items():
        metrics = {'model': model_name}

        # Accuracy Metrics
        if 'city' in df.columns and 'true_city' in df.columns:
            valid_cities = ~(df['city'].isna() | df['true_city'].isna())
            metrics['city_accuracy'] = (df.loc[valid_cities, 'city'].str.lower() == df.loc[valid_cities, 'true_city'].str.lower()).mean() if valid_cities.any() else np.nan
        else:
            metrics['city_accuracy'] = np.nan

        if 'state' in df.columns and 'true_state' in df.columns:
            valid_states = ~(df['state'].isna() | df['true_state'].isna())
            metrics['state_accuracy'] = (df.loc[valid_states, 'state'] == df.loc[valid_states, 'true_state']).mean() if valid_states.any() else np.nan
        else:
            metrics['state_accuracy'] = np.nan

        if 'country' in df.columns and 'true_country' in df.columns:
            valid_countries = ~(df['country'].isna() | df['true_country'].isna())
            metrics['country_accuracy'] = (df.loc[valid_countries, 'country'].str.lower() == df.loc[valid_countries, 'true_country'].str.lower()).mean() if valid_countries.any() else np.nan
        else:
            metrics['country_accuracy'] = np.nan

        # Distance Metrics
        if 'distance_km' in df.columns:
            valid_distances = df['distance_km'].dropna()
            metrics.update({
                'mean_distance_km': valid_distances.mean() if not valid_distances.empty else np.nan,
                'median_distance_km': valid_distances.median() if not valid_distances.empty else np.nan,
                'max_distance_km': valid_distances.max() if not valid_distances.empty else np.nan,
                'min_distance_km': valid_distances.min() if not valid_distances.empty else np.nan
            })
        else:
            metrics.update({
                'mean_distance_km': np.nan,
                'median_distance_km': np.nan,
                'max_distance_km': np.nan,
                'min_distance_km': np.nan
            })

        # Token and Cost Metrics
        if 'usage_info' in df.columns:
            usage_info_df = df['usage_info'].dropna().apply(pd.Series)
            metrics['total_tokens'] = usage_info_df['total_tokens'].sum()
            metrics['avg_tokens'] = usage_info_df['total_tokens'].mean()
            metrics['total_cost_usd'] = usage_info_df['cost_usd'].sum()
            metrics['avg_cost_usd'] = usage_info_df['cost_usd'].mean()
        else:
            metrics.update({
                'total_tokens': np.nan,
                'avg_tokens': np.nan,
                'total_cost_usd': np.nan,
                'avg_cost_usd': np.nan
            })

        performance_results.append(metrics)

    return pd.DataFrame(performance_results)


def standardize_confidence(value):
    """
    Converts a confidence score to a numeric type.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def create_distribution_visualization(all_data, rows, cols, figsize=None, model_order=None):
    """Creates a grid of distance error distribution plots."""
    if figsize is None:
        figsize = (cols * 7, rows * 5)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()
    model_names = _get_model_order(all_data, model_order)

    for i, model_name in enumerate(model_names):
        if i >= len(axes): break
        ax = axes[i]
        df = all_data[model_name]
        
        valid_distances = df['distance_km'].dropna()
        if not valid_distances.empty:
            sns.histplot(valid_distances, bins=30, ax=ax, color='cornflowerblue', kde=True)
            mean_dist = valid_distances.mean()
            ax.axvline(mean_dist, color='salmon', linestyle='--', label=f'Mean: {mean_dist:.2f} km')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No valid distance data', ha='center', va='center')

        ax.set_title(model_name)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Frequency')

    plt.show()



def create_confidence_visualization(all_data, rows, cols, figsize=None, model_order=None):
    """Creates a grid of distance vs. confidence boxplots."""
    if figsize is None:
        figsize = (cols * 7, rows * 5)
        
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()
    model_names = _get_model_order(all_data, model_order)
    standard_confidence_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    for i, model_name in enumerate(model_names):
        if i >= len(axes): break
        ax = axes[i]
        df = all_data[model_name]
        ax.set_title(model_name)

        if 'confidence' in df.columns:
            valid_conf_dist = df[['confidence', 'distance_km']].dropna()
            if not valid_conf_dist.empty:
                confidence_categories = pd.Categorical(valid_conf_dist['confidence'], 
                                                     categories=standard_confidence_levels, 
                                                     ordered=True)
                valid_conf_dist = valid_conf_dist.copy()
                valid_conf_dist['confidence_cat'] = confidence_categories
                
                sns.boxplot(x='confidence_cat', y='distance_km', data=valid_conf_dist, ax=ax,
                            color="lightblue", width=0.5,
                            boxprops=dict(alpha=.7))
                
                sns.stripplot(x='confidence_cat', y='distance_km', data=valid_conf_dist, ax=ax,
                              jitter=0.2, alpha=0.8, color='darkorange')
                
                ax.set_xticklabels([str(int(x)) for x in standard_confidence_levels])
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Distance Error (km)')
            else:
                ax.text(0.5, 0.5, 'No confidence/distance data', ha='center', va='center')
                ax.set_xlim(-0.5, 5.5)
                ax.set_xticks(range(5))
                ax.set_xticklabels(['1', '2', '3', '4', '5'])
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Distance Error (km)')
        else:
            ax.text(0.5, 0.5, 'Confidence column missing', ha='center', va='center')
            ax.set_xlim(-0.5, 4.5)
            ax.set_xticks(range(5))
            ax.set_xticklabels(['1', '2', '3', '4', '5'])
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Distance Error (km)')

    for j in range(len(model_names), len(axes)):
        axes[j].set_visible(False)

    plt.show()


def create_lat_lon_visualization(all_data, rows, cols, accuracy_df, dataset_name, coord_type='latitude', axis_range=None, figsize=None, model_order=None, save_path=None):
    """
    Creates a grid of true vs. predicted coordinate plots with final adjustments.
    """
    true_col = f'true_{coord_type}'
    pred_col = f'predicted_{coord_type}'
    df_pred_col = coord_type
    
    if axis_range and len(axis_range) == 2:
        lims = axis_range
    else:
        all_coords = pd.concat([df[[true_col, df_pred_col]].dropna() for df in all_data.values()])
        if not all_coords.empty:
            min_val = min(all_coords[true_col].min(), all_coords[df_pred_col].min())
            max_val = max(all_coords[true_col].max(), all_coords[df_pred_col].max())
            lims = [min_val, max_val]
        else:
            lims = [-90, 90] if coord_type == 'latitude' else [-180, 180]
    
    if figsize is None:
        figsize = (cols * 5, rows * 5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    model_names = _get_model_order(all_data, model_order)

    for i, model_name in enumerate(model_names):
        if i >= len(axes): break
        ax = axes[i]
        df = all_data[model_name]

        sns.scatterplot(
            data=df.dropna(subset=['confidence', true_col, df_pred_col]), 
            x=df_pred_col,
            y=true_col,
            ax=ax, 
            alpha=0.7,
            hue='confidence',
            palette='coolwarm',
            hue_norm=(1, 5),
            legend=False
        )
        
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        ax.set_title(model_name)
        ax.set_aspect('equal', adjustable='box')
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        model_stats = accuracy_df[accuracy_df['model'] == model_name]
        if not model_stats.empty:
            stats = model_stats.iloc[0]
            valid_points_df = df[[true_col, df_pred_col]].dropna()
            identification_count = len(valid_points_df)
            text_lines = [f"Identifications: {identification_count}"]

            city_acc = f"{stats.get('city_accuracy', 0):.1%}" if pd.notna(stats.get('city_accuracy')) else "N/A"
            text_lines.append(f"City Acc: {city_acc}")

            if dataset_name == 'dataset1' or dataset_name == 'dataset3':
                country_acc = f"{stats.get('country_accuracy', 0):.1%}" if pd.notna(stats.get('country_accuracy')) else "N/A"
                text_lines.append(f"Country Acc: {country_acc}")
            elif dataset_name == 'dataset2':
                state_acc = f"{stats.get('state_accuracy', 0):.1%}" if pd.notna(stats.get('state_accuracy')) else "N/A"
                text_lines.append(f"State Acc: {state_acc}")

            median_err = f"{stats.get('median_distance_km', 0):.1f} km" if pd.notna(stats.get('median_distance_km')) else "N/A"
            text_lines.append(f"Median Err: {median_err}")

            # corr_df = df[[true_col, df_pred_col]].dropna()
            # pearson_corr = corr_df.corr(method='pearson').iloc[0, 1]
            # corr_str = f"{pearson_corr:.2f}" if pd.notna(pearson_corr) else "N/A"
            # text_lines.append(f"Pearson: {corr_str}")

            stats_text = "\n".join(text_lines)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='ivory', alpha=0.8))

    for j in range(len(model_names), len(axes)):
        axes[j].set_visible(False)

    cmap = plt.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=1, vmax=5)
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                              label=f'{i}',
                              markerfacecolor=cmap(norm(i)), 
                              alpha=0.8,
                              markersize=8) for i in [1, 2, 3, 4, 5]]

    legend = fig.legend(handles=legend_elements,
                    loc='lower center',
                    ncol=5,
                    fontsize=14,
                    frameon=True,
                    bbox_to_anchor=(0.5, 0.0))

    # Manually add title to left of legend
    fig.text(0.23, 0.04, 'Confidence Score:', ha='left', va='center', fontsize=14)

    plt.tight_layout(h_pad=2, rect=[0, 0.08, 1, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()


def create_latitude_visualization(all_data, rows, cols, accuracy_df, dataset_name, 
                                  axis_range=None, figsize=None, model_order=None, save_path=None):
    """Helper function to create latitude plots."""
    create_lat_lon_visualization(all_data, rows, cols, accuracy_df, dataset_name=dataset_name,
                                 coord_type='latitude', axis_range=axis_range,
                                 figsize=figsize, model_order=model_order, save_path=save_path)


def create_longitude_visualization(all_data, rows, cols, accuracy_df, dataset_name, 
                                   axis_range=None, figsize=None, model_order=None, save_path=None):
    """Helper function to create longitude plots."""
    create_lat_lon_visualization(all_data, rows, cols, accuracy_df, dataset_name=dataset_name, 
                                 coord_type='longitude', axis_range=axis_range,
                                 figsize=figsize, model_order=model_order, save_path=save_path)