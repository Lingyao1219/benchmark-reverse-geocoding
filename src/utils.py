import re
import os
import sys
import json
import config
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt


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
        location_info = record.get('location_info', {}) or {}
        image_info = location_info.get('image_information', {}) or {}
        reasoning = location_info.get('reasoning', {}) or {}
        reverse_geocoding = location_info.get('reverse_geocoding', {}) or {}
        address = reverse_geocoding.get('address', {}) or {}
        coordinates = reverse_geocoding.get('coordinates', {}) or {}
        usage_info = record.get('usage_info', {}) or {}

        # Extract and clean coordinates
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
            'longitude': lon
        }
        processed_records.append(flat_record)

    # Create DataFrame directly from the list of flat dictionaries
    df = pd.DataFrame(processed_records)
    core_cols = [
        'image_file', 'model_used', 'confidence', 'street', 'city', 'state', 
        'country', 'latitude', 'longitude'
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
    
    # Distance analysis
    print("\n3. COORDINATE DISTANCE ANALYSIS:")
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