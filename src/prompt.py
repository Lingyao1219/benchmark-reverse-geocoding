SYSTEM_PROMPT = "You are an AI assistant specialized in geocoding analysis from images."

IMAGE_LOCATION_PROMPT = """
You are given an image. Use a step-by-step (chain-of-thought) reasoning process to infer the most likely location.

Structure your response in three sections:

1. **Image Information** — Describe the general setting of the image.
2. **Reasoning** — Provide detailed justifications using observable visual cues.
3. **Reverse Geocoding** — Output your best-guess location with a confidence score.

Your response must be valid JSON in this exact format:

{
  "image_information": {
    "environment": "indoor|outdoor",
    "scene_type": "building|scenery|street|mixed",
    "setting": "urban|suburban|rural|natural"
  },
  "reasoning": {
    "landmark_recognition": "e.g., iconic structures, architectural landmarks, natural featurese",
    "text_and_signage": "e.g., street signs, business names, license plates, visible text",
    "cultural_indicators": "e.g., architectural styles, regional patterns, cultural elements",
    "spatial_context": "e.g., geographic relationships, infrastructure patterns"
  },
  "reverse_geocoding": {
    "confidence": "1|2|3|4|5",
    "address": {
      "street": "street address",
      "city": "city name",
      "state": "state/province", 
      "country": "country name"
    },
    "coordinates": {
      "latitude": "decimal degrees",
      "longitude": "decimal degrees"
    }
  }
}

Examples:

{
  "image_information": {
    "environment": "outdoor",
    "scene_type": "building",
    "setting": "urban"
  },
  "reasoning": {
    "landmark_recognition": "Empire State Building with distinctive Art Deco spire clearly visible",
    "text_and_signage": "NYC taxi markings and New York license plates",
    "cultural_indicators": "Dense Manhattan urban layout with characteristic skyscraper arrangement",
    "spatial_context": "Midtown Manhattan street grid and building density patterns"
  },
  "reverse_geocoding": {
    "confidence": "5",
    "address": {
      "street": "350 5th Ave",
      "city": "New York",
      "state": "NY",
      "country": "USA"
    },
    "coordinates": {
      "latitude": "40.7484",
      "longitude": "-73.9857"
    }
  }
}

Guidance:
- Return only valid JSON, no additional text
- Retrun a confidence score between 1 (low) and 5 (high)
- For reverse geocoding, return only one location that you are mostly confident
"""