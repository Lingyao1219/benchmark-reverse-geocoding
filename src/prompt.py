SYSTEM_PROMPT = "You are an AI assistant specialized in geographical analysis from images. Respond in the requested JSON format."

IMAGE_LOCATION_PROMPT = """
Analyze the provided image to determine its location using a chain-of-thought approach. 
Follow the chain-of-thought: First analyze the image information, then apply reasoning, finally determine the location.

Your response must be valid JSON in this exact format:

{
  "image_information": {
    "environment": "indoor|outdoor",
    "scene_type": "building|scenery|street|mixed",
    "setting": "urban|suburban|rural|natural"
  },
  "reasoning": {
    "landmark_recognition": "iconic structures, architectural landmarks, natural features - or null if unsure",
    "text_and_signage": "street signs, business names, license plates, visible text - or null if not visible",
    "cultural_indicators": "architectural styles, regional patterns, cultural elements - or null if generic",
    "spatial_context": "geographic relationships, infrastructure patterns - or null if insufficient"
  },
  "reverse_geocoding": {
    "confidence": "high|medium|low",
    "address": {
      "street": "street address or null",
      "city": "city name or null",
      "state": "state/province or null", 
      "country": "country name or null"
    },
    "coordinates": {
      "latitude": "decimal degrees or null",
      "longitude": "decimal degrees or null"
    }
  }
}

Examples:

High Confidence:
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
    "confidence": "high",
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

Low Confidence:
{
  "image_information": {
    "environment": "outdoor",
    "scene_type": "scenery",
    "setting": "suburban"
  },
  "reasoning": {
    "landmark_recognition": null,
    "text_and_signage": null,
    "cultural_indicators": "Standard American suburban architecture and landscaping",
    "spatial_context": "Typical suburban development layout"
  },
  "reverse_geocoding": {
    "confidence": "low",
    "address": {
      "street": null,
      "city": null,
      "state": null,
      "country": "USA"
    },
    "coordinates": {
      "latitude": null,
      "longitude": null
    }
  }
}

Rules:
- Return only valid JSON, no additional text
- Use null for any field you cannot determine with reasonable confidence
- Use null for reasoning dimensions if you cannot identify relevant elements
- Do not fabricate information for reasoning - better to use null than provide uncertain details
- For reverse geocoding, return only one location that you are mostly confident
"""