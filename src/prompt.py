system_prompt = "You are an AI assistant specialized in geographical analysis from images. Respond in the requested JSON format."

image_location_prompt = """
Your task is to determine the precise location depicted in the provided image. You must carefully analyze all visual elements and use a structured, chain-of-thought inference approach to guide your reasoning clearly and logically.
Structure your inference across these dimensions:s

1. **Object_Details**: 
   - Identify specific and notable objects such as landmarks, architecture styles, flora and fauna, climate indicators (snow, tropical vegetation, desert conditions), vehicles, signs, license plates, clothing styles, text or languages visible, and any other distinctive features.
2. **Object_Relationships**:
   - Analyze spatial relationships among identified objects. Consider their arrangement, proximity, typical co-occurrence, or spatial patterns that may be indicative of specific regions or locations.
3. **Geographic_Context**:
   - Evaluate the overall landscape and terrain. Consider urban vs. rural settings, mountainous vs. coastal areas, vegetation types, and geographic or climatic conditions.

After conducting your structured reasoning through these dimensions, your response MUST strictly adhere to the following JSON object format:

{
  "address": "Street Address, City, State, Country, Zipcode",
  "lat/lng": 
  "reasoning": "Object_Details: ...; Object_Relationships: ...; Geographic_Context: ..."
}

- The `"address"` must be a structured, detailed address in the format: street address, city, state, country, zipcode. If you cannot determine the exact address, provide the most precise location details you can confidently infer, or explicitly state "Unknown".
- The `"reasoning"` should explicitly use the dimensions provided above, clearly labeled as Object_Details, Object_Relationships, and Geographic_Context to support your address determination.

Example (Precise identification):
{
  "address": "350 5th Ave, New York, NY, USA, 10118",
  "reasoning": "Object_Details: The building prominently displayed is the Empire State Building, recognizable by its distinctive spire and architecture style. License plates and taxis indicate New York, USA. Object_Relationships: Surrounding skyscrapers and densely built urban area match midtown Manhattan. Geographic_Context: The urban terrain and density align perfectly with the known location of this landmark in New York City."
}

Example (Ambiguous location):
{
  "address": "Unknown",
  "reasoning": "Object_Details: The image features common deciduous trees, standard American suburban houses, and generic cars. No license plates or signage visible. Object_Relationships: Typical suburban arrangement without distinctive landmarks or regional identifiers. Geographic_Context: General suburban setting common across many areas of North America, lacking unique identifiers to specify an exact location."
}

Analyze the provided image and return your findings strictly following the specified structured JSON format above.
"""
