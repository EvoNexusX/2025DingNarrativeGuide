import openai
import json
from tenacity import retry, stop_after_attempt, wait_fixed
import requests

# Define the chunking function
def split_script_into_chunks(script, chunk_size=3500):
    chunks = []
    for i in range(0, len(script), chunk_size):
        chunk = script[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# Define the polishing function
@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
def polish_chunk(chunk, context):
    try:
        chunk_with_context = context + chunk
        prompt = f"Please polish the following text to improve coherence, engagement, and flow:\n{chunk_with_context}"

        response = openai.ChatCompletion.create(
            model="gpt-4-32k",
            messages=[
                {"role": "system", "content": "You are a professional editor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=6000,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()
    except requests.exceptions.Timeout:
        print("API request timed out. Please check your network or try again later.")
        return chunk  
    except Exception as e:
        print(f"Error polishing chunk: {e}")
        return chunk 

# Define the function for polishing long texts
def polish_long_script_with_context(long_script):
    if len(long_script) <= 3500:
        #print("Input script is short, processing as a single chunk.")
        return polish_chunk(long_script, "")

    chunk_size = 3500
    chunks = split_script_into_chunks(long_script, chunk_size)
    polished_script = ""
    context = ""

    for i, chunk in enumerate(chunks):
        #print(f"Processing chunk {i + 1}/{len(chunks)} (length: {len(chunk)})")
        polished_chunk = polish_chunk(chunk, context)
        polished_script += polished_chunk + "\n"
        context = polished_chunk[-500:] if len(polished_chunk) >= 500 else polished_chunk

        #print(f"Chunk {i + 1}: {chunk[:100]}...")
        #print(f"Polished Chunk {i + 1}: {polished_chunk[:100]}...")

    return polished_script

# Load and combine scripts
def load_and_combine_scripts(opening_script_path, merged_script_path):
    with open(opening_script_path, "r", encoding="utf-8") as json_file:
        opening_script_data = json.load(json_file)
        opening_script = opening_script_data.get("opening_script", "")

    with open(merged_script_path, "r", encoding="utf-8") as file:
        merged_script = file.read()

    return opening_script + "\n" + merged_script
