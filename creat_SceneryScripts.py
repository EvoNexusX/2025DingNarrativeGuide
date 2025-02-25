import openai
import json
import os
import time
from py2neo import Graph

# Fetch all attractions data from the knowledge graph
def fetch_all_attractions_data():
    query = """
    MATCH (a:Attraction)
    RETURN a.name AS name, 
           a.history AS history, 
           a.culture AS culture, 
           a.story AS legends, 
           a.main_attractions AS main_attractions,
           a.location AS location
    """
    result = graph.run(query).data()
    #print("Knowledge graph extraction completed!")
    return result

# Load worldview and character settings from files
def load_worldview_and_characters(worldview_path, characters_path):
    with open(worldview_path, 'r', encoding='utf-8') as f:
        worldview = json.load(f)
    with open(characters_path, 'r', encoding='utf-8') as f:
        characters = json.load(f)
    return worldview, characters

# Generate script
def generate_script_with_llm(worldview, characters, attraction, n, max_retries=5, retry_delay=5):
    print(f"Generating script for attraction '{attraction['name']}'...")
    prompt = f"""
    1. Background and Worldview Setting:
    You will generate a historical tourism script that integrates modern technology with ancient wisdom. The worldview setting is as follows:
    {worldview}
    2. Character Setting:
    {characters}
    3. Attraction Setting:
    The script is set at {attraction['name']} in Nanjing, combining historical background: {attraction['history']} and cultural characteristics: {attraction['culture']} with historical stories: {attraction['legends']} to generate related story content.
    4. Script Requirements:
    Based on the above background and character settings, generate a **complete script**. Ensure that each part is clearly marked as “Intro:”, “Development:”, “Climax:”, “Ending:”. The content should feature vivid historical scenes, with attention to character interactions and emotional development, ensuring historical details and cultural background are accurate, while meeting the following points:
    - **Lively and fun style**, with a sense of adventure and exploration.
    - **The climax should include specific actions**, not just simple dialogue. The characters should have clear goals and actions, such as stopping a historical event from being changed or protecting cultural heritage.
    - The climax should include detailed tension-filled conflicts or task completion to enhance the story's intensity and immersion.
    - The ending need not elevate or connect to the next attraction but should complete the task independently.
    5. Special Requirements:
    - Emphasize **time-traveling feeling**: Each historical node should be a vivid historical scene, where the tourists (protagonists) feel the transition in time and space.
    - Incorporate the **historical stories** and **cultural features** of the attraction into the storyline.
    - **Natural and interesting dialogue**, with genuine character interactions that align with their personalities.
    - The protagonists already know each other before this script, no need to introduce their identities or backgrounds.
    - **Ensure the script is clearly divided into the following sections:**
        - Intro:
        - Development:
        - Climax:
        - Ending:
    """
    
    scripts = []
    attempt = 0
    
    while attempt < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "system", "content": "You are a travel script writer."
                }, {
                    "role": "user", "content": prompt
                }],
                max_tokens=4000,
                temperature=0.8,
                n=n
            )
            
            # Iterate through all generated choices
            for choice in response['choices']:
                script_content = choice['message']['content'].strip()
                
                # Parse the script structure (with error handling)
                try:
                    parts = {
                        'intro': script_content.split('Intro:')[1].split('Development:')[0].strip(),
                        'development': script_content.split('Development:')[1].split('Climax:')[0].strip(),
                        'climax': script_content.split('Climax:')[1].split('Ending:')[0].strip(),
                        'ending': script_content.split('Ending:')[1].strip(),
                    }
                    scripts.append(parts)
                except IndexError as e:
                    print(f"Failed to parse script structure: {e}. Retrying...")
                    raise ValueError("Invalid script format")
                
            print(f"Successfully generated {n} scripts for attraction '{attraction['name']}'.")
            return scripts
        
        except Exception as e:
            attempt += 1
            print(f"Script generation failed (attempt {attempt}/{max_retries}): {e}")
            time.sleep(retry_delay)
    
    print("Reached the maximum retry count, unable to generate the script.")
    return scripts



# Save script
def save_scripts(attraction, scripts, save_dir):
    scenery_dir = os.path.join(save_dir, attraction['name'])
    os.makedirs(scenery_dir, exist_ok=True)
    
    for i, script in enumerate(scripts):
        script_filename = os.path.join(scenery_dir, f"script_{i+1}.txt")
        with open(script_filename, 'w', encoding='utf-8') as f:
            f.write(f"Intro: {script['intro']}\n")
            f.write(f"Development: {script['development']}\n")
            f.write(f"Climax: {script['climax']}\n")
            f.write(f"Ending: {script['ending']}\n")

# Main function
def generate_and_save_all_scenery_scripts(worldview_path, characters_path, save_dir, n):
    # Load worldview and character settings
    worldview, characters = load_worldview_and_characters(worldview_path, characters_path)
    
    # Fetch all attractions data
    attractions = fetch_all_attractions_data()
    
    for attraction in attractions:
        # Generate scripts for each attraction
        scripts = generate_script_with_llm(worldview, characters, attraction, n)
        
        # Save the generated scripts
        save_scripts(attraction, scripts, save_dir)

