import openai
import json

#generate worldview
def generate_worldview(worldview_data):
    print("Generating worldview...")
    prompt = "Please construct a travel script worldview based on the following table:\n"
    for item in worldview_data:
        prompt += f"| Location       | Features/Culture/History/Legends |\n"
        prompt += f"|----------------|----------------------------------|\n"
        prompt += f"| {item['location']} | {item['features']} |\n"
    prompt += "\nThe worldview should connect various attractions' storylines. Describe basic information about this world.\n"
    prompt += """Example:
    Travel Script Worldview Setting
    Name: Time Journey: Dream Hunting in Jinling
    
    Background:
    At the intersection of modern technology and ancient wisdom exists a secret organization - 'Time Guardians'. 
    This group consists of individuals who can travel through historical periods to protect cultural heritage. 
    They can access a parallel world called 'Historical Realm' that preserves the most glorious cultural legacies 
    and captivating stories from history.
    
    In this realm, Nanjing (known as 'Jinling') is a mysterious place full of historical charm. Each attraction 
    represents a temporal node containing rich history and hidden passages to other eras. These passages only 
    appear during specific historical moments, and the Time Guardians' mission is to guide travelers through 
    these nodes while protecting cultural heritage from temporal erosion.
    
    World Rules:
    1. Historical Nodes: Each attraction becomes active during specific historical moments
    2. Temporal Passages: Secret connections between nodes that only open under certain conditions
    3. Time Guardians: Special guides with temporal travel capabilities
    4. Historical Realm: Parallel world preserving cultural heritage
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "system", "content": "You are a travel script assistant."
        }, {
            "role": "user", "content": prompt
        }],
        max_tokens=1000,
        temperature=0.7
    )
    print("Worldview generated!")
    return response['choices'][0]['message']['content'].strip()

#generate character
def generate_character_based_on_worldview(worldview):
    print("Generating characters...")
    prompt = f"""
    Please create two main characters based on this worldview:
    
    Worldview:
    {worldview}
    
    Character 1: User's Role
    - Name:
    - Identity (e.g., traveler, student, explorer):
    - Personality:
    - Background:
    - Travel Purpose:
    
    Character 2: Guide
    - Name:
    - Identity (e.g., local guide, historian):
    - Personality:
    - Background:
    - Relationship with User:

    Example:
    Character Settings
    
    Protagonist 1: User (Traveler)
    Name: Lin Yi (pseudonym)
    Background: A modern history enthusiast selected by the Time Guardians
    Traits: Curious, observant, with basic historical knowledge
    Goal: Experience historical charm and uncover secrets
    
    Protagonist 2: Guide (Time Guardian)
    Name: Murong Yun
    Background: Senior Time Guardian with deep historical knowledge
    Traits: Knowledgeable, calm, with temporal abilities
    Goal: Guide the traveler and protect historical nodes
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "system", "content": "You are a travel script assistant."
        }, {
            "role": "user", "content": prompt
        }],
        max_tokens=1000,
        temperature=0.7
    )
    print("Characters generated!")
    return response['choices'][0]['message']['content'].strip()

#generate opening script
def generate_opening_script(worldview, characters):
    print("Generating opening script...")
    prompt = f"""
    Travel Script Opening:  
    
    Worldview:  
    {worldview}  
    
    Characters:  
    {characters}
    
    Requirements:  
    - Brief introduction of worldview
    - First meeting between user and guide
    - Journey starting point and purpose
    - Preview of upcoming travels
    
    Make the opening engaging and intriguing.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{
            "role": "system", "content": "You are a travel script assistant."
        }, {
            "role": "user", "content": prompt
        }],
        max_tokens=1000,
        temperature=0.7
    )
    print("Opening script generated!")
    return response['choices'][0]['message']['content'].strip()

# Fetch attraction data from the Neo4j database
def fetch_attraction_data(neo4j_url, neo4j_auth):
    from py2neo import Graph
    graph = Graph(neo4j_url, auth=neo4j_auth)
    
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
    return result

# Generate worldview data based on attraction data
def generate_worldview_data(attractions):
    worldview_data = []
    for attraction in attractions:
        entry = {
            "location": attraction['name'],
            "features": f"History: {attraction.get('history', 'No history available')}, "
                        f"Culture: {attraction.get('culture', 'No cultural features')}, "
                        f"Legends: {attraction.get('legends', 'No legends available')}, "
                        f"Main Attractions: {attraction.get('main_attractions', 'No main attractions')}, "
                        f"Geographic Location: {attraction.get('location', 'No location information')}"
        }
        worldview_data.append(entry)
    return worldview_data

# Save worldview data to a file
def save_worldview_to_file(file_path, worldview):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump({"worldview": worldview}, file, ensure_ascii=False, indent=4)

# Save character settings to a file
def save_characters_to_file(file_path, characters):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump({"character_settings": characters}, file, ensure_ascii=False, indent=4)

# Save opening script to a file
def save_opening_script_to_file(file_path, opening_script):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump({"opening_script": opening_script}, file, ensure_ascii=False, indent=4)
