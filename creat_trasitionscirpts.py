import os
import pandas as pd
import openai
import time
from py2neo import Graph
from random import randint


def retry_api_call(func, *args, **kwargs):
    """
    Retry the OpenAI API call, retrying up to 5 times in case of errors.
    """
    max_retries = 5
    delay = 5

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
        except openai.error.APIError as e:
            print(f"API error encountered. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    print("Max retries reached. Proceeding with the next task.")
    return None


def get_scenic_scripts_from_local(path):
    """
    Read scenic spot scripts from the specified path and store each script by scenic spot name.
    """
    scenic_spots = {}
    
    # Get all scenic spot folders
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        
        if os.path.isdir(folder_path):
            scripts = []
            # Read each script file
            for i in range(1, 4):
                script_file = os.path.join(folder_path, f"script_{i}.txt")
                if os.path.exists(script_file):
                    with open(script_file, 'r', encoding='utf-8') as file:
                        scripts.append(file.read().strip())
            scenic_spots[folder_name] = scripts
    
    return scenic_spots


def get_scenic_spots_and_connections(graph):
    """
    Find all the connections between scenic spots in the knowledge graph.
    """
    query = """
    MATCH (a:Attraction)-[r]->(b:Attraction)
    RETURN DISTINCT a.name AS Attraction1, b.name AS Attraction2, r.type AS RelationType, r.description AS Description
    """
    results = graph.run(query)

    connected_scenic_spots = []
    for record in results:
        start = record['Attraction1']
        end = record['Attraction2']
        relation_type = record['RelationType']
        description = record['Description']
        
        # Store scenic spot connections and relationship descriptions
        connected_scenic_spots.append((start, end, relation_type, description))
    
    return connected_scenic_spots


def generate_transition_script(script1, script2):
    """
    Generate a transition script between two scenic spot scripts using the OpenAI GPT-4 API.
    """
    prompt = f"Please generate a well-crafted transition script that smoothly transitions between the following two scripts, and justifies the reason for moving from the current scenic spot to the next one:\nCurrent Scenic Spot Script: {script1}\nNext Scenic Spot Script: {script2}\nTransition Script:"
    
    # Use retry_api_call to handle potential API errors
    response = retry_api_call(openai.ChatCompletion.create, 
                              model="gpt-4",
                              messages=[ 
                                  {"role": "system", "content": "You are a helpful assistant."},
                                  {"role": "user", "content": prompt}
                              ],
                              max_tokens=4000,
                              temperature=0.7)
    
    if response:
        return response.choices[0].message["content"].strip()
    else:
        return "Transition script generation failed"

def calculate_weighted_score(scores):
    """
    Calculate weighted score, with equal weights for the four dimensions (each 0.25).
    scores is a list of average scores for the 4 dimensions.
    """
    # Weights for each dimension
    weights = [0.25, 0.25, 0.25, 0.25]
    
    # Calculate weighted score
    weighted_score = sum([score * weight for score, weight in zip(scores, weights)])
    
    return weighted_score



def score_with_transition_script(previous_script, transition_script, next_script, survey_text):
    """
    Combine the previous, transition, and next scripts and score them using the OpenAI GPT-4 API.
    If the AI returns an invalid response, retry the scoring process.
    """
    combined_script = f"Previous Script:\n{previous_script}\n\nTransition Script:\n{transition_script}\n\nNext Script:\n{next_script}"
    
    # Scoring prompt, incorporating survey questions
    prompt = f"Here is the combined script:\n{combined_script}\n\nHere are the survey questions:\n{survey_text}\n\nPlease score the script based on the survey questions, providing a score for each question. Don't say anything else. The response should be in the format: 4,5,3,2,3,4,1,2,3,1,3,3"  
    
    max_retries = 3  
    retry_count = 0
    
    while retry_count < max_retries:
        # Use retry_api_call to handle potential API errors
        response = retry_api_call(openai.ChatCompletion.create, 
                                  model="gpt-4",
                                  messages=[ 
                                      {"role": "system", "content": "You are a script evaluation expert."},
                                      {"role": "user", "content": prompt}
                                  ],
                                  max_tokens=500,
                                  temperature=0.7)
        
        if response:
            try:
                scores = [float(score) for score in response['choices'][0]['message']['content'].strip().split(',')]
                
                weighted_score = calculate_weighted_score(scores)
                
                return weighted_score
            except ValueError:
                retry_count += 1
                #print(f"Invalid response format. Retrying... (Attempt {retry_count}/{max_retries})")
                continue
        else:
            return "Scoring failed"
    
    return "Scoring failed after maximum retries"


def generate_scenic_pairing_scripts(scenic_spots, connected_scenic_spots, transition_output_path, score_output_path, survey_text):
    """
    Generate transition scripts for each pair of connected scenic spots and return the results as a DataFrame,
    while saving the transition scripts and scores to files.
    """
    transitions = []
    script_counter = 1  # Initialize transition script number
    if not os.path.exists(transition_output_path):
        os.makedirs(transition_output_path)
    if not os.path.exists(score_output_path):
        os.makedirs(score_output_path)
    
    print("Generating transition scripts...")
    for start, end, _, _ in connected_scenic_spots:
        scripts_start = scenic_spots.get(start, [])
        scripts_end = scenic_spots.get(end, [])
        
        # Generate transitions for each pair of scripts
        for script1_index, script1 in enumerate(scripts_start, start=1):
            for script2_index, script2 in enumerate(scripts_end, start=1):
                print(f"Generating transition script: {start} - {script1_index} and {end} - {script2_index}")  # Print the current transition script being generated
                
                # Generate transition for the current scenic spots
                transition_script = generate_transition_script(script1, script2)
                
                # Generate filename for the transition script
                transition_filename = f"transition_{start}_{script1_index}_to_{end}_{script2_index}_v{script_counter}.txt"
                transition_file_path = os.path.join(transition_output_path, transition_filename)
                
                # Save the transition script content
                with open(transition_file_path, 'w', encoding='utf-8') as f:
                    f.write(transition_script)
                
                # Get the previous and next scripts, if available
                previous_script = scripts_start[script1_index - 1] if script1_index > 1 else ""
                next_script = scripts_end[script2_index - 1] if script2_index < len(scripts_end) else ""
                
                # Score each transition script
                score = score_with_transition_script(previous_script, transition_script, next_script, survey_text)
                
                # Skip if scoring fails
                if score == "Scoring failed":
                    continue
                
                # Save the score result
                score_filename = f"score_{start}_script{script1_index}_{end}_script{script2_index}.txt"
                score_file_path = os.path.join(score_output_path, score_filename)
                with open(score_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"{score}\n")
                
                # Add the transition script and score to the DataFrame
                transitions.append({
                    "start_scenic": start,
                    "end_scenic": end,
                    "start_script_index": script1_index,
                    "end_script_index": script2_index,
                    "transition_filename": transition_filename,
                    "score": score
                })
                
                script_counter += 1

    return pd.DataFrame(transitions)



def get_survey_from_txt(file_path):
    """
    Read the survey text from the specified file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        survey = file.read().strip()
    return survey


def save_scores_to_txt(scoring_results_df, output_path):
    """
    Save the scoring results to a text file. Each scenic pair's score is saved in a separate file.
    Only the weighted score is saved.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print("Saving scoring results...")
    for _, row in scoring_results_df.iterrows():
        # Filename includes the start and end scenic spots' script indexes
        filename = f"score_{row['start_scenic']}_script{row['start_script_index']}_{row['end_scenic']}_script{row['end_script_index']}.txt"
        file_path = os.path.join(output_path, filename)
        
        # Save the weighted score
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{row['score']}\n")
