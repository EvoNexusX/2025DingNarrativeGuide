import openai
from py2neo import Graph
import config
from graph_operations import *
from script_generator import *
from creat_SceneryScripts import *
from creat_trasitionscirpts import *
from GA import *
from polish import *

# def main():
"""Main program that calls functions for building the knowledge graph and generating scripts."""
# Get configuration
csv_path = config.CSV_PATH
neo4j_url = config.NEO4J_URI
neo4j_auth = (config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    
# Set OpenAI API configuration
openai.api_base = config.OPENAI_API_BASE
openai.api_key = config.OPENAI_API_KEY
    
# Initialize Neo4j database connection
graph = Graph(neo4j_url, auth=neo4j_auth)

# Build knowledge graph
print("Starting to build the knowledge graph...")
build_knowledge_graph(csv_path, neo4j_url, neo4j_auth)
print("Knowledge graph construction completed!")

# Fetch attraction data
attractions = fetch_attraction_data(neo4j_url, neo4j_auth)
    
# Generate worldview data
worldview_data = generate_worldview_data(attractions)
worldview = generate_worldview(worldview_data)
    
# Generate character settings
characters = generate_character_based_on_worldview(worldview)
    
# Generate opening script
opening_script = generate_opening_script(worldview, characters)
    
# Save generated results
save_worldview_to_file(config.WORLDVIEW_OUTPUT_PATH, worldview)
save_characters_to_file(config.CHARACTERS_OUTPUT_PATH, characters)
save_opening_script_to_file(config.OPENING_SCRIPT_OUTPUT_PATH, opening_script)
    
print("All scripts and data have been saved!")

# Call the encapsulated function to generate and save scripts for all attractions
save_dir = "./SceneryScripts"
for attraction in attractions:
    scripts = generate_script_with_llm(worldview, characters, attraction, 3)
    # print(scripts)
    save_scripts(attraction, scripts, save_dir)
print("All attraction scripts have been generated!")

# Fetch scenic spot script data
scenic_spots = get_scenic_scripts_from_local("./SceneryScripts")

# Fetch scenic spot connection data
connected_scenic_spots = get_scenic_spots_and_connections(graph)

# Fetch survey text
survey_text = get_survey_from_txt("./fluency_survey.txt")

# Generate transition scripts between scenic spots and score them
transitions_df = generate_scenic_pairing_scripts(scenic_spots, connected_scenic_spots, 
                                                 transition_output_path="./transition_scripts", 
                                                 score_output_path="./scoring_results", 
                                                 survey_text=survey_text)

# Save scoring results
save_scores_to_txt(transitions_df, "./scoring_results")


travel_time_matrix = read_travel_times('./data/attraction transit time.xlsx')
smoothness_scores = read_smoothness_scores('./scoring_results')
prices = read_prices('./data/data.csv')
spot_names = list(travel_time_matrix.index)

# Get user input for maximum tour duration and budget
target_time = int(input("Enter the maximum tour duration (in minutes): "))
budget = float(input("Enter the maximum budget: "))
# target_time = 120
# budget = 30
# Set script folder path
script_folder = './SceneryScripts'
transition_script_folder = './transition_scripts'

# Run genetic algorithm
best_solution = genetic_algorithm(
    popnum=100, 
    generations=200, 
    min_len=4, 
    max_len=4, 
    spot_names=spot_names, 
    smoothness_scores=smoothness_scores, 
    travel_time_matrix=travel_time_matrix, 
    script_folder=script_folder,
    target_time=target_time,
    budget=budget,
    prices=prices
)

# Get and output the total duration and total cost
total_smoothness, total_time = calculate_travel_time(best_solution.path, travel_time_matrix, script_folder, smoothness_scores)
print(best_solution.path)
print(total_smoothness)
print(total_time)
output_file_path = './output/merged_script.txt'
merge_scripts_with_transitions(best_solution, script_folder, transition_script_folder, output_file_path)

try:
    combined_script = load_and_combine_scripts("./output/opening_script.json", "./output/merged_script.txt")
    # print(f"Combined Script Length: {len(combined_script)}")
    # print("Combined Script:", combined_script[:500])  # Print the first 500 characters of the combined script

    polished_script = polish_long_script_with_context(combined_script)
    with open("polished_long_script.txt", "w", encoding="utf-8") as output_file:
        output_file.write(polished_script)
    print("The polished long script has been saved.")
except Exception as e:
    print(f"An error occurred: {e}")
