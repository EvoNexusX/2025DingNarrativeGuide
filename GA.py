import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import copy

# Read travel times between scenic spots
def read_travel_times(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist!")
    df = pd.read_excel(file_path, index_col=0, header=0)
    return df

# Read smoothness scores
def read_smoothness_scores(smoothness_scores_folder):
    smoothness_data = {}

    try:
        # Check if the folder exists and is readable
        if not os.access(smoothness_scores_folder, os.R_OK):
            #print(f"Permission Denied: Cannot read from {smoothness_scores_folder}")
            return
        
        files = os.listdir(smoothness_scores_folder)
        
        for file in files:
            file_path = os.path.join(smoothness_scores_folder, file)
            
            try:
                # Use regular expressions to extract scenic spot and script numbers from the filename
                match = re.match(r"score_(.+?)_script(\d+)_(.+?)_script(\d+)\.txt", file)
                if match:
                    start_spot = match.group(1) 
                    script_id_start = int(match.group(2)) 
                    end_spot = match.group(3) 
                    script_id_end = int(match.group(4)) 
                    
                    # Read the numeric value in the file as the smoothness score
                    with open(file_path, 'r', encoding='utf-8') as f:
                        score = float(f.read().strip())  # Assume the file content is a number (smoothness score), convert to float
                    
                    # Store the smoothness score in the dictionary, key as (start_spot, script_id_start, end_spot, script_id_end)
                    smoothness_data[(start_spot, script_id_start, end_spot, script_id_end)] = score
                else:
                    print(f"Filename format mismatch: {file}")
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        
        return smoothness_data
    
    except PermissionError as e:
        print(f"PermissionError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

# Calculate script length
def get_script_length(script_folder, start_spot, script_id_start):
    script_path = os.path.join(script_folder, str(start_spot), f"script_{str(script_id_start)}.txt")
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            script = f.read()
        return len(script)
    except FileNotFoundError:
        print(f"Error: The script file {script_path} was not found.")
        return 0

# Calculate total travel time, including script time and transport time
def calculate_travel_time(path, travel_time_matrix, script_folder, smoothness_scores):
    total_time = 0
    total_smoothness = 0
    total_script_time = 0
    total_transport_time = 0
    
    # Traverse through the scenic spots in the path
    for i in range(len(path) - 1):
        start_spot, script_id_start = path[i]
        end_spot, script_id_end = path[i+1]
        
        # Calculate smoothness
        smoothness = smoothness_scores.get((start_spot, script_id_start, end_spot, script_id_end), 3)
        total_smoothness += smoothness
        
        # Calculate script time
        script_time_start = get_script_length(script_folder, start_spot, script_id_start) / 400
        script_time_end = get_script_length(script_folder, end_spot, script_id_end) / 400
        total_script_time += script_time_start + script_time_end
        
        # Calculate transport time
        transport_time = travel_time_matrix.loc[start_spot, end_spot]
        total_transport_time += transport_time

    # Total travel time
    total_time = total_script_time + total_transport_time
    
    # Calculate average smoothness
    average_smoothness = total_smoothness / (len(path) - 1)

    return average_smoothness, total_time

# Read ticket price data for scenic spots
def read_prices(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist!")
    df = pd.read_csv('./data/attraction ticket prices.csv', encoding='utf-8')
    #print(f"DataFrame shape: {df.shape}") 
    #print(f"Columns: {df.columns}") 
    #print(df.iloc[:, 1])
    return df.iloc[:, 1]

# Calculate the total price of the path
def calculate_total_price(path, prices):
    total_price = 0
    for spot, _ in path:
        total_price += prices.get(spot, 0)
    return total_price

# Update individual class
class Individual:
    def __init__(self, path=None, travel_time_matrix=None, script_folder=None, smoothness_scores=None, solution=None, target_time=None, budget=None, prices=None):
        self.path = path
        self.travel_time_matrix = travel_time_matrix
        self.script_folder = script_folder
        self.smoothness_scores = smoothness_scores
        self.target_time = target_time
        self.budget = budget
        self.prices = prices
        
        if solution is None:
            self.solution = self.path 
        else:
            self.solution = solution

        self.objective = self.calculate_objective()

    def calculate_objective(self):
        if self.travel_time_matrix is not None and self.script_folder is not None and self.smoothness_scores is not None and self.target_time is not None:
            # Calculate the objective value
            total_smoothness, total_time = calculate_travel_time(self.path, self.travel_time_matrix, self.script_folder, self.smoothness_scores)
            #total_price = calculate_total_price(self.path, self.prices)
            total_price = 0
            w_f = 1
            w_t = 60/60
            objective_1 = -total_smoothness*w_f+total_time*w_t
            objective_2 = 0 
            objective_3 = 0  
            #print(self.path)
            #print({'Objective1': objective_1, 'Objective2': objective_2, 'Objective3': objective_3})
            return {'Objective1': objective_1, 'Objective2': objective_2, 'Objective3': objective_3}
        return {'Objective1': 0, 'Objective2': 0, 'Objective3': 0}

# Update population initialization function
def initialize_population(popnum, min_len, max_len, spot_names, smoothness_scores, travel_time_matrix, script_folder, target_time, budget, prices):
    population = []
    for _ in range(popnum):
        path = generate_random_path(min_len, max_len, spot_names, allow_duplicates=False)
        
        path_with_scripts = [(spot, random.choice([1, 2, 3])) for spot in path]
        
        individual = Individual(path=path_with_scripts, travel_time_matrix=travel_time_matrix, script_folder=script_folder, smoothness_scores=smoothness_scores, target_time=target_time, budget=budget, prices=prices)
        
        population.append(individual)
    return population

def generate_random_path(min_len, max_len, spot_names, allow_duplicates=False):
    """
    Generate a random path
    
    :param min_len: Minimum length of the path
    :param max_len: Maximum length of the path
    :param spot_names: List of scenic spot names
    :param allow_duplicates: Whether to allow duplicate spots
    :return: Random path (list of scenic spot names)
    """
    path_length = random.randint(min_len, max_len)
    if allow_duplicates:
        return random.choices(spot_names, k=path_length)
    else:
        return random.sample(spot_names, k=path_length)

# Elite selection mechanism
def elite_selection(population, popnum):
    population.sort(key=lambda x: (x.objective['Objective1'], x.objective['Objective2']))
    return population[:popnum]

# Crossover operation: exchange part of the paths between two individuals
def crossover(individual1, individual2):
    # Randomly select the crossover point
    crossover_point = random.randint(1, min(len(individual1.path), len(individual2.path)) - 1)

    # Exchange the parts of the paths
    new_path1 = individual1.path[:crossover_point] + individual2.path[crossover_point:]
    new_path2 = individual2.path[:crossover_point] + individual1.path[crossover_point:]

    
    # Generate new individuals
    return Individual(path=new_path1, travel_time_matrix=individual1.travel_time_matrix, script_folder=individual1.script_folder, smoothness_scores=individual1.smoothness_scores, target_time=individual1.target_time), \
           Individual(path=new_path2, travel_time_matrix=individual2.travel_time_matrix, script_folder=individual2.script_folder, smoothness_scores=individual2.smoothness_scores, target_time=individual2.target_time)


def crossover_new(individual1, individual2):
    # Randomly select the crossover point
    crossover_point = random.randint(1, min(len(individual1.path), len(individual2.path)) - 1)

    choice_index = random.sample(range(len(individual1.path)),k=2)
    node1 = min(choice_index)
    node2 = max(choice_index)
    new_path1 = individual1.path
    new_path2 = individual2.path

    for i in range(node1,node2+1):
        if new_path1[i][0] not in [p[0] for p in new_path2]:
            new_path2[i] = copy.deepcopy(new_path1[i])
        if new_path2[i][0] not in [p[0] for p in new_path1]:
            new_path1[i] = copy.deepcopy(new_path2[i])            
    
    # Generate new individuals
    return Individual(path=new_path1, travel_time_matrix=individual1.travel_time_matrix, script_folder=individual1.script_folder, smoothness_scores=individual1.smoothness_scores, target_time=individual1.target_time), \
           Individual(path=new_path2, travel_time_matrix=individual2.travel_time_matrix, script_folder=individual2.script_folder, smoothness_scores=individual2.smoothness_scores, target_time=individual2.target_time)

def bird_cross_CCIP_new(bird, choice, spot_names):
    '''
    Particle crossover
    '''
    choice_index = random.sample(range(4), k=2)
    node1 = min(choice_index)
    node2 = max(choice_index)
    new_bird = copy.deepcopy(bird)
    for i in range(node1, node2 + 1):
        new_bird[i] = choice[i]
    
    # Check for duplicate POI positions
    rep_poi = [0]
    rep_index = []
    for i in range(node1, node2 + 1):
        if new_bird[i] not in rep_poi:
            num = new_bird.count(new_bird[i])
            if num > 1:
                rep_index.append(i)
                rep_poi.append(new_bird[i])
    
    city_pois = []
    if type(1) == int:
        for i in spot_names:
            if i // 100 == spot_names:
                city_pois.append(i)
    else:
        for i in spot_names:
            if True:
                city_pois.append(i)

    cantSelect = [i for i in bird if i != 0]
    canSelect = [i for i in city_pois if i not in cantSelect]
    mut_pois = random.sample(canSelect, k=len(rep_index))
    for index, i in enumerate(rep_index):
        new_bird[i] = mut_pois[index]
    # check_num_poi(new_bird, (len(new_bird) // max_poi_day))
    return new_bird


# Mutation operation: randomly modify a scenic spot in the individual
def mutation(individual, spot_names, smoothness_scores, travel_time_matrix, script_folder, target_time):
    new_path = individual.path.copy()
    
    # Randomly select a position to mutate
    mutation_point = random.randint(0, len(new_path) - 1)
    
    # Randomly choose a new scenic spot
    new_spot = random.choice([p for p in spot_names if p not in [k[0] for k in individual.path]])
    new_path[mutation_point] = (new_spot, random.choice([1, 2, 3]))  # Randomly assign a script

    # Generate a new individual
    return Individual(path=new_path, travel_time_matrix=travel_time_matrix, script_folder=script_folder, smoothness_scores=smoothness_scores, target_time=target_time)

# Update the call in the genetic algorithm
def genetic_algorithm(popnum, generations, min_len, max_len, spot_names, smoothness_scores, travel_time_matrix, script_folder, target_time, budget, prices):
    # Initialize the population
    population = initialize_population(popnum, min_len, max_len, spot_names, smoothness_scores, travel_time_matrix, script_folder, target_time, budget, prices)
    
    best_solution = None
    best_objective = None
    
    for generation in range(generations):
        #print(generation)
        # Elite selection retains the best individuals
        population = elite_selection(population, popnum // 2)
        
        # Crossover operation
        new_population = population.copy()
        while len(new_population) < popnum:
            parent1, parent2 = random.sample(population, 2)
            offspring1, offspring2 = crossover_new(parent1, parent2)
            new_population.extend([offspring1, offspring2])
        
        # Mutation operation
        for individual in new_population:
            if random.random() < 0.3:  # Mutation probability
                mutated_individual = mutation(individual, spot_names, smoothness_scores, travel_time_matrix, script_folder, target_time)
                new_population.append(mutated_individual)
        
        # Update the population
        population = copy.deepcopy(new_population)
        
        # Update the best solution
        '''
        for individual in population:
            if best_solution is None or individual.objective['Objective1'] < best_objective['Objective1'] or \
                    (individual.objective['Objective1'] == best_objective['Objective1'] and individual.objective['Objective2'] < best_objective['Objective2']) or \
                    (individual.objective['Objective1'] == best_objective['Objective1'] and individual.objective['Objective2'] == best_objective['Objective2'] and individual.objective['Objective3'] < best_objective['Objective3']):
                best_solution = individual
                best_objective = individual.objective        
        '''
        min_fitness = 100000
        for ind in population:
            if ind.objective['Objective1'] < min_fitness:
                best_solution = ind
                best_objective = ind.objective
        #print(best_objective)
    
    return best_solution


def merge_scripts_with_transitions(best_solution, script_folder, transition_script_folder, output_file_path):
    """
    Merge selected scripts (including transition scripts) and add labels to the transition parts.
    
    :param best_solution: The best solution, containing scenic spots and scripts
    :param script_folder: Path to the scenic spot script folder
    :param transition_script_folder: Path to the transition script folder
    :param output_file_path: Path to the output file to save the merged complete script
    """
    full_script = ""
    
    # Traverse through the spots in the path, add scripts one by one, and add transition scripts between spots
    for i in range(len(best_solution.path) - 1):
        start_spot, script_id_start = best_solution.path[i]
        end_spot, script_id_end = best_solution.path[i+1]
        
        # Add the script for the start spot
        script_path_start = os.path.join(script_folder, str(start_spot), f"script_{str(script_id_start)}.txt")
        try:
            with open(script_path_start, 'r', encoding='utf-8') as f:
                script_content_start = f.read()
            full_script += f"\n\n--- {start_spot} - Script {script_id_start} ---\n\n"
            full_script += script_content_start
        except FileNotFoundError:
            print(f"Error: The script file {script_path_start} was not found.")
        
        # Add the transition script
        transition_script_filename = f"transition_{start_spot}_{script_id_start}_to_{end_spot}_{script_id_end}"
        transition_script_path = find_transition_script(transition_script_folder, transition_script_filename)
        
        if transition_script_path:
            try:
                with open(transition_script_path, 'r', encoding='utf-8') as f:
                    transition_script_content = f.read()
                full_script += f"\n\n--- Transition from {start_spot} to {end_spot} ---\n\n"
                full_script += transition_script_content
            except FileNotFoundError:
                print(f"Error: The transition script file {transition_script_path} was not found.")
    
    # Add the script for the last spot
    end_spot, script_id_end = best_solution.path[-1]
    script_path_end = os.path.join(script_folder, str(end_spot), f"script_{str(script_id_end)}.txt")
    try:
        with open(script_path_end, 'r', encoding='utf-8') as f:
            script_content_end = f.read()
        full_script += f"\n\n--- {end_spot} - Script {script_id_end} ---\n\n"
        full_script += script_content_end
    except FileNotFoundError:
        print(f"Error: The script file {script_path_end} was not found.")
    
    # Save the full script to a file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(full_script)
        print(f"The complete script has been saved to: {output_file_path}")
    except Exception as e:
        print(f"Error occurred while saving the script: {e}")


def find_transition_script(transition_script_folder, base_filename):
    """
    Find the transition script file, ignoring the version number part.
    
    :param transition_script_folder: Path to the transition script folder
    :param base_filename: The base filename (without version number)
    :return: The full path of the found transition script, or None if not found
    """
    # Define the regular expression to ignore the version number part, matching the last "_v<number>" part
    version_pattern = re.compile(r'(.*)(_v\d+)$')
    
    #print(f"Searching for transition script with base filename: {base_filename}")
    
    # Traverse the transition script folder
    for filename in os.listdir(transition_script_folder):
        #print(f"Checking file: {filename}")
        if filename.startswith(base_filename):
            match = version_pattern.match(filename)
            if match:
                filename_without_version = match.group(1)
                if filename_without_version == base_filename:
                    #print(f"Found transition script: {filename}")
                    return os.path.join(transition_script_folder, filename)
            else:
                if filename.startswith(base_filename):
                    #print(f"Found transition script without version: {filename}")
                    return os.path.join(transition_script_folder, filename)
    
    #print(f"No transition script found for {base_filename}")
    return None
