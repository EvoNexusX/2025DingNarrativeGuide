# Narrative-Driven Travel Planning: Geocultural-Grounded Script Generation with Evolutionary Itinerary Optimization

## Overview

This repository contains the implementation  of the paper "Narrative-Driven Travel Planning: Geocultural-Grounded Script Generation with Evolutionary Itinerary Optimization".

### Key Contributions

- Proposed a narrative-driven travel planning framework called **NarrativeGuide** that generates a geoculturally-grounded narrative script for travelers.
- Developed a knowledge graph for city attractions and used it to configure the worldview, character setting, and exposition.
- Modeled narrative-driven travel planning as an optimization problem and utilized a genetic algorithm (GA) to refine the itinerary.


## Getting Started

### Requirements

- openai==0.28.0
- py2neo==2021.2.4
- pandas==2.2.3
- numpy==1.24.4
- matplotlib==3.5.1

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Narrative-Driven-Travel-Planning.git
   cd Narrative-Driven-Travel-Planning

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Configure the settings:
   Before running the script, you need to configure the API keys and database settings in `config.py.`

4. Run the script:
   ```bash
   python run.py

### Usage
To use this repository, simply run the `run.py` script. This will generate a geoculturally-grounded travel itinerary based on the narrative-driven framework. The final output will be saved as `polished_long_script.txt` in the root directory of the repository.
