import pandas as pd
from py2neo import Graph, Node, Relationship

def build_knowledge_graph(csv_path, neo4j_url, neo4j_auth):
    # Read the CSV file
    data = pd.read_csv(csv_path, encoding='utf-8')
    
    # Initialize Neo4j database connection
    g = Graph(neo4j_url, auth=neo4j_auth)

    # Iterate through each row in the DataFrame
    for index, row in data.iterrows():
        # Get the attraction name
        attraction_name = row['Attraction Name']
        
        # Create an attraction node and add all attributes as node properties
        attraction_node = Node("Attraction", name=attraction_name,
                               history=row['Historical Background'],
                               culture=row['Cultural Significance'],
                               story=row['Historical Stories'],
                               main_attractions=row['Main Attractions'],
                               location=row['Geographical Location'])
        
        # Merge the attraction node into the graph
        g.merge(attraction_node, "Attraction", "name")
        
        # Iterate through other columns (historical background, cultural significance, etc.) and add them as node properties
        for column, value in row.items():
            if column != 'Attraction Name':
                # Create attribute nodes
                attribute_node = Node(column, name=column, value=value)
                g.merge(attribute_node, column, "value")
                
                # Create relationships: Attraction -> Attribute
                relation = Relationship(attraction_node, column, attribute_node)
                g.create(relation)

    # Add associations between attractions: historical, cultural, and geographical connections
    associations = [
        # Historical associations
        ("Charles Bridge", "Prague Castle", "Historical Association", "Charles Bridge was commissioned by Emperor Charles IV, who also initiated the construction of St. Vitus Cathedral within Prague Castle, both being significant landmarks in Prague's medieval history."),
        ("Prague Castle", "Old Town Square", "Historical Association", "Prague Castle and Old Town Square have both been central to Prague's history, with the castle serving as the seat of power and the square as a key location for public events and executions."),
        ("St. Vitus Cathedral", "Prague Castle", "Historical Association", "St. Vitus Cathedral is located within the Prague Castle complex and has been the site of numerous royal coronations, reflecting the intertwined history of the cathedral and the castle."),
        ("Berlin Cathedral", "Reichstag Building", "Historical Association", "Both the Berlin Cathedral and the Reichstag Building have witnessed significant events in German history, from the era of the German Empire to the reunification of Germany, symbolizing the nation's political and religious heritage."),
        ("Brandenburg Gate", "Reichstag Building", "Historical Association", "The Brandenburg Gate and the Reichstag Building both stand as symbols of German history, with the gate representing peace and unity, and the Reichstag serving as the seat of German democracy."),
        
        # Cultural associations
        ("Charles Bridge", "St. Vitus Cathedral", "Cultural Association", "Charles Bridge and St. Vitus Cathedral are both adorned with Baroque statues and represent Prague's rich cultural heritage, with the bridge connecting the city's historical districts and the cathedral as a Gothic masterpiece."),
        ("Prague Castle", "Golden Lane", "Cultural Association", "Prague Castle and the Golden Lane both offer insights into Prague's medieval urban life and craftsmanship, with the castle representing royal power and the lane showcasing the living conditions of artisans and workers."),
        ("Old Town Square", "Prague Astronomical Clock", "Cultural Association", "Old Town Square is home to the Prague Astronomical Clock, one of the oldest and most intricate astronomical clocks in the world, reflecting Prague's cultural and scientific achievements."),
        ("Berlin Cathedral", "Museum Island", "Cultural Association", "Berlin Cathedral and Museum Island both house significant cultural and historical collections, with the cathedral representing religious heritage and the island showcasing European art and archaeology."),
        ("Brandenburg Gate", "East Side Gallery", "Cultural Association", "The Brandenburg Gate and the East Side Gallery both symbolize freedom and unity, with the gate representing historical peace and the gallery showcasing artistic expressions of post-Cold War reconciliation."),
    ]

    # Iterate through the association data and create relationships in Neo4j
    for attraction1, attraction2, relation_type, description in associations:
        attraction_node1 = g.nodes.match("Attraction", name=attraction1).first()
        attraction_node2 = g.nodes.match("Attraction", name=attraction2).first()
        
        if attraction_node1 and attraction_node2:
            relation = Relationship(attraction_node1, relation_type, attraction_node2, description=description)
            g.create(relation)

    #print("The association between the attractions has been successfully added to the knowledge graph!")
