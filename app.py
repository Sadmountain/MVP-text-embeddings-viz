import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sentence_transformers import SentenceTransformer
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px
import numpy as np

# Function to preprocess the text (lowercasing, removing punctuation, and tokenizing)
def preprocess(text):
    if text is not None:
        text = text.lower().replace('.', '').replace(',', '').replace(':', '')
        tokens = text.split()
        return ' '.join(tokens)
    return ""

# Read the preprocessed dataset from a CSV file
data_path = "data/van_de_Schoot_2018_with_authors.csv"  # Update the path accordingly
papers_df = pd.read_csv(data_path)

# Create a subset with at least all the relevant datapoints
relevant_papers = papers_df[papers_df['label_included'] == 1]
num_additional_papers = 100 - len(relevant_papers)
if num_additional_papers > 0:
    non_relevant_papers = papers_df[papers_df['label_included'] != 1]
    additional_papers = non_relevant_papers.sample(min(len(non_relevant_papers), num_additional_papers), random_state=42)
    subset_df = pd.concat([relevant_papers, additional_papers])
else:
    subset_df = relevant_papers
papers_df = subset_df.reset_index(drop=True)

# Function to add a tooltip with information
def add_tooltip(slider_label):
    return f"""
    <style>
    .tooltip {{
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }}

    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 220px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 100%; /* Position the tooltip above the text */
        left: 50%;
        margin-left: -110px;
        opacity: 0;
        transition: opacity 0.3s;
    }}

    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>

    <div class="tooltip">{slider_label}
      <span class="tooltiptext">
        Adjusting the similarity threshold changes the criteria for connecting papers. 
        - Lower thresholds (e.g., 0.2) connect papers that are highly similar (cosine similarity > 0.8).
        - Higher thresholds (e.g., 0.7) connect papers that are less similar (cosine similarity > 0.3).
      </span>
    </div>
    """

# model options for sentence transformers
model_options = [
    'stsb-roberta-base-v2',
    'all-MiniLM-L6-v2'
]
# Sidebar for model selection
st.sidebar.title("Model Options")
selected_model = st.sidebar.selectbox("Select Embedding Model", model_options)

# Function to load the selected SentenceTransformer model
@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

# Load the selected model
model = load_model(selected_model)

# Function to compute embeddings for a list of texts
@st.cache_data
def compute_embeddings(texts, model):
    return model.encode(texts)

# Compute embeddings for the processed abstracts
abstract_embeddings = model.encode(papers_df['processed_abstract'].tolist())

# Initialize session state for storing ratings and reviewed indices
if 'ratings' not in st.session_state:
    st.session_state['ratings'] = {}
    
if 'index_reviewed' not in st.session_state:
    st.session_state['index_reviewed'] = []
    
if 'selected_node_ratings' not in st.session_state:
    st.session_state['selected_node_ratings'] = {}
    
if 'selected_node_reviewed' not in st.session_state:
    st.session_state['selected_node_reviewed'] = []
    
if 'next_relevant_paper' not in st.session_state:
    st.session_state['next_relevant_paper'] = None

# Function to calculate distances from a specific node to all other nodes --> for the network graph
def calculate_node_distances(selected_node_id, embeddings, distance_metric):
    selected_idx = papers_df.index[papers_df['title'] == selected_node_id].tolist()[0] # Get the index of the selected node
    selected_embedding = embeddings[selected_idx].reshape(1, -1) # Resahpe the embedding of the selected paper into a 2D array with one row and as many cols as needed.
    # Calculate the distance from the selected embedding and all other embeddings using the specified metric
    distances = calculate_distances(embeddings, selected_embedding, distance_metric)
    distances[selected_idx] = np.inf  # Set the distance to itself as infinity
    return distances

# Determine the next most relevant paper to scan

def get_next_paper(question_embedding, abstract_embeddings, distance_metric, reviewed_indices):
    # Check if there are already reviewed papers
    if reviewed_indices:
        rated_indices = [idx for idx, rated in st.session_state['ratings'].items() if rated]
        if rated_indices:
            # Calculate the average embedding of the research question and relevant papers
            relevant_embeddings = np.array(abstract_embeddings)[rated_indices]
            avg_embedding = np.mean(np.vstack([question_embedding, relevant_embeddings]), axis=0).reshape(1, -1)
        else:
            # If no relevant papers, use the question_embedding as the avg_embedding
            avg_embedding = question_embedding.reshape(1, -1)
    else:
        # If no papers have been reviewed, use the question_embedding directly
        avg_embedding = question_embedding.reshape(1, -1)
    
    # Calculate distances between the average embedding and all papers
    distances = calculate_distances(abstract_embeddings, avg_embedding, distance_metric)
    # Find the closest unreviewed paper
    best_next = sorted([(idx, dist) for idx, dist in enumerate(distances) if idx not in reviewed_indices], key=lambda x: x[1])
    return best_next[0][0] if best_next else None

    
# Determine the next most relevant paper to scan for the selected node
def get_next_paper_selected_node(selected_node_embedding, abstract_embeddings, distance_metric, reviewed_indices, selected_node_index):
    # Check if there are already reviewed papers
    if reviewed_indices:
        rated_indices = [idx for idx, rated in st.session_state['selected_node_ratings'].items() if rated]
        if rated_indices:
            # Calculate the average embedding of the selected node and relevant papers
            relevant_embeddings = np.array(abstract_embeddings)[rated_indices]
            avg_embedding = np.mean(np.vstack([selected_node_embedding, relevant_embeddings]), axis=0).reshape(1, -1)
        else:
            # If no relevant papers, use the selected_node_embedding as the avg_embedding
            avg_embedding = selected_node_embedding.reshape(1, -1)
    else:
        # If no papers have been reviewed, use the selected_node_embedding directly
        avg_embedding = selected_node_embedding.reshape(1, -1)
    
    # Calculate distances between the average embedding and all papers
    distances = calculate_distances(abstract_embeddings, avg_embedding, distance_metric)
    distances[selected_node_index] = np.inf  # Ensure selected node is not considered
    
    # Find the closest unreviewed paper
    best_next = sorted([(idx, dist) for idx, dist in enumerate(distances) if idx not in reviewed_indices], key=lambda x: x[1])
    return best_next[0][0] if best_next else None



# Sidebar for clustering options
st.sidebar.title("Clustering Options")
clustering_method = st.sidebar.selectbox("Select Clustering Method", ["KMeans", "Hierarchical", "DBSCAN"])
if clustering_method == "KMeans":
    num_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5, step=1)
elif clustering_method == "Hierarchical":
    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    distance_threshold = st.sidebar.slider("Distance Threshold", 0.1, 2.0, 0.5)
elif clustering_method == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (DBSCAN)", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("Min samples (DBSCAN)", 1, 10, 5)

# Sidebar for distance metric options
st.sidebar.title("Distance Metric Options")
distance_metric = st.sidebar.selectbox("Select Distance Metric", ["Cosine", "Manhattan", "Euclidean"])

# Sidebar for network creation method
network_method = st.sidebar.selectbox("Select Network Creation Method", ["Distances and Threshold", "Author Relations"])

# Processing research question as the other papers
research_question = st.text_input("Enter your research question", "")
processed_question = preprocess(research_question)
question_embedding = model.encode([processed_question])

# Function to calculate distances between embeddings and the research question embedding (+ other already screened relevant papers)
@st.cache_data
def calculate_distances(embeddings, question_embedding, metric):
    if metric == "Euclidean":
        distances = euclidean_distances(embeddings, question_embedding).flatten()
    elif metric == "Manhattan":
        distances = manhattan_distances(embeddings, question_embedding).flatten()
    elif metric == "Cosine":
        distances = cosine_similarity(embeddings, question_embedding).flatten() # flatten --> conversts 2d array of distances into a 1d array
        distances = 1 - distances  # Convert similarity to distance
    return distances

if research_question:
    # Calculate distances based on selected metric
    distances = calculate_distances(np.array(abstract_embeddings), question_embedding, distance_metric)
    
    # Calculate the pairwise distance matrix for clustering
    distance_matrix = euclidean_distances(abstract_embeddings) if distance_metric == "Euclidean" else \
                      manhattan_distances(abstract_embeddings) if distance_metric == "Manhattan" else \
                      1 - cosine_similarity(abstract_embeddings)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(abstract_embeddings)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(reduced_features, columns=['PCA1', 'PCA2'])
    pca_df['Title'] = papers_df['title']
    pca_df['Authors'] = papers_df['authors']  # Add authors to the DataFrame
    pca_df['label_included'] = papers_df['label_included']
    pca_df['Is Relevant'] = [st.session_state['ratings'].get(i, False) for i in range(len(papers_df))]
    pca_df['Is Relevant from Selected Node'] = [st.session_state['selected_node_ratings'].get(i, False) for i in range(len(papers_df))]
    pca_df['Type'] = ['Paper'] * len(papers_df)
    
    # Append the research question embedding reduced features
    rq_reduced = pca.transform(question_embedding)
    rq_df = pd.DataFrame(rq_reduced, columns=['PCA1', 'PCA2'])
    rq_df['Title'] = ['Research Question']
    rq_df['Type'] = ['Research Question']
    
    pca_df = pd.concat([pca_df, rq_df], ignore_index=True)

    # Apply selected clustering techniques
    if clustering_method == "KMeans":
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(reduced_features)
        clusters = np.append(clusters, -1)  # Append cluster label for research question
    elif clustering_method == "Hierarchical":
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage=linkage_method)
        clusters = clustering.fit_predict(reduced_features)
        clusters = np.append(clusters, -1)  # Append cluster label for research question
    elif clustering_method == "DBSCAN":
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = clustering.fit_predict(reduced_features)
        clusters = np.append(clusters, -1)  # Append cluster label for research question

    pca_df['Cluster'] = clusters

    # Determine the next relevant paper
    next_paper_index = get_next_paper(question_embedding, abstract_embeddings, distance_metric, st.session_state['index_reviewed'])
    while next_paper_index is not None and next_paper_index in st.session_state['index_reviewed']:
        next_paper_index = get_next_paper(question_embedding, abstract_embeddings, distance_metric, st.session_state['index_reviewed'])
    if next_paper_index is not None:
        next_paper_title = papers_df.iloc[next_paper_index]['title']
    else:
        next_paper_title = None
        
    st.markdown("""
**Legend:**
- **Red Border:** Papers marked as relevant by the user.
- **Green Border:** Papers marked as relevant to the selected node.
- **Purple Border:** The next paper to review.
""")

    # Visualize with Plotly first plot with clusters
    fig = px.scatter(pca_df, x='PCA1', y='PCA2',
                     color='Cluster',
                     hover_name='Title',
                     title='2D PCA of research paper abstracts and review question',
                     template="simple_white",
                     symbol='Type',
                     hover_data={'label_included': True},
                     symbol_map={'Paper': 'circle', 'Research Question': 'diamond'},
                     )

    # Update traces to only add red, green, and purple borders to papers marked as relevant or next paper
    def get_border_color(i):
        if i == next_paper_index:
            return 'purple'
        elif st.session_state['ratings'].get(i, False):
            return 'red'
        elif st.session_state['selected_node_ratings'].get(i, False):
            return 'green'
        else:
            return '#FFFFFF'

    # Set a constant border width
    constant_border_width = 2

    fig.update_traces(marker=dict(size=12,  # Increase the size of the symbols
                                line=dict(width=constant_border_width,
                                            color=[get_border_color(i) for i in range(len(papers_df))])),
                    selector=dict(mode='markers'))


    fig.update_layout(
        legend_title_text='Relevant Papers',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='LightSteelBlue',
            bordercolor='Black',
            borderwidth=2
        )
    )

    st.plotly_chart(fig)

## Network graph for paper review
st.header("Review Papers and Find the Most Relevant Paper")

if research_question:
    # Find the next paper to review
    next_paper_index = get_next_paper(question_embedding, abstract_embeddings, distance_metric, st.session_state['index_reviewed'])
    # make sure the selected paper has not already been reviewed
    while next_paper_index is not None and next_paper_index in st.session_state['index_reviewed']:
        next_paper_index = get_next_paper(question_embedding, abstract_embeddings, distance_metric, st.session_state['index_reviewed'])

    # Check if there are more papers to review
    if next_paper_index is not None:
        # Dsiplay meta data papers
        paper_title = papers_df.iloc[next_paper_index]['title']
        paper_abstract = papers_df.iloc[next_paper_index]['abstract']
        
        st.write(f"Title: {paper_title}")
        st.write(f"Abstract: {paper_abstract}")
        relevant = st.radio("Is this paper relevant to your research?", ("Yes", "No"))

        if st.button('Submit Rating'):
            st.session_state['ratings'][next_paper_index] = (relevant == "Yes")
            st.session_state['index_reviewed'].append(next_paper_index)
            papers_df.at[next_paper_index, 'Is Relevant'] = True
            st.experimental_rerun()
    else:
        st.write("No more papers to rate or adjust your settings to refine recommendations.")
else:
    st.write("Please enter a research question to start reviewing papers.")

# Selected Node Screening
st.header("Select a Node and Find the Most Relevant Paper")
selected_node_id = st.selectbox("Select a node by ID", papers_df['title'].tolist())

# Function to create a network graph based on distances and threshold
def create_graph_distances(papers_df, distance_metric, question_embedding, threshold, selected_node_id):
    # Initialize an empty graph
    G = nx.Graph()
    # Determine number of papers in the dataframe
    num_papers = len(papers_df)
    # Calculate distances between abstract embeddings and the research question embedding
    question_distances = calculate_distances(np.array(abstract_embeddings), question_embedding, distance_metric)

    selected_node_distances = None
    if selected_node_id:
        selected_node_distances = calculate_node_distances(selected_node_id, abstract_embeddings, distance_metric)

    # Add paper nodes
    
    for i in range(num_papers):
        # Retrieve wheter a paper is included and wheter it is mareked as "isrelevant" or "is_relevant_from_selected_node"
        included = int(papers_df.iloc[i].get('label_included', 0))  # Default to 0 if column or value is missing
        is_relevant = st.session_state['ratings'].get(i, False)
        is_relevant_from_selected_node = st.session_state['selected_node_ratings'].get(i, False)
        # add each paper as node in the graph with these attributes
        G.add_node(str(papers_df.iloc[i]['title']), type='paper', label_included=included, is_relevant=is_relevant, is_relevant_from_selected_node=is_relevant_from_selected_node)

    # Add edges between papers based on threshold
    for i in range(num_papers):
        # Calculate pairwise distances between papers
        for j in range(i + 1, num_papers):
            # Calculate distance based on selected metric
            distance = float(calculate_distances(np.array([abstract_embeddings[i]]), np.array([abstract_embeddings[j]]), distance_metric)[0])
            similarity = 1 - distance  # Calculate similarity
            if distance < threshold:
                G.add_edge(str(papers_df.iloc[i]['title']), str(papers_df.iloc[j]['title']), weight=distance, title=f'Similarity: {similarity:.4f}')

    # Add research question node
    G.add_node("Research Question", type='question', label_included=0, is_relevant=False)

    # Connect research question to papers based on similarity threshold
    for i in range(num_papers):
        distance = float(question_distances[i])
        similarity = 1 - distance  # Calculate similarity
        if distance < threshold:
            G.add_edge("Research Question", str(papers_df.iloc[i]['title']), weight=distance, title=f'Similarity: {similarity:.4f}')

    # Add the slected node and its connections (if any)
    if selected_node_id:
        G.add_node(selected_node_id, type='selected_node')
        for i in range(num_papers):
            distance = float(selected_node_distances[i])
            similarity = 1 - distance  # Calculate similarity
            if distance < threshold:
                G.add_edge(selected_node_id, str(papers_df.iloc[i]['title']), weight=distance, title=f'Similarity: {similarity:.4f}')

    return G

# Function to create a network graph based on author relations
def create_graph_authors(papers_df, selected_node_id=None):
    G = nx.Graph()
    num_papers = len(papers_df)
    
    # Add paper nodes
    for i in range(num_papers):
        included = int(papers_df.iloc[i].get('label_included', 0))  # Default to 0 if column or value is missing
        is_relevant = st.session_state['ratings'].get(i, False)
        is_relevant_from_selected_node = st.session_state['selected_node_ratings'].get(i, False)  # Check if paper is relevant from selected node
        
        G.add_node(str(papers_df.iloc[i]['title']), type='paper', label_included=included, is_relevant=is_relevant, is_relevant_from_selected_node=is_relevant_from_selected_node)
    
    # Add edges based on common authors
    for i in range(num_papers):
        authors_i = papers_df.iloc[i]['authors']
        # If authors are listed as a string, split them into set of individual authors
        if isinstance(authors_i, str):
            authors_i = set(authors_i.split(', '))
        else:
            authors_i = set()
        for j in range(i + 1, num_papers):
            authors_j = papers_df.iloc[j]['authors']
            if isinstance(authors_j, str):
                authors_j = set(authors_j.split(', '))
            else:
                authors_j = set()
            common_authors = authors_i.intersection(authors_j)
            if common_authors:
                common_authors_str = ', '.join(common_authors)
                G.add_edge(str(papers_df.iloc[i]['title']), str(papers_df.iloc[j]['title']), title=f'Common Authors: {common_authors_str}')
                
    # Highlight the selected node
    if selected_node_id:
        G.add_node(selected_node_id, type='selected_node')
    
    return G

# Function to plot the network graph using Pyvis
def plot_network_graph_pyvis(G, next_paper_title=None):
    net = Network(height='750px', width='100%', notebook=True)
    net.from_nx(G)
    for node in net.nodes:
        if G.nodes[node['id']]['type'] == 'question':
            node['color'] = 'red'
        elif G.nodes[node['id']]['type'] == 'selected_node':
            node['color'] = 'green'
        elif G.nodes[node['id']]['label_included'] == 1:
            node['color'] = 'blue'
            if G.nodes[node['id']]['is_relevant']:
                node['borderWidth'] = 2
                node['borderWidthSelected'] = 2
                node['color'] = {
                    'border': 'red',
                }
        if G.nodes[node['id']]['type'] == 'paper' and G.nodes[node['id']]['is_relevant_from_selected_node']:
            node['borderWidth'] = 2
            node['borderWidthSelected'] = 2
            node['color'] = {
                'border': 'green'
            }
        if next_paper_title and node['id'] == next_paper_title:
            node['borderWidth'] = 3
            node['borderWidthSelected'] = 3
            node['color'] = {
                'border': 'purple'
            }
        node['title'] = f"{node['id']}"  # Make the title visible
        node['font'] = {"color": "rgba(0,0,0,0)"}  # Make title text transparent by default

    for edge in net.edges:
        edge['title'] = edge['title']  # Ensure edge title is displayed for distance and similarity

    net.show_buttons(filter_=['physics'])
    net_html = net.generate_html()
    return net_html

# Display legend for the network graph
st.markdown("""
**Legend:**
- **Red Border:** Papers marked as relevant by the user.
- **Green Border:** Papers marked as relevant to the selected node.
- **Purple Border:** The next paper to review.
""")

# Plot the network graph
st.markdown(add_tooltip("Similarity Threshold"), unsafe_allow_html=True)
threshold = st.slider('Similarity Threshold', min_value=0.0, max_value=1.0, value=0.33, step=0.01)
# Determine the network creation method
if network_method == "Distances and Threshold":
    G = create_graph_distances(papers_df, distance_metric, question_embedding, threshold, selected_node_id)
else:
    G = create_graph_authors(papers_df, selected_node_id)

next_paper_title = None
if research_question:
    next_paper_index = get_next_paper(question_embedding, abstract_embeddings, distance_metric, st.session_state['index_reviewed'])
    while next_paper_index is not None and next_paper_index in st.session_state['index_reviewed']:
        next_paper_index = get_next_paper(question_embedding, abstract_embeddings, distance_metric, st.session_state['index_reviewed'])
    if next_paper_index is not None:
        next_paper_title = papers_df.iloc[next_paper_index]['title']

net_html = plot_network_graph_pyvis(G, next_paper_title)
components.html(net_html, height=800)

if selected_node_id:
    selected_node_index = papers_df.index[papers_df['title'] == selected_node_id].tolist()[0]
    selected_node_embedding = abstract_embeddings[selected_node_index]
    next_paper_index = get_next_paper_selected_node(selected_node_embedding, abstract_embeddings, distance_metric, st.session_state['selected_node_reviewed'], selected_node_index)
    
    # If there are reviewed papers, use the average embedding
    

    while next_paper_index is not None and next_paper_index in st.session_state['selected_node_reviewed']:
        next_paper_index = get_next_paper_selected_node(selected_node_embedding, abstract_embeddings, distance_metric, st.session_state['selected_node_reviewed'], selected_node_index)

    if next_paper_index is not None:
        closest_node_title = papers_df.iloc[next_paper_index]['title']
        closest_node_abstract = papers_df.iloc[next_paper_index]['abstract']
        
        # Display the most relevant paper message based on screened relevant papers
        if st.session_state['selected_node_reviewed']:
            st.write(f"The most relevant paper to the average of '{selected_node_id}' and other screened relevant papers is '{closest_node_title}'")
        else:
            st.write(f"The most relevant paper to '{selected_node_id}' is '{closest_node_title}'")
        
        st.write(f"Abstract: {closest_node_abstract}")

        relevant = st.radio("Is this paper relevant to your selected node?", ("Yes", "No"))

        if st.button('Submit Selected Node Rating'):
            st.session_state['selected_node_ratings'][next_paper_index] = (relevant == "Yes")
            st.session_state['selected_node_reviewed'].append(next_paper_index)
            papers_df.at[next_paper_index, 'Is Relevant from Selected Node'] = True
            st.rerun()  

