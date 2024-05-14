import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sentence_transformers import SentenceTransformer
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Read the dataset from a CSV file
data_path = "data/van_de_Schoot_2018.csv"  # Update the path accordingly
papers_df = pd.read_csv(data_path)

# Sample with at least all the relevant datapoints
relevant_papers = papers_df[papers_df['label_included'] == 1]
num_additional_papers = 100 - len(relevant_papers)
if num_additional_papers > 0:
    non_relevant_papers = papers_df[papers_df['label_included'] != 1]
    additional_papers = non_relevant_papers.sample(min(len(non_relevant_papers), num_additional_papers), random_state=42)
    subset_df = pd.concat([relevant_papers, additional_papers])
else:
    subset_df = relevant_papers
papers_df = subset_df.reset_index(drop=True)

# Drop rows with NaN values in the abstract column
papers_df = papers_df.dropna(subset=['abstract'])

# Function to preprocess the text (lowercasing, removing punctuation, and tokenizing)
def preprocess(text):
    if text is not None:
        text = text.lower().replace('.', '').replace(',', '').replace(':', '')
        tokens = text.split()
        return ' '.join(tokens)
    return ""

#Apply the preprocessing function to the abstract column
papers_df['processed_abstract'] = papers_df['abstract'].apply(preprocess)

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



# Available models for selection
model_options = [
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2',
    'para-distilroberta-base-v1',
    'paraphrase-MiniLM-L6-v2',
    'stsb-roberta-base-v2',
]
# Sidebar for model selection
st.sidebar.title("Model Options")
selected_model = st.sidebar.selectbox("Select Embedding Model", model_options)

# Function to load the selected SentenceTransformer model
def load_model(model_name):
    return SentenceTransformer(model_name)

# Load the selected model
model = load_model(selected_model)

# Function to compute embeddings for a list of texts
@st.cache_data #used to cach data computations
def compute_embeddings(texts, model):
    return model.encode(texts)

# Compute embeddings for the processed abstracts
abstract_embeddings = model.encode(papers_df['processed_abstract'].tolist())

# Initialize session state for storing ratings and reviewed indices
if 'ratings' not in st.session_state:
    st.session_state['ratings'] = {}
if 'index_reviewed' not in st.session_state:
    st.session_state['index_reviewed'] = []

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
        distances = cosine_similarity(embeddings, question_embedding).flatten()
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
    pca_df['label_included'] = papers_df['label_included']
    pca_df['Is Relevant'] = [st.session_state['ratings'].get(i, False) for i in range(len(papers_df))]
    pca_df['Type'] = ['Paper'] * len(papers_df)
    
    # Append the research question embedding reduced features
    rq_reduced = pca.transform(question_embedding)
    rq_df = pd.DataFrame(rq_reduced, columns=['PCA1', 'PCA2'])
    rq_df['Title'] = ['Research Question']
    rq_df['label_included'] = [0]
    rq_df['Is Relevant'] = [False]
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

    # Add a 'Size' column to the DataFrame where larger values are assigned if 'label_included' is 1
    pca_df['Size'] = pca_df['label_included'].apply(lambda x: 5 if x == 1 else 1)

    st.dataframe(pca_df, width=700, height=300)

    # Visualize with Plotly first plot with clusters
    fig = px.scatter(pca_df, x='PCA1', y='PCA2',
                     color='Cluster',
                     hover_name='Title',
                     title='2D PCA of Research Paper Abstracts',
                     template="simple_white",
                     symbol='Type',
                     size='Size',
                     symbol_map={'Paper': 'circle', 'Research Question': 'diamond'},
                     color_discrete_sequence=px.colors.qualitative.Set1)

    fig.update_traces(marker=dict(line=dict(width=[2 if x == 1 else 1 for x in pca_df['label_included']],
                                            color=['red' if x == 1 else '#FFFFFF' for x in pca_df['label_included']])),
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
    
    
# Display distances 

# Function to create a network graph
def create_graph(papers_df, distance_metric, question_embedding, threshold):
    G = nx.Graph()
    num_papers = len(papers_df)
    question_distances = calculate_distances(np.array(abstract_embeddings), question_embedding, distance_metric)

    # Add paper nodes
    for i in range(num_papers):
        included = papers_df.iloc[i].get('label_included', 0)  # Default to 0 if column or value is missing
        is_relevant = st.session_state['ratings'].get(i, False)
        G.add_node(papers_df.iloc[i]['title'], type='paper', label_included=included, is_relevant=is_relevant)
        
    # Add edges between papers based on threshold
    for i in range(num_papers):
        # Calculate pairwise distances between papers
        for j in range(i + 1, num_papers):
            # Calculate distance based on selected metric
            distance = calculate_distances(np.array([abstract_embeddings[i]]), np.array([abstract_embeddings[j]]), distance_metric)[0]
            if distance < threshold:
                G.add_edge(papers_df.iloc[i]['title'], papers_df.iloc[j]['title'], weight=distance)

    # Add research question node
    G.add_node("Research Question", type='question', label_included=0, is_relevant=False)

    # Connect research question to papers based on similarity threshold
    for i in range(num_papers):
        if question_distances[i] < threshold:
            G.add_edge("Research Question", papers_df.iloc[i]['title'], weight=question_distances[i])

    return G

# Function to plot the network graph
def plot_network_graph(G):
    pos = nx.spring_layout(G)  # Use spring layout for positioning of nodes
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    node_hover_text = []  # Hover text information
    node_color = []
    node_border_color = []  # Border color for nodes

    # Extract positions and properties for edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # None to create disjoint line segments
        edge_y.extend([y0, y1, None])

    # Extract positions and properties for nodes
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_hover_text.append(node)
        # Set node color and border color based on 'label_included' and 'is_relevant'
        if data['label_included'] == 1:
            node_color.append('blue')  # blue for included papers
            if data['is_relevant']:
                node_border_color.append('red')  # red border for relevant papers
            else:
                node_border_color.append('black')  # black border for included but not relevant papers
        elif data['type'] == 'question':
            node_color.append('red')  # Distinct color for the research question
            node_border_color.append('black')  # black border for research question
        else:
            node_color.append('gray')  # Default color for regular papers
            node_border_color.append('black')  # black border for regular papers
            if data['is_relevant']:
                    node_border_color.append('red')  # red border for relevant papers
            else:
                node_border_color.append('black')  # black border for included but not relevant papers

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        mode='lines',
        hoverinfo='none'
    )

    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_hover_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=20,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(color=node_border_color, width=2)  # Set the border color for nodes
        )
    )

    # Create figure and update layout
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[{
            'text': "",
            'showarrow': False,
            'xref': "paper", 'yref': "paper",
            'x': 0.005, 'y': -0.002
        }],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))
    

    return fig


# Determine the next most relevant paper to scan
def get_next_paper(question_embedding, abstract_embeddings, distance_metric):
    #Check if there are already reviewed papers
    if st.session_state['index_reviewed']:
        rated_indices = [idx for idx, rated in st.session_state['ratings'].items() if rated]
        if rated_indices:
            # Calculate the average embedding of the research question and relevant papers
            relevant_embeddings = np.array(abstract_embeddings)[rated_indices]
            avg_embedding = np.mean(np.vstack([question_embedding, relevant_embeddings]), axis=0).reshape(1, -1)  # Reshape here
            # Calculate distances between the average embedding and all papers
            distances = calculate_distances(abstract_embeddings, avg_embedding, distance_metric)
            # Find the closest Unreviewed paper
            best_next = sorted([(idx, dist) for idx, dist in enumerate(distances) if idx not in st.session_state['index_reviewed']], key=lambda x: x[1])
            return best_next[0][0] if best_next else None
        else:
            return None
    else:
        # Calculate pairwise distances between research question and all papers
        distances = calculate_distances(abstract_embeddings, question_embedding, distance_metric)
        return np.argmin(distances)

# Display the next paper to scan
if research_question:
    next_paper_index = get_next_paper(question_embedding, abstract_embeddings, distance_metric)
    if next_paper_index is not None and next_paper_index not in st.session_state['index_reviewed']:
        paper_title = papers_df.iloc[next_paper_index]['title']
        paper_abstract = papers_df.iloc[next_paper_index]['abstract']
        
        st.write(f"Title: {paper_title}")
        st.write(f"Abstract: {paper_abstract}")
        relevant = st.radio("Is this paper relevant to your research?", ("Yes", "No"))

        if st.button('Submit Rating'):
            st.session_state['ratings'][next_paper_index] = (relevant == "Yes")
            st.session_state['index_reviewed'].append(next_paper_index)
            papers_df.at[next_paper_index, 'Is Relevant'] = True  # Update the DataFrame
            st.experimental_rerun()
    else:
        st.write("No more papers to rate or adjust your settings to refine recommendations.")
else:
    st.write("Please enter a research question to start reviewing papers.")

# Plot the network graph
# Display tooltip with slider
st.markdown(add_tooltip("Similarity Threshold"), unsafe_allow_html=True)
threshold = st.slider('', min_value=0.0, max_value=1.0, value=0.30, step=0.01)
G = create_graph(papers_df, distance_metric, question_embedding, threshold)
fig = plot_network_graph(G)
st.plotly_chart(fig)