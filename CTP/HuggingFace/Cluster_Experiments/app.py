import gradio as gr
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from pyclustering.cluster.clarans import clarans
from sklearn.metrics.pairwise import euclidean_distances
import nltk
from nltk.tokenize import sent_tokenize
import os
import time

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

model = SentenceTransformer("all-MiniLM-L6-v2")

def run_all_clustering(text, query=None):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return "Please enter at least two sentences.", None, None, None, None, None, "", "", ""
    
    # Embed sentences
    embeddings = model.encode(sentences, normalize_embeddings=True)
    
    # Reduce to 2D for visualization
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embeddings)
    
    # --- HDBSCAN Clustering ---
    start_time = time.time()
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    hdbscan_labels = hdbscan_clusterer.fit_predict(reduced)
    hdbscan_time = time.time() - start_time
    
    plt.figure(figsize=(7,5))
    plt.scatter(reduced[:,0], reduced[:,1], c=hdbscan_labels, cmap='tab10', s=60)
    plt.title("HDBSCAN Clustering")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    for i in range(len(sentences)):
        plt.annotate(str(i), (reduced[i,0], reduced[i,1]))
    plt.tight_layout()
    plt.savefig("hdbscan_plot.png")
    plt.close()
    
    hdbscan_grouped = {}
    for sent, lbl in zip(sentences, hdbscan_labels):
        hdbscan_grouped.setdefault(int(lbl), []).append(sent)
    hdbscan_summary = f"⏱️ Time: {hdbscan_time:.2f}s\n\n" + "\n\n".join([f"Cluster {c}:\n" + "\n".join(v) for c, v in hdbscan_grouped.items()])
    
    # --- KMeans Clustering ---
    start_time = time.time()
    k = min(5, len(sentences))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(reduced)
    kmeans_time = time.time() - start_time
    
    plt.figure(figsize=(7,5))
    plt.scatter(reduced[:,0], reduced[:,1], c=kmeans_labels, cmap='tab10', s=60)
    plt.title(f"KMeans Clustering (k={k})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    for i in range(len(sentences)):
        plt.annotate(str(i), (reduced[i,0], reduced[i,1]))
    plt.tight_layout()
    plt.savefig("kmeans_plot.png")
    plt.close()
    
    kmeans_grouped = {}
    for sent, lbl in zip(sentences, kmeans_labels):
        kmeans_grouped.setdefault(int(lbl), []).append(sent)
    kmeans_summary = f"⏱️ Time: {kmeans_time:.2f}s\n\n" + "\n\n".join([f"Cluster {c}:\n" + "\n".join(v) for c, v in kmeans_grouped.items()])
    
    # --- CLARANS Clustering ---
    start_time = time.time()
    k_clarans = min(3, len(sentences))
    distance_matrix = euclidean_distances(reduced).tolist()
    clarans_instance = clarans(distance_matrix, number_clusters=k_clarans, numlocal=2, maxneighbor=10)
    clarans_instance.process()
    clusters = clarans_instance.get_clusters()
    clarans_time = time.time() - start_time
    
    clarans_labels = np.full(len(sentences), -1)
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            clarans_labels[idx] = cluster_id
    
    plt.figure(figsize=(7,5))
    plt.scatter(reduced[:,0], reduced[:,1], c=clarans_labels, cmap='tab10', s=60)
    plt.title(f"CLARANS Clustering (k={k_clarans})")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    for i in range(len(sentences)):
        plt.annotate(str(i), (reduced[i,0], reduced[i,1]))
    plt.tight_layout()
    plt.savefig("clarans_plot.png")
    plt.close()
    
    clarans_grouped = {}
    for sent, lbl in zip(sentences, clarans_labels):
        clarans_grouped.setdefault(int(lbl), []).append(sent)
    clarans_summary = f"⏱️ Time: {clarans_time:.2f}s\n\n" + "\n\n".join([f"Cluster {c}:\n" + "\n".join(v) for c, v in clarans_grouped.items()])
    
    # --- Query Results (separate for each algorithm to show differences) ---
    hdbscan_query = kmeans_query = clarans_query = ""
    if query:
        query_emb = model.encode([query], normalize_embeddings=True)
        sims = np.dot(embeddings, query_emb.T).flatten()
        top_idx = np.argsort(-sims)[:5]
        
        # Base query results
        base_results = [f"{i+1}. {sentences[idx]} (similarity: {sims[idx]:.3f})" for i, idx in enumerate(top_idx)]
        
        # Add cluster info for each algorithm
        hdbscan_query = "Top 5 sentences matching query:\n\n" + "\n\n".join(
            [f"{base_results[i]} [HDBSCAN Cluster: {hdbscan_labels[top_idx[i]]}]" for i in range(len(base_results))]
        )
        
        kmeans_query = "Top 5 sentences matching query:\n\n" + "\n\n".join(
            [f"{base_results[i]} [KMeans Cluster: {kmeans_labels[top_idx[i]]}]" for i in range(len(base_results))]
        )
        
        clarans_query = "Top 5 sentences matching query:\n\n" + "\n\n".join(
            [f"{base_results[i]} [CLARANS Cluster: {clarans_labels[top_idx[i]]}]" for i in range(len(base_results))]
        )
    
    return (hdbscan_summary, "hdbscan_plot.png", hdbscan_query,
            kmeans_summary, "kmeans_plot.png", kmeans_query,
            clarans_summary, "clarans_plot.png", clarans_query)

# Gradio interface with all three algorithms
demo = gr.Interface(
    fn=run_all_clustering,
    inputs=[
        gr.Textbox(lines=10, placeholder="Paste article or chat transcript here...", label="Text Input"),
        gr.Textbox(lines=1, placeholder="Enter query to search relevant sentences (optional)", label="Query (Optional)")
    ],
    outputs=[
        gr.Textbox(label="HDBSCAN: Clustered Sentences (with timing)"),
        gr.Image(label="HDBSCAN: Cluster Graph"),
        gr.Textbox(label="HDBSCAN: Query Results"),
        gr.Textbox(label="KMeans: Clustered Sentences (with timing)"),
        gr.Image(label="KMeans: Cluster Graph"),
        gr.Textbox(label="KMeans: Query Results"),
        gr.Textbox(label="CLARANS: Clustered Sentences (with timing)"),
        gr.Image(label="CLARANS: Cluster Graph"),
        gr.Textbox(label="CLARANS: Query Results")
    ],
    title="Clustering Comparison: HDBSCAN vs KMeans vs CLARANS",
    description="Compare three clustering algorithms side-by-side. Enter text to cluster sentences and optionally query for relevant sentences. Timing and cluster assignments are shown for each algorithm."
)

if __name__ == "__main__":
    demo.launch()