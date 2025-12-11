import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import io
import hashlib
import re
from typing import List, Union
import openai

st.set_page_config(page_title="Semantic Map 3D", layout="wide")
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}

def text_hash(text: str, model_key: str):
    return hashlib.sha256((model_key + text).encode("utf-8")).hexdigest()
@st.cache_resource
def load_local_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_openai_embeddings(texts: List[str], api_key: str, model_name: str = "text-embedding-3-small") -> np.ndarray:
    try:
        client = openai.OpenAI(api_key=api_key)
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Błąd podczas pobierania embeddingów z OpenAI: {e}")
        raise

def encode_texts(texts: List[str], embedding_model: str, api_key: str = None, model_name: str = None) -> np.ndarray:
    cache = st.session_state.embedding_cache
    model_identifier = embedding_model + (model_name if model_name else "")
    
    embeddings_list = [None] * len(texts)
    missing_texts = []
    missing_indices = []

    for i, txt in enumerate(texts):
        h = text_hash(txt, model_identifier)
        if h in cache:
            embeddings_list[i] = cache[h]
        else:
            missing_texts.append(txt)
            missing_indices.append(i)

    if missing_texts:
        new_embeddings = []
        if embedding_model == "Lokalny (SentenceTransformer)":
            model = load_local_model()
            new_embeddings = model.encode(missing_texts)
        elif embedding_model == "OpenAI API":
            if not api_key:
                raise ValueError("API key OpenAI jest wymagany")
            new_embeddings = get_openai_embeddings(missing_texts, api_key, model_name or "text-embedding-3-small")
        else:
            raise ValueError(f"Nieznany model: {embedding_model}")

        for idx, emb, txt in zip(missing_indices, new_embeddings, missing_texts):
            h = text_hash(txt, model_identifier)
            cache[h] = emb
            embeddings_list[idx] = emb
            
    return np.array(embeddings_list)

@st.cache_data
def chunk_text_fixed(text, chunk_size=500, overlap=True):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            chunks.append(sentence)
            continue
        
        test_chunk = current_chunk + [sentence]
        test_length = len(" ".join(test_chunk))
        
        if test_length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                if overlap == True:
                    current_chunk = [current_chunk[-1]]
                else:
                    current_chunk = []
        
        current_chunk.append(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

@st.cache_data
def semantic_chunking(text, threshold_percentile=90, min_chunk_len=100):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    if len(sentences) == 1:
        return sentences

    model = load_local_model()
    embeddings = model.encode(sentences)
    
    distances = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        distance = 1 - sim
        distances.append(distance)
        
    if distances:
        breakpoint_distance_threshold = np.percentile(distances, threshold_percentile)
    else:
        breakpoint_distance_threshold = 0

    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        dist = distances[i-1]
        current_chunk_len = sum(len(s) for s in current_chunk)
        
        if dist > breakpoint_distance_threshold and current_chunk_len > min_chunk_len:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

@st.cache_data
def read_pdf(file_bytes, filename):
    file = io.BytesIO(file_bytes)
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            text += txt + "\n"
    return text

@st.cache_data
def read_text_file(file_bytes, filename):
    return file_bytes.decode("utf-8")

st.title("Eksplorator Przestrzeni Semantycznej")

with st.sidebar:
    st.header("Dane wejściowe")
    
    default_text = """King
Queen
Man
Woman"""
    manual_input = st.text_area("Wpisz krótkie frazy:", value=default_text, height=150)
    
    uploaded_files = st.file_uploader("Lub wgraj dokumenty (PDF, TXT)", type=['txt', 'pdf'], accept_multiple_files=True)
    
    file_processing_mode = st.radio(
        "Tryb przetwarzania plików:",
        ["Chmura chunków", "Jeden punkt"],
        help="Chmura chunków: każdy fragment pliku jako osobny punkt. Jeden punkt: chunki z pliku są uśredniane do jednego punktu w przestrzeni."
    )
    
    st.markdown("---")
    st.subheader("Ustawienia Chunkowania")
    
    chunking_method = st.radio(
            "Metoda podziału (Chunking):",
            ["Sztywna", "Semantyczna (AI)"],
            help="""Metody podziału tekstu:
            \n**Sztywna:** Deterministyczny podział tekstu na segmenty o zbliżonej długości (z zachowaniem granic zdań). Miękka granica limitu znaków.
            \n **Semantyczna:** Inteligentna analiza znaczenia sterowana przez AI. Algorytm grupuje logicznie powiązane zdania i tworzy nowy fragment dopiero w momencie wykrycia zmiany tematu."""
        )
    
    chunk_len = 500
    semantic_threshold = 90
    semantic_min_len = 100
    
    if chunking_method == "Sztywna":
        chunk_len = st.number_input("Długość chunka (znaki)", 100, 2000, 500)
    else:
        semantic_threshold = st.slider("Czułość podziału (Percentyl)", 50, 99, 90, 
                                     help="Im wyższa wartość (np. 95), tym rzadziej tnie (tylko przy dużych zmianach tematu).")
        semantic_min_len = st.number_input("Min. długość chunka (znaki)", 50, 1000, 100)

    st.markdown("---")
    st.subheader("Parametry Wizualizacji")
    
    model_options = ["Lokalny (SentenceTransformer)", "OpenAI API"]
    embedding_model = st.radio(
        "Model do embeddingów:",
        model_options,
        help="Wybierz model do tworzenia embeddingów. OpenAI API wymaga klucza API."
    )
    
    openai_api_key = None
    openai_model_name = "text-embedding-3-small"
    if embedding_model == "OpenAI API":
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        openai_model_name = st.selectbox(
            "Model OpenAI", 
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            help="Wybierz model embeddingów OpenAI"
        )
        if not openai_api_key:
            st.warning("Wprowadź klucz API OpenAI")
    
    view_mode = st.radio("Widok:", ["3D", "2D"], horizontal=True)
    
    show_neighbor_connections = st.toggle("Połączenia do najbliższych sąsiadów", value=False,
                                          help="Pokazuje cienkie linie łączące każdy punkt z jego najbliższymi sąsiadami.")
    num_neighbors = 3
    if show_neighbor_connections:
        num_neighbors = st.number_input("Liczba najbliższych sąsiadów", 1, 10, 1)
    
    n_neighbors = st.slider(
        "UMAP: Sąsiedztwo",
        min_value=2, 
        max_value=50, 
        value=15,
        help="""Definiuje balans między analizą szczegółową a ogólną:
        \n**Niska wartość (Mikro):** Algorytm skupia się na lokalnych niuansach. Pozwala wyodrębnić nawet małe, specyficzne podgrupy tematyczne.
        \n**Wysoka wartość (Makro):** Algorytm analizuje szerszy kontekst. Priorytetyzuje ogólną topologię danych, lepiej ukazując relacje między odległymi tematami."""
        )
    min_dist = st.slider(
        "UMAP: Dystans",
        min_value=0.0,
        max_value=0.99,
        value=0.1,
        step=0.01,
        help="""Kontroluje stopień zagęszczenia danych w przestrzeni:
        \n **Niska wartość:** Punkty o wysokim podobieństwie są grupowane bardzo ciasno. Podkreśla to silną przynależność do klastra.
        \n **Wysoka wartość:** Punkty są rozmieszczane luźniej. Pozwala to na łatwiejszą obserwację wewnętrznej struktury grup oraz subtelnych różnic między zbliżonymi klastrami.""")
    n_clusters = st.slider("K-Means: Ilość klastrów", 2, 10, 3)

    st.markdown("---")
    if st.button("Wyczyść cache embeddingów"):
        st.session_state.embedding_cache.clear()
        st.success("Cache wyczyszczony!")

all_texts = []
source_labels = [] 
file_to_chunks_map = {} 

if manual_input:
    lines = [s.strip() for s in manual_input.split('\n') if s.strip()]
    all_texts.extend(lines)
    source_labels.extend(["Ręczny wpis"] * len(lines))

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.endswith('.pdf'):
            raw_text = read_pdf(file_bytes, uploaded_file.name)
        else:
            raw_text = read_text_file(file_bytes, uploaded_file.name)
        uploaded_file.seek(0)
        
        if chunking_method == "Sztywna":
            file_chunks = chunk_text_fixed(raw_text, chunk_size=chunk_len)
        else:
            file_chunks = semantic_chunking(
                raw_text, 
                threshold_percentile=semantic_threshold, 
                min_chunk_len=semantic_min_len
            )
        
        if file_chunks:
            start_idx = len(all_texts)
            all_texts.extend(file_chunks)
            end_idx = len(all_texts)
            source_labels.extend([f"Plik: {uploaded_file.name}"] * len(file_chunks))
            file_to_chunks_map[uploaded_file.name] = list(range(start_idx, end_idx))

if len(all_texts) >= 3:
    cache_info = f"(W cache: {len(st.session_state.embedding_cache)})"
    with st.spinner(f'Przetwarzanie {len(all_texts)} fragmentów tekstu... {cache_info}'):
        
        try:
            embeddings = encode_texts(all_texts, embedding_model, openai_api_key, openai_model_name)
        except Exception as e:
            st.error(f"Błąd podczas tworzenia embeddingów: {e}")
            st.stop()
        
        if file_processing_mode == "Jeden punkt" and file_to_chunks_map:
            averaged_texts = []
            averaged_source_labels = []
            averaged_embeddings = []
            
            manual_indices = [i for i, label in enumerate(source_labels) if label == "Ręczny wpis"]
            for idx in manual_indices:
                averaged_texts.append(all_texts[idx])
                averaged_source_labels.append(source_labels[idx])
                averaged_embeddings.append(embeddings[idx])
            
            for filename, chunk_indices in file_to_chunks_map.items():
                file_embeddings = embeddings[chunk_indices]
                averaged_embedding = np.mean(file_embeddings, axis=0)
                
                first_chunk_text = all_texts[chunk_indices[0]][:200]
                chunk_info = f"[{chunking_method}]"
                if len(chunk_indices) > 1:
                    text_preview = f"{first_chunk_text}... {chunk_info} [Plik: {len(chunk_indices)} fragm.]"
                else:
                    text_preview = first_chunk_text
                
                averaged_texts.append(text_preview)
                averaged_source_labels.append(f"Plik: {filename}")
                averaged_embeddings.append(averaged_embedding)
            
            all_texts = averaged_texts
            source_labels = averaged_source_labels
            embeddings = np.array(averaged_embeddings)

        n_neighbors_actual = min(n_neighbors, len(embeddings) - 1)
        if n_neighbors_actual < 2: n_neighbors_actual = 2
        
        init_method = 'spectral' if len(embeddings) > 5 else 'random'
        n_components = 2 if view_mode == "2D" else 3

        umap_reducer = UMAP(
            n_components=n_components, 
            n_neighbors=n_neighbors_actual, 
            min_dist=min_dist, 
            random_state=42,
            init=init_method
        )
        projections = umap_reducer.fit_transform(embeddings)

        n_clusters_actual = min(n_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        if view_mode == "2D":
            df = pd.DataFrame(projections, columns=['x', 'y'])
            df['z'] = 0
        else:
            df = pd.DataFrame(projections, columns=['x', 'y', 'z'])
            
        df['text'] = all_texts
        df['source'] = source_labels
        df['cluster'] = clusters
        df['short_text'] = df['text'].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
        
        neighbor_connections = {}
        if show_neighbor_connections:
            similarity_matrix = cosine_similarity(embeddings)
            for i in range(len(embeddings)):
                sims = similarity_matrix[i].copy()
                sims[i] = -1 
                top_n = sims.argsort()[-num_neighbors:][::-1]
                neighbor_connections[i] = top_n.tolist()

    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("🔍 Znajdź w dokumentach")
        search_query = st.text_input("Szukana fraza:")
        top_n = st.number_input("Liczba wyników", min_value=1, max_value=min(20, len(embeddings)), value=3, step=1)
        
    search_traces = []
    
    if search_query:
        try:
            query_vec = encode_texts([search_query], embedding_model, openai_api_key, openai_model_name)
            sims = cosine_similarity(query_vec, embeddings)[0]
        except Exception as e:
            st.error(f"Błąd podczas embedowania zapytania: {e}")
            query_vec = None
            sims = None
    
        if query_vec is not None and sims is not None:
            top_n_actual = min(top_n, len(embeddings))
            top_n_indices = sims.argsort()[-top_n_actual:][::-1]
            
            matched_coords = projections[top_n_indices]
            virtual_center = np.mean(matched_coords, axis=0)
            
            if view_mode == "2D":
                search_point = go.Scatter(
                    x=[virtual_center[0]], y=[virtual_center[1]],
                    mode='markers+text',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    text=[f"Szukane: {search_query}"],
                    textposition="top center",
                    name='Szukana fraza'
                )
            else:
                search_point = go.Scatter3d(
                    x=[virtual_center[0]], y=[virtual_center[1]], z=[virtual_center[2]],
                    mode='markers+text',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    text=[f"Szukane: {search_query}"],
                    textposition="top center",
                    name='Szukana fraza'
                )
            search_traces.append(search_point)

            for idx in top_n_indices:
                target = projections[idx]
                found_text_short = all_texts[idx][:50] + "..."
                
                if view_mode == "2D":
                    search_traces.append(
                        go.Scatter(
                            x=[virtual_center[0], target[0]],
                            y=[virtual_center[1], target[1]],
                            mode='lines',
                            line=dict(color='rgba(255, 50, 50, 0.6)', width=5),
                            hoverinfo='none',
                            showlegend=False
                        )
                    )
                    search_traces.append(
                        go.Scatter(
                            x=[target[0]], y=[target[1]],
                            mode='markers',
                            marker=dict(size=8, color='orange', symbol='circle-open', line=dict(width=2)),
                            text=[f"Trafienie: {found_text_short}"],
                            hoverinfo='text',
                            showlegend=False
                        )
                    )
                else:
                    search_traces.append(
                        go.Scatter3d(
                            x=[virtual_center[0], target[0]],
                            y=[virtual_center[1], target[1]],
                            z=[virtual_center[2], target[2]],
                            mode='lines',
                            line=dict(color='rgba(255, 50, 50, 0.6)', width=5),
                            hoverinfo='none',
                            showlegend=False
                        )
                    )
                    search_traces.append(
                        go.Scatter3d(
                            x=[target[0]], y=[target[1]], z=[target[2]],
                            mode='markers',
                            marker=dict(size=8, color='orange', symbol='circle-open', line=dict(width=2)),
                            text=[f"Trafienie: {found_text_short}"],
                            hoverinfo='text',
                            showlegend=False
                        )
                    )

    color_mode = st.radio("Koloruj według:", ["Tematyki (Klastry)", "Źródła pliku"], horizontal=True)
    
    fig = go.Figure()

    cluster_colors = pc.qualitative.Set3
    
    if view_mode == "2D":
        if color_mode == "Tematyki (Klastry)":
            unique_clusters = sorted(df['cluster'].unique())
            for cluster_id in unique_clusters:
                subset = df[df['cluster'] == cluster_id]
                color_idx = cluster_id % len(cluster_colors)
                fig.add_trace(go.Scatter(
                    x=subset['x'], y=subset['y'],
                    mode='markers',
                    text=subset['source'] + ":<br>" + subset['short_text'],
                    hoverinfo='text',
                    marker=dict(size=5, color=cluster_colors[color_idx], opacity=0.8),
                    name=f'Klaster {cluster_id}'
                ))
        else:
            unique_sources = df['source'].unique()
            for source in unique_sources:
                subset = df[df['source'] == source]
                fig.add_trace(go.Scatter(
                    x=subset['x'], y=subset['y'],
                    mode='markers',
                    text=subset['short_text'],
                    hoverinfo='text',
                    marker=dict(size=5, opacity=0.8),
                    name=source
                ))
    else:
        if color_mode == "Tematyki (Klastry)":
            unique_clusters = sorted(df['cluster'].unique())
            for cluster_id in unique_clusters:
                subset = df[df['cluster'] == cluster_id]
                color_idx = cluster_id % len(cluster_colors)
                fig.add_trace(go.Scatter3d(
                    x=subset['x'], y=subset['y'], z=subset['z'],
                    mode='markers',
                    text=subset['source'] + ":<br>" + subset['short_text'],
                    hoverinfo='text',
                    marker=dict(size=5, color=cluster_colors[color_idx], opacity=0.8),
                    name=f'Klaster {cluster_id}'
                ))
        else:
            unique_sources = df['source'].unique()
            for source in unique_sources:
                subset = df[df['source'] == source]
                fig.add_trace(go.Scatter3d(
                    x=subset['x'], y=subset['y'], z=subset['z'],
                    mode='markers',
                    text=subset['short_text'],
                    hoverinfo='text',
                    marker=dict(size=5, opacity=0.8),
                    name=source
                ))

    if show_neighbor_connections and neighbor_connections:
        for point_idx, neighbor_indices in neighbor_connections.items():
            point_coords = projections[point_idx]
            for neighbor_idx in neighbor_indices:
                neighbor_coords = projections[neighbor_idx]
                if view_mode == "2D":
                    connection_trace = go.Scatter(
                        x=[point_coords[0], neighbor_coords[0]],
                        y=[point_coords[1], neighbor_coords[1]],
                        mode='lines',
                        line=dict(color='rgba(100, 200, 255, 0.3)', width=1),
                        hoverinfo='none',
                        showlegend=False,
                    )
                else:
                    connection_trace = go.Scatter3d(
                        x=[point_coords[0], neighbor_coords[0]],
                        y=[point_coords[1], neighbor_coords[1]],
                        z=[point_coords[2], neighbor_coords[2]],
                        mode='lines',
                        line=dict(color='rgba(100, 200, 255, 0.3)', width=1),
                        hoverinfo='none',
                        showlegend=False,
                    )
                fig.add_trace(connection_trace)

    for trace in search_traces:
        fig.add_trace(trace)

    if view_mode == "2D":
        fig.update_layout(
            xaxis_title='X',
            yaxis_title='Y',
            xaxis=dict(gridcolor='gray', showgrid=True),
            yaxis=dict(gridcolor='gray', showgrid=True),
            plot_bgcolor='black',
            margin=dict(l=0, r=0, b=0, t=0),
            height=700,
            paper_bgcolor="black",
            font=dict(color="white"),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")
            )
        )
    else:
        fig.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                bgcolor='black'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=700,
            paper_bgcolor="black",
            font=dict(color="white"),
            legend=dict(
                yanchor="top", y=0.99, xanchor="left", x=0.01,
                bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")
            )
        )

    st.plotly_chart(fig, use_container_width=True)
    
    if search_query and query_vec is not None and sims is not None:
        with st.expander(f"📄 Wyniki wyszukiwania dla: '{search_query}'", expanded=True):
            for idx in top_n_indices:
                row = df.iloc[idx]
                similarity_score = sims[idx]
                
                st.markdown(f"**Trafienie (Podobieństwo: {similarity_score:.4f})** | Źródło: *{row['source']}*")
                st.code(row['text'], language="text")
                st.divider()
    
    st.markdown("---")
    st.subheader("Inspektor punktów")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_cluster = st.multiselect(
            "Filtruj wg klastra (tematu):", 
            options=sorted(df['cluster'].unique()),
            default=sorted(df['cluster'].unique())
        )
    with col_f2:
        selected_source = st.multiselect(
            "Filtruj wg źródła:",
            options=sorted(df['source'].unique()),
            default=sorted(df['source'].unique())
        )

    filtered_df = df[
        (df['cluster'].isin(selected_cluster)) & 
        (df['source'].isin(selected_source))
    ]

    st.dataframe(
        filtered_df[['source', 'cluster', 'text']],
        column_config={
            "source": "Źródło",
            "cluster": "Klaster",
            "text": st.column_config.TextColumn(
                "Treść punktu",
                width="large"
            )
        },
        use_container_width=True,
        hide_index=True,
        height=400
    )

else:
    st.info("Wgraj pliki lub wpisz tekst (minimum 3 fragmenty), aby wygenerować mapę.")