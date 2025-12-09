import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import io
import hashlib
from typing import List, Union

# Opcjonalny import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Konfiguracja strony
st.set_page_config(page_title="Semantic Map 3D", layout="wide")

# --- 0. CACHE INIT (Nowość z app.py) ---
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}

def text_hash(text: str, model_key: str):
    """Tworzy hash unikalny dla tekstu i użytego modelu"""
    return hashlib.sha256((model_key + text).encode("utf-8")).hexdigest()

# --- 1. Funkcje do embedowania ---
@st.cache_resource
def load_local_model():
    """Ładuje lokalny model SentenceTransformer"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_openai_embeddings(texts: List[str], api_key: str, model_name: str = "text-embedding-3-small") -> np.ndarray:
    """Pobiera embeddingi z API OpenAI"""
    if not OPENAI_AVAILABLE:
        raise ImportError("Biblioteka 'openai' nie jest zainstalowana. Zainstaluj ją przez: pip install openai")
    try:
        client = openai.OpenAI(api_key=api_key)
        # OpenAI API przyjmuje listy tekstów, ale ma limit batch size
        embeddings = []
        batch_size = 100  # OpenAI ma limit na batch
        
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
    """
    Uniwersalna funkcja do embedowania tekstów z CACHINGIEM.
    Sprawdza, czy tekst był już przeliczony dla danego modelu.
    """
    cache = st.session_state.embedding_cache
    # Klucz modelu do hasha (np. "Lokalny" lub "OpenAI APItext-embedding-3-small")
    model_identifier = embedding_model + (model_name if model_name else "")
    
    embeddings_list = [None] * len(texts)
    missing_texts = []
    missing_indices = []

    # 1. Sprawdź cache
    for i, txt in enumerate(texts):
        h = text_hash(txt, model_identifier)
        if h in cache:
            embeddings_list[i] = cache[h]
        else:
            missing_texts.append(txt)
            missing_indices.append(i)

    # 2. Oblicz brakujące
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

        # 3. Zapisz nowe do cache i do listy wynikowej
        for idx, emb, txt in zip(missing_indices, new_embeddings, missing_texts):
            h = text_hash(txt, model_identifier)
            cache[h] = emb
            embeddings_list[idx] = emb
            
    return np.array(embeddings_list)

# --- Funkcje pomocnicze do dokumentów ---
def chunk_text(text, chunk_size=500, overlap=50):
    """Dzieli tekst na mniejsze kawałki z zakładką (overlap), aby nie ciąć zdań w połowie kontekstu."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        # Próba znalezienia końca zdania lub spacji, żeby nie ciąć wyrazów
        if end < text_len:
            last_space = chunk.rfind(' ')
            if last_space != -1:
                end = start + last_space + 1
                chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start += chunk_size - overlap # Przesuwamy okno, zachowując overlap
        
    return chunks

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            text += txt + "\n"
    return text

st.title("🌌 Eksplorator Przestrzeni Semantycznej (Docs Support + Cache)")

# --- 2. Panel boczny - Dane ---
with st.sidebar:
    st.header("Dane wejściowe")
    
    # A. Tekst ręczny
    default_text = """Kot pije mleko.
Pies szczeka na listonosza.
Sztuczna inteligencja zmienia świat.
Algorytmy uczenia maszynowego."""
    manual_input = st.text_area("Wpisz krótkie frazy:", value=default_text, height=150)
    
    # B. Upload plików
    uploaded_files = st.file_uploader("Lub wgraj dokumenty (PDF, TXT)", type=['txt', 'pdf'], accept_multiple_files=True)
    
    # Opcja wyboru trybu przetwarzania plików
    file_processing_mode = st.radio(
        "Tryb przetwarzania plików:",
        ["Chmura chunków", "Jeden punkt"],
        help="Chmura chunków: każdy fragment pliku jako osobny punkt. Jeden punkt: chunki z pliku są uśredniane do jednego punktu w przestrzeni."
    )
    
    st.markdown("---")
    st.subheader("Parametry")
    
    # Opcja wyboru modelu do embeddingów
    model_options = ["Lokalny (SentenceTransformer)"]
    if OPENAI_AVAILABLE:
        model_options.append("OpenAI API")
    else:
        st.info("💡 Aby używać OpenAI API, zainstaluj bibliotekę: `pip install openai`")
    
    embedding_model = st.radio(
        "Model do embeddingów:",
        model_options,
        help="Wybierz model do tworzenia embeddingów. OpenAI API wymaga klucza API."
    )
    
    # Pole na API key OpenAI (tylko gdy wybrano OpenAI)
    openai_api_key = None
    openai_model_name = "text-embedding-3-small"
    if embedding_model == "OpenAI API":
        openai_api_key = st.text_input("OpenAI API Key", type="password", 
                                       help="Wprowadź swój klucz API OpenAI")
        openai_model_name = st.selectbox(
            "Model OpenAI",
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            help="Wybierz model embeddingowy OpenAI"
        )
        if not openai_api_key:
            st.warning("⚠️ Wprowadź klucz API OpenAI, aby używać modelu OpenAI")
    
    # Opcja wyboru widoku
    view_mode = st.radio("Widok:", ["3D", "2D"], horizontal=True)
    
    # Opcja połączeń do najbliższych sąsiadów
    show_neighbor_connections = st.toggle("Połączenia do najbliższych sąsiadów", value=False, 
                                           help="Pokaż cienkie linie łączące każdy punkt z jego najbliższymi sąsiadami")
    
    # Liczba najbliższych sąsiadów (tylko gdy opcja włączona)
    num_neighbors = 3
    if show_neighbor_connections:
        num_neighbors = st.number_input("Liczba najbliższych sąsiadów", min_value=1, max_value=10, value=3, step=1)
    
    n_neighbors = st.slider("UMAP: Sąsiedztwo", 2, 50, 15) # Zwiększamy domyślne, bo przy chunkach będzie więcej punktów
    min_dist = st.slider("UMAP: Dystans", 0.0, 1.0, 0.1)
    n_clusters = st.slider("K-Means: Ilość klastrów", 2, 10, 3)
    
    # Długość chunka - zawsze używana, bo pliki zawsze są dzielone na chunki
    chunk_len = st.number_input("Długość chunka (znaki)", 100, 2000, 500)

    st.markdown("---")
    # Przycisk czyszczenia cache
    if st.button("🧹 Wyczyść cache embeddingów"):
        st.session_state.embedding_cache.clear()
        st.success("Cache wyczyszczony!")

# --- 3. Przygotowanie Danych ---
all_texts = []
source_labels = [] # Lista przechowująca nazwę pliku/źródła dla każdego chunka

# 1. Przetwarzanie ręcznego tekstu
if manual_input:
    lines = [s.strip() for s in manual_input.split('\n') if s.strip()]
    all_texts.extend(lines)
    source_labels.extend(["Ręczny wpis"] * len(lines))

# 2. Przetwarzanie plików
# Zawsze dzielimy pliki na chunki (ze względu na ograniczenia embeddingów)
file_to_chunks_map = {}  # Mapa: nazwa_pliku -> lista indeksów chunków w all_texts
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.pdf'):
            raw_text = read_pdf(uploaded_file)
        else:
            raw_text = uploaded_file.read().decode("utf-8")
        
        # Zawsze dzielimy na chunki
        file_chunks = chunk_text(raw_text, chunk_size=chunk_len)
        if file_chunks:
            start_idx = len(all_texts)
            all_texts.extend(file_chunks)
            end_idx = len(all_texts)
            source_labels.extend([f"Plik: {uploaded_file.name}"] * len(file_chunks))
            
            # Zapisujemy mapowanie pliku do indeksów jego chunków
            file_to_chunks_map[uploaded_file.name] = list(range(start_idx, end_idx))

# --- Główna pętla programu ---
if len(all_texts) >= 3:
    # Komunikat o ilości tekstów i cache
    cache_info = f"(W cache: {len(st.session_state.embedding_cache)})"
    with st.spinner(f'Przetwarzanie {len(all_texts)} fragmentów tekstu... {cache_info}'):
        
        # A. Wektoryzacja - zawsze embedujemy wszystkie chunki (z użyciem cache)
        try:
            embeddings = encode_texts(all_texts, embedding_model, openai_api_key, openai_model_name)
        except Exception as e:
            st.error(f"Błąd podczas tworzenia embeddingów: {e}")
            st.stop()
        
        # Jeśli tryb "Jeden punkt", uśredniamy embeddingi chunków z tego samego pliku
        if file_processing_mode == "Jeden punkt" and file_to_chunks_map:
            # Tworzymy nowe listy dla uśrednionych danych
            averaged_texts = []
            averaged_source_labels = []
            averaged_embeddings = []
            
            # Zachowujemy ręczne wpisy (nie są plikami)
            manual_indices = [i for i, label in enumerate(source_labels) if label == "Ręczny wpis"]
            for idx in manual_indices:
                averaged_texts.append(all_texts[idx])
                averaged_source_labels.append(source_labels[idx])
                averaged_embeddings.append(embeddings[idx])
            
            # Uśredniamy chunki z każdego pliku
            for filename, chunk_indices in file_to_chunks_map.items():
                file_embeddings = embeddings[chunk_indices]
                averaged_embedding = np.mean(file_embeddings, axis=0)
                
                # Tekst to pierwsze 200 znaków z pierwszego chunka + info o ilości chunków
                first_chunk_text = all_texts[chunk_indices[0]][:200]
                if len(chunk_indices) > 1:
                    text_preview = f"{first_chunk_text}... [Plik podzielony na {len(chunk_indices)} fragmentów]"
                else:
                    text_preview = first_chunk_text
                
                averaged_texts.append(text_preview)
                averaged_source_labels.append(f"Plik: {filename}")
                averaged_embeddings.append(averaged_embedding)
            
            # Aktualizujemy dane
            all_texts = averaged_texts
            source_labels = averaged_source_labels
            embeddings = np.array(averaged_embeddings)

        # B. Redukcja (Zabezpieczenie: n_neighbors nie może być większe niż ilość danych)
        n_neighbors_actual = min(n_neighbors, len(embeddings) - 1)
        if n_neighbors_actual < 2: n_neighbors_actual = 2
        
        # FIX: Dla bardzo małych zbiorów danych (N <= n_components + 2) metoda 'spectral' wyrzuca błąd.
        init_method = 'spectral' if len(embeddings) > 5 else 'random'
        
        # Wybór liczby wymiarów w zależności od trybu widoku
        n_components = 2 if view_mode == "2D" else 3

        umap_reducer = UMAP(
            n_components=n_components, 
            n_neighbors=n_neighbors_actual, 
            min_dist=min_dist, 
            random_state=42,
            init=init_method
        )
        projections = umap_reducer.fit_transform(embeddings)

        # C. Klasteryzacja (tylko semantyczna)
        n_clusters_actual = min(n_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # D. DataFrame - różne kolumny w zależności od trybu
        if view_mode == "2D":
            df = pd.DataFrame(projections, columns=['x', 'y'])
            df['z'] = 0  # Dla kompatybilności z kodem wyszukiwania
        else:
            df = pd.DataFrame(projections, columns=['x', 'y', 'z'])
        df['text'] = all_texts
        df['source'] = source_labels
        df['cluster'] = clusters
        
        # Skracanie tekstu do wyświetlania w tooltipie
        df['short_text'] = df['text'].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
        
        # Obliczenie najbliższych sąsiadów (dla połączeń)
        neighbor_connections = {}
        if show_neighbor_connections:
            # Używamy cosine similarity na embeddings do znalezienia najbliższych sąsiadów
            similarity_matrix = cosine_similarity(embeddings)
            for i in range(len(embeddings)):
                # Pobierz podobieństwa dla punktu i, pomiń sam punkt
                sims = similarity_matrix[i].copy()
                sims[i] = -1  # Pomiń sam punkt
                # Znajdź N najbardziej podobnych
                top_n = sims.argsort()[-num_neighbors:][::-1]
                neighbor_connections[i] = top_n.tolist()

    # --- 4. Funkcja Wyszukiwania ---
    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("🔍 Znajdź w dokumentach")
        search_query = st.text_input("Szukana fraza:")
        top_n = st.number_input("Liczba wyników", min_value=1, max_value=min(20, len(embeddings)), value=3, step=1)
        
    search_traces = []
    
    if search_query:
        try:
            # Używamy tej samej funkcji z cachem do zapytania
            query_vec = encode_texts([search_query], embedding_model, openai_api_key, openai_model_name)
            sims = cosine_similarity(query_vec, embeddings)[0]
        except Exception as e:
            st.error(f"Błąd podczas embedowania zapytania: {e}")
            query_vec = None
            sims = None
    
        if query_vec is not None and sims is not None:
            
            # Znajdź N najlepszych fragmentów
            top_n_actual = min(top_n, len(embeddings))
            top_n_indices = sims.argsort()[-top_n_actual:][::-1]
            
            matched_coords = projections[top_n_indices]
            virtual_center = np.mean(matched_coords, axis=0)
            
            # Punkt centralny wyszukiwania - różne typy dla 2D i 3D
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

            # Linie do znalezionych fragmentów
            for idx in top_n_indices:
                target = projections[idx]
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
                    # Podświetlenie znalezionych punktów
                    found_text_short = all_texts[idx][:50] + "..."
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
                    # Podświetlenie znalezionych punktów
                    found_text_short = all_texts[idx][:50] + "..."
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

    # --- 5. Wizualizacja ---
    
    # Tryb kolorowania: Semantyczny (Klastry) vs Źródłowy (Pliki)
    color_mode = st.radio("Koloruj według:", ["Tematyki (Klastry)", "Źródła pliku"], horizontal=True)
    
    fig = go.Figure()

    if view_mode == "2D":
        # Widok 2D
        if color_mode == "Tematyki (Klastry)":
            # Rysujemy wszystko jedną serią, kolor zależy od klastra
            fig.add_trace(go.Scatter(
                x=df['x'], y=df['y'],
                mode='markers',
                text=df['source'] + ":<br>" + df['short_text'],
                hoverinfo='text',
                marker=dict(size=5, color=df['cluster'], colorscale='Viridis', opacity=0.8),
                name='Fragmenty'
            ))
        else:
            # Rysujemy oddzielną serię dla każdego pliku (żeby mieć legendę kolorów)
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
        # Widok 3D
        if color_mode == "Tematyki (Klastry)":
            # Rysujemy wszystko jedną serią, kolor zależy od klastra
            fig.add_trace(go.Scatter3d(
                x=df['x'], y=df['y'], z=df['z'],
                mode='markers',
                text=df['source'] + ":<br>" + df['short_text'],
                hoverinfo='text',
                marker=dict(size=5, color=df['cluster'], colorscale='Viridis', opacity=0.8),
                name='Fragmenty'
            ))
        else:
            # Rysujemy oddzielną serię dla każdego pliku (żeby mieć legendę kolorów)
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

    # Dodanie linii połączeń do najbliższych sąsiadów (jeśli opcja włączona)
    if show_neighbor_connections and neighbor_connections:
        # Dodajemy linie dla każdego punktu do jego najbliższych sąsiadów
        for point_idx, neighbor_indices in neighbor_connections.items():
            point_coords = projections[point_idx]
            for neighbor_idx in neighbor_indices:
                neighbor_coords = projections[neighbor_idx]
                if view_mode == "2D":
                    connection_trace = go.Scatter(
                        x=[point_coords[0], neighbor_coords[0]],
                        y=[point_coords[1], neighbor_coords[1]],
                        mode='lines',
                        line=dict(color='rgba(100, 200, 255, 0.3)', width=1),  # Cienkie linie z niską opacity
                        hoverinfo='none',
                        showlegend=False,
                        name=f'connection_{point_idx}_{neighbor_idx}'
                    )
                else:
                    connection_trace = go.Scatter3d(
                        x=[point_coords[0], neighbor_coords[0]],
                        y=[point_coords[1], neighbor_coords[1]],
                        z=[point_coords[2], neighbor_coords[2]],
                        mode='lines',
                        line=dict(color='rgba(100, 200, 255, 0.3)', width=1),  # Cienkie linie z niską opacity
                        hoverinfo='none',
                        showlegend=False,
                        name=f'connection_{point_idx}_{neighbor_idx}'
                    )
                fig.add_trace(connection_trace)

    # Dodanie elementów wyszukiwania
    for trace in search_traces:
        fig.add_trace(trace)

    # Layout w zależności od trybu widoku
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
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(color="white")
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
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(color="white")
            )
        )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Wgraj pliki lub wpisz tekst (minimum 3 fragmenty), aby wygenerować mapę.")