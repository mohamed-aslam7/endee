import streamlit as st
import requests
import PyPDF2
from io import BytesIO
from sentence_transformers import SentenceTransformer
import json
import time
import numpy as np

class RealEndeeClient:
    """Real Endee Vector Database Client"""
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_collection(self, name, dimension, metric="cosine"):
        """Create a collection in Endee"""
        payload = {
            "name": name,
            "dimension": dimension,
            "metric": metric
        }
        try:
            response = self.session.post(f"{self.base_url}/collections", json=payload)
            return response.status_code == 200
        except requests.ConnectionError:
            st.error("❌ Cannot connect to Endee server. Using fallback mode.")
            return False
    
    def insert_vectors(self, collection_name, vectors_data):
        """Insert multiple vectors into collection"""
        payload = {
            "collection": collection_name,
            "vectors": vectors_data
        }
        try:
            response = self.session.post(f"{self.base_url}/vectors", json=payload)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
    
    def search_vectors(self, collection_name, query_vector, top_k=5):
        """Search for similar vectors"""
        payload = {
            "collection": collection_name,
            "vector": query_vector,
            "top_k": top_k
        }
        try:
            response = self.session.post(f"{self.base_url}/search", json=payload)
            if response.status_code == 200:
                return response.json()
            return []
        except requests.ConnectionError:
            return []
    
    def get_collection_info(self, collection_name):
        """Get collection statistics"""
        try:
            response = self.session.get(f"{self.base_url}/collections/{collection_name}")
            if response.status_code == 200:
                return response.json()
            return None
        except requests.ConnectionError:
            return None

class FallbackMockEndee:
    """Fallback when real Endee is not available"""
    def __init__(self):
        self.collections = {}
        self.vectors = {}
        st.warning("⚠️ Using fallback mode - Real Endee server not available")
    
    def create_collection(self, name, dimension, metric="cosine"):
        self.collections[name] = {"dimension": dimension, "count": 0, "metric": metric}
        self.vectors[name] = []
        return True
    
    def insert_vectors(self, collection_name, vectors_data):
        if collection_name not in self.vectors:
            return False
        self.vectors[collection_name].extend(vectors_data)
        self.collections[collection_name]["count"] += len(vectors_data)
        return True
    
    def search_vectors(self, collection_name, query_vector, top_k=5):
        if collection_name not in self.vectors:
            return []
        
        results = []
        query_vec = np.array(query_vector)
        
        for item in self.vectors[collection_name]:
            item_vec = np.array(item["vector"])
            similarity = np.dot(query_vec, item_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(item_vec))
            results.append({
                "id": item["id"],
                "score": float(similarity),
                "metadata": item["metadata"]
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_collection_info(self, collection_name):
        if collection_name in self.collections:
            return self.collections[collection_name]
        return None

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource  
def init_endee_client():
    """Initialize Endee client with fallback"""
    client = RealEndeeClient()
    
    # Test connection
    try:
        test_response = requests.get("http://localhost:8080/health", timeout=2)
        if test_response.status_code == 200:
            st.success("✅ Connected to Real Endee Database!")
            return client
    except:
        pass
    
    # Fallback to mock
    return FallbackMockEndee()

def process_pdf(uploaded_file):
    """Extract and chunk PDF content"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    # Create chunks
    words = text.split()
    chunk_size = 150  # Smaller chunks for better precision
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
    
    return chunks

def main():
    st.set_page_config(
        page_title="Real Endee Document Q&A",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Real Endee Vector Database - Document Q&A")
    st.write("**Powered by Production-Grade Endee Vector Database**")
    
    # Initialize
    encoder = load_model()
    endee_client = init_endee_client()
    collection_name = "documents"
    
    # Show connection status
    col1, col2 = st.columns([3, 1])
    with col1:
        if isinstance(endee_client, RealEndeeClient):
            st.success("🔗 **Status**: Connected to Real Endee Server")
        else:
            st.warning("🔄 **Status**: Using Fallback Mode (Mock)")
    
    with col2:
        if st.button("🔄 Reconnect"):
            st.cache_resource.clear()
            st.rerun()
    
    # Create collection
    if endee_client.create_collection(collection_name, 384):
        st.info("✅ Collection ready")
    
    # Show collection stats
    stats = endee_client.get_collection_info(collection_name)
    if stats:
        st.sidebar.write("**📊 Database Stats:**")
        st.sidebar.write(f"Dimension: {stats.get('dimension', 384)}")
        st.sidebar.write(f"Vectors: {stats.get('count', 0)}")
        st.sidebar.write(f"Metric: {stats.get('metric', 'cosine')}")
    
    # File upload
    st.header("1. 📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing with Endee..."):
            chunks = process_pdf(uploaded_file)
            
            # Prepare vectors for batch insertion
            vectors_data = []
            progress_bar = st.progress(0)
            
            for i, chunk in enumerate(chunks):
                vector = encoder.encode(chunk).tolist()
                vectors_data.append({
                    "id": f"{uploaded_file.name}_{i}",
                    "vector": vector,
                    "metadata": {
                        "text": chunk,
                        "source": uploaded_file.name,
                        "chunk_id": i
                    }
                })
                progress_bar.progress((i + 1) / len(chunks))
            
            # Insert vectors in batch (more efficient)
            if endee_client.insert_vectors(collection_name, vectors_data):
                st.success(f"✅ Stored {len(vectors_data)} vectors in Endee!")
            else:
                st.error("❌ Failed to store vectors")
    
    # Query section
    st.header("2. 🔍 Ask Questions")
    question = st.text_input("Enter your question:")
    
    if question and st.button("🚀 Search with Endee", type="primary"):
        with st.spinner("Searching Endee database..."):
            start_time = time.time()
            
            # Encode question
            question_vector = encoder.encode(question).tolist()
            
            # Search using Endee
            results = endee_client.search_vectors(collection_name, question_vector, top_k=3)
            
            search_time = time.time() - start_time
            
            if results:
                st.write("### 💡 Results:")
                st.write(f"*Search completed in {search_time:.3f} seconds*")
                
                for i, result in enumerate(results):
                    with st.expander(f"Result {i+1} - Score: {result.get('score', 0):.3f}"):
                        if 'metadata' in result:
                            st.write(result['metadata'].get('text', 'No text available'))
                            st.write(f"**Source:** {result['metadata'].get('source', 'Unknown')}")
            else:
                st.info("No results found. Upload a document first or try a different question.")
    
    # Performance comparison
    with st.expander("🔬 Why Real Endee vs Mock?"):
        st.write("""
        **Real Endee Advantages:**
        
        🚀 **Performance**: 10-1000x faster search
        💾 **Persistence**: Data survives restarts  
        📈 **Scalability**: Handle millions of vectors
        🔒 **Production Ready**: Thread-safe, robust
        ⚡ **Advanced Algorithms**: HNSW, IVF indexing
        🌐 **Multi-user**: Concurrent access support
        
        **Mock Implementation Limitations:**
        
        🐌 **Slow**: Python loops for search
        💨 **Volatile**: Lost on restart
        📉 **Limited Scale**: Memory constraints
        🧪 **Demo Only**: Not production ready
        """)

if __name__ == "__main__":
    main()