# RAG System Architecture Diagram

## Complete System Flow

```mermaid
graph TB
    subgraph "DOCUMENT INGESTION"
        A[📄 Alice Uploads Documents] --> B[🔐 AltaStata Encrypted Storage]
        B --> C[🔔 SHARE Event Triggered]
        C --> D[👂 Bob Event Listener]
    end
    
    subgraph "INDEXING PIPELINE"
        D --> E[📖 Read Document from AltaStata]
        E --> F[✂️ Chunk Document with LangChain]
        F --> G[🧠 Generate Embeddings]
        G --> H[💾 Store in Vertex AI Index]
        H --> I[🏷️ Encode Metadata in Datapoint ID]
    end
    
    subgraph "QUERY PROCESSING"
        J[❓ User Query] --> K[🧠 Query Embedding Generation]
        K --> L[🔍 Vertex AI Vector Search]
        L --> M[📊 Apply Similarity Threshold]
        M --> N[🎯 Retrieve Relevant Chunks]
        N --> O[📋 Extract Metadata from Datapoint ID]
        O --> P[📖 Read Full Document via AltaStata]
        P --> Q[✂️ Re-chunk Document on Demand]
        Q --> R[📄 Extract Specific Chunk]
        R --> S[📝 Build Context for LLM]
        S --> T[🤖 Generate Response with Gemini 2.5 Flash]
    end
    
    subgraph "LANGCHAIN INTEGRATION"
        U[📚 Document Loader] --> V[✂️ Text Splitter]
        V --> W[🧠 Embeddings Generation]
        W --> X[💾 Vector Store Operations]
    end
    
    subgraph "VERTEX AI VECTOR SEARCH"
        Y[📊 Vector Index] --> Z[🔍 Similarity Search]
        Z --> AA[🏷️ Metadata in Datapoint ID]
        AA --> BB[🚀 Endpoint Deployment]
    end
    
    subgraph "ALTASTATA INTEGRATION"
        CC[📁 fsspec Filesystem Interface] --> DD[🔐 Encrypted File Operations]
        DD --> EE[⚡ Event-Driven Architecture]
        EE --> FF[🔔 Real-time Notifications]
    end
    
    %% Data Flow Connections
    B --> E
    E --> P
    H --> L
    N --> O
    O --> P
    P --> Q
    Q --> R
    R --> S
    S --> T
    
    %% Integration Connections
    F --> V
    G --> W
    H --> X
    L --> Z
    E --> CC
    P --> CC
    
    %% Styling
    classDef ingestion fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef indexing fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef query fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef langchain fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef vertex fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef altastata fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    
    class A,B,C,D ingestion
    class E,F,G,H,I indexing
    class J,K,L,M,N,O,P,Q,R,S,T query
    class U,V,W,X langchain
    class Y,Z,AA,BB vertex
    class CC,DD,EE,FF altastata
```

## Detailed Component Flow

```mermaid
sequenceDiagram
    participant User
    participant Alice
    participant AltaStata
    participant Bob
    participant VertexAI
    participant Gemini
    
    Note over Alice,AltaStata: Document Ingestion
    Alice->>AltaStata: Upload Documents
    AltaStata->>AltaStata: Encrypt & Store
    AltaStata->>Bob: SHARE Event
    
    Note over Bob,VertexAI: Indexing Pipeline
    Bob->>AltaStata: Read Document
    Bob->>Bob: Chunk with LangChain
    Bob->>VertexAI: Generate Embeddings
    Bob->>VertexAI: Store in Index
    Bob->>VertexAI: Encode Metadata in Datapoint ID
    
    Note over User,Gemini: Query Processing
    User->>Bob: Submit Query
    Bob->>VertexAI: Generate Query Embedding
    Bob->>VertexAI: Vector Search
    Bob->>Bob: Apply Similarity Threshold
    Bob->>AltaStata: Read Full Documents
    Bob->>Bob: Re-chunk on Demand
    Bob->>Gemini: Generate Response
    Gemini->>User: Return Answer
```

## Key Architecture Features

### 🔐 **Secure Document Storage**
- Documents stored encrypted in AltaStata
- No local metadata files needed
- Real-time event-driven processing

### 🚀 **On-Demand Retrieval**
- Full documents retrieved via fsspec when needed
- Re-chunking on demand for exact content extraction
- Metadata encoded in Vertex AI datapoint IDs

### 🎯 **Intelligent Search**
- Similarity threshold filtering (0.6)
- Vector similarity search in Vertex AI
- Context-aware document retrieval

### ⚡ **Performance Optimized**
- No large local storage requirements
- Scalable cloud-native architecture
- Event-driven real-time processing

### 🔄 **LangChain Integration**
- Native support for document loaders
- Standard text splitting and embeddings
- Compatible with existing LangChain workflows
