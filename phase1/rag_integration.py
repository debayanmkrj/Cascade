"""RAG Integration - Technical Implementation Extraction"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import requests
import json
import re
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from pathlib import Path

from config import GRAPH_FILE, EMBEDDING_MODEL, TOP_K_UI_NODES, TOP_K_NODE_ARCHETYPES, OLLAMA_URL, MODEL_NAME_REASONING


class UINodeLibrary:
    """RAG-based retrieval for technical implementation details"""

    def __init__(self, graph_file: Path = GRAPH_FILE):
        print(f"Initializing UINodeLibrary...")
        self.graph_file = graph_file
        self.graph = None
        self.chunks_index = {}
        self.concepts_index = {}
        self.st_model: Optional[SentenceTransformer] = None

        if graph_file.exists():
            with open(graph_file, 'rb') as f:
                data = pickle.load(f)
                self.graph = data['graph']
                self.chunks_index = data['chunks_index']
                self.concepts_index = data['concepts_index']
            print(f"  Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        else:
            print(f"  Warning: Graph file not found at {graph_file}")

    def _ensure_embeddings(self):
        """Lazy load sentence transformer"""
        if self.st_model is None and self.graph is not None:
            print("  Loading sentence transformer for RAG...")
            self.st_model = SentenceTransformer(EMBEDDING_MODEL)

    def retrieve_chunks(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant chunks from knowledge graph via semantic similarity"""
        if not self.graph:
            return []

        self._ensure_embeddings()
        query_emb = self.st_model.encode([query])[0]

        retrieved = []
        for chunk_id, chunk in self.chunks_index.items():
            chunk_text = chunk.get('content', '')[:500]
            if not chunk_text.strip():
                continue

            try:
                chunk_emb = self.st_model.encode([chunk_text])[0]
                similarity = float(np.dot(query_emb, chunk_emb) /
                                 (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-8))
            except Exception:
                continue

            retrieved.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'similarity': similarity,
                'header': chunk.get('header', ''),
                'source': chunk.get('source', '')
            })

        retrieved.sort(key=lambda x: x['similarity'], reverse=True)
        return retrieved[:top_k]

    def _assemble_context(self, chunks: List[Dict], max_length: int = 3000) -> str:
        """Assemble retrieved chunks into context string"""
        if not chunks:
            return "No relevant information found."

        parts = []
        current_len = 0

        for i, chunk in enumerate(chunks, 1):
            entry = f"[Source {i}: {chunk.get('source', 'Unknown')}]\n{chunk.get('text', '')}\n"
            if current_len + len(entry) > max_length:
                break
            parts.append(entry)
            current_len += len(entry)

        return "\n".join(parts)

    def retrieve_implementation_details(self, query: str, top_k: int = TOP_K_NODE_ARCHETYPES) -> Dict:
        """
        RAG pipeline: Retrieve context then ask LLM to extract technical implementation details.
        Returns tech stack for Shader/Taichi coding.
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retrieve_chunks(query, top_k)

        # Step 2: Assemble context
        context = self._assemble_context(chunks, max_length=3000)

        # Step 3: Ask LLM to extract pure coding logic
        prompt = f"""[INST] You are a Graphics Engineer.
Extract the technical implementation details for "{query}" from the context below.

CONTEXT:
{context}

TASK:
Identify the specific MATH, ALGORITHMS, and LOGIC needed to code this in a Shader.
- Math: (e.g. "Use dot product for lighting", "Use sin(time) for pulse")
- Logic: (e.g. "Advect velocity first, then diffuse")
- Functions: (e.g. "Use smoothstep for edges")

OUTPUT:
Return a concise, bulleted "Tech Stack" list. No fluff.
[/INST]
"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    'model': MODEL_NAME_REASONING,
                    'prompt': prompt,
                    'stream': False,
                    'temperature': 0.3,
                    'options': {'num_predict': 500}
                },
                timeout=60
            )

            if response.status_code == 200:
                result_text = response.json().get('response', '')
                result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL)
                result_text = result_text.strip()

                return {
                    'implementation_hint': result_text,
                    'retrieved_chunks': chunks,
                    'num_sources': len(chunks)
                }

        except Exception as e:
            print(f"  RAG LLM error: {e}")

        return {
            'implementation_hint': 'Standard shader logic',
            'retrieved_chunks': chunks,
            'num_sources': len(chunks)
        }

    def retrieve_ui_concepts(self, query: str, brand_values: Dict, top_k: int = TOP_K_UI_NODES) -> List[Dict]:
        """Retrieve UI control concepts from knowledge graph"""
        chunks = self.retrieve_chunks(f"{query} UI controls interaction design", top_k)

        ui_concepts = []
        for chunk in chunks:
            ui_concepts.append({
                'concept': chunk.get('header', 'control'),
                'text': chunk.get('text', ''),
                'grounding_score': chunk.get('similarity', 0.5),
                'source': chunk.get('source', 'Knowledge Base')
            })

        return ui_concepts

    def retrieve_node_archetypes(self, query: str, brand_values: Dict, top_k: int = TOP_K_NODE_ARCHETYPES) -> List[Dict]:
        """Retrieve node archetype concepts from knowledge graph using pure semantic similarity"""

        # Use LLM to expand query for better retrieval (no hardcoded keywords)
        expanded_query = self._expand_query_via_llm(query)

        # Pure RAG retrieval - no keyword boosting, no hardcoded rules
        chunks = self.retrieve_chunks(expanded_query, top_k)

        node_concepts = []
        for chunk in chunks:
            node_concepts.append({
                'concept': chunk.get('header', 'node'),
                'text': chunk.get('text', ''),
                'grounding_score': chunk.get('similarity', 0.5),
                'source': chunk.get('source', 'Knowledge Base')
            })

        print(f"  [RAG] Retrieved {len(node_concepts)} node archetypes via semantic similarity")
        return node_concepts

    def _expand_query_via_llm(self, query: str) -> str:
        """Use LLM to deeply analyze query and expand into comprehensive search terms"""
        try:
            prompt = f"""You are a visual effects technical expert. Analyze this creative request:

REQUEST: "{query}"

DEEP ANALYSIS - Think about:
1. VISUAL ELEMENTS: What's being created? (particles, meshes, shapes, fields, etc.)
2. TECHNICAL SYSTEMS: What components are needed? (emitters, renderers, generators, operators)
3. ALGORITHMS: What math/logic? (physics, noise, simulation, procedural, forces)
4. MOTION QUALITIES: How does it move? (organic, fluid, rigid, chaotic, smooth, flowing)
5. AESTHETIC PROPERTIES: How does it look? (neon, glow, dark, bright, colorful, minimal)

GENERATE KEYWORDS: Comprehensive space-separated technical search terms (30-50 words).

EXAMPLES:
Input: "particle system"
Output: particle emitter spawn generator physics simulation update integrate render points sprites buffer geometry instance three_js webgl graphics

Input: "flowing organic motion"
Output: fluid flow field vector velocity curl noise turbulence swirl drift advection natural smooth organic brownian perlin simplex displacement

Input: "neon colors glowing"
Output: neon vibrant saturated glow bloom luminance bright hdr color grade hue shift post process effect light emission

Input: "glitch distortion"
Output: glitch distortion aberration chromatic displace corrupt digital artifact noise pixelate rgb split offset feedback

YOUR TURN - Be thorough and technical:
Request: "{query}"
Keywords:"""

            response = requests.post(
                OLLAMA_URL,
                json={
                    'model': MODEL_NAME_REASONING,
                    'prompt': prompt,
                    'stream': False,
                    'temperature': 0.5,  # Higher for comprehensive expansion
                    'options': {'num_predict': 150}
                },
                timeout=20
            )

            if response.status_code == 200:
                expanded = response.json().get('response', '').strip()
                # Clean - take first line only (LLM might add explanation)
                expanded = expanded.split('\n')[0].strip()
                # Remove markdown/quotes
                expanded = expanded.replace('`', '').replace('"', '').replace("'", '')
                # Combine original + expanded
                combined = f"{query} {expanded}"
                keyword_count = len(expanded.split())
                print(f"  [RAG] Query expansion: {keyword_count} keywords generated")
                return combined
        except Exception as e:
            print(f"  [RAG] Query expansion failed: {e}")

        # Fallback to original query
        return query

    def query_node_archetypes_with_generation(self, query: str, brand_values: Dict,
                                              top_k: int = TOP_K_NODE_ARCHETYPES) -> Dict:
        """Full RAG pipeline for node archetypes with LLM generation"""
        result = self.retrieve_implementation_details(query, top_k)
        result['recommendations'] = []
        return result

    def query_ui_controls_with_generation(self, query: str, brand_values: Dict,
                                          top_k: int = TOP_K_UI_NODES) -> Dict:
        """Full RAG pipeline for UI controls"""
        chunks = self.retrieve_ui_concepts(query, brand_values, top_k)
        return {
            'recommendations': [],
            'retrieved_chunks': chunks,
            'num_sources': len(chunks)
        }
