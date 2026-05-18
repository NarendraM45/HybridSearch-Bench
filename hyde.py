import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

class HyDEQueryExpander:
    """
    Hypothetical Document Embeddings (Gao et al., 2022).
    """
    def __init__(self, llm, embedder, blend_alpha: float = 0.5, fallback_on_error: bool = True):
        self.llm = llm
        self.embedder = embedder
        self.blend_alpha = blend_alpha
        self.fallback_on_error = fallback_on_error
        
        self.prompt_template = (
            "Write a short paragraph that would answer the following question: {query}"
        )

    def expand(self, query: str) -> Tuple[str, list[float]]:
        """
        Returns (hypothetical_doc, blended_embedding_vector).
        """
        # 1. Get original embedding
        try:
            original_vector = self.embedder.embed([query])[0]
        except Exception as e:
            logger.error(f"Failed to embed original query: {e}")
            raise e

        # 2. Generate hypothetical doc
        hypothetical_doc = query
        try:
            # We assume llm is an Ollama client wrapper or ChatOllama that responds to a simple prompt
            # If it's a ChatOllama instance, we can invoke it.
            if hasattr(self.llm, "invoke"):
                from langchain_core.messages import HumanMessage
                res = self.llm.invoke([HumanMessage(content=self.prompt_template.format(query=query))])
                hypothetical_doc = res.content.strip()
            else:
                # Fallback to direct ollama client if llm is the module/client itself
                import ollama
                from settings import get_settings
                s = get_settings()
                client = ollama.Client(host=s.ollama_base_url)
                res = client.chat(
                    model=s.ollama_model,
                    messages=[{"role": "user", "content": self.prompt_template.format(query=query)}]
                )
                hypothetical_doc = res["message"]["content"].strip()
                
            logger.debug(f"HyDE generated doc: {hypothetical_doc[:100]}...")
        except Exception as e:
            logger.error(f"HyDE LLM generation failed: {e}")
            if self.fallback_on_error:
                return query, original_vector
            raise e

        # 3. Embed hypothetical doc
        try:
            hyde_vector = self.embedder.embed([hypothetical_doc])[0]
        except Exception as e:
            logger.error(f"HyDE embedding failed: {e}")
            if self.fallback_on_error:
                return query, original_vector
            raise e

        # 4. Blend embeddings
        v1 = np.array(original_vector)
        v2 = np.array(hyde_vector)
        
        # blend_alpha=1.0 -> pure HyDE. blend_alpha=0.0 -> pure original
        blended = (1.0 - self.blend_alpha) * v1 + self.blend_alpha * v2
        
        # normalize
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm
            
        return hypothetical_doc, blended.tolist()
