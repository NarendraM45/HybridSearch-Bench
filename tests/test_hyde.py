import pytest
import numpy as np
from hyde import HyDEQueryExpander
from unittest.mock import MagicMock

def test_hyde_blend_math():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "Hypothetical Doc"
    
    mock_embedder = MagicMock()
    # Original query embedding [1.0, 0.0]
    # Hyp doc embedding [0.0, 1.0]
    mock_embedder.embed.side_effect = [[[1.0, 0.0]], [[0.0, 1.0]]]
    
    hyde = HyDEQueryExpander(mock_llm, mock_embedder, blend_alpha=0.5)
    doc, vec = hyde.expand("Query")
    
    assert doc == "Hypothetical Doc"
    # Blend should be [0.5, 0.5], normalized -> [0.707, 0.707]
    assert np.isclose(vec[0], 0.707106)
    assert np.isclose(vec[1], 0.707106)

def test_hyde_fallback():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("LLM Error")
    
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[1.0, 0.0]]
    
    hyde = HyDEQueryExpander(mock_llm, mock_embedder, fallback_on_error=True)
    doc, vec = hyde.expand("Query")
    
    assert doc == "Query"  # fell back to query
    assert vec == [1.0, 0.0]
