import pytest
from chroma_store import ChromaStore
from chunker import ParentChildChunk

def test_parent_child_upsert_and_retrieve(monkeypatch):
    # Mock settings
    monkeypatch.setattr("settings.get_settings", lambda: type("Settings", (), {"chroma_persist_dir": "./test_db", "collection_name": "test"})())
    
    store = ChromaStore()
    # Using magic mock to avoid real chromadb init
    from unittest.mock import MagicMock
    store._client = MagicMock()
    
    mock_child_coll = MagicMock()
    mock_parent_coll = MagicMock()
    
    def mock_get_or_create(name, **kwargs):
        if "children" in name: return mock_child_coll
        if "parents" in name: return mock_parent_coll
    
    store._client.get_or_create_collection.side_effect = mock_get_or_create
    store._client.get_collection.side_effect = mock_get_or_create
    
    chunks = [
        ParentChildChunk(
            child_id="c1", parent_id="p1", 
            child_text="child1", parent_text="parent1", 
            source="test", metadata={"parent_id": "p1"}
        )
    ]
    
    store.upsert_parent_child(chunks)
    
    assert mock_parent_coll.upsert.called
    
    # Test query_children
    mock_child_coll.count.return_value = 1
    mock_child_coll.query.return_value = {
        "ids": [["c1"]],
        "documents": [["child1"]],
        "metadatas": [[{"parent_id": "p1"}]],
        "distances": [[0.1]]
    }
    
    res = store.query_children([1.0], top_k=1)
    assert len(res) == 1
    assert res[0].document.doc_id == "c1"
    
    # Test get_parents
    mock_parent_coll.get.return_value = {
        "ids": ["p1"],
        "documents": ["parent1"],
        "metadatas": [{"parent_id": "p1"}]
    }
    
    parents = store.get_parents(["p1"])
    assert len(parents) == 1
    assert parents[0].text == "parent1"
