import pytest
from utils import StructureAwareChunker, RecursiveCharacterTextSplitter, detect_document_category

def test_recursive_character_splitter():
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    text = "Line 1 paragraph.\n\nLine 2 paragraph.\n\nLine 3 paragraph."
    splits = splitter.split_text(text)
    assert isinstance(splits, list)
    assert len(splits) > 0

def test_structure_aware_chunker_network_config(chunker):
    doc = {
        "id": 1,
        "title": "cisco_router.cfg",
        "source_type": "Network Configuration",
        "content": "interface GigabitEthernet0/1\n description Primary Uplink\n ip address 192.168.1.1 255.255.255.0\n!\nrouter bgp 65000\n bgp log-neighbor-changes\n"
    }
    chunks = chunker.chunk_document(doc)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert "filename" in chunks[0]
    assert "page" in chunks[0]
    assert "section" in chunks[0]
    assert "chunk_id" in chunks[0]

def test_zero_shot_document_categorization():
    cat1 = detect_document_category("switch.cfg", "interface GigabitEthernet0/1\n switchport mode access")
    assert isinstance(cat1, str)
    assert len(cat1) > 0
