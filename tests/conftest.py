import sys
import os
import pytest

workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from utils import FAISSVectorStore, StructureAwareChunker, EnterpriseEmbeddingEngine
from security import SecurityManager
from fastapi.testclient import TestClient
from api import app

@pytest.fixture(scope="session")
def vector_store():
    vs = FAISSVectorStore()
    sample_doc = {
        "id": 1,
        "source_type": "Network Configuration",
        "title": "cisco_switch.cfg",
        "pages": [{"page": 1, "content": "interface GigabitEthernet0/1\n switchport access vlan 10\n shutdown"}],
        "content": "interface GigabitEthernet0/1\n switchport access vlan 10\n shutdown"
    }
    vs.build_index([sample_doc])
    return vs

@pytest.fixture(scope="session")
def chunker():
    return StructureAwareChunker()

@pytest.fixture(scope="session")
def api_client():
    return TestClient(app)

@pytest.fixture(scope="session")
def auth_header():
    token = SecurityManager.create_access_token({"username": "test_admin", "role": "IT_Operator"})
    return {"Authorization": f"Bearer {token}"}
