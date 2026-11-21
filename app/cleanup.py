import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services import eval_store

eval_store.clear_evaluations(session_id="test_session_ragas")
print("Cleaned test session records from SQLite.")

if os.path.exists("app/test_ragas.py"):
    os.remove("app/test_ragas.py")
    print("Removed app/test_ragas.py")

if os.path.exists("app/test_ragas_e2e.py"):
    os.remove("app/test_ragas_e2e.py")
    print("Removed app/test_ragas_e2e.py")
