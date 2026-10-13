from pathlib import Path

def test_project_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app.py").exists()
    assert (root / "main.py").exists()
    assert (root / "requirements.txt").exists()
