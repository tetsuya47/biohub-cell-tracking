import tempfile
from pathlib import Path
import pandas as pd
from src.submission.writer import SubmissionWriter

def test_submission_writer() -> None:
    """
    Verifies that the SubmissionWriter correctly accumulates nodes and edges
    and serializes them to the correct CSV structure.
    """
    writer = SubmissionWriter()
    
    # Add a mock track of a cell dividing: Node 0 -> Node 1 & Node 2
    writer.add_node(node_id="n0", t=0, z=10.0, y=20.0, x=30.0)
    writer.add_node(node_id="n1", t=1, z=11.0, y=21.0, x=31.0)
    writer.add_node(node_id="n2", t=1, z=9.0,  y=19.0, x=29.0)
    
    writer.add_edge(parent_id="n0", child_id="n1")
    writer.add_edge(parent_id="n0", child_id="n2")

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "submission.csv"
        writer.save_csv(csv_path)

        assert csv_path.exists()
        
        # Read and check file content
        df = pd.read_csv(csv_path)
        
        # Expected columns: node_id, parent_id, t, z, y, x
        assert list(df.columns) == ["node_id", "parent_id", "t", "z", "y", "x"]
        assert len(df) == 3
        
        # Verify node properties
        assert "n0" in df["node_id"].values
        assert "n1" in df["node_id"].values
        assert "n2" in df["node_id"].values

        # Check edge parenting
        n1_row = df[df["node_id"] == "n1"].iloc[0]
        n2_row = df[df["node_id"] == "n2"].iloc[0]
        n0_row = df[df["node_id"] == "n0"].iloc[0]

        assert n1_row["parent_id"] == "n0"
        assert n2_row["parent_id"] == "n0"
        assert n0_row["parent_id"] == "-1"  # No parent
