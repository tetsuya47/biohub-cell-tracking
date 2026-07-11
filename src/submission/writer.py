"""Module for compiling and writing tracking lineage graphs to the official submission format."""

import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)


class SubmissionWriter:
    """Submission format writer.

    Accumulates track nodes and parent-child edges, aligning them and writing
    them out to the official CSV format required by the competition.
    """

    def __init__(self) -> None:
        """Initialize the SubmissionWriter."""
        logger.info("Initializing SubmissionWriter")
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, str]] = []

    def add_node(self, node_id: str, t: int, z: float, y: float, x: float) -> None:
        """Add a cell node prediction to the submission.

        Args:
            node_id: Unique string identifier for the node.
            t: Timepoint index.
            z: Depth coordinate.
            y: Height coordinate.
            x: Width coordinate.
        """
        logger.debug("Adding node %s at t=%d", node_id, t)
        self.nodes[node_id] = {
            "node_id": node_id,
            "parent_id": "-1",  # Default to no parent
            "t": t,
            "z": z,
            "y": y,
            "x": x,
        }

    def add_edge(self, parent_id: str, child_id: str) -> None:
        """Add a directed parent-child tracking edge to the lineage graph.

        Args:
            parent_id: Parent cell node ID.
            child_id: Child cell node ID.
        """
        logger.debug("Adding tracking edge: %s -> %s", parent_id, child_id)
        self.edges.append({"parent_id": parent_id, "child_id": child_id})

    def save_csv(self, path: Path) -> None:
        """Serialize the accumulated nodes and edges to the target CSV file.

        Args:
            path: Destination path on disk.
        """
        logger.info("Saving submission CSV to %s", path)

        # Apply parent-child edges to nodes
        for edge in self.edges:
            parent_id = edge["parent_id"]
            child_id = edge["child_id"]
            if child_id in self.nodes:
                self.nodes[child_id]["parent_id"] = parent_id
            else:
                logger.warning("Child node %s referenced in edge but not found in node list", child_id)

        # Convert to pandas DataFrame and save
        df = pd.DataFrame(list(self.nodes.values()))

        # Ensure column order matches competition specs: node_id, parent_id, t, z, y, x
        columns = ["node_id", "parent_id", "t", "z", "y", "x"]
        if not df.empty:
            df = df[columns]
        else:
            df = pd.DataFrame(columns=columns)

        df.to_csv(path, index=False)
