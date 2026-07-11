# Sprint 7 Architecture Review

---

# 1. Competition Goal

The objective of this project is to build an end-to-end AI pipeline capable of tracking cells over time from 3D microscopy images.

The final system must:

- detect every visible cell
- assign a persistent identity to every cell
- follow each cell through every timepoint
- detect cell division events
- reconstruct the lineage graph
- generate the official competition submission format

The project is designed for correctness, reproducibility, modularity and maintainability.

---

# 2. Input Format

Each training sample contains one Zarr dataset.

Image dimensions are

(T, Z, Y, X)

where

T = time

Z = depth

Y = height

X = width

Example

(100,64,256,256)

Voxel spacing is stored inside zarr metadata.

Images are grayscale uint16 microscopy volumes.

---

# 3. Ground Truth Format

Ground truth is stored inside .geff datasets.

A .geff dataset contains

Nodes

representing individual cells.

Each node contains

- cell id
- t
- z
- y
- x

Edges connect parent and child nodes across time.

The graph represents the complete cell lineage.

---

# 4. Output Format

The final prediction is NOT a segmentation mask.

The final output is a lineage graph describing

cell identity

cell position

cell lifetime

cell division

parent-child relationships

This graph will later be converted into the official submission format.

---

# 5. Evaluation Metric

The official evaluation metric is defined by the Biohub Cell Tracking Challenge documentation.

This project MUST optimize the official competition evaluation protocol rather than generic computer vision metrics.

Architecture decisions must always prioritize compatibility with the official evaluation methodology.

The following image segmentation metrics are NOT project objectives by themselves:

- Dice
- IoU
- Pixel Accuracy

Detection, Tracking, Cell Division and Lineage Reconstruction must always be designed according to the official competition evaluation protocol.

---

# 6. Detection Task

Input

3D microscopy volume

Output

Cell center coordinates

Detection is treated as the first stage of the pipeline.

No tracking is performed here.

---

# 7. Tracking Task

Tracking associates detections belonging to the same biological cell over multiple timepoints.

Input

Detected cells

Output

Cell trajectories

Tracking preserves cell identity across time.

---

# 8. Cell Division Task

When one biological cell divides

Parent

↓

Child A

Child B

The model must detect the event and construct the correct lineage relationship.

---

# 9. Planned Baseline Architecture

Stage 1

Dataset Reader

↓

Stage 2

Transforms

↓

Stage 3

Torch Dataset

↓

Stage 4

DataLoader

↓

Stage 5

Detection Model

↓

Stage 6

Tracking

↓

Stage 7

Division Detection

↓

Stage 8

Submission Writer

Every stage must remain independent and replaceable.

---

# 10. Sprint 8 Implementation Plan

Sprint 8 will implement the first Detection Baseline.

Only the detection model will be developed.

Tracking

Division

Submission

will NOT be implemented during Sprint 8.

Sprint 8 deliverables

src/models/detection/

tests/

notebooks/

The objective is obtaining the first working detection model.
