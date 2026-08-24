from src.edge_detection import EDGE_ALGORITHMS, compare_edges, detect_edges
def test_every_edge_detector(rgb_image):
    for name in EDGE_ALGORITHMS:
        edges, metrics = detect_edges(rgb_image, name); assert edges.ndim == 2 and metrics["edge_pixels"] >= 0
def test_comparison(rgb_image):
    images, rows = compare_edges(rgb_image); assert len(images) == len(rows) == 5
