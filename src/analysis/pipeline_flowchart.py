from graphviz import Digraph

# Create directed graph
dot = Digraph(comment="JWST Clustering Pipeline", format="png")
dot.attr(rankdir="LR", size="8,5")

# Define nodes
dot.node("A", "Data Ingestion\n(ASTRODEEP FITS)", shape="box")
dot.node("B", "Preprocessing & QA\n(zphot, tier slicing)", shape="box")
dot.node("C", "Comoving Mapping\n(RA,Dec,z → X,Y,Z)", shape="box")
dot.node("D", "Pair Counts\n(DD, DR, RR)", shape="box")
dot.node("E", "Correlation Function ξ(d)\n(Landy–Szalay)", shape="box")
dot.node("F", "Fourier Analysis\n(P(k), periodicity)", shape="box")
dot.node("G", "Anisotropy / Arc Scans\n(2D FFT, ring tests)", shape="box")
dot.node("H", "Inter-field Variance\n(mock comparisons)", shape="box")
dot.node("I", "Quality Cuts & Depth Normalization\n(SNR≥10, de-starring, photo-z QC)", shape="box")
dot.node("J", "Results & Interpretation\n(overdensities, proto-clusters?)", shape="ellipse")

# Define edges (flow)
dot.edges([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
dot.edges([("E", "F"), ("E", "G"), ("E", "H")])
dot.edge("H", "I")
dot.edge("I", "J")
dot.edge("F", "J")
dot.edge("G", "J")

# Render
output_path = "/mnt/data/jwst_clustering_pipeline"
dot.render(output_path, format="png", cleanup=True)
output_path + ".png"
