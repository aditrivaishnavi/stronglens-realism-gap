# Training expects 64x64 images, but pipeline generates 101x101 cutouts
# Center crop is applied during preprocessing (see preprocess.py)
STAMP_SIZE = 64
CUTOUT_SIZE = 101  # Size from Legacy Survey cutout generation

BANDS = ("g", "r", "z")
CORE_BOX = 10
CORE_RADIUS = 5
SEED_DEFAULT = 1337
