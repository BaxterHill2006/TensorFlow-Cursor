"""Single place to choose which dataset training uses.

Change ACTIVE_DATASET to switch loaders without editing the rest of the pipeline.
"""

# One of: "cifar10" | "image_directory"
ACTIVE_DATASET = "cifar10"
