from enum import Enum
# from nerfstudio.field_components.field_heads import FieldHeadNames

class DepthLERFFieldHeadNames(Enum):
    """Possible field outputs"""
    HASHGRID = "hashgrid"
    CLIP = "clip"
    DINO = "dino"