from pydantic import BaseModel, field_validator


class SegmentationSettings(BaseModel):
    diameter: float | None = None
    channel_cyto: int = 0
    channel_nuc: int = 0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    do_3d: bool = False
    anisotropy: float | None = None

    @field_validator("diameter", mode="before")
    @classmethod
    def _parse_diameter(cls, v):
        if v == "" or v is None:
            return None
        v = float(v)
        return v if v > 0 else None

    @field_validator("anisotropy", mode="before")
    @classmethod
    def _parse_anisotropy(cls, v):
        if v == "" or v is None:
            return None
        return float(v)
