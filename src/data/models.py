from dataclasses import dataclass

@dataclass
class Protein:
    subcellular_locations: []


@dataclass
class SubcellularLocations:
    mitochondia: bool
    nucleus: bool