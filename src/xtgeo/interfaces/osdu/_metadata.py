# -*- coding: utf-8 -*-
"""OSDU metadata mapping between RESQML property kinds and Eclipse/OSDU identifiers.

Adapted and refactored from tmp/ecletp/osdu_mapping.py to be a clean, reusable module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class OsduPropertyMapping:
    """Mapping result for a RESQML property to OSDU/Eclipse identity."""

    ecl_keyword: str
    osdu_reference: Optional[str] = None
    osdu_name: str = ""
    uom_family: str = ""
    is_discrete: bool = False


# ---------------------------------------------------------------------------
# Canonical OSDU property name types
# ---------------------------------------------------------------------------
_OSDU_PROPERTIES: Dict[str, OsduPropertyMapping] = {
    # --- Petrophysics (continuous, fraction) ---
    "POROSITY": OsduPropertyMapping(
        ecl_keyword="PORO",
        osdu_reference="osdu:reference-data--PropertyNameType:Porosity:1.0.0",
        osdu_name="Porosity",
        uom_family="fraction",
    ),
    "NET_TO_GROSS_RATIO": OsduPropertyMapping(
        ecl_keyword="NTG",
        osdu_reference="osdu:reference-data--PropertyNameType:NetToGrossRatio:1.0.0",
        osdu_name="Net to Gross Ratio",
        uom_family="fraction",
    ),
    "PERMEABILITY_X": OsduPropertyMapping(
        ecl_keyword="PERMX",
        osdu_reference="osdu:reference-data--PropertyNameType:PermeabilityX:1.0.0",
        osdu_name="Permeability X",
        uom_family="mD",
    ),
    "PERMEABILITY_Y": OsduPropertyMapping(
        ecl_keyword="PERMY",
        osdu_reference="osdu:reference-data--PropertyNameType:PermeabilityY:1.0.0",
        osdu_name="Permeability Y",
        uom_family="mD",
    ),
    "PERMEABILITY_Z": OsduPropertyMapping(
        ecl_keyword="PERMZ",
        osdu_reference="osdu:reference-data--PropertyNameType:PermeabilityZ:1.0.0",
        osdu_name="Permeability Z",
        uom_family="mD",
    ),
    # --- Saturations (continuous, fraction) ---
    "WATER_SATURATION": OsduPropertyMapping(
        ecl_keyword="SWAT",
        osdu_reference="osdu:reference-data--PropertyNameType:WaterSaturation:1.0.0",
        osdu_name="Water Saturation",
        uom_family="fraction",
    ),
    "OIL_SATURATION": OsduPropertyMapping(
        ecl_keyword="SOIL",
        osdu_reference="osdu:reference-data--PropertyNameType:OilSaturation:1.0.0",
        osdu_name="Oil Saturation",
        uom_family="fraction",
    ),
    "GAS_SATURATION": OsduPropertyMapping(
        ecl_keyword="SGAS",
        osdu_reference="osdu:reference-data--PropertyNameType:GasSaturation:1.0.0",
        osdu_name="Gas Saturation",
        uom_family="fraction",
    ),
    "CRITICAL_WATER_SATURATION": OsduPropertyMapping(
        ecl_keyword="SWCR",
        osdu_reference="osdu:reference-data--PropertyNameType:CriticalWaterSaturation:1.0.0",
        osdu_name="Critical Water Saturation",
        uom_family="fraction",
    ),
    "INITIAL_WATER_SATURATION": OsduPropertyMapping(
        ecl_keyword="SWINIT",
        osdu_reference="osdu:reference-data--PropertyNameType:InitialWaterSaturation:1.0.0",
        osdu_name="Initial Water Saturation",
        uom_family="fraction",
    ),
    "CONNATE_WATER_SATURATION": OsduPropertyMapping(
        ecl_keyword="SWL",
        osdu_reference="osdu:reference-data--PropertyNameType:ConnateWaterSaturation:1.0.0",
        osdu_name="Connate Water Saturation",
        uom_family="fraction",
    ),
    "CRITICAL_OIL_IN_WATER_SATURATION": OsduPropertyMapping(
        ecl_keyword="SOWCR",
        osdu_reference="osdu:reference-data--PropertyNameType:CriticalOilInWaterSaturation:1.0.0",
        osdu_name="Critical Oil In Water Saturation",
        uom_family="fraction",
    ),
    "CRITICAL_GAS_SATURATION": OsduPropertyMapping(
        ecl_keyword="SGCR",
        osdu_reference="osdu:reference-data--PropertyNameType:CriticalGasSaturation:1.0.0",
        osdu_name="Critical Gas Saturation",
        uom_family="fraction",
    ),
    "MAXIMUM_GAS_SATURATION": OsduPropertyMapping(
        ecl_keyword="SGU",
        osdu_reference="osdu:reference-data--PropertyNameType:MaximumGasSaturation:1.0.0",
        osdu_name="Maximum Gas Saturation",
        uom_family="fraction",
    ),
    # --- Pressure and PVT (continuous) ---
    "PRESSURE": OsduPropertyMapping(
        ecl_keyword="PRESSURE",
        osdu_reference="osdu:reference-data--PropertyNameType:Pressure:1.0.0",
        osdu_name="Pressure",
        uom_family="bar",
    ),
    "INITIAL_PRESSURE": OsduPropertyMapping(
        ecl_keyword="PINIT",
        osdu_reference="osdu:reference-data--PropertyNameType:InitialPressure:1.0.0",
        osdu_name="Initial Pressure",
        uom_family="bar",
    ),
    "SOLUTION_GAS_OIL_RATIO": OsduPropertyMapping(
        ecl_keyword="RS",
        osdu_reference="osdu:reference-data--PropertyNameType:SolutionGasOilRatio:1.0.0",
        osdu_name="Solution Gas-Oil Ratio",
        uom_family="sm3/sm3",
    ),
    "VAPORIZED_OIL_GAS_RATIO": OsduPropertyMapping(
        ecl_keyword="RV",
        osdu_reference="osdu:reference-data--PropertyNameType:VaporizedOilGasRatio:1.0.0",
        osdu_name="Vaporized Oil-Gas Ratio",
        uom_family="sm3/sm3",
    ),
    "OIL_VISCOSITY": OsduPropertyMapping(
        ecl_keyword="OILVISCOSITY",
        osdu_reference="osdu:reference-data--PropertyNameType:OilViscosity:1.0.0",
        osdu_name="Oil Viscosity",
        uom_family="cP",
    ),
    "OIL_DENSITY": OsduPropertyMapping(
        ecl_keyword="OILDENSITY",
        osdu_reference="osdu:reference-data--PropertyNameType:OilDensity:1.0.0",
        osdu_name="Oil Density",
        uom_family="kg/m3",
    ),
    "OIL_FORMATION_VOLUME_FACTOR": OsduPropertyMapping(
        ecl_keyword="BO",
        osdu_reference="osdu:reference-data--PropertyNameType:OilFormationVolumeFactor:1.0.0",
        osdu_name="Oil Formation Volume Factor",
        uom_family="rm3/sm3",
    ),
    "TEMPERATURE": OsduPropertyMapping(
        ecl_keyword="TEMP",
        osdu_reference="osdu:reference-data--PropertyNameType:Temperature:1.0.0",
        osdu_name="Temperature",
        uom_family="degC",
    ),
    # --- Transmissibility / multipliers (continuous) ---
    "TRANSMISSIBILITY_X": OsduPropertyMapping(
        ecl_keyword="TRANX",
        osdu_reference="osdu:reference-data--PropertyNameType:TransmissibilityX:1.0.0",
        osdu_name="Transmissibility X",
        uom_family="cP.rm3/day/bar",
    ),
    "TRANSMISSIBILITY_Y": OsduPropertyMapping(
        ecl_keyword="TRANY",
        osdu_reference="osdu:reference-data--PropertyNameType:TransmissibilityY:1.0.0",
        osdu_name="Transmissibility Y",
        uom_family="cP.rm3/day/bar",
    ),
    "TRANSMISSIBILITY_Z": OsduPropertyMapping(
        ecl_keyword="TRANZ",
        osdu_reference="osdu:reference-data--PropertyNameType:TransmissibilityZ:1.0.0",
        osdu_name="Transmissibility Z",
        uom_family="cP.rm3/day/bar",
    ),
    "TRANSMISSIBILITY_MULTIPLIER_Z": OsduPropertyMapping(
        ecl_keyword="MULTZ",
        osdu_reference="osdu:reference-data--PropertyNameType:TransmissibilityMultiplierZ:1.0.0",
        osdu_name="Transmissibility Multiplier Z",
        uom_family="unitless",
    ),
    "PORE_VOLUME_MULTIPLIER": OsduPropertyMapping(
        ecl_keyword="MULTPV",
        osdu_reference="osdu:reference-data--PropertyNameType:PoreVolumeMultiplier:1.0.0",
        osdu_name="Pore Volume Multiplier",
        uom_family="unitless",
    ),
    # --- Geometry / depth (continuous) ---
    "DEPTH": OsduPropertyMapping(
        ecl_keyword="DEPTH",
        osdu_reference="osdu:reference-data--PropertyNameType:Depth:1.0.0",
        osdu_name="Depth",
        uom_family="m",
    ),
    "TOP_DEPTH": OsduPropertyMapping(
        ecl_keyword="TOPS",
        osdu_reference="osdu:reference-data--PropertyNameType:TopDepth:1.0.0",
        osdu_name="Top Depth",
        uom_family="m",
    ),
    "THICKNESS": OsduPropertyMapping(
        ecl_keyword="DZ",
        osdu_reference="osdu:reference-data--PropertyNameType:Thickness:1.0.0",
        osdu_name="Thickness",
        uom_family="m",
    ),
    "BULK_VOLUME": OsduPropertyMapping(
        ecl_keyword="BULKVOL",
        osdu_reference="osdu:reference-data--PropertyNameType:BulkVolume:1.0.0",
        osdu_name="Bulk Volume",
        uom_family="m3",
    ),
    "PORE_VOLUME": OsduPropertyMapping(
        ecl_keyword="PORV",
        osdu_reference="osdu:reference-data--PropertyNameType:PoreVolume:1.0.0",
        osdu_name="Pore Volume",
        uom_family="rm3",
    ),
    # --- Region numbers (discrete) ---
    "ACTIVE_CELL": OsduPropertyMapping(
        ecl_keyword="ACTNUM",
        osdu_reference="osdu:reference-data--PropertyNameType:ActiveCell:1.0.0",
        osdu_name="Active Cell",
        uom_family="unitless",
        is_discrete=True,
    ),
    "FLOW_REGION_INDEX": OsduPropertyMapping(
        ecl_keyword="FIPNUM",
        osdu_reference="osdu:reference-data--PropertyNameType:FlowRegionIndex:1.0.0",
        osdu_name="Flow Region Index",
        uom_family="unitless",
        is_discrete=True,
    ),
    "SATURATION_NUMBER": OsduPropertyMapping(
        ecl_keyword="SATNUM",
        osdu_reference="osdu:reference-data--PropertyNameType:SaturationNumber:1.0.0",
        osdu_name="Saturation Number",
        uom_family="unitless",
        is_discrete=True,
    ),
    "EQUILIBRIUM_NUMBER": OsduPropertyMapping(
        ecl_keyword="EQLNUM",
        osdu_reference="osdu:reference-data--PropertyNameType:EquilibriumNumber:1.0.0",
        osdu_name="Equilibrium Number",
        uom_family="unitless",
        is_discrete=True,
    ),
    "PVT_NUMBER": OsduPropertyMapping(
        ecl_keyword="PVTNUM",
        osdu_reference="osdu:reference-data--PropertyNameType:PVTNumber:1.0.0",
        osdu_name="PVT Number",
        uom_family="unitless",
        is_discrete=True,
    ),
    "ROCK_TYPE_INDEX": OsduPropertyMapping(
        ecl_keyword="ROCKNUM",
        osdu_reference="osdu:reference-data--PropertyNameType:RockTypeIndex:1.0.0",
        osdu_name="Rock Type Index",
        uom_family="unitless",
        is_discrete=True,
    ),
    "IMBIBITION_NUMBER": OsduPropertyMapping(
        ecl_keyword="IMBNUM",
        osdu_reference="osdu:reference-data--PropertyNameType:ImbibitionNumber:1.0.0",
        osdu_name="Imbibition Number",
        uom_family="unitless",
        is_discrete=True,
    ),
    "ENDPOINT_SCALING_NUMBER": OsduPropertyMapping(
        ecl_keyword="ENDNUM",
        osdu_reference="osdu:reference-data--PropertyNameType:EndpointScalingNumber:1.0.0",
        osdu_name="Endpoint Scaling Number",
        uom_family="unitless",
        is_discrete=True,
    ),
    "FACIES": OsduPropertyMapping(
        ecl_keyword="FACIES",
        osdu_reference="osdu:reference-data--PropertyNameType:Facies:1.0.0",
        osdu_name="Facies",
        uom_family="unitless",
        is_discrete=True,
    ),
    "ZONE_INDEX": OsduPropertyMapping(
        ecl_keyword="ZONE",
        osdu_reference="osdu:reference-data--PropertyNameType:ZoneIndex:1.0.0",
        osdu_name="Zone Index",
        uom_family="unitless",
        is_discrete=True,
    ),
}

# ---------------------------------------------------------------------------
# Eclipse keyword -> canonical key (title synonyms)
# Covers Eclipse keywords, common RMS property names, and xtgeo conventions.
# ---------------------------------------------------------------------------
_TITLE_SYNONYMS: Dict[str, str] = {
    # Direct Eclipse keywords
    "PORO": "POROSITY",
    "NTG": "NET_TO_GROSS_RATIO",
    "PERMX": "PERMEABILITY_X",
    "PERMY": "PERMEABILITY_Y",
    "PERMZ": "PERMEABILITY_Z",
    "SWAT": "WATER_SATURATION",
    "SOIL": "OIL_SATURATION",
    "SGAS": "GAS_SATURATION",
    "SWCR": "CRITICAL_WATER_SATURATION",
    "SWINIT": "INITIAL_WATER_SATURATION",
    "SWL": "CONNATE_WATER_SATURATION",
    "SOWCR": "CRITICAL_OIL_IN_WATER_SATURATION",
    "SGCR": "CRITICAL_GAS_SATURATION",
    "SGU": "MAXIMUM_GAS_SATURATION",
    "PRESSURE": "PRESSURE",
    "PINIT": "INITIAL_PRESSURE",
    "RS": "SOLUTION_GAS_OIL_RATIO",
    "RV": "VAPORIZED_OIL_GAS_RATIO",
    "BO": "OIL_FORMATION_VOLUME_FACTOR",
    "TEMP": "TEMPERATURE",
    "TRANX": "TRANSMISSIBILITY_X",
    "TRANY": "TRANSMISSIBILITY_Y",
    "TRANZ": "TRANSMISSIBILITY_Z",
    "MULTZ": "TRANSMISSIBILITY_MULTIPLIER_Z",
    "MULTPV": "PORE_VOLUME_MULTIPLIER",
    "ACTNUM": "ACTIVE_CELL",
    "FIPNUM": "FLOW_REGION_INDEX",
    "SATNUM": "SATURATION_NUMBER",
    "EQLNUM": "EQUILIBRIUM_NUMBER",
    "PVTNUM": "PVT_NUMBER",
    "ROCKNUM": "ROCK_TYPE_INDEX",
    "IMBNUM": "IMBIBITION_NUMBER",
    "ENDNUM": "ENDPOINT_SCALING_NUMBER",
    "DEPTH": "DEPTH",
    "TOPS": "TOP_DEPTH",
    "DZ": "THICKNESS",
    "PORV": "PORE_VOLUME",
    "BULKVOL": "BULK_VOLUME",
    "FACIES": "FACIES",
    "ZONE": "ZONE_INDEX",
    # Common RMS/xtgeo-style names (often title-cased or full words)
    "POROSITY": "POROSITY",
    "NET/GROSS": "NET_TO_GROSS_RATIO",
    "NET_GROSS": "NET_TO_GROSS_RATIO",
    "NET TO GROSS": "NET_TO_GROSS_RATIO",
    "PERM_X": "PERMEABILITY_X",
    "PERM_Y": "PERMEABILITY_Y",
    "PERM_Z": "PERMEABILITY_Z",
    "KLOGH": "PERMEABILITY_X",
    "SW": "WATER_SATURATION",
    "SO": "OIL_SATURATION",
    "SG": "GAS_SATURATION",
    "WATER_SATURATION": "WATER_SATURATION",
    "OIL_SATURATION": "OIL_SATURATION",
    "GAS_SATURATION": "GAS_SATURATION",
    "INITIAL_WATER_SATURATION": "INITIAL_WATER_SATURATION",
    "THICKNESS": "THICKNESS",
    "TOP_DEPTH": "TOP_DEPTH",
    "TEMPERATURE": "TEMPERATURE",
    "BULK_VOLUME": "BULK_VOLUME",
    "PORE_VOLUME": "PORE_VOLUME",
    "FACIES_CODE": "FACIES",
    "ZONE_LOG": "ZONE_INDEX",
    "ZONE_INDEX": "ZONE_INDEX",
}

# RESQML property kind titles -> canonical key
_KIND_SYNONYMS: Dict[str, str] = {
    "POROSITY": "POROSITY",
    "NET TO GROSS RATIO": "NET_TO_GROSS_RATIO",
    "NET_TO_GROSS_RATIO": "NET_TO_GROSS_RATIO",
    "PERMEABILITY ROCK": "PERMEABILITY_X",  # needs facet for direction
    "PERMEABILITY": "PERMEABILITY_X",
    "PERMEABILITY THICKNESS": "PERMEABILITY_X",
    "PRESSURE": "PRESSURE",
    "PORE PRESSURE": "PRESSURE",
    "WATER SATURATION": "WATER_SATURATION",
    "OIL SATURATION": "OIL_SATURATION",
    "GAS SATURATION": "GAS_SATURATION",
    "DEPTH": "DEPTH",
    "THICKNESS": "THICKNESS",
    "CELL THICKNESS": "THICKNESS",
    "TEMPERATURE": "TEMPERATURE",
    "SOLUTION GAS-OIL RATIO": "SOLUTION_GAS_OIL_RATIO",
    "FORMATION VOLUME FACTOR": "OIL_FORMATION_VOLUME_FACTOR",
    "VISCOSITY": "OIL_VISCOSITY",
    "DENSITY": "OIL_DENSITY",
    "TRANSMISSIBILITY": "TRANSMISSIBILITY_X",
    "PORE VOLUME": "PORE_VOLUME",
    "BULK VOLUME": "BULK_VOLUME",
    "FACIES": "FACIES",
    "ROCK TYPE": "ROCK_TYPE_INDEX",
    "ZONE": "ZONE_INDEX",
    "ACTIVE": "ACTIVE_CELL",
    "REGION": "FLOW_REGION_INDEX",
}


def resolve_property_mapping(
    title: Optional[str] = None,
    property_kind: Optional[str] = None,
    facet_direction: Optional[str] = None,
) -> Optional[OsduPropertyMapping]:
    """Resolve a RESQML property to its OSDU/Eclipse mapping.

    Resolution order:
      1. Title-based lookup (most direct)
      2. PropertyKind-based lookup + facet direction
      3. Return None if unmapped (caller should use title as keyword)
    """
    # 1. Title-based
    if title:
        upper_title = title.strip().upper()
        canon_key = _TITLE_SYNONYMS.get(upper_title)
        if canon_key and canon_key in _OSDU_PROPERTIES:
            return _OSDU_PROPERTIES[canon_key]
        # Direct match
        if upper_title in _OSDU_PROPERTIES:
            return _OSDU_PROPERTIES[upper_title]

    # 2. PropertyKind-based
    if property_kind:
        upper_kind = property_kind.strip().upper()
        canon_key = _KIND_SYNONYMS.get(upper_kind)
        if canon_key:
            # Apply facet direction for permeability/transmissibility
            if "PERMEABILITY" in canon_key and facet_direction:
                facet_map = {"I": "X", "J": "Y", "K": "Z", "X": "X", "Y": "Y", "Z": "Z"}
                xyz = facet_map.get(facet_direction.strip().upper(), "")
                dir_key = f"PERMEABILITY_{xyz}" if xyz else canon_key
                if dir_key in _OSDU_PROPERTIES:
                    return _OSDU_PROPERTIES[dir_key]
            if "TRANSMISSIBILITY" in canon_key and facet_direction:
                facet_map = {"I": "X", "J": "Y", "K": "Z", "X": "X", "Y": "Y", "Z": "Z"}
                xyz = facet_map.get(facet_direction.strip().upper(), "")
                dir_key = f"TRANSMISSIBILITY_{xyz}" if xyz else canon_key
                if dir_key in _OSDU_PROPERTIES:
                    return _OSDU_PROPERTIES[dir_key]
            if canon_key in _OSDU_PROPERTIES:
                return _OSDU_PROPERTIES[canon_key]

    return None


def ecl_keyword_to_osdu(ecl_keyword: str) -> Optional[OsduPropertyMapping]:
    """Look up OSDU mapping from an Eclipse keyword."""
    upper = ecl_keyword.strip().upper()
    canon_key = _TITLE_SYNONYMS.get(upper)
    if canon_key and canon_key in _OSDU_PROPERTIES:
        return _OSDU_PROPERTIES[canon_key]
    return None


def osdu_reference_to_mapping(osdu_ref: str) -> Optional[OsduPropertyMapping]:
    """Look up mapping from an OSDU reference-data URI.

    Example input: "osdu:reference-data--PropertyNameType:Porosity:1.0.0"
    """
    if not osdu_ref:
        return None
    ref_upper = osdu_ref.upper()
    for mapping in _OSDU_PROPERTIES.values():
        if mapping.osdu_reference and mapping.osdu_reference.upper() == ref_upper:
            return mapping
    # Fuzzy: extract the property name part and try matching
    # e.g. "...PropertyNameType:Porosity:1.0.0" -> "POROSITY"
    parts = osdu_ref.split(":")
    if len(parts) >= 3:
        name_part = parts[-2].upper()
        # Try direct canonical key match
        if name_part in _OSDU_PROPERTIES:
            return _OSDU_PROPERTIES[name_part]
        # Try converting CamelCase to UPPER_SNAKE
        import re

        snake = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", parts[-2]).upper()
        if snake in _OSDU_PROPERTIES:
            return _OSDU_PROPERTIES[snake]
    return None


def osdu_name_to_ecl_keyword(osdu_name: str) -> Optional[str]:
    """Reverse lookup: OSDU human name -> Eclipse keyword.

    Example: "Porosity" -> "PORO", "Water Saturation" -> "SWAT"
    """
    if not osdu_name:
        return None
    upper = osdu_name.strip().upper()
    for mapping in _OSDU_PROPERTIES.values():
        if mapping.osdu_name.upper() == upper:
            return mapping.ecl_keyword
    return None


def list_supported_properties() -> List[OsduPropertyMapping]:
    """Return all supported property mappings."""
    return list(_OSDU_PROPERTIES.values())


@dataclass
class OsduWorkProductMetadata:
    """OSDU Work Product Component metadata for a data object."""

    uuid: str = ""
    kind: str = (
        ""  # e.g. "osdu:wks:work-product-component--ResqmlIjkGridRepresentation:1.0.0"
    )
    name: str = ""
    description: str = ""
    legal_tags: List[str] = None  # type: ignore[assignment]
    acl_viewers: List[str] = None  # type: ignore[assignment]
    acl_owners: List[str] = None  # type: ignore[assignment]
    ancestry_inputs: List[str] = None  # type: ignore[assignment]
    crs_epsg: Optional[int] = None
    data_partition: str = ""

    def __post_init__(self):
        if self.legal_tags is None:
            self.legal_tags = []
        if self.acl_viewers is None:
            self.acl_viewers = []
        if self.acl_owners is None:
            self.acl_owners = []
        if self.ancestry_inputs is None:
            self.ancestry_inputs = []

    def to_osdu_record(self) -> Dict:
        """Return OSDU-style record dict for API submission."""
        return {
            "id": self.uuid,
            "kind": self.kind,
            "legal": {
                "legaltags": self.legal_tags,
                "otherRelevantDataCountries": [],
            },
            "acl": {
                "viewers": self.acl_viewers,
                "owners": self.acl_owners,
            },
            "data": {
                "Name": self.name,
                "Description": self.description,
                "ExtensionProperties": {},
            },
        }
