"""Department registry for multi-department support."""

from departments.blast_furnace import BlastFurnaceDepartment
from departments.sintering import SinteringDepartment
from departments.energy import EnergyDepartment

# Minimal registry of available departments
DEPARTMENTS = {
    "blast_furnace": BlastFurnaceDepartment,
    "sintering": SinteringDepartment,
    "energy": EnergyDepartment,
}

__all__ = ["DEPARTMENTS", "BlastFurnaceDepartment", "SinteringDepartment", "EnergyDepartment"]
