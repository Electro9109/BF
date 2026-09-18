"""Department registry for multi-department support."""

from departments.blast_furnace import BlastFurnaceDepartment

# Minimal registry of available departments
DEPARTMENTS = {
    "blast_furnace": BlastFurnaceDepartment,
}

__all__ = ["DEPARTMENTS", "BlastFurnaceDepartment"]
