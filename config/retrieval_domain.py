"""BF/Sinter vocabulary and section policy for retrieval."""

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "sinter": ["sinter", "sintering", "coke breeze", "sfca", "traveling grate"],
    "pellet": ["pellet", "pellets", "pelletizing", "induration", "balling"],
    "basicity": ["basicity", "cao/sio2", "binary basicity", "flux", "fluxing"],
    "cohesive_zone": ["cohesive zone", "cohesive", "softening zone"],
    "softening_melting": ["softening", "melting", "tm-ts", "tm_ts"],
    "hydrogen_reduction": ["hydrogen", "h2", "h2 reduction", "hydrogen reduction"],
    "reducibility": ["reducibility", "reducible", "reduction degree", "rdi"],
    "pressure_drop": ["pressure drop", "permeability", "delp", "del p", "ndm"],
    "slag_formation": ["slag", "slag formation", "viscosity", "liquidus"],
    "burden_distribution": ["burden", "burden distribution", "charging", "stock line"],
}

EXPLANATORY_SECTIONS = {
    "overview", "definition", "process", "key mechanisms", "mechanism",
    "production", "microstructure", "blast furnace significance",
    "key takeaways", "summary", "effect", "effects",
}