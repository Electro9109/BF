from config.models import EMBED_MODEL_NAME, LLM_MODEL_NAME
from config.paths import DATA_FILE, PROJECT_ROOT
from config.retrieval import TOP_K
from config.retrieval_domain import TOPIC_KEYWORDS
from config.sinter import ML_FEATURES, ML_TARGET


def test_paths_are_repository_anchored():
    assert PROJECT_ROOT.is_dir()
    assert DATA_FILE.parent == PROJECT_ROOT / "data_files"


def test_infrastructure_model_config_excludes_sinter_features():
    import config.models as model_config

    assert EMBED_MODEL_NAME
    assert LLM_MODEL_NAME
    assert not hasattr(model_config, "ML_FEATURES")
    assert not hasattr(model_config, "ML_TARGET")


def test_domain_configuration_remains_explicit():
    assert ML_FEATURES[-1] == "Basicity"
    assert ML_TARGET == "Tm-Ts"
    assert "sinter" in TOPIC_KEYWORDS
    assert TOP_K > 0