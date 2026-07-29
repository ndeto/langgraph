import os
from typing import TypedDict

from dotenv import load_dotenv

from atlasai.util.utils import load_file

load_dotenv()

SRC_DIR = os.path.dirname(os.path.dirname(__file__))


class ModelConfig(TypedDict):
    provider: str
    model: str
    base_url: str
    api_key: str


class SysConfig(TypedDict):
    model: ModelConfig
    soul: str | None
    user_instruction_prompt: str | None
    cg_api_key: str | None
    pg_store: str  # PGVector for Vector DB
    pg_vector_store: str  # Normal Checkpoint ans Store


def get_env(_var: str) -> str:
    if _var is None:
        raise ValueError("Fetching empty var!")
    var = os.getenv(_var)
    if var is None:
        raise ValueError("Env variable {_var} is missing!")
    return var


def get_model_config() -> ModelConfig:
    model_provider = get_env("MODEL_PROVIDER")
    model = get_env("MODEL")
    provider_base_url = get_env("MODEL_PROVIDER_BASE_URL")
    api_key = get_env("MODEL_API_KEY")

    # Run validations
    return ModelConfig(
        provider=model_provider,
        model=model,
        base_url=provider_base_url,
        api_key=api_key,
    )


def get_soul() -> str:
    path = get_env("SOUL_PATH")
    return load_file(os.path.join(SRC_DIR, path))


def get_sys_prompt() -> str:
    path = get_env("SYSTEM_PROMPT_PATH")
    return load_file(os.path.join(SRC_DIR, path))


def bootstrap_config() -> SysConfig:
    _model_config = get_model_config()
    _soul = get_soul()
    _sys_prompt = get_sys_prompt()
    _cg_api_key = get_env("CG_API_KEY")
    _pg_store = get_env("DB_CONN")
    _pg_vector_store = get_env(
        "PGVECTOR_CONNECTION"
    )  # Always ensure the extension is enabled at start

    config = SysConfig(
        model=_model_config,
        soul=_soul,
        user_instruction_prompt=_sys_prompt,
        cg_api_key=_cg_api_key,
        pg_store=_pg_store,
        pg_vector_store=_pg_vector_store,
    )

    return config
