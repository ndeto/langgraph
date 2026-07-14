from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from config.config import SysConfig

def system_model(config: SysConfig) -> ChatOpenAI :
    return ChatOpenAI(
        model=config["model"]["model"],
        temperature=0,
        timeout=None,
        max_retries=2,
        api_key=SecretStr(config["model"]["api_key"]),
        base_url=config["model"]["base_url"],
    )
