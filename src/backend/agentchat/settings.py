import os
import re
from pathlib import Path

import yaml
from typing import Literal, Optional
from loguru import logger
from types import SimpleNamespace
from pydantic.v1 import BaseModel, BaseSettings, Field, confloat, conint

from agentchat.schema.common import MultiModels, ModelConfig, Tools, Rag, StorageConfig


class AgentExecutionConfig(BaseModel):
    recursion_limit: conint(strict=True, gt=0) = 25


class MemoryConfig(BaseModel):
    recent_history_count: conint(strict=True, gt=0) = 6
    semantic_memory_limit: conint(strict=True, gt=0) = 5
    memory_min_score: confloat(ge=0, le=1) = 0.2
    context_token_budget: conint(strict=True, gt=0) = 2000


class Settings(BaseSettings):
    redis: dict = {}
    mysql: dict = {}
    server: dict = {}
    langfuse: dict = {}
    whitelist_paths: list = []
    wechat_config: dict = {}
    default_config: dict = {}
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    rag: Optional[Rag] = None
    tools: Optional[Tools] = None
    storage: Optional[StorageConfig] = None
    multi_models: Optional[MultiModels] = None
    agent_execution: AgentExecutionConfig = Field(default_factory=AgentExecutionConfig)


app_settings = Settings()

ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def _find_env_file(config_path: Path) -> Optional[Path]:
    search_roots = [config_path.parent, Path.cwd().resolve()]
    visited = set()
    for search_root in search_roots:
        for directory in (search_root, *search_root.parents):
            if directory in visited:
                continue
            visited.add(directory)
            env_path = directory / ".env"
            if env_path.is_file():
                return env_path
    return None


def _load_env_file(env_path: Path) -> None:
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key.isidentifier():
            raise ValueError(f"Invalid .env entry at {env_path}:{line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'\"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _expand_environment_variables(config_value):
    missing_variables = set()

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, str):
            return value

        def replace(match):
            variable_name = match.group(1)
            environment_value = os.environ.get(variable_name)
            if not environment_value:
                missing_variables.add(variable_name)
                return match.group(0)
            return environment_value

        return ENV_VAR_PATTERN.sub(replace, value)

    expanded_value = expand(config_value)
    if missing_variables:
        missing = ", ".join(sorted(missing_variables))
        raise ValueError(f"Missing required environment variables: {missing}")
    return expanded_value

async def initialize_app_settings(file_path: str = None):
    global app_settings

    file_path = file_path or "agentchat/config.yaml"
    try:
        config_path = Path(file_path).resolve()
        env_path = _find_env_file(config_path)
        if env_path is not None:
            _load_env_file(env_path)

        with config_path.open('r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            data = _expand_environment_variables(data)
            if data is None:
                logger.error("YAML 文件解析为空")
                return

            # 特殊处理multi_models配置
            if "multi_models" in data:
                data["multi_models"] = MultiModels(**data["multi_models"])

            if "tools" in data:
                data["tools"] = Tools(**data["tools"])

            if "rag" in data:
                data["rag"] = Rag(**data["rag"])

            if "storage" in data:
                data["storage"] = StorageConfig(**data["storage"])

            if "agent_execution" in data:
                data["agent_execution"] = AgentExecutionConfig(**data["agent_execution"])

            if "memory" in data:
                data["memory"] = MemoryConfig(**data["memory"])

            for key, value in data.items():
                setattr(app_settings, key, value)
    except Exception as e:
        logger.error(f"Yaml file loading error: {e}")
