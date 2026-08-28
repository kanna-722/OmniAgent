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

async def initialize_app_settings(file_path: str = None):
    global app_settings

    file_path = file_path or "agentchat/config.yaml"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
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
