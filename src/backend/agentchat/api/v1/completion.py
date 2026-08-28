import json
import loguru
from starlette.types import Receive
from fastapi.responses import StreamingResponse
from typing import List, Callable
from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from agentchat.core.agents.general_agent import GeneralAgent, AgentConfig
from agentchat.api.services.history import HistoryService
from agentchat.api.services.dialog import DialogService
from agentchat.api.services.user import UserPayload, get_login_user
from agentchat.prompts.completion import SYSTEM_PROMPT
from agentchat.schema.completion import CompletionReq
from agentchat.services.memory.client import memory_client
from agentchat.utils.contexts import set_user_id_context, set_agent_name_context
from agentchat.utils.helpers import build_completion_system_prompt, build_completion_user_input
from agentchat.services.memory.context import (
    build_memory_write_kwargs,
    estimate_context_tokens,
    format_memory_context,
    retrieve_mixed_memories,
)
from agentchat.settings import app_settings

router = APIRouter(tags=["Completion"])

class WatchedStreamingResponse(StreamingResponse):
    """
    重写 StreamingResponse类 保证流式输出的时候可随时暂停
    """
    def __init__(
        self,
        content,
        callback: Callable = None,
        status_code: int = 200,
        headers = None,
        media_type: str | None = None,
        background = None,
    ):
        super().__init__(content, status_code, headers, media_type, background)

        self.callback = callback

    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                loguru.logger.info("http.disconnect. stop task and streaming")

                if self.callback:
                    self.callback()

                break

@router.post("/completion", description="对话接口")
async def completion(
    *,
    req: CompletionReq,
    login_user: UserPayload = Depends(get_login_user)
):
    """
    与AI助手进行实时对话的核心接口

    该接口支持流式响应，能够实时返回AI生成的内容，
    同时处理历史对话记录和上下文管理
    """
    # 根据对话ID获取智能体配置信息
    db_config = await DialogService.get_agent_by_dialog_id(dialog_id=req.dialog_id)
    agent_config_data = dict(db_config)
    agent_config_data["agent_id"] = agent_config_data.get(
        "agent_id",
        agent_config_data.get("id", ""),
    )
    agent_config = AgentConfig(**agent_config_data)

    # 设置全局变量统计调用
    set_user_id_context(login_user.user_id)
    set_agent_name_context(agent_config.name)

    # 将agent_config的配置改成请求的用户ID
    agent_config.user_id = login_user.user_id

    # 基于配置创建流式对话智能体实例
    chat_agent = GeneralAgent(agent_config)
    await chat_agent.init_agent()

    # 备份用户原始输入，用于后续数据库存储和记忆检索
    original_user_input = req.user_input

    # 整合用户输入内容，将文本和附件URL合并处理
    req.user_input = build_completion_user_input(
        file_url=req.file_url,
        user_input=req.user_input
    )

    # 构建系统提示词基础指令
    system_prompt = (
        agent_config.system_prompt
        if agent_config.system_prompt.strip()
        else SYSTEM_PROMPT
    )

    recent_history_count = app_settings.memory.recent_history_count
    semantic_memory_limit = app_settings.memory.semantic_memory_limit
    memory_min_score = app_settings.memory.memory_min_score

    # 无论是否开启长期记忆，都保留最近的数据库历史
    recent_messages = await HistoryService.select_history(
        dialog_id=req.dialog_id,
        top_k=recent_history_count,
    ) or []
    long_term_memories = []
    if agent_config.enable_memory and agent_config.agent_id:
        long_term_memories = await retrieve_mixed_memories(
            memory_client,
            query=original_user_input,
            user_id=login_user.user_id,
            agent_id=agent_config.agent_id,
            dialog_id=req.dialog_id,
            recent_messages=recent_messages,
            limit=semantic_memory_limit,
            min_score=memory_min_score,
        )
    elif agent_config.enable_memory:
        loguru.logger.error(
            "Long-term memory disabled for this request: agent_id is missing"
        )

    memory_context = format_memory_context(recent_messages, long_term_memories)
    loguru.logger.info(
        "Completion context: recent_messages={}, semantic_memories={}, estimated_tokens={}",
        len(recent_messages),
        len(long_term_memories),
        estimate_context_tokens(memory_context),
    )
    system_prompt = build_completion_system_prompt(system_prompt, memory_context)

    # 构建完整消息列表（System → Human 的标准对话结构）
    messages: List[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=req.user_input),
    ]

    # 事件队列：收集工具调用、状态变更等非文本事件
    events = []

    async def general_generate():
        """
        流式响应生成器

        实时处理AI助手的响应流，将内容按SSE格式返回给前端，
        同时收集和处理各种事件（工具调用、心跳等）
        """
        response_content = " "  # 累积完整响应文本，用于后续持久化

        try:
            async for event in chat_agent.astream(messages):
                if event.get("type") == "response_chunk":
                    # 文本片段：按SSE标准格式封装并流式传输
                    yield f'data: {json.dumps(event)}\n\n'
                    response_content += event["data"].get("chunk")
                else:
                    # 其他事件（工具调用、状态更新等）：记录并同步传输
                    events.append(event)
                    yield f'data: {json.dumps(event)}\n\n'
        finally:
            # 无论流式响应是否完整，都要保存对话记录
            if agent_config.enable_memory and agent_config.agent_id:
                # 只从用户明确表达中提取长期事实，避免助手推测和工具错误污染记忆
                try:
                    await memory_client.add(
                        **build_memory_write_kwargs(
                            user_input=original_user_input,
                            user_id=login_user.user_id,
                            agent_id=agent_config.agent_id,
                            dialog_id=req.dialog_id,
                        )
                    )
                except Exception as memory_err:
                    loguru.logger.error(f"Save long-term memory failed: {memory_err}")

            # 持久化到MySQL数据库
            await HistoryService.save_chat_history(
                role="assistant",
                content=response_content,
                events=events,
                dialog_id=req.dialog_id,
                memory_enable=agent_config.enable_memory
            )

    # 先保存用户输入到数据库（确保对话完整性）
    await HistoryService.save_chat_history(
        role="user",
        content=original_user_input,
        events=events,
        dialog_id=req.dialog_id,
        memory_enable=agent_config.enable_memory
    )

    # 返回SSE流式响应
    return WatchedStreamingResponse(
        content=general_generate(),
        callback=chat_agent.stop_streaming_callback,
        media_type="text/event-stream"
    )
