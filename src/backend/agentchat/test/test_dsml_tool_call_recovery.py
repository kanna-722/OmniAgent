from agentchat.core.agents.general_agent import (
    DSMLStreamSanitizer,
    recover_dsml_tool_calls,
)
from agentchat.tools.web_search.tavily_search.action import tavily_search


def test_recovers_dsml_tool_call_and_removes_protocol_text():
    content = r"""我将查询天气，请稍等。
<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="web_search">
<｜｜DSML｜｜parameter name="query" string="true">成都市成华区最近7天天气\</｜｜DSML｜｜parameter>
\</｜｜DSML｜｜invoke>
\</｜｜DSML｜｜tool_calls>"""

    cleaned_content, tool_calls = recover_dsml_tool_calls(content)

    assert cleaned_content == "我将查询天气，请稍等。"
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "web_search"
    assert tool_calls[0]["args"] == {"query": "成都市成华区最近7天天气"}
    assert tool_calls[0]["id"].startswith("call_dsml_")


def test_leaves_normal_model_content_unchanged():
    content = "今天成都天气晴朗。"

    cleaned_content, tool_calls = recover_dsml_tool_calls(content)

    assert cleaned_content == content
    assert tool_calls == []


def test_stream_sanitizer_hides_split_dsml_block():
    sanitizer = DSMLStreamSanitizer()
    chunks = [
        "我将查询天气。\n<",
        "｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name=\"web_search\">",
        "<｜｜DSML｜｜parameter name=\"query\" string=\"true\">成都天气",
        "</｜｜DSML｜｜parameter></｜｜DSML｜｜invoke>",
        "</｜｜DSML｜｜tool_calls>",
    ]

    visible = "".join(sanitizer.feed(chunk) for chunk in chunks)

    assert visible == "我将查询天气。\n"


def test_stream_sanitizer_preserves_non_dsml_angle_brackets():
    sanitizer = DSMLStreamSanitizer()

    visible = sanitizer.feed("比较结果：1 < 2，使用 <div> 示例。")

    assert visible == "比较结果：1 < 2，使用 <div> 示例。"


def test_stream_sanitizer_handles_complete_block_in_one_chunk():
    sanitizer = DSMLStreamSanitizer()
    content = (
        "调用前。<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name=\"web_search\">"
        "</｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>调用后。"
    )

    assert sanitizer.feed(content) == "调用前。调用后。"


def test_web_search_optional_arguments_are_not_required():
    schema = tavily_search.tool_call_schema.model_json_schema()

    assert schema["required"] == ["query"]
    assert schema["properties"]["topic"]["default"] == "general"
    assert schema["properties"]["max_results"]["default"] == 5
