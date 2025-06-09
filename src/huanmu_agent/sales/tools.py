import datetime
from typing import List
from langchain_core.tools import tool
import zoneinfo

@tool
def get_current_time(target_timezone: str = "Asia/Shanghai") -> str:
    """
    获取指定IANA时区的当前时间。
    该工具通过获取服务器的UTC时间，然后转换为您指定的目标时区，来确保时间的准确性，特别是处理夏令时。

    Args:
        target_timezone: 目标时区的名称，遵循IANA时区数据库格式 (例如, 'Asia/Singapore', 'America/New_York', 'Europe/London').

    Returns:
        返回格式为 'YYYY-MM-DD HH:MM:SS' 的目标时区当前时间字符串, 
        如果时区名称无效，则返回错误消息。
    """
    try:
        # 使用UTC时间作为可靠基准
        utc_now = datetime.datetime.now(datetime.timezone.utc)

        # 转换为目标时区
        target_tz = zoneinfo.ZoneInfo(target_timezone)
        target_time = utc_now.astimezone(target_tz)

        return target_time.strftime('%Y-%m-%d %H:%M:%S')
    except zoneinfo.ZoneInfoNotFoundError:
        return f"无效的时区: '{target_timezone}'. 请使用有效的 IANA 时区名称 (例如, 'Asia/Singapore')."
    except Exception as e:
        return f"出现错误: {e}"

