import datetime
from typing import List
from langchain_core.tools import tool
import zoneinfo

@tool
def get_current_time(target_timezone: str = "Asia/Shanghai") -> str:
    """
    Get the current time in a specified IANA timezone.
    This tool ensures time accuracy by getting the server's UTC time and then converting it to your specified target timezone, especially handling daylight saving time.

    Args:
        target_timezone: The name of the target timezone, following IANA timezone database format (e.g., 'Asia/Singapore', 'America/New_York', 'Europe/London').

    Returns:
        Returns the current time string in the target timezone formatted as 'YYYY-MM-DD HH:MM:SS',
        or an error message if the timezone name is invalid.
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

