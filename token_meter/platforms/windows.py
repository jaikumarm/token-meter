"""Windows per-user path, process, update, and trash policy."""

import ntpath

from .base import (
    CpuActivity,
    PlatformCapabilities,
    PlatformPaths,
    ProcessOptions,
    ProcessPurpose,
    TrashPlan,
    UpdatePlan,
)


CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


class WindowsPlatformServices:
    platform_id = "windows"

    def __init__(self, environment, home):
        self.environment = dict(environment or {})
        self.home = str(
            home or self.environment.get("USERPROFILE") or
            ntpath.expanduser("~")
        )

    def _environment_path(self, name, default):
        value = str(self.environment.get(name) or default)
        if value == "~":
            return self.home
        if value.startswith("~\\") or value.startswith("~/"):
            return ntpath.join(self.home, value[2:])
        return ntpath.normpath(value)

    def resolve_paths(self):
        roaming = self._environment_path(
            "APPDATA", ntpath.join(self.home, "AppData", "Roaming")
        )
        local = self._environment_path(
            "LOCALAPPDATA", ntpath.join(self.home, "AppData", "Local")
        )
        return PlatformPaths(
            config_home=roaming,
            data_home=local,
            cache_home=local,
            claude_desktop_data_roots=(
                ntpath.join(roaming, "Claude"),
                ntpath.join(roaming, "Claude-3p"),
            ),
            cursor_state_db=ntpath.join(
                roaming, "Cursor", "User", "globalStorage", "state.vscdb"
            ),
            cursor_request_logs=ntpath.join(roaming, "Cursor", "logs"),
            opencode_data_root=ntpath.join(local, "opencode"),
            opencode_cache_root=ntpath.join(local, "opencode"),
            default_trash_dir=ntpath.join(local, "Token Meter", "Trash"),
        )

    def process_options(self, purpose):
        try:
            purpose = ProcessPurpose(purpose)
        except ValueError:
            return ProcessOptions(
                supported=False,
                error_code="unsupported_process_purpose",
                message="This process operation is unavailable.",
            )
        flags = CREATE_NO_WINDOW
        if purpose is ProcessPurpose.DETACHED:
            flags |= CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
        return ProcessOptions(
            close_fds=True,
            hidden_window=True,
            creation_flags=flags,
        )

    def trash_plan(self, path, override=None, command_available=False):
        destination = self.resolve_paths().default_trash_dir
        if override:
            destination = str(override)
            if destination == "~":
                destination = self.home
            elif destination.startswith("~\\") or destination.startswith("~/"):
                destination = ntpath.join(self.home, destination[2:])
            destination = ntpath.normpath(destination)
        return TrashPlan(
            supported=True,
            strategy="move",
            destination_root=destination,
            destination_label="Trash",
        )

    def update_plan(self, source_root, checkout, status_path):
        script = ntpath.join(str(source_root), "scripts", "update-windows.ps1")
        return UpdatePlan(
            supported=True,
            script_path=script,
            command=(
                "powershell.exe", "-NoLogo", "-NoProfile",
                "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
                "-File", script, str(checkout), str(status_path),
            ),
        )

    def agent_launcher(self, source_root):
        return ntpath.join(str(source_root), "scripts", "run-token-meter-mcp.cmd")

    def capabilities(self):
        return PlatformCapabilities(paths=True, detached_process=True, trash=True)

    def lower_background_priority(self):
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel32.SetPriorityClass.restype = wintypes.BOOL
            return bool(kernel32.SetPriorityClass(
                kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS,
            ))
        except (AttributeError, OSError, ImportError, ValueError):
            return False

    def set_background_mode(self, yielding):
        return False

    def cpu_activity(self, times_reader=None):
        try:
            times = (times_reader or _system_times)()
            if not times:
                return None
            idle, kernel, user = (int(value) for value in times)
        except (AttributeError, OSError, ImportError, TypeError, ValueError):
            return None
        # Kernel time includes idle time on Windows.
        return CpuActivity(busy=max(0, kernel - idle) + user, idle=idle)

    def power_constrained(self, status_reader=None):
        try:
            status = (status_reader or _system_power_status)()
            if status is None:
                return None
            ac_line, saver = (int(value) for value in status)
        except (AttributeError, OSError, ImportError, TypeError, ValueError):
            return None
        if saver == 1 or ac_line == 0:
            return True
        if ac_line == 1:
            return False
        return None


def _system_times():
    import ctypes

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    idle, kernel, user = FileTime(), FileTime(), FileTime()
    if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    return tuple((value.high << 32) | value.low for value in (idle, kernel, user))


def _system_power_status():
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return int(status.ACLineStatus), int(status.SystemStatusFlag)
