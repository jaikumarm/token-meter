import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from token_meter.platforms import linux, macos, windows
from token_meter.platforms.base import CpuActivity
from token_meter.platforms.registry import platform_services


PLATFORMS = ("darwin", "linux", "win32", "plan9")
BATTERY = "Now drawing from 'Battery Power'\n -InternalBattery-0\t80%; discharging"
MAINS = "Now drawing from 'AC Power'\n -InternalBattery-0\t100%; charged"


class BackgroundWorkContractTests(unittest.TestCase):
    def test_every_platform_implements_the_background_work_hooks(self):
        for name in PLATFORMS:
            with self.subTest(platform=name):
                platform = platform_services(name, environment={}, home="/home/example")
                for method in ("lower_background_priority", "power_constrained",
                               "cpu_activity", "set_background_mode"):
                    self.assertTrue(callable(getattr(platform, method, None)), method)

    def test_unsupported_platform_reports_unknown_and_changes_nothing(self):
        platform = platform_services("plan9", environment={}, home="/home/example")
        self.assertFalse(platform.lower_background_priority())
        self.assertFalse(platform.set_background_mode(True))
        self.assertIsNone(platform.power_constrained())
        self.assertIsNone(platform.cpu_activity())

    def test_posix_priority_is_best_effort(self):
        platform = platform_services("linux", environment={}, home="/home/example")
        with mock.patch.object(os, "nice") as nice:
            self.assertTrue(platform.lower_background_priority())
        nice.assert_called_once_with(10)
        with mock.patch.object(os, "nice", side_effect=OSError("denied")):
            self.assertFalse(platform.lower_background_priority())


class MacOSBackgroundWorkTests(unittest.TestCase):
    def setUp(self):
        self.platform = platform_services("darwin", environment={}, home="/Users/example")

    def test_pmset_power_states(self):
        cases = [
            ((BATTERY, " powermode 0\n"), True),
            ((MAINS, " powermode 0\n"), False),
            ((MAINS, " powermode 1\n"), True),
            ((MAINS, " powermode 2\n"), False),
            ((MAINS, " lowpowermode 1\n"), True),
            (("", " lowpowermode 0\n"), False),
            ((MAINS, ""), False),
            (("", ""), None),
            (("unexpected", "garbage"), None),
        ]
        for outputs, expected in cases:
            with self.subTest(outputs=outputs):
                self.assertEqual(macos.parse_pmset_power(*outputs), expected)

    def test_pmset_failures_mean_unknown(self):
        for error in (OSError("missing"), subprocess.TimeoutExpired("pmset", 2)):
            with self.subTest(error=type(error).__name__):
                def run(args, error=error):
                    raise error
                self.assertIsNone(self.platform.power_constrained(run=run))
        outputs = {("-g", "batt"): BATTERY, ("-g",): ""}
        self.assertTrue(self.platform.power_constrained(run=outputs.__getitem__))

    def test_host_cpu_ticks_include_nice_in_busy(self):
        activity = self.platform.cpu_activity(reader=lambda: (10, 5, 80, 3))
        self.assertEqual(activity, CpuActivity(busy=18, idle=80))
        self.assertIsNone(self.platform.cpu_activity(reader=lambda: None))

        def broken():
            raise OSError("no mach")
        self.assertIsNone(self.platform.cpu_activity(reader=broken))


class LinuxBackgroundWorkTests(unittest.TestCase):
    def setUp(self):
        self.platform = platform_services("linux", environment={}, home="/home/example")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def supply(self, name, **fields):
        folder = self.root / name
        folder.mkdir()
        for field, value in fields.items():
            (folder / field).write_text(value + "\n")

    def test_proc_stat_counts_nice_as_busy_and_iowait_as_idle(self):
        text = "cpu  100 20 30 400 50 6 7 8 0 0\ncpu0 1 2 3 4 5 6 7 8\n"
        self.assertEqual(linux.parse_proc_stat(text), CpuActivity(busy=171, idle=450))
        self.assertEqual(linux.parse_proc_stat("cpu 1 2 3 4\n"), CpuActivity(busy=6, idle=4))
        self.assertIsNone(linux.parse_proc_stat("cpu 1 2\n"))
        self.assertIsNone(linux.parse_proc_stat("cpu a b c d\n"))
        self.assertIsNone(linux.parse_proc_stat(""))
        self.assertIsNone(self.platform.cpu_activity(path=str(self.root / "missing")))

    def test_power_supply_states(self):
        self.assertIsNone(self.platform.power_constrained(root=str(self.root / "missing")))
        self.assertIsNone(self.platform.power_constrained(root=str(self.root)))
        self.supply("BAT0", type="Battery", status="Discharging")
        self.assertTrue(self.platform.power_constrained(root=str(self.root)))
        self.supply("AC", type="Mains", online="1")
        self.assertFalse(self.platform.power_constrained(root=str(self.root)))

    def test_offline_adapter_means_battery(self):
        self.supply("AC", type="Mains", online="0")
        self.supply("BAT0", type="Battery", status="Not charging")
        self.assertTrue(self.platform.power_constrained(root=str(self.root)))


class WindowsBackgroundWorkTests(unittest.TestCase):
    def setUp(self):
        self.platform = platform_services("win32", environment={}, home=r"C:\Users\example")

    def test_power_status(self):
        cases = [((0, 0), True), ((1, 1), True), ((1, 0), False), ((255, 0), None)]
        for status, expected in cases:
            with self.subTest(status=status):
                self.assertEqual(
                    self.platform.power_constrained(status_reader=lambda status=status: status),
                    expected,
                )
        self.assertIsNone(self.platform.power_constrained(status_reader=lambda: None))

        def broken():
            raise AttributeError("no windll")
        self.assertIsNone(self.platform.power_constrained(status_reader=broken))

    def test_system_times_exclude_idle_from_kernel_time(self):
        activity = self.platform.cpu_activity(times_reader=lambda: (70, 100, 40))
        self.assertEqual(activity, CpuActivity(busy=70, idle=70))

    def test_priority_and_mode_are_safe_off_windows(self):
        if os.name == "nt":
            self.skipTest("exercises the non-Windows fallback")
        self.assertFalse(self.platform.lower_background_priority())
        self.assertFalse(self.platform.set_background_mode(True))
        self.assertEqual(windows.BELOW_NORMAL_PRIORITY_CLASS, 0x4000)


if __name__ == "__main__":
    unittest.main()
