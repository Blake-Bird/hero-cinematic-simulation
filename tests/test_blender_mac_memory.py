import unittest
from unittest.mock import patch
import core.blender as blender


class BlenderMacMemoryTests(unittest.TestCase):
    def read(self,text):
        with patch.object(blender.sys,'platform','darwin'),patch('subprocess.check_output',return_value=text),patch.object(blender,'cgroup_available_memory',return_value=None):
            return blender._available_memory_bytes()

    def test_only_lower_bound_file_cache_is_added(self):
        text='Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 10.\nPages speculative: 5.\nPages active: 100.\nPages inactive: 80.\nPages wired down: 20.\nFile-backed pages: 200.\n'
        self.assertEqual(self.read(text),75*4096)

    def test_anonymous_inactive_memory_is_not_counted(self):
        text='Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages speculative: 5.\nPages active: 100.\nPages inactive: 5000.\nPages wired down: 20.\nFile-backed pages: 20.\n'
        self.assertEqual(self.read(text),15*16384)
