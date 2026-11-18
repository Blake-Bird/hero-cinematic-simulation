import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bootstrap


class InstallationProgressTests(unittest.TestCase):
    def run_install(self, directory, *, failure):
        state = Path(directory)
        environment = state / 'environments/default'
        (environment / 'bin').mkdir(parents=True)
        (environment / 'bin/python').touch()
        packages = [{'name': 'first'}, {'name': 'second'}]
        first = {'name': 'first', 'revision': 'recorded-source'}
        second = {'name': 'second', 'revision': 'another-source'}
        responses = [
            json.dumps({'prefix': str(environment), 'version': [3, 13, 0]}),
            'first==1\n',
            json.dumps({'version': [3, 13, 0], 'implementation': 'cpython', 'machine': 'arm64'}),
            'first==1\nsecond==1\n',
            json.dumps({'version': [3, 13, 0], 'implementation': 'cpython', 'machine': 'arm64'}),
        ]
        installs = [[first], RuntimeError('second build failed') if failure else [second]]
        with patch.object(bootstrap, 'paths', return_value=(state, environment)), \
                patch.object(bootstrap, 'dependency_plan', return_value=(packages, {})), \
                patch.object(bootstrap, 'install_packages', side_effect=installs), \
                patch.object(bootstrap.subprocess, 'check_output', side_effect=responses):
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'second build failed'):
                    bootstrap.main(['--install', '--profile', 'all'])
            else:
                self.assertEqual(bootstrap.main(['--install', '--profile', 'all']), 0)
        return json.loads((environment / 'installation.json').read_text())

    def test_later_failure_preserves_successful_package_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = self.run_install(directory, failure=True)
        self.assertEqual(receipt['installation_status'], 'PARTIAL')
        self.assertEqual(receipt['installed_profiles'], [])
        self.assertEqual([row['name'] for row in receipt['bundled_packages']], ['first'])

    def test_completed_profile_retains_every_successful_package(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = self.run_install(directory, failure=False)
        self.assertEqual(receipt['installation_status'], 'COMPLETE')
        self.assertEqual(receipt['installed_profiles'], ['all'])
        self.assertEqual([row['name'] for row in receipt['bundled_packages']], ['first', 'second'])
