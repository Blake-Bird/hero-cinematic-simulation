"""Verification: organized validation and retained numerical controls."""
from __future__ import annotations

import sys as _entry_sys
import core as _entry_core  # Consolidated source namespace registration; no controls run at import.
if __name__ == '__main__':
    _entry_sys.modules.setdefault('verification', _entry_sys.modules[__name__])

# ========================================================================
# test_visualization_numpy_contracts
from pathlib import Path
import json,sys,tempfile,unittest
import numpy as np
test_visualization_numpy_contracts_ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(test_visualization_numpy_contracts_ROOT/'core'),str(test_visualization_numpy_contracts_ROOT)]
from core.foundation import verify_numpy_container
from core.data import numpy_archive_contract,verify_numpy_archive_contract
from blender import verify_timeline_assets
from core.foundation import atomic_json, file_sha256 as digest

class VisualizationNumpyTests(unittest.TestCase):
    def identity_contract(self,root):
        path=root/'identities.npz';arrays={'matched_vertex_ids':np.array([2**64-1,2**64-1],dtype=np.uint64),
            'matched_ids':np.array([2**64-1],dtype=np.uint64),'left_only_ids':np.array([0],dtype=np.uint64),
            'right_only_ids':np.array([],dtype=np.uint64)}
        np.savez(path,**arrays)
        contract=numpy_archive_contract(path,{name:{'units':'identity','quantity':name,'identity_namespace':'native:dm'} for name in arrays})
        return path,contract
    def test_stdlib_container_reader_accepts_exact_wide_id_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,contract=self.identity_contract(root)
            self.assertEqual(verify_numpy_container(path,contract),verify_numpy_archive_contract(path,contract))
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):verify_numpy_container(path,contract)
    def test_scene_consumer_rejects_wrong_identity_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,contract=self.identity_contract(root)
            manifest={'data_contract':{'schema':'darkmattersim-data-contract-v1'},'identity_namespace':'native:stars',
                'assets':[{'path':path.name,'sha256':digest(path),'size_bytes':path.stat().st_size}],
                'snapshots':[],'intervals':[{'identity_asset':path.name,'identity_contract':contract}]}
            target=root/'timeline.json';atomic_json(target,manifest)
            (root/'timeline.json.sha256').write_text(digest(target)+'\n')
            with self.assertRaisesRegex(ValueError,'namespace/dtype'):verify_timeline_assets(target)
    def test_alternate_geometry_npz_carries_coordinates_and_text_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'skeleton.npz'
            np.savez(path,points_mpc_h=np.zeros((2,3)),lines=np.array([2,0,1]),metadata_json=np.asarray('{}'))
            contract=numpy_archive_contract(path,{'points_mpc_h':{'units':'comoving Mpc/h','quantity':'skeleton points'},
                'lines':{'units':'VTK connectivity','quantity':'polyline row indices'},
                'metadata_json':{'units':'UTF-8 JSON','quantity':'metadata'}},epoch={'actual_scale_factor':.5})
            verify_numpy_archive_contract(path,contract)
            self.assertEqual(json.loads(contract['datasets']['points_mpc_h']['epoch_json'])['actual_scale_factor'],.5)

# ========================================================================
# test_product_contract_cli
from unittest.mock import patch
test_product_contract_cli_ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(test_product_contract_cli_ROOT),str(test_product_contract_cli_ROOT/'core')]
from commands import audit_data_contract_main as product_contract_cli_main

class ProductContractCliTests(unittest.TestCase):
    def test_product_audit_retains_explicit_legacy_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();target=root/'table.dat';out=root/'audit.json'
            argv=['commands.py',str(root),'--product',str(target),'--product-kind','numeric-table','--output',str(out)]
            with patch('sys.argv',argv),patch('core.data.verify_numeric_table',return_value={'status':'LEGACY_UNSEALED'}):product_contract_cli_main()
            self.assertEqual(json.loads(out.read_text())['status'],'LEGACY_UNSEALED')
    def test_explicit_generation_routes_to_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();target=root/'product.h5';generation=root/'.hdf5_product_history/one';out=root/'audit.json'
            argv=['commands.py',str(root),'--product',str(target),'--product-kind','hdf5','--recover-generation',str(generation),'--output',str(out)]
            with patch('sys.argv',argv),patch('core.data.recover_hdf5_product',return_value={'schema':'hdf5-product-publication-v1'}) as replay:product_contract_cli_main()
            replay.assert_called_once_with(target,generation)
            self.assertEqual(json.loads(out.read_text())['science_acceptance'],'NOT_ACCEPTED')
    def test_product_and_snapshot_modes_are_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();out=root/'audit.json'
            with patch('sys.argv',['commands.py',str(root),'--product','table.dat','--product-kind','numeric-table','--snapshot','0','--output',str(out)]):
                with self.assertRaises(SystemExit):product_contract_cli_main()
            self.assertFalse(out.exists())

    def test_numeric_audit_routes_to_shared_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();target=root/'velocity.json';out=root/'result.json'
            argv=['commands.py',str(root),'--product',str(target),'--product-kind','numeric-audit','--output',str(out)]
            with patch('sys.argv',argv),patch('core.data.verify_numeric_audit',return_value={'status':'PASS','science_acceptance':'NOT_ACCEPTED'}) as reader:
                product_contract_cli_main()
            reader.assert_called_once_with(target)
            self.assertEqual(json.loads(out.read_text())['status'],'INTEGRITY_VERIFIED')

    def test_numeric_audit_replay_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();out=root/'result.json'
            argv=['commands.py',str(root),'--product','velocity.json','--product-kind','numeric-audit','--recover-generation','old','--output',str(out)]
            with patch('sys.argv',argv):
                with self.assertRaises(SystemExit):product_contract_cli_main()
            self.assertFalse(out.exists())

    def test_native_ic_audit_routes_to_saved_source_verifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();target=root/'ic-audit.json';out=root/'result.json'
            argv=['commands.py',str(root),'--product',str(target),'--product-kind','native-ic-audit','--output',str(out)]
            with patch('sys.argv',argv),patch('zoom.verify_music_ic_audit',return_value={'status':'STRUCTURAL_PASS','science_acceptance':'NOT_ACCEPTED'}) as reader:
                product_contract_cli_main()
            reader.assert_called_once_with(target)
            self.assertEqual(json.loads(out.read_text())['status'],'INTEGRITY_VERIFIED')

# ========================================================================
# test_zoom_target
import configparser

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from core.foundation import KPC_CM, MSUN_G
from core.foundation import file_sha256
from zoom import periodic_region, trace_aperture, apply_traced_region


class TargetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ic, self.snap = self.root / "parent.hdf5", self.root / "snapshot_000.hdf5"
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "traced"
        self.ids = np.array([2**63 + 11, 2**63 + 12, 2**63 + 13], dtype=np.uint64)
        counts = np.array([0, 3, 0, 0, 0, 0], dtype=np.uint64)
        self.q = np.array([[98, 40, 50], [2, 41, 51], [50, 25, 25]], dtype=np.float64)
        header = dict(
            NumFilesPerSnapshot=1,
            NumPart_ThisFile=counts,
            NumPart_Total=counts,
            MassTable=np.array([0, 1, 0, 0, 0, 0]),
            BoxSize=100.0,
            HubbleParam=0.7,
            Omega0=0.3,
            OmegaLambda=0.7,
            OmegaBaryon=0.05,
        )
        with h5py.File(self.ic, "w") as f:
            f.create_group("Header").attrs.update(
                **header,
                Time=0.01,
                Music_lagrangian_schema=1,
                Music_lagrangian_unitlength_cm=KPC_CM,
                Music_levelmin=4,
            )
            group = f.create_group("PartType1")
            group["ParticleIDs"] = self.ids
            group["Coordinates"] = (self.q + 7) % 100
            group["LagrangianCoordinates"] = self.q
        with h5py.File(self.snap, "w") as f:
            f.create_group("Header").attrs.update(**header, Time=1.0, Redshift=0.0)
            f.create_group("Parameters").attrs.update(
                UnitLength_in_cm=KPC_CM,
                UnitMass_in_g=1e10 * MSUN_G,
                UnitVelocity_in_cm_per_s=1e5,
                ComovingIntegrationOn=1,
                HubbleParam=0.7,
                Omega0=0.3,
                OmegaLambda=0.7,
                OmegaBaryon=0.05,
                InitCondFile=str(self.ic.with_suffix("")),
            )
            group = f.create_group("PartType1")
            group["ParticleIDs"] = self.ids
            group["Coordinates"] = np.array(
                [[99, 0, 0], [1, 0, 0], [50, 0, 0]], dtype=np.float64
            )
            group["Velocities"] = np.zeros((3, 3))
            group["Coordinates"].attrs.update(
                to_cgs=KPC_CM, a_scaling=1.0, h_scaling=-1.0
            )
            group["Velocities"].attrs.update(to_cgs=1e5, a_scaling=0.5, h_scaling=0.0)
        self.record_handoff()

    def record_handoff(self):
        self.manifest.write_text(
            json.dumps(
                {
                    "gadget_handoff": {
                        "initial_conditions": [
                            {"path": str(self.ic), "sha256": file_sha256(self.ic)}
                        ],
                        "initial_condition_prefix_passed_to_gadget": str(
                            self.ic.with_suffix("")
                        ),
                    }
                }
            )
        )

    def trace(self, **kwargs):
        return trace_aperture(
            self.ic,
            [self.snap],
            self.manifest,
            self.output,
            center_comoving_kpc_h=[0, 0, 0],
            radius_proper_kpc=3,
            padding_comoving_kpc_h=2,
            **kwargs,
        )

    def test_wrapped_aperture_uses_exact_q_and_all_selected_ids(self):
        report = self.trace()
        self.assertEqual(report["status"], "TRACED_PROPOSAL")
        self.assertEqual(report["selected_counts"], {"1": 2, "2": 0})
        np.testing.assert_allclose(report["music"]["ref_center"], [0, 0.405, 0.505])
        np.testing.assert_allclose(report["music"]["ref_extent"], [0.205, 0.175, 0.175])
        with h5py.File(self.output / "selected_particles.hdf5") as f:
            np.testing.assert_array_equal(f["PartType1/ParticleIDs"][:], self.ids[:2])
            np.testing.assert_array_equal(
                f["PartType1/LagrangianCoordinates"][:], self.q[:2]
            )
        self.assertEqual(report["science_acceptance"], "NOT_ACCEPTED")

    def test_regeneration_receives_traced_bounds_and_rejects_stale_sources(self):
        self.trace()
        cfg = configparser.ConfigParser()
        cfg.read_dict(
            {
                "setup": {"boxlength": "0.1"},
                "output": {"format": "arepo"},
                "cosmology": {
                    "H0": "70",
                    "Omega_m": "0.3",
                    "Omega_b": "0.05",
                    "Omega_L": "0.7",
                },
            }
        )
        apply_traced_region(cfg, self.output / "region.json")
        self.assertEqual(cfg.get("setup", "region"), "box")
        self.assertTrue(cfg.getboolean("output", "shift_back"))
        self.assertAlmostEqual(
            float(cfg.get("setup", "ref_center").split(",")[1]), 0.405
        )
        with h5py.File(self.snap, "r+") as f:
            f["PartType1/Coordinates"][0] = [2, 0, 0]
        with self.assertRaisesRegex(ValueError, "source changed"):
            apply_traced_region(cfg, self.output / "region.json")

    def test_missing_and_duplicate_parent_matches_fail_atomically(self):
        with h5py.File(self.ic, "r+") as f:
            f["PartType1/ParticleIDs"][1] = self.ids[0]
        self.record_handoff()
        with self.assertRaisesRegex(ValueError, "missing or duplicated"):
            self.trace()
        self.assertFalse(self.output.exists())

    def test_float_ids_and_changed_parent_lineage_are_rejected(self):
        with h5py.File(self.snap, "r+") as f:
            del f["PartType1/ParticleIDs"]
            f["PartType1/ParticleIDs"] = self.ids.astype(float)
        with self.assertRaisesRegex(ValueError, "integer"):
            self.trace()
        with h5py.File(self.snap, "r+") as f:
            del f["PartType1/ParticleIDs"]
            f["PartType1/ParticleIDs"] = self.ids
        with h5py.File(self.ic, "r+") as f:
            f["PartType1/Coordinates"][0] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "unchanged"):
            self.trace()
        self.assertFalse(self.output.exists())

    def test_memory_gate_precedes_particle_allocation_and_noncompact_region_fails(self):
        with patch('analysis.validate_gadget_snapshot_set') as validator:
            with self.assertRaises(MemoryError):
                self.trace(memory_gib=0.001)
            validator.assert_not_called()
        with self.assertRaisesRegex(ValueError, "half-box"):
            periodic_region(np.array([[0, 0, 0], [50, 0, 0]]), 100, 1)
        self.assertFalse(self.output.exists())

    def test_traced_reader_rejects_missing_seal_and_changed_figure(self):
        from zoom import verify_traced_region
        self.trace()
        report = self.output / 'region.json'
        verify_traced_region(report)
        seal = self.output / 'region.sha256'
        saved = seal.read_bytes()
        seal.unlink()
        with self.assertRaisesRegex(ValueError, 'seal is missing'):
            verify_traced_region(report)
        seal.write_bytes(saved)
        figure = self.output / 'traced_zoom_region.png'
        figure.write_bytes(figure.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'figure checksum'):
            verify_traced_region(report)
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from zoom import verify_traced_region

class TargetPublicationTests(TargetTests):
    def test_canonical_particle_contract_and_epochs(self):
        report=self.trace();verify_traced_region(self.output/'region.json')
        self.assertEqual(report['initial_epoch']['actual_scale_factor'],.01)
        self.assertEqual(report['target_epoch']['actual_scale_factor'],1.)
        with h5py.File(self.output/'selected_particles.hdf5') as source:
            ids=source['PartType1/ParticleIDs'];q=source['PartType1/LagrangianCoordinates']
            self.assertEqual(ids.dtype,np.dtype('uint64'));self.assertTrue(ids.fletcher32)
            self.assertTrue(q.fletcher32);self.assertIn('payload_sha256',q.attrs)

    def test_numeric_corruption_rejected_even_with_updated_whole_file_hash(self):
        self.trace();path=self.output/'selected_particles.hdf5'
        with h5py.File(path,'r+') as source:source['PartType1/LagrangianCoordinates'][0,0]+=1
        manifest=self.output/'region.json';record=json.loads(manifest.read_text())
        record['selected_particles_sha256']=digest(path);manifest.write_text(json.dumps(record))
        (self.output/'region.sha256').write_text(digest(manifest)+'\n')
        with self.assertRaises(ValueError):verify_traced_region(manifest)

    def test_failure_retains_staged_particle_product(self):
        with patch('zoom._plot_region',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):self.trace()
        self.assertFalse(self.output.exists())
        self.assertTrue(list(self.root.glob('.target-*/selected_particles.hdf5')))
        self.assertTrue(list(self.root.glob('.target-*/failure.json')))

# ========================================================================
# test_native_source_state
import shlex
import os
import subprocess
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.data import read_native_checkpoint_layout, verify_native_checkpoints
test_native_source_state_ROOT=Path(__file__).resolve().parent

class NativeSourceStateTests(unittest.TestCase):
    def compile_run(self,root,flags):
        (root/'gadgetconfig.h').write_text('#define IDS_64BIT\n#define POSITIONS_IN_64BIT\n#define DOUBLEPRECISION 1\n#define MULTIPOLE_ORDER 2\n'+flags)
        includes=shlex.split(subprocess.check_output(['pkg-config','--cflags','gsl','hdf5'],text=True))
        command=['mpicxx','-std=c++11','-I',str(root),'-I',str(test_native_source_state_ROOT/'core/software/gadget4/src'),str(test_native_source_state_ROOT/'core/native_source_state_control.cc'),'-o',str(root/'control'),*includes]
        result=subprocess.run(command,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        payload=root/'restart.0'
        result=subprocess.run([str(root/'control'),str(payload)],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('PASS',result.stdout)
        return payload

    def test_exact_native_state_roundtrip_dark_matter_and_stellar_bh_builds(self):
        for flags in ('','#define STARFORMATION\n#define COOLING\n#define BLACK_HOLES\n'):
            with self.subTest(flags=flags),tempfile.TemporaryDirectory() as tmp:
                path=self.compile_run(Path(tmp),flags)
                layout=read_native_checkpoint_layout(path,require_layout=True)
                self.assertEqual(layout['status'],'VERSIONED_STATE_LAYOUT')
                self.assertEqual(layout['source_state_schema'],'BUILD_FINGERPRINT_REQUIRED')
                self.assertEqual(layout['ranks'],1)
                self.assertGreater(layout['particle_bytes'],160)
                record={'schema':'gadget-rank-checkpoint-v1','rank':0,'ranks':1,'generation':3,'a':.375,'size_bytes':path.stat().st_size,'crc32':'%08x'%zlib.crc32(path.read_bytes())}
                Path(str(path)+'.checksum.json').write_text(json.dumps(record,separators=(',',':'))+'\n')
                self.assertEqual(verify_native_checkpoints(path.parent,1,require_receipts=True)['files'][0]['state_layout'],layout)
                with self.assertRaisesRegex(ValueError,'expected build'):read_native_checkpoint_layout(path,expected={**layout,'particle_bytes':1})

    def test_legacy_and_truncated_headers_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'restart.0';path.write_bytes(b'legacy')
            self.assertEqual(read_native_checkpoint_layout(path)['status'],'LEGACY_ABI_UNGUARDED')
            with self.assertRaises(ValueError):read_native_checkpoint_layout(path,require_layout=True)
            path.write_bytes(b'R2RST001')
            with self.assertRaisesRegex(ValueError,'Truncated'):read_native_checkpoint_layout(path)

# ========================================================================
# test_gas_initial_temperature
import copy
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from evolution import hydro_sph_parameters, gadget_param_text


class InitialTemperatureTests(unittest.TestCase):
    def test_reference_temperature_preserves_homogeneous_entropy(self):
        gas = {'initial_gas_temperature': 1.,
               'initial_gas_temperature_reference_scale_factor': .02}
        entropies = []
        for a in (.02, .01, .005):
            temperature = hydro_sph_parameters(gas, start_scale_factor=a)['InitGasTemp']
            # rho scales as a^-3; monatomic gamma=5/3 gives T/rho^(gamma-1).
            entropies.append(temperature / (a ** -3) ** (2 / 3))
        for entropy in entropies:
            self.assertAlmostEqual(entropy, .02 ** 2, places=14)
        self.assertEqual(hydro_sph_parameters(gas)['InitGasTemp'], 1.)

    def test_public_native_parameters_use_actual_start_epoch(self):
        root = Path(__file__).resolve().parent
        cfg = yaml.safe_load((root / 'core/config.yaml').read_text())
        cfg['cosmology'].update(model='lcdm', H0=70., omega_b=.05*.7**2,
                                omega_cdm=.25*.7**2, T_cmb=2.7255, N_ur=3.046)
        cfg['hardware'].update(mpi_ranks=1, memory_limit_gb=2., cpu_threads=1, omp_threads=1)
        cfg['simulation'].update(physics='hydro', start_redshift=49., end_redshift=9.,
            num_files_per_snapshot=1, time_limit_cpu_seconds=600., restart_every_seconds=60,
            snapshot_redshift_list=[19., 9.], box_mpc_h=10., softening_kpc_h=1., softening_max_physical_kpc_h=1.,
            err_tol_theta=.5, err_tol_force_acc=.005, err_tol_int_accuracy=.025,
            max_size_timestep_ln_a=.02)
        cfg['hydro'].update(initial_gas_temperature=1.,
                            initial_gas_temperature_reference_scale_factor=.02)
        with tempfile.TemporaryDirectory() as directory:
            for redshift, expected in ((49., 1.), (99., 4.), (199., 16.)):
                branch = copy.deepcopy(cfg)
                branch['simulation']['start_redshift'] = redshift
                text = gadget_param_text(branch, directory, Path(directory) / 'ics.hdf5', dry_run=True)
                actual = next(float(line.split()[1]) for line in text.splitlines()
                              if line.split() and line.split()[0] == 'InitGasTemp')
                self.assertEqual(actual, expected)

    def test_legacy_at_start_policy_and_invalid_reference(self):
        self.assertEqual(hydro_sph_parameters({'initial_gas_temperature': 12.},
                                             start_scale_factor=.005)['InitGasTemp'], 12.)
        for reference in (False, 0, -1, 2, float('nan'), '0.02'):
            with self.assertRaises(ValueError):
                hydro_sph_parameters({'initial_gas_temperature_reference_scale_factor': reference})
        for a in (False, 0, -1, 2, float('nan')):
            with self.assertRaises(ValueError):
                hydro_sph_parameters({'initial_gas_temperature_reference_scale_factor': .02},
                                     start_scale_factor=a)

# ========================================================================
# test_analysis_ownership
sys.path[:0]=[str(Path(__file__).resolve().parent/'core')]
from core.native import native_run_lock
test_analysis_ownership_ROOT=Path(__file__).resolve().parent

class AnalysisOwnershipTests(unittest.TestCase):
    def test_competing_public_analysis_cannot_create_workflow_or_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run';run.mkdir();(run/'sentinel').write_bytes(b'unchanged')
            script="\nimport sys\nsys.path[:0]=[sys.argv[1],sys.argv[1]+'/core']\nfrom commands import process_run\nprocess_run(sys.argv[2],['hmf'])\n"
            with native_run_lock(run,operation='native writer control'):
                result=subprocess.run([sys.executable,'-c',script,str(test_analysis_ownership_ROOT),str(run)],capture_output=True,text=True,timeout=60)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('already owned',result.stderr)
            self.assertEqual([p.name for p in run.iterdir()],['sentinel'])
            self.assertEqual((run/'sentinel').read_bytes(),b'unchanged')

# ========================================================================
# test_zoom_initial_conditions

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from zoom import audit_music_ic, write_audit, verify_music_ic_audit
from core.foundation import G_MPC_KMS2_MSUN


class ZoomAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.cfg, self.ic = base / "fixture.conf", base / "fixture.hdf5"
        self.cfg.write_text("""[setup]
boxlength=1
zstart=99
levelmin=1
levelmax=2
baryons=yes
[cosmology]
H0=70
Omega_m=0.3
Omega_b=0.05
m_nu1=0
m_nu2=0
m_nu3=0
transfer=CAMB
[output]
format=arepo
arepo_coarsetype=2
arepo_doubleprec=yes
arepo_longids=yes
arepo_num_files=1
""")
        rho = 3 * 100**2 / (8 * np.pi * G_MPC_KMS2_MSUN) / 1e10
        masses = np.array([0.05 * rho / 64, 0.25 * rho / 64, 0, 0, 0, 0])
        counts = np.array([8, 8, 7, 0, 0, 0], dtype=np.uint32)
        with h5py.File(self.ic, "w") as f:
            f.create_group("Header").attrs.update(
                NumFilesPerSnapshot=1,
                NumPart_ThisFile=counts,
                NumPart_Total=counts,
                NumPart_Total_HighWord=np.zeros(6, dtype=np.uint32),
                MassTable=masses,
                BoxSize=1000.0,
                Time=0.01,
                HubbleParam=0.7,
                Omega0=0.3,
            )
            offset = 2**63
            for t, n in enumerate(counts[:3]):
                group = f.create_group(f"PartType{t}")
                group["ParticleIDs"] = np.arange(n, dtype=np.uint64) + np.uint64(offset)
                offset += int(n)
                group["Coordinates"] = np.zeros((n, 3), dtype=np.float64)
                group["Velocities"] = np.zeros((n, 3), dtype=np.float64)
                if t == 2:
                    group["Masses"] = np.full(n, 0.3 * rho / 8)

    def test_native_mass_levels_and_types(self):
        report = audit_music_ic(self.ic, self.cfg)
        self.assertEqual(report["status"], "STRUCTURAL_PASS")
        self.assertLess(report["mass_closure_relative_error"], 1e-14)
        self.assertEqual(report["black_hole_particle_type"], 5)
        self.assertEqual(report["linear_transfers"]["status"], "UNVALIDATED")

    def test_duplicate_ids_and_mass_loss_rejected(self):
        with h5py.File(self.ic, "r+") as f:
            f["PartType2/ParticleIDs"][0] = f["PartType1/ParticleIDs"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_music_ic(self.ic, self.cfg)
        with h5py.File(self.ic, "r+") as f:
            f["PartType2/ParticleIDs"][0] = np.uint64(2**63 + 16)
            f["PartType2/Masses"][:] = f["PartType2/Masses"][:] * 8
        with self.assertRaisesRegex(ValueError, "closure"):
            audit_music_ic(self.ic, self.cfg)

    def test_bh_type_collision_rejected(self):
        self.cfg.write_text(
            self.cfg.read_text().replace("arepo_coarsetype=2", "arepo_coarsetype=5")
        )
        with self.assertRaisesRegex(ValueError, "reserving type 5"):
            audit_music_ic(self.ic, self.cfg)

    def test_single_coarse_level_mass_table_supported(self):
        with h5py.File(self.ic, "r+") as f:
            masses = f["Header"].attrs["MassTable"]
            masses[2] = f["PartType2/Masses"][0]
            f["Header"].attrs["MassTable"] = masses
            del f["PartType2/Masses"]
        self.assertEqual(audit_music_ic(self.ic, self.cfg)["status"], "STRUCTURAL_PASS")

    def make_uniform_parent(self):
        self.cfg.write_text(self.cfg.read_text().replace('levelmax=2','levelmax=1'))
        with h5py.File(self.ic,'r+') as source:
            counts=source['Header'].attrs['NumPart_Total'].copy()
            counts[2]=0
            source['Header'].attrs['NumPart_Total']=counts
            source['Header'].attrs['NumPart_ThisFile']=counts
            masses=source['Header'].attrs['MassTable'].copy()
            masses[:2]*=8
            source['Header'].attrs['MassTable']=masses
            source['Header'].attrs.update(Music_lagrangian_schema=1,Music_lagrangian_unitlength_cm=KPC_CM)
            del source['PartType2']
            q=(np.indices((2,2,2)).reshape(3,-1).T+.5)*500
            for t in (0,1):
                source[f'PartType{t}/LagrangianCoordinates']=q.astype(np.float64)

    def test_uniform_parent_requires_explicit_mode_and_real_native_q(self):
        self.make_uniform_parent()
        with self.assertRaisesRegex(ValueError,'distinct'):
            audit_music_ic(self.ic,self.cfg)
        report=audit_music_ic(self.ic,self.cfg,input_kind='parent')
        self.assertEqual(report['input_kind'],'parent')
        with h5py.File(self.ic,'r+') as source:
            source['PartType1/LagrangianCoordinates'][0,0]=1001.
        with self.assertRaisesRegex(ValueError,'wrapped'):
            audit_music_ic(self.ic,self.cfg,input_kind='parent')

    def test_uniform_parent_species_mass_is_not_just_total_closure(self):
        self.make_uniform_parent()
        with h5py.File(self.ic,'r+') as source:
            masses=source['Header'].attrs['MassTable'].copy()
            masses[1]-=masses[0]
            masses[0]*=2
            source['Header'].attrs['MassTable']=masses
        with self.assertRaisesRegex(ValueError,'species mass'):
            audit_music_ic(self.ic,self.cfg,input_kind='parent')

    def test_parent_lattice_cannot_duplicate_or_invent_cells(self):
        self.make_uniform_parent()
        with h5py.File(self.ic,'r+') as source:
            q=source['PartType1/LagrangianCoordinates']
            saved=q[1].copy()
            q[1]=q[0]
        with self.assertRaisesRegex(ValueError,'Duplicate.*lattice'):
            audit_music_ic(self.ic,self.cfg,input_kind='parent')
        with h5py.File(self.ic,'r+') as source:
            source['PartType1/LagrangianCoordinates'][1]=saved+np.array([.1,0,0])
        with self.assertRaisesRegex(ValueError,'uniform periodic lattice'):
            audit_music_ic(self.ic,self.cfg,input_kind='parent')

    def test_integer_phase_space_cannot_pass_double_precision_requirement(self):
        with h5py.File(self.ic,'r+') as source:
            del source['PartType1/Velocities']
            source['PartType1/Velocities']=np.zeros((8,3),dtype=np.int64)
        with self.assertRaisesRegex(ValueError,'double phase-space'):
            audit_music_ic(self.ic,self.cfg)


    def test_immutable_standalone_report_and_source_changes(self):
        from core.foundation import file_sha256 as digest
        destination=self.ic.parent/'audit.json';before=digest(self.ic)
        report=write_audit(self.ic,self.cfg,destination)
        self.assertEqual(verify_music_ic_audit(destination),report)
        self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')
        self.assertEqual(digest(self.ic),before)
        with self.assertRaises(FileExistsError):write_audit(self.ic,self.cfg,destination)
        self.cfg.write_text(self.cfg.read_text()+'\n# changed original input\n')
        with self.assertRaisesRegex(ValueError,'input changed'):verify_music_ic_audit(destination)

    def test_interrupted_seal_preserves_failed_bytes_and_rejects_read(self):
        from unittest.mock import patch
        from core.foundation import publish_file
        destination=self.ic.parent/'audit.json'
        def fail_seal(stage,target):
            if str(target).endswith('.sha256'):raise OSError('simulated seal interruption')
            return publish_file(stage,target)
        with patch('zoom.publish_file',side_effect=fail_seal):
            with self.assertRaisesRegex(OSError,'seal interruption'):write_audit(self.ic,self.cfg,destination)
        self.assertTrue(destination.is_file())
        self.assertTrue(list(destination.parent.glob('.audit.json-stage-*/failure.json')))
        self.assertTrue(list(destination.parent.glob('.audit.json-stage-*/audit.sha256')))
        with self.assertRaises(FileNotFoundError):verify_music_ic_audit(destination)

# ========================================================================
# test_ic_wdm
from initial_conditions import apply_native_wdm_thermal,verify_native_wdm_thermal
from core.cosmology import wdm_thermal_velocity_from_ids,wdm_fd_speed_moments,_fermi_dirac_momentum_table
import verification as audit_fixtures


def configuration():
    return {'cosmology':{'model':'wdm','H0':70.,'omega_b':.05*.7**2,'omega_cdm':.25*.7**2,'T_cmb':2.7255},
            'spectrum':{'source':'boltzmann'},
            'wdm':{'mode':'class_ncdm','model':'wdm_thermal_relic','thermal_relic_mass_keV':2.,'include_thermal_velocities':True},
            'simulation':{'random_seed':2**63+71}}


class ThermalTests(unittest.TestCase):
    def setUp(self):
        self.f=audit_fixtures.ZoomAuditTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.cfg=configuration()
        with h5py.File(self.f.ic,'r+') as source:
            source['Header'].attrs['OmegaBaryon']=.05

    def test_high_ids_mass_levels_and_exact_recomputation(self):
        before=file_sha256(self.f.ic)
        report=apply_native_wdm_thermal(self.f.ic,self.cfg)
        self.assertEqual(report['status'],'PASS')
        self.assertNotEqual(before,file_sha256(self.f.ic))
        self.assertEqual(report,verify_native_wdm_thermal(self.f.ic,self.cfg))
        self.assertEqual(len(report['mass_level_counts']),2)
        with h5py.File(self.f.ic) as source:
            for t in (1,2):
                np.testing.assert_array_equal(source['PartType'+str(t)+'/Velocities'][:],
                                              source['PartType'+str(t)+'/ThermalVelocities'][:])
        with self.assertRaisesRegex(ValueError,'already assigned'):
            apply_native_wdm_thermal(self.f.ic,self.cfg)

    def test_failed_speed_gate_does_not_modify_original(self):
        self.cfg['wdm']['max_thermal_speed_fraction_c']=1e-15
        before=file_sha256(self.f.ic)
        with self.assertRaisesRegex(ValueError,'Newtonian'):
            apply_native_wdm_thermal(self.f.ic,self.cfg)
        self.assertEqual(before,file_sha256(self.f.ic))

    def test_changed_payload_or_seed_rejected(self):
        apply_native_wdm_thermal(self.f.ic,self.cfg)
        self.cfg['simulation']['random_seed']+=1
        with self.assertRaisesRegex(ValueError,'prescription mismatch'):
            verify_native_wdm_thermal(self.f.ic,self.cfg)
        self.cfg['simulation']['random_seed']-=1
        with h5py.File(self.f.ic,'r+') as source:
            source['PartType1/ThermalVelocities'][0]=[1,2,3]
        with self.assertRaisesRegex(ValueError,'changed'):
            verify_native_wdm_thermal(self.f.ic,self.cfg)

    def test_fd_cdf_speed_moments_and_epoch_scaling(self):
        ids=np.arange(100000,dtype=np.uint64)+np.uint64(2**63)
        v,y,beta=wdm_thermal_velocity_from_ids(ids,self.cfg,.01)
        grid,cdf=_fermi_dirac_momentum_table()
        distance=np.max(abs(np.interp(np.sort(y),grid,cdf)-np.arange(1,len(y)+1)/len(y)))
        self.assertLess(distance,.008)
        speed=np.linalg.norm(v,axis=1)*.1
        expected=wdm_fd_speed_moments(self.cfg,.01)
        self.assertLess(abs(np.sqrt(np.mean(speed**2))/expected['rms_speed_km_s']-1),.01)
        self.assertLess(np.max(abs(np.mean(v,axis=0)))/np.sqrt(np.mean(v*v)),.02)
        slower,_,_=wdm_thermal_velocity_from_ids(ids,self.cfg,.02)
        np.testing.assert_allclose(v/slower,np.sqrt(8),rtol=1e-5)

    def test_particle_reordering_and_chunking_do_not_change_sample(self):
        ids=np.arange(128,dtype=np.uint64)+np.uint64(2**63)
        v,_,_=wdm_thermal_velocity_from_ids(ids,self.cfg,.01)
        shuffled,_,_=wdm_thermal_velocity_from_ids(ids[::-1],self.cfg,.01)
        np.testing.assert_array_equal(v,shuffled[::-1])
        chunks=np.concatenate([wdm_thermal_velocity_from_ids(x,self.cfg,.01)[0] for x in np.array_split(ids,7)])
        np.testing.assert_array_equal(v,chunks)

# ========================================================================
# test_modular_cosmology_publication
test_modular_cosmology_publication_ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(test_modular_cosmology_publication_ROOT),str(test_modular_cosmology_publication_ROOT/'core')]
from core.linear import export_power, export_background, verify_spectrum_directory, checksum
from core.data import verify_numeric_table

class ModularCosmologyTests(unittest.TestCase):
    def test_power_uses_actual_saved_epoch_and_headerless_ngenic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.dat';source.write_text('# redshift z=63\n.01 2\n.1 3\n1 4\n')
            export_power(source,root,False)
            record=verify_numeric_table(root/'pk_class.txt')
            self.assertEqual(record['context']['epoch']['actual_scale_factor'],1/64)
            self.assertFalse((root/'pk_gadget.txt').read_text().startswith('#'))
    def test_background_rejects_undeclared_extrapolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.dat';source.write_text('hashed fixture')
            a=np.array([.01,.1,.5,1.]);data=np.column_stack((1/a-1,a**-1.5,a,np.ones_like(a)))
            with patch('core.linear.read_background',return_value=({'z':0,'H':1,'D':2,'f':3},data)):
                with self.assertRaisesRegex(ValueError,'earliest'):export_background(source,root,63.)
            self.assertFalse((root/'gadget_background.txt').exists())
    def test_pending_run_rejects_public_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);atomic_json(root/'execution_status.json',{'status':'PENDING'})
            with self.assertRaisesRegex(ValueError,'incomplete'):verify_spectrum_directory(root)
    def test_complete_manifest_rejects_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);asset=root/'solver.log';asset.write_text('fixture log')
            record={'status':'complete','checksums_sha256':{'solver.log':checksum(asset)}}
            atomic_json(root/'spectrum.json',record);manifest_hash=checksum(root/'spectrum.json')
            (root/'spectrum.sha256').write_text(manifest_hash+'\n')
            atomic_json(root/'execution_status.json',{'status':'COMPLETE','manifest_sha256':manifest_hash})
            self.assertEqual(verify_spectrum_directory(root)['integrity_status'],'SEALED_ARTIFACTS_VERIFIED')
            asset.write_text('changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):verify_spectrum_directory(root)

# ========================================================================
# test_comparison_contracts
test_comparison_contracts_ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(test_comparison_contracts_ROOT/'core'))
from core.data import attach_comparison_contracts
from core.data import epoch_contract,verify_hdf5_dataset_payloads

class ComparisonContractTests(unittest.TestCase):
    def test_domains_wide_rows_missingness_and_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'comparison.h5'
            context={'run_a':{'identity_namespaces':{'halo':'run-A:halo'}},
                'run_b':{'identity_namespaces':{'halo':'run-B:halo'}},'analyses':{}}
            epochs={'pair_0000':{'run_a':epoch_contract(.5),'run_b':epoch_contract(.50000001)}}
            with h5py.File(path,'w') as handle:
                pair=handle.create_group('pairs/pair_0000')
                pair.attrs.update(snapshot_a='snap_003',snapshot_b='snap_009')
                match=pair.create_group('matched_halos')
                match['group_index_a']=np.array([0,2**63+1],dtype=np.uint64)
                match['group_index_b']=np.array([9,10],dtype=np.uint64)
                match['mass_a_msun_h']=np.array([1e12,2e12])
                history=pair.create_group('formation_history')
                history['snapshot_redshift_a']=np.array([3.,1.])
                history['progenitor_group_index_a']=np.array([[-1,0],[2,3]],dtype=np.int64)
                history['progenitor_mass_msun_h_a']=np.ones((2,2))
                attach_comparison_contracts(handle,context,epochs)
                a=match['group_index_a'];b=match['group_index_b']
                self.assertEqual(a.dtype,np.dtype('uint64'))
                self.assertEqual(int(a[1]),2**63+1)
                self.assertNotEqual(a.attrs['identity_namespace'],b.attrs['identity_namespace'])
                self.assertIn('snap_003',a.attrs['identity_namespace'])
                self.assertTrue(match['group_index_a_available'][0])
                self.assertFalse(history['progenitor_group_index_a_available'][0,0])
                self.assertTrue(history['progenitor_group_index_a_available'][0,1])
                self.assertEqual(match['mass_a_msun_h'].attrs['units'],'Msun/h')
                self.assertEqual(json.loads(a.attrs['epoch_json']),epochs['pair_0000'])
                ancestry=json.loads(history['progenitor_group_index_a'].attrs['ancestry_json'])
                self.assertEqual(ancestry['column_epochs'][0]['scale_factor'],.25)
            self.assertFalse(verify_hdf5_dataset_payloads(path,require_contract=True)['legacy_unsealed'])

    def test_unknown_quantities_stay_explicit_and_ratio_is_dimensionless(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'comparison.h5'
            with h5py.File(path,'w') as handle:
                handle['new_unknown_quantity']=np.array([1.])
                handle['hmf_ratio_log10_ratio']=np.array([[0.,np.nan]])
                attach_comparison_contracts(handle,{'analyses':{}},{})
                self.assertEqual(handle['new_unknown_quantity'].attrs['quantity_definition_status'],'UNDECLARED')
                self.assertEqual(handle['hmf_ratio_log10_ratio'].attrs['units'],'dimensionless log10(B/A)')
                self.assertTrue(handle['hmf_ratio_log10_ratio'].fletcher32)

# ========================================================================
# test_zoom_staging

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from zoom import stage_zoom_inputs, verify_staged_zoom_inputs
from core.cosmology import ede_parameters


class StagingTests(unittest.TestCase):
    def test_repeated_raw_level_streams_rejected_before_publication(self):
        with self.assertRaisesRegex(ValueError,'reuse raw Gaussian cube seed streams'):
            stage_zoom_inputs(self.manifest,self.destination,box_mpc_h=10.,parent_level=5,
                refinement_level=7,seeds_by_level={'5':421,'6':422,'7':423},padding_cells=4,
                poisson_accuracy=1e-10)
        self.assertFalse(self.destination.exists())

    def test_complete_frozen_files_reach_parent_and_zoom_recipes(self):
        from evolution import freeze_phase_realization
        records={}
        for level in (5,6):
            directory=self.root/f'noise{level}'
            freeze_phase_realization(level,directory,grid_size=2**level,memory_gib=1.)
            path=directory/'noise.grafic'
            records[str(level)]={'path':str(path),'sha256':file_sha256(path)}
        stage_zoom_inputs(self.manifest,self.destination,box_mpc_h=10.,parent_level=5,
            refinement_level=6,seeds_by_level={'5':421,'6':422},padding_cells=4,
            poisson_accuracy=1e-10,phase_noise_by_level=records)
        for name,levels in (('parent.conf',(5,)),('zoom_template.conf',(5,6))):
            cfg=configparser.ConfigParser();cfg.read(self.destination/name)
            # Gas velocities alone are insufficient: MUSIC must also emit
            # gas positions and identities, selected before plugin startup.
            self.assertTrue(cfg.getboolean('setup','do_SPH'))
            for level in levels:self.assertEqual(cfg['random'][f'seed[{level}]'],records[str(level)]['path'])
        self.assertEqual(verify_staged_zoom_inputs(self.destination)['phase_noise_by_level'],records)

    def test_incomplete_frozen_fine_noise_rejected_before_publication(self):
        with self.assertRaisesRegex(ValueError, 'files at every parent/refinement level'):
            stage_zoom_inputs(self.manifest,self.destination,box_mpc_h=10.,parent_level=5,
                refinement_level=6,seeds_by_level={'5':421,'6':422},padding_cells=4,
                poisson_accuracy=1e-10,phase_noise_by_level={'6':{'path':'not-consumed.bin','sha256':'0'*64}})
        self.assertFalse(self.destination.exists())

    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.destination=self.root/'staged'
        self.manifest=self.root/'spectrum.json'
        k=np.geomspace(1e-5,1000,128)
        self.rows=np.column_stack([k,-k*k,-2*k*k,k*k*.1,k*k*.2,np.zeros_like(k),np.zeros_like(k)])
        np.savetxt(self.root/'transfer.dat',self.rows,header='transfer functions at redshift z = 99\n1:k 2:d_b 3:d_cdm 4:t_b 5:t_cdm 6:h_t_nb_prime 7:k2gamma_nb')
        np.savetxt(self.root/'background.dat',[[200.,.5,.01,1.,9e-5],[99.,.3,.02,1.,9e-5],[0.,.0002,1.,.5,9e-5]],
            header='1:z 2:H [1/Mpc] 3:gr.fac. D 4:gr.fac. f 5:Omega_r(z)')
        self.info={'schema_version':2,'status':'complete','model':'lcdm','solver':'CLASS',
            'parameters':{'model':'lcdm','z_start':99.,'H0':70.,'Omega_b':.05,'Omega_cdm':.25,'A_s':2.1e-9,'n_s':.96},
            'class_parameters':{'start':{'H0':70.,'omega_b':.05*.7**2,'omega_cdm':.25*.7**2,'A_s':2.1e-9,
                'n_s':.96,'z_pk':99.,'output':'mPk,mTk,vTk','nbody_gauge_transfer_functions':'yes',
                'T_cmb':2.7255,'N_ur':3.046}},
            'source_files':{'start':{'transfer':'transfer.dat','background':'background.dat','ini':'input.ini'}},
            'checksums_sha256':{name:file_sha256(self.root/name) for name in ('transfer.dat','background.dat')}}
        self.save_solver_input()
        self.save_manifest()

    def save_solver_input(self):
        path=self.root/'input.ini'
        path.write_text(''.join(f'{key} = {value}\n' for key,value in self.info['class_parameters']['start'].items()))
        self.info['checksums_sha256']['input.ini']=file_sha256(path)

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.info))

    def stage(self):
        return stage_zoom_inputs(self.manifest,self.destination,box_mpc_h=10.,parent_level=5,refinement_level=7,
                                 seeds_by_level={'5':150,'6':160,'7':170},padding_cells=2,poisson_accuracy=1e-14)

    def test_exact_species_transfers_shared_parent_seed_and_no_fake_target(self):
        report=self.stage()
        self.assertEqual(report['status'],'STAGED_MODEL_MATCHED_INPUTS')
        self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')
        staged=np.loadtxt(self.destination/'transfers.txt')
        np.testing.assert_allclose(staged,self.rows[:,[0,2,1,4,3]],rtol=1e-15,atol=0)
        configs=[]
        for name in ('parent','zoom_template'):
            cfg=configparser.ConfigParser()
            cfg.optionxform=str
            cfg.read(self.destination/(name+'.conf'))
            configs.append(cfg)
            self.assertEqual(cfg.get('random','seed[5]'),'150')
            self.assertFalse(cfg.getboolean('cosmology','ZeroRadiation'))
            self.assertFalse(cfg.getboolean('setup','use_2LPT'))
            self.assertFalse(cfg.has_option('setup','ref_center'))
            self.assertEqual(cfg.get('cosmology','transfer_file'),str(self.destination.resolve()/'transfers.txt'))
        self.assertEqual(configs[0].getint('setup','levelmax'),5)
        self.assertEqual(configs[1].getint('setup','levelmax'),7)
        self.assertTrue(configs[1].getboolean('darkmattersim','requires_traced_region'))
        with self.assertRaises(FileExistsError):
            self.stage()

    def test_wrong_normalization_and_changed_source_fail_before_allocating(self):
        self.info['parameters']['A_s']=3e-9
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'normalization mismatch'):
            self.stage()
        self.info['parameters']['A_s']=2.1e-9
        self.save_manifest()
        (self.root/'transfer.dat').write_text('changed source')
        with self.assertRaisesRegex(ValueError,'Changed'):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_finest_mesh_coverage_failure_prevents_staging(self):
        # Table covers the 32-cell parent but not the 128-cell zoom diagonal.
        rows=self.rows.copy()
        rows[:,0]=np.geomspace(1e-5,25.,128)
        np.savetxt(self.root/'transfer.dat',rows,header='transfer functions at redshift z = 99\n1:k 2:d_b 3:d_cdm 4:t_b 5:t_cdm 6:h_t_nb_prime 7:k2gamma_nb')
        self.info['checksums_sha256']['transfer.dat']=file_sha256(self.root/'transfer.dat')
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'finest mesh'):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_wrong_engine_or_gauge_cannot_stage_ede(self):
        self.info['model']='ede'
        self.info['parameters']['model']='ede'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'engine'):
            self.stage()
        self.info['solver']='AxiCLASS'
        self.info['class_parameters']['start']['nbody_gauge_transfer_functions']='no'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'N-body gauge'):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_saved_input_cannot_disagree_even_with_updated_hash(self):
        path=self.root/'input.ini'
        original=path.read_text()
        for changed in (original.replace('n_s = 0.96','n_s = 0.98'),
                        original+'H0 = 70.0\n', original+'Omega_k = 0.01\n'):
            with self.subTest(input=changed):
                path.write_text(changed)
                self.info['checksums_sha256']['input.ini']=file_sha256(path)
                self.save_manifest()
                with self.assertRaisesRegex(ValueError,'solver'):
                    self.stage()
                self.assertFalse(self.destination.exists())

    def test_changed_or_missing_saved_input_is_rejected(self):
        path=self.root/'input.ini'
        path.write_text('changed bytes')
        with self.assertRaisesRegex(ValueError,'Changed or missing solver source: ini'):
            self.stage()
        path.unlink()
        with self.assertRaisesRegex(ValueError,'Changed or missing solver source: ini'):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_unsupported_background_or_fractional_massive_count_is_not_truncated(self):
        solver=self.info['class_parameters']['start']
        for key,value in (('Omega_k',.01),('wa_fld',.1),('N_ncdm',.5),('Omega_scf',.01)):
            with self.subTest(key=key):
                solver[key]=value
                self.save_solver_input()
                self.save_manifest()
                with self.assertRaises(ValueError):
                    self.stage()
                self.assertFalse(self.destination.exists())
                del solver[key]

    def test_requested_ede_cannot_disagree_with_consistently_hashed_solver_input(self):
        self.info['model']='ede'
        self.info['solver']='AxiCLASS'
        self.info['parameters'].update(model='ede',f_ede=.1,log10_zc=np.log10(3500.),theta_i=2.8,n_axion=3.)
        self.info['class_parameters']['start'].update(ede_parameters({'ede':{
            'parameterization':'axion','f_ede':.1,'z_c':3500.,'theta_i':2.8,'n_axion':3.,'extra_axiclass':{}}}))
        self.info['parameters']['f_ede']=.2
        self.save_solver_input()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError,'Requested EDE parameters differ'):
            self.stage()
        self.assertFalse(self.destination.exists())


    def test_staging_reader_rejects_changed_recipe_and_missing_seal(self):
        self.stage()
        report=verify_staged_zoom_inputs(self.destination)
        self.assertEqual(report['status'],'STAGED_MODEL_MATCHED_INPUTS')
        self.assertIn('epoch',report)
        path=self.destination/'parent.conf'
        original=path.read_bytes();path.write_bytes(original+b'\n# changed\n')
        with self.assertRaisesRegex(ValueError,'asset changed'):
            verify_staged_zoom_inputs(self.destination)
        path.write_bytes(original)
        (self.destination/'staging.sha256').unlink()
        with self.assertRaisesRegex(ValueError,'seal is missing'):
            verify_staged_zoom_inputs(self.destination)

    def test_staging_reader_rejects_extra_and_changed_nested_assets(self):
        self.stage()
        unexpected=self.destination/'undeclared.txt';unexpected.write_text('unsealed recipe')
        with self.assertRaisesRegex(ValueError,'inventory changed'):
            verify_staged_zoom_inputs(self.destination)
        unexpected.unlink()
        nested=next((self.destination/'.numeric_table_history').rglob('columns.npz'))
        nested.write_bytes(b'changed numeric inputs')
        with self.assertRaisesRegex(ValueError,'asset changed'):
            verify_staged_zoom_inputs(self.destination)

# ========================================================================
# test_observation_model
import hashlib
from scipy.integrate import quad
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from stars import (to_measurements, from_measurements, gaussian_log_likelihood,
                               gaussian_mock, selected_mixture_log_likelihood, export_mock_observations)


class ObservationModelTests(unittest.TestCase):
    def test_chart_roundtrip_across_longitude_wrap_and_near_pole(self):
        for center, angles in (([0., 0.], [[2*np.pi-.05, .02], [.04, -.03]]),
                               ([1.4, 1.5], [[2.8, 1.53], [.1, 1.52]])):
            data = np.column_stack((angles, [2., 4.], [12., -5.], [1., 2.], [-2., 3.]))
            chart = to_measurements(data, chart_center=center)
            np.testing.assert_array_equal(chart[:, 2], [.5, .25])
            restored = from_measurements(chart, chart_center=center)
            difference = restored - data
            difference[:, 0] = (difference[:, 0] + np.pi) % (2*np.pi) - np.pi
            np.testing.assert_allclose(difference, 0., atol=3.e-15)

    def test_chart_boundary_and_negative_parallax_cannot_be_inverted(self):
        for longitude in (np.pi/2., np.pi):
            with self.assertRaisesRegex(ValueError, 'hemisphere'):
                to_measurements([[longitude, 0., 1., 0., 0., 0.]], chart_center=[0., 0.])
        with self.assertRaisesRegex(ValueError, 'positive parallax'):
            from_measurements([[0., 0., -.01, 0., 0., 0.]], chart_center=[0., 0.])

    def test_missing_component_uses_marginal_not_conditional_covariance(self):
        covariance = np.array([[4., 4.8], [4.8, 9.]])
        data = np.array([[3., np.nan], [np.nan, -2.], [np.nan, np.nan]])
        mask = np.isfinite(data)
        got = gaussian_log_likelihood(data, np.zeros((3, 2)), covariance, mask)
        expected = [-.5 * (9./4. + np.log(2*np.pi*4.)),
                    -.5 * (4./9. + np.log(2*np.pi*9.)), 0.]
        np.testing.assert_allclose(got, expected, atol=2.e-15)
        # Explicit integration of the joint correlated density gives the marginal.
        def joint(second):
            return np.exp(gaussian_log_likelihood([[3., second]], [[0., 0.]], covariance,
                                                  np.ones((1, 2), dtype=bool))[0])
        integrated = quad(joint, -np.inf, np.inf, epsabs=1.e-11)[0]
        self.assertAlmostEqual(integrated, np.exp(got[0]), places=11)

    def test_unit_scales_do_not_invalidate_full_gaussian_density(self):
        scales = np.array([1.e-8, 1.e-7, .02, 100., .03, .04])
        correlation = np.eye(6); correlation[0, 5] = correlation[5, 0] = .5
        covariance = scales[:, None] * correlation * scales[None, :]
        data = scales[None, :] * np.array([[1., 0., -1., 2., 0., .5]])
        got = gaussian_log_likelihood(data, np.zeros((1, 6)), covariance, np.ones((1, 6), dtype=bool))[0]
        standardized = data[0] / scales
        expected = -.5 * (standardized @ np.linalg.solve(correlation, standardized)
                          + np.linalg.slogdet(correlation)[1] + 2*np.log(scales).sum()
                          + 6*np.log(2*np.pi))
        self.assertAlmostEqual(got, expected, places=12)

    def test_correlated_mock_has_declared_covariance_and_retains_negative_parallax(self):
        covariance = np.array([[.09, .042], [.042, .04]])
        prediction = np.zeros((60000, 2))
        mask = np.ones(prediction.shape, dtype=bool)
        first = gaussian_mock(prediction, covariance, mask, seed=80192)
        np.testing.assert_array_equal(first, gaussian_mock(prediction, covariance, mask, seed=80192))
        np.testing.assert_allclose(np.cov(first, rowvar=False), covariance, rtol=.025)
        self.assertGreater(np.count_nonzero(first[:, 0] < 0.), 29000)
        likelihood = gaussian_log_likelihood(first, prediction, covariance, mask)
        constant = np.linalg.slogdet(covariance)[1] + 2*np.log(2*np.pi)
        chi2 = -2*likelihood - constant
        self.assertAlmostEqual(chi2.mean(), 2., delta=.025)
        missing = mask[:3].copy(); missing[1, 1] = False; missing[2] = False
        drawn = gaussian_mock(prediction[:3], covariance, missing, seed=18)
        self.assertTrue(np.all(np.isnan(drawn[~missing])))
        self.assertTrue(np.all(np.isfinite(gaussian_log_likelihood(drawn, prediction[:3], covariance, missing))))

    def test_selected_contaminated_density_is_normalized_and_changes_mixture_weight(self):
        # Two unit Gaussians centred at +/-1, with selection y>0. The true
        # member fraction before selection is .3; retained fractions differ.
        fraction = .3; em, ec = ndtr(1.), ndtr(-1.)
        def component(y, mean):
            return -.5*((y-mean)**2 + np.log(2*np.pi))
        def selected(y):
            return np.exp(selected_mixture_log_likelihood(component(y, 1.), component(y, -1.),
                          member_fraction=fraction, selection_probability=float(y>0.),
                          member_efficiency=em, contaminant_efficiency=ec))
        integrated = quad(selected, 0., np.inf, epsabs=1.e-11)[0]
        self.assertAlmostEqual(integrated, 1., places=11)
        self.assertEqual(selected(-1.), 0.)
        post_fraction = fraction*em/(fraction*em+(1-fraction)*ec)
        self.assertGreater(post_fraction, .69)
        # Omitting selection normalization would integrate to only ~.36.
        self.assertLess(fraction*em+(1-fraction)*ec, .37)

    def test_mixture_limits_and_invalid_efficiencies(self):
        for fraction, expected in ((0., -3.), (1., -2.)):
            got = selected_mixture_log_likelihood(-2., -3., member_fraction=fraction,
                     selection_probability=1., member_efficiency=1., contaminant_efficiency=1.)
            self.assertEqual(got, expected)
        with self.assertRaisesRegex(ValueError, 'zero inclusion'):
            selected_mixture_log_likelihood(-2., -3., member_fraction=.5,
                     selection_probability=1., member_efficiency=0., contaminant_efficiency=0.)
        with self.assertRaisesRegex(ValueError, 'probabilities'):
            selected_mixture_log_likelihood(-2., -3., member_fraction=1.1,
                     selection_probability=1., member_efficiency=1., contaminant_efficiency=1.)

    def test_invalid_covariance_and_silently_ignored_data_fail(self):
        mask = np.ones((1, 2), dtype=bool)
        for covariance in ([[1., 2.], [2., 1.]], [[1., 1.], [1., 1.]],
                           [[1., .1], [.2, 1.]], [[-1., 0.], [0., 1.]]):
            with self.assertRaises(ValueError):
                gaussian_log_likelihood([[0., 0.]], [[0., 0.]], covariance, mask)
        with self.assertRaisesRegex(ValueError, 'missing values'):
            gaussian_log_likelihood([[0., 8.]], [[0., 0.]], np.eye(2), [[True, False]])
        with self.assertRaisesRegex(ValueError, 'shapes'):
            gaussian_log_likelihood([[0., 0.]], [[0., 0.]], np.eye(2), [[1, 1]])

    def _truth_source(self, root, present=True):
        source = Path(root) / 'truth'
        source.mkdir()
        n = 200
        ids = np.arange(n, dtype=np.uint64) + np.uint64(2**63 + 98)
        truth = np.zeros((n, 6)); truth[:, 2] = 100.
        if not present:
            truth[:] = np.nan
        np.savez_compressed(source / 'truth.npz', particle_ids=ids,
                            present=np.full(n, present, dtype=bool), observables=truth)
        report = {'schema': 'darkmattersim-stellar-observer-v1', 'status': 'TRUTH_CATALOGUE',
                  'catalogue_sha256': hashlib.sha256((source/'truth.npz').read_bytes()).hexdigest(),
                  'present_count': n if present else 0, 'selected_count': n,
                  'epoch': {'scope': 'synthetic test fixture'}, 'frame': {'scope': 'synthetic test fixture'}}
        (source / 'report.json').write_text(json.dumps(report))
        return source, ids

    def test_mock_export_keeps_truth_separate_and_missing_values_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            source, ids = self._truth_source(root)
            covariance = np.diag([1.e-8, 1.e-8, 1., 4., .04, .04])
            args = dict(chart_center=[0., 0.], covariance=covariance,
                        measured=[True, True, True, False, True, True],
                        selection_bounds=[[None, None]]*6, seed=5801)
            report = export_mock_observations(source, output_directory=Path(root)/'observed', **args)
            export_mock_observations(source, output_directory=Path(root)/'repeat', **args)
            with np.load(Path(root)/'observed/observed.npz', allow_pickle=False) as data, \
                 np.load(Path(root)/'repeat/observed.npz', allow_pickle=False) as repeat:
                self.assertEqual(set(data.files), {'particle_ids', 'measurements', 'measured', 'covariance'})
                np.testing.assert_array_equal(data['particle_ids'], ids)
                np.testing.assert_array_equal(data['measurements'], repeat['measurements'])
                self.assertTrue(np.all(np.isnan(data['measurements'][:, 3])))
                self.assertGreater(np.count_nonzero(data['measurements'][:, 2] < 0.), 80)
                with np.load(Path(root)/'observed/mock_truth.npz', allow_pickle=False) as private:
                    replay = gaussian_mock(private['prediction'], private['covariance'],
                                           np.ones(private['prediction'].shape, dtype=bool), seed=args['seed'])
                    np.testing.assert_array_equal(replay, private['unmasked_draw'])
                    likelihood = gaussian_log_likelihood(data['measurements'], private['prediction'],
                                                        data['covariance'], data['measured'])
                    self.assertTrue(np.isfinite(likelihood).all())
            self.assertEqual(report['selection']['efficiency_status'], 'NOT_CALIBRATED')
            with self.assertRaises(FileExistsError):
                export_mock_observations(source, output_directory=Path(root)/'observed', **args)

    def test_mock_export_rejects_selection_on_missing_component_and_modified_truth(self):
        with tempfile.TemporaryDirectory() as root:
            source, _ = self._truth_source(root)
            bounds = [[None, None] for _ in range(6)]; bounds[3] = [0., None]
            args = dict(chart_center=[0., 0.], covariance=np.eye(6),
                        measured=[True, True, True, False, True, True],
                        selection_bounds=bounds, seed=8, output_directory=Path(root)/'observed')
            with self.assertRaisesRegex(ValueError, 'latent-selection'):
                export_mock_observations(source, **args)
            self.assertFalse((Path(root)/'observed').exists())
            with (source/'truth.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash disagrees'):
                export_mock_observations(source, **args)

    def test_no_saved_stars_cannot_generate_a_mock_population(self):
        with tempfile.TemporaryDirectory() as root:
            source, _ = self._truth_source(root, present=False)
            with self.assertRaisesRegex(ValueError, 'no present saved stars'):
                export_mock_observations(source, chart_center=[0., 0.], covariance=np.eye(6),
                        measured=[True]*6, selection_bounds=[[None, None]]*6,
                        seed=8, output_directory=Path(root)/'observed')
            self.assertFalse((Path(root)/'observed').exists())

# ========================================================================
# test_observer_publication
sys.path[:0]=[str(Path(__file__).resolve().parent/'core'),str(Path(__file__).resolve().parent)]
from stars import plot_mock_response


class ObserverPublicationTests(unittest.TestCase):
    def arguments(self):
        return dict(chart_center=[0.,0.],covariance=np.eye(6),measured=[True]*6,
                    selection_bounds=[[None,None]]*6,seed=812)

    def test_public_mock_writes_verifiable_contracts_and_seals(self):
        with tempfile.TemporaryDirectory() as root:
            source,ids=ObservationModelTests()._truth_source(root)
            destination=Path(root)/'observed'
            report=export_mock_observations(source,output_directory=destination,**self.arguments())
            for name,contract in report['array_contracts'].items():
                verify_numpy_archive_contract(destination/name,contract)
                self.assertEqual(contract['datasets']['particle_ids']['dtype'],'uint64')
            self.assertTrue((destination/'report.sha256').is_file())
            with np.load(destination/'observed.npz',allow_pickle=False) as observed:
                np.testing.assert_array_equal(observed['particle_ids'],ids)
            with (destination/'mock_truth.npz').open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                plot_mock_response(destination,Path(root)/'figures')
            self.assertFalse((Path(root)/'figures').exists())

    def test_failed_directory_publication_retains_numeric_staging(self):
        with tempfile.TemporaryDirectory() as root:
            source,_=ObservationModelTests()._truth_source(root)
            destination=Path(root)/'observed'
            with patch('core.foundation.publish_directory',side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    export_mock_observations(source,output_directory=destination,**self.arguments())
            self.assertFalse(destination.exists())
            stages=list(Path(root).glob('.mock-observation-*'))
            self.assertEqual(len(stages),1)
            for name in ['observed.npz','mock_truth.npz','report.json','export_failure.json']:
                self.assertTrue((stages[0]/name).is_file())
            export_mock_observations(source,output_directory=destination,**self.arguments())
            self.assertTrue(stages[0].is_dir())

    def test_another_thread_cannot_own_destination_during_public_export(self):
        import concurrent.futures
        with tempfile.TemporaryDirectory() as root:
            source,_=ObservationModelTests()._truth_source(root)
            destination=Path(root)/'observed'
            with native_run_lock(destination),concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(export_mock_observations,source,output_directory=destination,**self.arguments())
                with self.assertRaisesRegex(RuntimeError,'owned'):future.result(timeout=10)
            self.assertFalse(destination.exists())

# ========================================================================
# test_zoom_catalogue_target
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from zoom import trace_catalogue_host
import verification as test_zoom_target


class CatalogueTargetTests(unittest.TestCase):
    def setUp(self):
        fixture = test_zoom_target.TargetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        (self.root / 'data').mkdir()
        self.snap = self.root / 'data/snapshot_000.hdf5'
        shutil.copy2(fixture.snap, self.snap)
        self.catalogue = self.root / 'data/fof_subhalo_tab_000.hdf5'
        with h5py.File(self.snap) as source, h5py.File(self.catalogue, 'w') as output:
            output.create_group('Header').attrs.update(Time=1., BoxSize=100., NumFiles=1,
                                                      Ngroups_Total=2, Ngroups_ThisFile=2)
            output.create_group('Parameters').attrs.update(dict(source['Parameters'].attrs))
            groups = output.create_group('Group')
            groups['GroupPos'] = [[0., 0., 0.], [50., 0., 0.]]
            groups['Group_R_Crit200'] = [.7, 1.]
            offsets = np.zeros((2,6),dtype=np.uint64)
            offsets[1,1] = 2
            lengths = np.zeros((2,6),dtype=np.uint64)
            lengths[:,1] = [2,1]
            groups['GroupOffsetType'] = offsets
            groups['GroupLenType'] = lengths

    def trace(self, **kwargs):
        return trace_catalogue_host(self.root,0,0,self.fixture.ic,self.fixture.manifest,self.fixture.output,
                                    aperture_r200c_factor=1.,padding_comoving_kpc_h=2.,**kwargs)

    def test_all_exact_host_members_retained_even_outside_r200c_aperture(self):
        report = self.trace()
        self.assertEqual(report['selected_counts'],{'1':2,'2':0})
        self.assertEqual(report['catalogue_binding']['host_fof_row'],0)
        self.assertAlmostEqual(report['selection']['radius_proper_kpc'],1.)
        # Both native particles lie 1/.7 proper kpc from the centre: aperture
        # alone would be empty. Actual FOF member IDs must still be traced.
        with h5py.File(self.fixture.output/'selected_particles.hdf5') as source:
            np.testing.assert_array_equal(source['PartType1/ParticleIDs'],self.fixture.ids[:2])
            np.testing.assert_array_equal(source['PartType1/SelectedHostMember'],[True,True])

    def test_grouped_native_output_follows_exact_source_lineage(self):
        grouped=self.root/'data/grouped_snapshot_000.hdf5'
        shutil.copy2(self.snap, grouped)
        with h5py.File(grouped,'r+') as source:
            source['Parameters'].attrs['InitCondFile']=str(self.snap.with_suffix(''))
        (self.root/'input').mkdir()
        receipt=self.root/'input/halo_finding.json'
        atomic_json(receipt,{'status':'COMPLETE','snapshots':[{
            'inputs_sha256':{str(self.snap):digest(self.snap)},
            'outputs':[{'path':str(grouped),'sha256':digest(grouped)}]}]})
        report=self.trace()
        bound={item['path'] for item in report['data_contract']['input_hashes']}
        self.assertIn(str(receipt.resolve()),bound)
        self.assertIn(str(self.snap.resolve()),bound)
        with h5py.File(self.snap,'r+') as source:
            source['Parameters'].attrs['InitCondFile']='unrelated-parent'
        with self.assertRaisesRegex(ValueError,'native source changed'):
            trace_catalogue_host(self.root,0,0,self.fixture.ic,self.fixture.manifest,self.root/'rejected',
                aperture_r200c_factor=1.,padding_comoving_kpc_h=2.)

    def test_changed_catalogue_blocks_regeneration(self):
        self.trace()
        cfg=configparser.ConfigParser()
        cfg.optionxform=str
        cfg.read_dict({'setup':{'boxlength':'.1'},'output':{'format':'arepo'},
                       'cosmology':{'H0':'70','Omega_m':'.3','Omega_b':'.05','Omega_L':'.7'}})
        apply_traced_region(cfg,self.fixture.output/'region.json')
        with h5py.File(self.catalogue,'r+') as source:
            source['Group/Group_R_Crit200'][0] = .8
        with self.assertRaisesRegex(ValueError,'source changed'):
            apply_traced_region(cfg,self.fixture.output/'region.json')

    def test_overlapping_host_offsets_and_wrong_epoch_rejected(self):
        with h5py.File(self.catalogue,'r+') as source:
            source['Group/GroupOffsetType'][1,1]=1
        with self.assertRaisesRegex(ValueError,'overlaps'):
            self.trace()
        with h5py.File(self.catalogue,'r+') as source:
            source['Group/GroupOffsetType'][1,1]=2
            source['Header'].attrs['Time']=.5
        with self.assertRaisesRegex(ValueError,'mismatch'):
            self.trace()
        self.assertFalse(self.fixture.output.exists())

    def test_incomplete_shards_and_memory_fail_before_tracing(self):
        with self.assertRaises(MemoryError):
            self.trace(memory_gib=.001)
        with h5py.File(self.catalogue,'r+') as source:
            source['Header'].attrs['NumFiles']=2
        with self.assertRaisesRegex(ValueError,'incomplete'):
            self.trace()
        self.assertFalse(self.fixture.output.exists())

    def add_galaxy(self):
        with h5py.File(self.snap,'r+') as source:
            counts = source['Header'].attrs['NumPart_Total'].copy()
            counts[4] = 1
            source['Header'].attrs['NumPart_Total'] = counts
            source['Header'].attrs['NumPart_ThisFile'] = counts
            stars = source.create_group('PartType4')
            stars['ParticleIDs'] = np.array([2**63+50],dtype=np.uint64)
            stars['Coordinates'] = [[0.,0.,0.]]
            stars['Velocities'] = [[0.,0.,0.]]
            stars['Masses'] = [1.]
            stars['Masses'].attrs.update(to_cgs=float(source['Parameters'].attrs['UnitMass_in_g']),
                                          a_scaling=0.,h_scaling=-1.)
            stars['Coordinates'].attrs.update(dict(source['PartType1/Coordinates'].attrs))
            stars['Velocities'].attrs.update(dict(source['PartType1/Velocities'].attrs))
        with h5py.File(self.catalogue,'r+') as source:
            source['Group/GroupLenType'][0,4] = 1
            source['Header'].attrs.update(Nsubhalos_Total=1,Nsubhalos_ThisFile=1)
            sub = source.create_group('Subhalo')
            sub['SubhaloGroupNr'] = [0]
            sub['SubhaloLenType'] = np.array([[0,2,0,0,1,0]],dtype=np.uint64)
            sub['SubhaloOffsetType'] = np.zeros((1,6),dtype=np.uint64)
            sub['SubhaloIDMostbound'] = self.fixture.ids[:1]

    def test_galaxy_has_exact_bound_members_and_core(self):
        self.add_galaxy()
        report = self.trace(galaxy_subhalo_row=0)
        self.assertEqual(report['catalogue_binding']['galaxy']['most_bound_particle_id'],int(self.fixture.ids[0]))
        with h5py.File(self.fixture.output/'selected_particles.hdf5') as source:
            np.testing.assert_array_equal(source['SelectedGalaxyBoundMembers/PartType4/ParticleIDs'],[2**63+50])

    def test_missing_galaxy_core_or_false_stellar_count_rejected(self):
        self.add_galaxy()
        with h5py.File(self.catalogue,'r+') as source:
            source['Subhalo/SubhaloIDMostbound'][0] = 2**63+999
        with self.assertRaisesRegex(ValueError,'most-bound ID'):
            self.trace(galaxy_subhalo_row=0)
        with h5py.File(self.catalogue,'r+') as source:
            source['Subhalo/SubhaloIDMostbound'][0] = self.fixture.ids[0]
            source['Subhalo/SubhaloLenType'][0,4] = 2
        with self.assertRaisesRegex(ValueError,'outside'):
            self.trace(galaxy_subhalo_row=0)
        self.assertFalse(self.fixture.output.exists())

# ========================================================================
# test_halo_shape_profiles

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from halos import radial_density_shapes
from halos import export_native_halo_shapes
import verification as test_halo_shape_profiles_fixtures


class HaloShapeProfilesTests(unittest.TestCase):
    def setUp(self):
        n = 1200
        z = 1-2*(np.arange(n)+.5)/n
        phi = np.arange(n)*np.pi*(3-np.sqrt(5))
        directions = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z))
        self.spherical = (directions[None,:,:]*np.array([.3,.75,1.5,2.5,3.5])[:,None,None]).reshape(-1,3)
        self.x = self.spherical*np.array([1.,.7,.45])/(.7*.45)**(1/3)
        self.mass = np.ones(len(self.x))
        self.controls = dict(support_radius_kpc=6., window='iterative_ellipsoidal_shell',
             estimator='mass_quadrupole', minimum_particles=100, tolerance=1.e-5,
             maximum_iterations=150, minimum_axis_ratio=.1, bootstrap_realizations=0,
             seed=991, minimum_bootstrap_success_fraction=.9, minimum_eigenvalue_gap=.02,
             maximum_orientation_error_deg=20.)

    def profile(self, x=None, mass=None, edges=None, **changes):
        return radial_density_shapes(self.x if x is None else x, self.mass if mass is None else mass,
                      [0.,.5,1.,2.,3.] if edges is None else edges, **(self.controls|changes))

    def test_ellipsoidal_shells_recover_density_axes_and_conserve_window_volume(self):
        result = self.profile()
        self.assertEqual(result['status'], 'MEASURED')
        self.assertEqual(result['science_acceptance'], 'NOT_ACCEPTED')
        for shell in result['shells']:
            self.assertAlmostEqual(shell['b_over_a'], .7, delta=3.e-6)
            self.assertAlmostEqual(shell['c_over_a'], .45, delta=1.e-6)
            axes = np.asarray(shell['axes_columns'])
            np.testing.assert_allclose(axes@axes.T, np.eye(3), atol=1.e-15)
            self.assertAlmostEqual(np.linalg.det(axes), 1., places=13)
            self.assertAlmostEqual(np.prod(shell['boundary_semiaxes_kpc']), shell['upper_radius_kpc']**3, places=11)
            metric = np.asarray(shell['selection_metric'])
            r2 = np.einsum('ni,ij,nj->n', self.x, metric, self.x)
            count = np.count_nonzero((r2>=shell['lower_radius_kpc']**2)&(r2<shell['upper_radius_kpc']**2))
            self.assertEqual(count, shell['particle_count'])

    def test_rotated_reduced_estimator_recovers_axes_with_sign_invariance(self):
        angle=.63
        rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
        result=self.profile(x=self.x@rotation.T, estimator='reduced_mass_quadrupole', tolerance=1.e-8)
        self.assertEqual(result['status'], 'MEASURED')
        for shell in result['shells']:
            self.assertAlmostEqual(shell['b_over_a'], .7, delta=3.e-6)
            self.assertAlmostEqual(shell['c_over_a'], .45, delta=1.e-6)
            np.testing.assert_allclose(abs(np.sum(np.asarray(shell['axes_columns'])*rotation,axis=0)),1.,atol=2.e-8)

    def test_spherical_moment_does_not_claim_intrinsic_ellipsoidal_shape(self):
        spherical=self.profile(window='spherical_shell')
        self.assertEqual(spherical['status'], 'MEASURED')
        # Spherical clipping of a triaxial density changes the measured moments.
        self.assertGreater(abs(spherical['shells'][2]['b_over_a']-.7), .02)
        np.testing.assert_array_equal(spherical['shells'][2]['boundary_semiaxes_kpc'],[2.]*3)

    def test_bootstrap_refits_shells_and_resolves_distinct_axes(self):
        result=self.profile(edges=[.5,1.], bootstrap_realizations=32)
        shell=result['shells'][0]
        self.assertEqual(shell['sampling_uncertainty']['status'],'ESTIMATED')
        self.assertEqual(shell['sampling_uncertainty']['success_fraction'],1.)
        self.assertEqual(shell['orientation_sampling_resolved'],[True]*3)
        intervals=np.asarray(shell['sampling_uncertainty']['axis_ratio_percentile_16_50_84'])
        self.assertTrue(np.all(intervals[0]<intervals[2]))
        repeat=self.profile(edges=[.5,1.], bootstrap_realizations=32)
        self.assertEqual(result,repeat)

    def test_nearly_spherical_orientation_is_not_sampling_resolved(self):
        result=self.profile(x=self.spherical,edges=[.5,1.],bootstrap_realizations=32)
        shell=result['shells'][0]
        self.assertEqual(shell['status'],'MEASURED')
        self.assertEqual(shell['orientation_sampling_resolved'],[False]*3)

    def test_support_shortage_nonconvergence_and_small_bins_are_retained(self):
        result=self.profile(support_radius_kpc=3.)
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(result['shells'][-1]['status'],'OUTSIDE_COMPLETE_SUPPORT')
        incomplete=self.profile(maximum_iterations=1)
        self.assertEqual(incomplete['shells'][0]['status'],'NOT_CONVERGED')
        missing=self.profile(edges=[.01,.02])
        self.assertEqual(missing['shells'][0]['status'],'INSUFFICIENT_PARTICLES')

    def test_mass_domination_and_invalid_definitions_fail(self):
        mass=self.mass.copy();mass[0]=1.e12
        result=self.profile(mass=mass,edges=[0.,.5],window='spherical_shell')
        self.assertEqual(result['shells'][0]['status'],'INSUFFICIENT_EFFECTIVE_MASS_COUNT')
        for changes in ({'window':'undefined'}, {'minimum_axis_ratio':0.}, {'seed':True},
                        {'bootstrap_realizations':12}, {'support_radius_kpc':2.}):
            with self.assertRaises(ValueError):
                self.profile(**changes)
        mass[0]=-1.
        with self.assertRaises(ValueError): self.profile(mass=mass)

    def test_native_adapter_preserves_units_exact_ids_and_incomplete_measurements(self):
        fixture=test_halo_shape_profiles_fixtures.CatalogueTargetTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with h5py.File(fixture.snap,'r+') as source:
            source['PartType1/Coordinates'][:]=[[99.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            source['PartType1/ParticleIDs'][:]=np.array([2**63+7,2**63+8,2**63+9],dtype=np.uint64)
        controls=self.controls|dict(radial_edges_kpc=[0.,2.],support_radius_kpc=3.,
                                    window='spherical_shell',minimum_particles=3)
        destination=fixture.root/'shape-result'
        report=export_native_halo_shapes(fixture.root,0,0,output_directory=destination,**controls)
        self.assertEqual(report['parent_definition'],'all type-1 particles in complete proper support sphere; includes substructure and non-bound matter')
        with np.load(destination/'parent.npz',allow_pickle=False) as parent:
            np.testing.assert_array_equal(parent['particle_ids'],[2**63+7,2**63+8,2**63+9])
            np.testing.assert_allclose(parent['offsets_proper_kpc'],np.diag([-1.,1.,1.])/.7,atol=1.e-15)
            np.testing.assert_allclose(parent['masses_msun'],np.full(3,1.e10/.7))
        self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')
        self.assertEqual(json.loads((destination/'report.json').read_text())['snapshot'],0)
        with self.assertRaises(FileExistsError):
            export_native_halo_shapes(fixture.root,0,0,output_directory=destination,**controls)
        with self.assertRaisesRegex(ValueError,'half-box'):
            export_native_halo_shapes(fixture.root,0,0,output_directory=fixture.root/'unsafe',
                                     **(controls|{'support_radius_kpc':100.}))

    def test_native_adapter_rejects_duplicate_selected_ids(self):
        fixture=test_halo_shape_profiles_fixtures.CatalogueTargetTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with h5py.File(fixture.snap,'r+') as source:
            source['PartType1/Coordinates'][:]=[[99.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            source['PartType1/ParticleIDs'][:]=np.array([8,8,9],dtype=np.uint64)
        with self.assertRaisesRegex(ValueError,'duplicate IDs'):
            export_native_halo_shapes(fixture.root,0,0,output_directory=fixture.root/'duplicate-shape',
                        **(self.controls|dict(radial_edges_kpc=[0.,2.],support_radius_kpc=3.,minimum_particles=3)))
        self.assertFalse((fixture.root/'duplicate-shape').exists())

    def test_changed_native_catalogue_cannot_publish_shape_report(self):
        fixture=test_halo_shape_profiles_fixtures.CatalogueTargetTests();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        def changed(*args,**kwargs):
            result=radial_density_shapes(*args,**kwargs)
            with h5py.File(fixture.catalogue,'r+') as source:
                source['Group/Group_R_Crit200'][0]=.8
            return result
        with patch('halos.radial_density_shapes',side_effect=changed):
            with self.assertRaisesRegex(RuntimeError,'sources changed'):
                export_native_halo_shapes(fixture.root,0,0,output_directory=fixture.root/'stale-shape',
                        **(self.controls|dict(radial_edges_kpc=[0.,2.],support_radius_kpc=3.,minimum_particles=3)))
        self.assertFalse((fixture.root/'stale-shape').exists())

# ========================================================================
# test_stellar_history

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from stars import exact_ids, bound_rows, read_epoch, trace_stellar_candidates


def test_stellar_history_snapshot(root, number, a, ids, xyz, *, bound_count=None):
    path = root / "data" / f"snapshot_{number:03d}.hdf5"
    path.parent.mkdir(exist_ok=True)
    parameters = {"UnitLength_in_cm": KPC_CM, "UnitMass_in_g": 1e10*MSUN_G,
                  "UnitVelocity_in_cm_per_s": 1e5, "HubbleParam": .5,
                  "Omega0": .3, "OmegaLambda": .7, "OmegaBaryon": .05,
                  "ComovingIntegrationOn": 1}
    with h5py.File(path, "w") as f:
        counts = np.array([0, 0, 0, 0, len(ids), 0], dtype=np.uint64)
        hdr = f.create_group("Header").attrs
        hdr.update({"Time": a, "Redshift": 1/a-1, "BoxSize": 100., "MassTable": np.zeros(6),
                    "NumFilesPerSnapshot": 1, "NumPart_Total": counts, "NumPart_ThisFile": counts})
        f.create_group("Parameters").attrs.update(parameters)
        stars = f.create_group("PartType4")
        stars["ParticleIDs"] = np.array(ids, dtype=np.uint64)
        for key, data, unit, ascale, hscale in (
                ("Coordinates", xyz, KPC_CM, 1., -1.),
                ("Velocities", np.tile([10., 0., 0.], (len(ids), 1)), 1e5, .5, 0.),
                ("Masses", np.ones(len(ids)), 1e10*MSUN_G, 0., -1.)):
            stars[key] = np.asarray(data, dtype=float)
            stars[key].attrs.update({"to_cgs": unit, "a_scaling": ascale, "h_scaling": hscale})
    if bound_count is not None:
        with h5py.File(root / "data" / f"fof_subhalo_tab_{number:03d}.hdf5", "w") as f:
            hdr = f.create_group("Header").attrs
            hdr.update({"Time": a, "BoxSize": 100., "NumFiles": 1,
                        "Nsubhalos_ThisFile": 1, "Nsubhalos_Total": 1})
            f.create_group("Parameters").attrs.update(parameters)
            sub = f.create_group("Subhalo")
            sub["SubhaloOffsetType"] = np.zeros((1, 6), dtype=np.uint64)
            sub["SubhaloLenType"] = np.array([[0, 0, 0, 0, bound_count, 0]], dtype=np.uint64)


class StellarHistoryTests(unittest.TestCase):
    def test_integer_ids_are_never_rounded(self):
        ids = exact_ids(np.array([2**63+1, 2**63+2], dtype=np.uint64))
        self.assertNotEqual(ids[0], ids[1])
        with self.assertRaises(ValueError):
            exact_ids(np.array([1., 2.]))

    def test_bound_span_overlap_rejected(self):
        offsets, lengths = np.zeros((2, 6), dtype=np.uint64), np.zeros((2, 6), dtype=np.uint64)
        offsets[:, 4], lengths[:, 4] = [0, 2], [3, 3]
        with self.assertRaises(ValueError):
            bound_rows(np.array([1]), offsets, lengths, 5)

    def fixture(self, root):
        first, second = 2**63+1, 2**63+2
        test_stellar_history_snapshot(root, 0, .5, [first], [[99, 5, 5]], bound_count=1)
        test_stellar_history_snapshot(root, 1, 1., [second, first], [[3, 5, 5], [1, 5, 5]], bound_count=1)
        catalogue = root / "candidates.hdf5"
        with h5py.File(catalogue, "w") as f:
            f.attrs["provenance_json"] = json.dumps({"snapshot": 1, "scale_factor": 1., "h": .5,
                "host_center_comoving_kpc_h": [0, 0, 0]})
            f["ParticleIDs"] = np.array([first, second], dtype=np.uint64)
            f["CandidateID"] = np.array([1, 1], dtype=np.uint64)
            f["Coordinates"] = [[2., 10., 10.], [6., 10., 10.]]
        return catalogue, np.array([first, second], dtype=np.uint64)

    def test_history_keeps_missing_and_unbound_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue, ids = self.fixture(root)
            xyz, vel, mass, present, rows, _ = read_epoch(root, 0, ids, chunk_size=1)
            np.testing.assert_array_equal(present, [True, False])
            np.testing.assert_array_equal(rows, [0, -2])
            self.assertTrue(np.isnan(xyz[1]).all())
            self.assertAlmostEqual(vel[0, 0], 10*np.sqrt(.5))
            self.assertEqual(mass[0], 2e10)
            result = trace_stellar_candidates(root, catalogue=catalogue, snapshots=[0, 1], candidate_ids=[1])
            with h5py.File(result["catalogue"], "r") as f:
                np.testing.assert_array_equal(f["ParticleIDs"][:], ids)
                np.testing.assert_array_equal(f["Epochs/001/BoundSubhaloRow"][:], [-1, 0])
                np.testing.assert_array_equal(f["Epochs/001/Coordinates"][:, 0], [2, 6])
            self.assertTrue(Path(result["figure"]).is_file())
            with self.assertRaises(FileExistsError):
                trace_stellar_candidates(root, catalogue=catalogue, snapshots=[0, 1], candidate_ids=[1])

    def test_missing_catalogue_is_not_unbound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, ids = self.fixture(root)
            (root / "data" / "fof_subhalo_tab_000.hdf5").unlink()
            *_, rows, meta = read_epoch(root, 0, ids)
            np.testing.assert_array_equal(rows, [-3, -2])
            self.assertFalse(meta["catalogue_available"])

    def test_duplicate_matching_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test_stellar_history_snapshot(root, 0, .5, [9, 9], [[1, 2, 3], [2, 3, 4]])
            with self.assertRaisesRegex(ValueError, "more than once"):
                read_epoch(root, 0, np.array([9], dtype=np.uint64), chunk_size=1)

    def test_selected_ids_and_memberships_across_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue, ids = self.fixture(root)
            original = root / "data" / "snapshot_001.hdf5"
            with h5py.File(original, "r") as source:
                for shard in range(2):
                    with h5py.File(original.with_name(f"snapshot_001.{shard}.hdf5"), "w") as target:
                        source.copy("Header", target)
                        source.copy("Parameters", target)
                        target["Header"].attrs["NumFilesPerSnapshot"] = 2
                        target["Header"].attrs["NumPart_ThisFile"] = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint64)
                        stars = target.create_group("PartType4")
                        for key, dataset in source["PartType4"].items():
                            stars[key] = dataset[shard:shard+1]
                            stars[key].attrs.update(dict(dataset.attrs))
            original.unlink()
            xyz, _, _, present, rows, _ = read_epoch(root, 1, ids, chunk_size=1)
            np.testing.assert_array_equal(present, [True, True])
            np.testing.assert_array_equal(xyz[:, 0], [2, 6])
            np.testing.assert_array_equal(rows, [-1, 0])
            # An incomplete shard set must be rejected before membership use.
            (root / "data" / "snapshot_001.1.hdf5").unlink()
            with self.assertRaisesRegex(ValueError, "declares 2 files"):
                read_epoch(root, 1, ids)

    def test_catalogue_epoch_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, ids = self.fixture(root)
            with h5py.File(root / "data" / "fof_subhalo_tab_000.hdf5", "a") as f:
                f["Header"].attrs["Time"] = .75
            with self.assertRaisesRegex(ValueError, "epochs/box/h"):
                read_epoch(root, 0, ids)

    def test_wrong_run_and_reverse_time_leave_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "increasing"):
                trace_stellar_candidates(root, catalogue=catalogue, snapshots=[1, 0], candidate_ids=[1])
            with h5py.File(catalogue, "a") as f:
                f["Coordinates"][0] = [50, 50, 50]
            with self.assertRaisesRegex(ValueError, "positions"):
                trace_stellar_candidates(root, catalogue=catalogue, snapshots=[0, 1], candidate_ids=[1])
            self.assertFalse((root / "results" / "stellar_histories").exists())

# ========================================================================
# test_research_outputs
import csv

test_research_outputs_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(test_research_outputs_ROOT), str(test_research_outputs_ROOT / "core")]
from halos import export_subhalo_catalogue, subhalo_table
from stars import candidate_morphology, membership_transitions, saved_subhalo_proximity
from analysis import summarize_halo_shapes, _analyze_one_product, _write_product_hdf5
from visualization import _blender_halo_overlays
from commands import process_run


def catalogue_fixture(root):
    test_stellar_history_snapshot(root, 0, .5, [2**63+1], [[99., 1., 1.]], bound_count=1)
    # The mocked DM analysis workflow still needs actual DM input support.
    # Keep these native-format rows explicitly synthetic; catalogue properties
    # below are software fixtures, not a self-consistent physical halo model.
    with h5py.File(root/'data/snapshot_000.hdf5','r+') as source:
        for name in ('NumPart_Total','NumPart_ThisFile'):
            counts=source['Header'].attrs[name];counts[1]=25
            source['Header'].attrs[name]=counts
        dm=source.create_group('PartType1')
        dm['ParticleIDs']=np.arange(25,dtype=np.uint64)+np.uint64(2**60+1)
        for name,values,unit,ae,he in [
                ('Coordinates',np.tile([99.,1.,1.],(25,1)),KPC_CM,1.,-1.),
                ('Velocities',np.zeros((25,3)),1e5,.5,0.),
                ('Masses',np.ones(25),1e10*MSUN_G,0.,-1.)]:
            dm.create_dataset(name,data=values).attrs.update(to_cgs=unit,a_scaling=ae,h_scaling=he)

    path = root / "data/fof_subhalo_tab_000.hdf5"
    with h5py.File(path, "r+") as source:
        source["Header"].attrs.update(Nsubhalos_ThisFile=2, Nsubhalos_Total=2)
        sub = source["Subhalo"]
        del sub["SubhaloLenType"]
        sub["SubhaloLenType"] = [[10, 20, 0, 0, 2, 1], [0, 5, 0, 0, 0, 0]]
        sub["SubhaloPos"] = [[99., 1., 1.], [1., 1., 1.]]
        sub["SubhaloMass"] = [3., 2.]
        sub["SubhaloMassType"] = [[.5, 1., 0, 0, 1., .5], [0, 2., 0, 0, 0, 0]]
        sub["SubhaloHalfmassRad"] = [4., 2.]
        sub["SubhaloGroupNr"] = [0, 0]
        sub["SubhaloRankInGr"] = [0, 1]
        sub["SubhaloVel"] = [[10., 20., 30.], [1., 2., 3.]]
        sub["SubhaloVelDisp"] = [8., 4.]
        sub["SubhaloVmax"] = [20., 10.]
        sub["SubhaloVmaxRad"] = [5., 3.]
        sub["SubhaloIDMostbound"] = np.array([2**63+1, 2**63+2], dtype=np.uint64)
    config = {"simulation": {"box_mpc_h": .1}, "analysis": {"hmf_mass_definition": "fof"}, "visualization": {}}
    import yaml
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    return path, config


class ResearchOutputTests(unittest.TestCase):
    def test_all_subhalo_rows_units_and_exact_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue_fixture(root)
            table = export_subhalo_catalogue(root, 0)
            with (root / "results" / table["artifact"]).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["most_bound_particle_id"], str(2**63+1))
            self.assertEqual(float(rows[0]["mass_msun_h"]), 3e10)
            self.assertEqual(float(rows[0]["half_mass_radius_proper_kpc"]), 4.)
            self.assertEqual(rows[1]["central"], "False")
            self.assertEqual(rows[1]["n_stars"], "0")
            with h5py.File(root / "result.h5", "w") as output:
                _write_product_hdf5(output, "subhalos", table)
                self.assertEqual(int(output["subhalos/fields/SubhaloIDMostbound"][0]), 2**63+1)

    def test_corrupt_species_mass_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _ = catalogue_fixture(root)
            with h5py.File(path, "r+") as source:
                source["Subhalo/SubhaloMass"][0] = 9.
            with self.assertRaisesRegex(ValueError, "sum to bound mass"):
                subhalo_table(root, 0)

    def test_epoch_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _ = catalogue_fixture(root)
            with h5py.File(path, "r+") as source:
                source["Header"].attrs["Time"] = .4
            with self.assertRaisesRegex(ValueError, "epoch/box/h"):
                subhalo_table(root, 0)

    def test_missing_catalogue_shard_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, _ = catalogue_fixture(root)
            with h5py.File(path, "r+") as source:
                source["Header"].attrs["NumFiles"] = 2
            with self.assertRaisesRegex(ValueError, "incomplete"):
                subhalo_table(root, 0)

    def test_blender_subhalo_size_species_velocity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, cfg = catalogue_fixture(root)
            item = _blender_halo_overlays(root, 0, cfg, .1)["subhalos"]["items"][0]
            self.assertAlmostEqual(item["half_mass_radius_mpc_h"], .004)
            self.assertEqual(item["velocity_km_s"], [10., 20., 30.])
            self.assertEqual(item["mass_by_type_msun_h"], [5e9, 1e10, 0., 0., 1e10, 5e9])
            self.assertEqual(item["particle_count_by_type"], [10, 20, 0, 0, 2, 1])

    def test_subhalo_product_connected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, cfg = catalogue_fixture(root)
            table = _analyze_one_product(root, cfg, 0, [], "subhalos", None)
            self.assertEqual(table["count"], 2)
            self.assertTrue((root / "results" / table["artifact"]).is_file())

    def test_periodic_morphology_and_proper_scale(self):
        xyz = np.array([[98., 1., 1.], [0., 1., 1.], [2., 1., 1.]])
        result = candidate_morphology(xyz, np.zeros((3, 3)), np.ones(3), 100., .5)
        self.assertAlmostEqual(result["rms_axes_proper_kpc"][0], np.sqrt(2/3))
        self.assertAlmostEqual(result["center_comoving_kpc"][0], 0.)
        self.assertEqual(result["velocity_rms_km_s"], 0.)
        self.assertEqual(result["b_over_a"], 0.)

    def test_ambiguous_periodic_morphology_unavailable(self):
        result = candidate_morphology(np.array([[0., 0., 0.], [40., 0., 0.], [60., 0., 0.]]),
                                      np.zeros((3, 3)), np.ones(3), 100., 1.)
        self.assertEqual(result["status"], "UNAVAILABLE")

    def test_missing_memberships_never_imply_stripping(self):
        masks = membership_transitions([0, 0, -3, -2, -1, 3], [-1, -3, -1, -1, 9, 2])
        np.testing.assert_array_equal(masks["bound_to_outside"], [True, False, False, False, False, False])
        np.testing.assert_array_equal(masks["bound_at_both"], [False, False, False, False, False, True])
        self.assertEqual(int(masks["comparable"].sum()), 3)

    def test_proximity_wrap_and_exclude_progenitor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue_fixture(root)
            result = saved_subhalo_proximity(root, 0, np.array([[198., 2., 2.]]), [0], 200., .5, .5)
            row = result["closest_other_subhalos"][0]
            self.assertEqual(row["subhalo_row"], 1)
            self.assertAlmostEqual(row["distance_proper_kpc"], 2.)
            self.assertEqual(row["bound_mass_msun"], 4e10)
            bound = result["bound_subhalo_properties"][0]
            self.assertEqual(bound["bound_mass_msun"], 6e10)
            self.assertEqual(bound["half_mass_radius_proper_kpc"], 4.)
            self.assertEqual(bound["host_fof_row"], 0)

    def test_shape_population_excludes_degenerate_triaxiality(self):
        result = summarize_halo_shapes([
            {"mass_msun_h": 1., "b_over_a": .8, "c_over_a": .6, "triaxiality": .5},
            {"mass_msun_h": 3., "b_over_a": 1., "c_over_a": 1., "triaxiality": np.nan}])
        self.assertAlmostEqual(result["metrics"]["b_over_a"]["mean"], .9)
        self.assertAlmostEqual(result["metrics"]["b_over_a"]["mass_weighted_mean"], .95)
        self.assertEqual(result["metrics"]["triaxiality"]["count"], 1)
        self.assertIsNone(result["metrics"]["triaxiality"]["standard_deviation"])

    def test_all_requires_saved_run_configuration_before_applicability(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "configuration"):
                process_run(Path(temp), ["all"])

    def test_incomplete_analysis_retains_failure_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue_fixture(root)
            with patch('commands.analyze_run', return_value={"output_directory": "fixture", "skipped": {"shape_000": "unresolved"}}):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    process_run(root, ["shape"])
            record = json.loads(next((root / "results").glob("workflow_*.json")).read_text())
            self.assertEqual(record["status"], "INCOMPLETE")
            self.assertEqual(record["unavailable"]["shape_000"], "unresolved")

    def test_failed_contamination_gate_cannot_report_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue_fixture(root)
            request = {"regions": [], "fine_dm_mass_msun": 1., "maximum_coarse_mass_fraction": 0.}
            with patch('zoom.audit_zoom_regions', return_value={"status": "FAIL", "report": "retained.json"}):
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    process_run(root, ["zoom_contamination"], zoom_regions=request)
            record = json.loads(next((root / "results").glob("workflow_*.json")).read_text())
            self.assertEqual(record["status"], "INCOMPLETE")
            self.assertEqual(record["artifacts"]["zoom_contamination"]["status"], "FAIL")

# ========================================================================
# test_scientific_table_publication
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.data import publish_numeric_table,recover_numeric_table
from core.cosmology import convert_pk_to_ngenic,_load_numeric_table,write_gadget_background_table

class ScientificTableTests(unittest.TestCase):
    def write(self,path,values):
        return publish_numeric_table(path,values,{'k':{'units':'h/Mpc','quantity':'Fourier k'},
            'P':{'units':'(Mpc/h)^3','quantity':'Linear P(k)'}},algorithm=__file__,fmt='%.12e')
    def test_serialized_columns_and_prior_generations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary).resolve()/'pk.dat';first=self.write(path,np.array([[1.,2.],[3.,4.]]))
            second=self.write(path,np.array([[1.,5.],[3.,6.]]))
            self.assertNotEqual(first['generation'],second['generation'])
            self.assertEqual(verify_numeric_table(path),second)
            self.assertEqual(digest(path.parent/first['generation']/path.name),first['table_sha256'])
    def test_changed_native_text_rejected_by_public_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary).resolve()/'pk.dat';self.write(path,np.array([[1.,2.],[3.,4.]]))
            path.write_text('1 9\n3 9\n')
            with self.assertRaises(ValueError):_load_numeric_table(path)
    def test_initial_interrupted_publication_closes_discovery(self):
        from core.foundation import atomic_copy
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary).resolve()/'pk.dat'
            def interrupt(source,destination):
                result=atomic_copy(source,destination)
                if Path(destination)==path:raise KeyboardInterrupt()
                return result
            with patch('core.data.atomic_copy',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):self.write(path,np.array([[1.,2.],[3.,4.]]))
            with self.assertRaisesRegex(ValueError,'pending'):verify_numeric_table(path)
            journal=json.loads(Path(str(path)+'.publication.json').read_text())
            self.assertEqual(journal['state'],'PENDING')
            recover_numeric_table(path,journal['generation'])
            self.assertEqual(json.loads(Path(str(path)+'.publication.json').read_text())['state'],'COMPLETE')

    def test_explicit_generation_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary).resolve()/'pk.dat';record=self.write(path,np.array([[1.,2.],[3.,4.]]))
            path.write_text('partial')
            recovered=recover_numeric_table(path,record['generation'])
            self.assertEqual(recovered,record)

    def test_failed_publication_retains_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary).resolve()/'pk.dat'
            with patch('core.data.publish_directory',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):self.write(path,np.array([[1.,2.],[3.,4.]]))
            self.assertFalse(path.exists());self.assertTrue(list(path.parent.glob('.numeric-table-*/failure.json')))
    def test_native_background_has_canonical_epoch_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'background.dat';source.write_text('hashed synthetic background')
            a=np.array([.01,.05,.1,.2,.5,1.]);data=np.column_stack((1/a-1,a**-1.5,a,np.ones_like(a)))
            with patch('core.cosmology.class_background_table',return_value=({'H':1,'growth_D':2,'growth_f':3},data)):
                record=write_gadget_background_table(source,root/'gadget.dat',.05)
            verified=verify_numeric_table(root/'gadget.dat')
            self.assertEqual(verified['context']['canonical_time'],'scale_factor')
            self.assertEqual(verified['columns'],['a','E','D','f'])
            np.testing.assert_allclose(np.loadtxt(root/'gadget.dat')[-1],[1,1,1,1])

    def test_ngenic_preserves_headerless_native_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'pk.dat';np.savetxt(source,[[1.,2.],[3.,4.]])
            target=root/'ngenic.dat';convert_pk_to_ngenic(source,target,.7)
            self.assertFalse(target.read_text().startswith('#'));verify_numeric_table(target)
            np.testing.assert_allclose(np.loadtxt(target)[:,1],np.log10(np.array([1.,3.])**3*np.array([2.,4.])/(2*np.pi**2)))

    def test_verified_staging_keeps_numeric_history_and_source_bytes(self):
        from core.data import copy_numeric_table
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source=root/'original/pk.dat';target=root/'native/pk.dat'
            record=self.write(source,np.array([[1.,2.],[3.,4.]]));before=digest(source)
            copy_numeric_table(source,target)
            self.assertEqual(verify_numeric_table(target),record)
            self.assertEqual(digest(source),before)
            self.assertTrue((target.parent/record['generation']/'columns.npz').is_file())
            with self.assertRaises(FileExistsError):copy_numeric_table(source,target)
            (target.parent/record['generation']/'columns.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):verify_numeric_table(target)

# ========================================================================
# test_analysis_contract
import errno
test_analysis_contract_ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(test_analysis_contract_ROOT/'core'),str(test_analysis_contract_ROOT)]
from analysis import analyze_selected_products,verify_analysis_bundle
from figures import export_figure,verify_figure_export
from matplotlib import pyplot as plt


class AnalysisContractTests(unittest.TestCase):
    def export(self,root):
        with patch('figures.generate_single_run_story_figures',return_value={'status':'UNAVAILABLE','reason':'unit fixture has no large-scale field validation'}):
            summary=analyze_selected_products(root,['subhalos'],include_pdf_report=False)
        return Path(summary['output_directory'])

    def test_native_text_inputs_and_portable_artifact_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);catalogue_fixture(root);trees=root/'data/tree.list';trees.write_text('# saved native text catalogue\n')
            output=self.export(root);record=verify_analysis_bundle(output,verify_sources=True)
            self.assertIn(str(trees.resolve()),[item['path'] for item in record['data_contract']['input_hashes']])
            with h5py.File(output/'analysis_products.h5') as f:
                self.assertIn('data_contract_json',f.attrs)
                self.assertTrue(f['snap_000/subhalos/data/fields/SubhaloIDMostbound'].fletcher32)
            import shutil
            portable=root/'portable';shutil.copytree(output,portable);verify_analysis_bundle(portable)
            csv=next(portable.rglob('*.csv'));csv.write_text(csv.read_text()+'corruption\n')
            with self.assertRaisesRegex(ValueError,'artifact missing or changed'):verify_analysis_bundle(portable)

    def test_changed_or_added_inputs_preserve_failed_measurement_and_do_not_publish(self):
        for change in ['modified','added']:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);catalogue_fixture(root)
                import analysis as analyze
                original=analyze._analyze_one_product
                def mutate(*args,**kwargs):
                    result=original(*args,**kwargs)
                    if change=='added':(root/'data/new.tree').write_text('new unmeasured tree')
                    else:
                        with h5py.File(root/'data/snapshot_000.hdf5','r+') as f:f['PartType4/Coordinates'][0,0]=98.
                    return result
                with patch('analysis._analyze_one_product',side_effect=mutate):
                    with self.assertRaisesRegex(ValueError,'input inventory or content changed'):self.export(root)
                self.assertFalse(list((root/'results').glob('analysis_products_*')))
                failures=list(root.glob('.analysis_products_*/analysis_failure.json'));self.assertEqual(len(failures),1)
                self.assertTrue((failures[0].parent/'results/analysis_products.h5').is_file())

    def test_readable_hash_valid_output_still_requires_intact_hdf5_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);catalogue_fixture(root);output=self.export(root);h5=output/'analysis_products.h5'
            with h5py.File(h5) as f:offset=f['snap_000/subhalos/data/fields/SubhaloIDMostbound'].id.get_chunk_info_by_coord((0,)).byte_offset
            with h5.open('r+b') as stream:
                stream.seek(offset);value=stream.read(1);stream.seek(offset);stream.write(bytes([value[0]^1]))
            p=output/'analysis_provenance.json';record=json.loads(p.read_text())
            for item in record['output_artifacts']:
                if item['relative_path']==h5.name:item['sha256']=digest(h5)
            atomic_json(p,record);p.with_name(p.name+'.sha256').write_text(digest(p)+'\n')
            with self.assertRaises(OSError):verify_analysis_bundle(output)

    def test_figure_plotted_data_masks_checksums_and_prior_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fig,ax=plt.subplots();ax.plot([1,2,3],np.ma.array([2.,4.,6.],mask=[False,True,False]));ax.set_xlabel('radius [kpc]');ax.set_ylabel('mass [Msun]')
            export_figure(fig,root/'profile.png',formats=['png'])
            p=root/'profile.figure.json';old=digest(p);record=verify_figure_export(p)
            with np.load(root/record['plot_data'],allow_pickle=False) as data:
                np.testing.assert_array_equal(data['axis_0_line_0_y_mask'],[False,True,False])
                np.testing.assert_array_equal(data['axis_0_line_0_y'],[2.,4.,6.])
            fig,ax=plt.subplots();ax.plot([1,2],[3,4]);export_figure(fig,root/'profile.png',formats=['png'])
            historical=next((root/'.figure_history').glob('*/profile.figure.json'));self.assertEqual(digest(historical),old);verify_figure_export(historical)
            asset=root/'profile.plotdata.npz';asset.write_bytes(asset.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'asset missing or changed'):verify_figure_export(p)

    def test_failed_render_keeps_previous_manifest_and_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);fig,ax=plt.subplots();ax.plot([1,2],[3,4]);export_figure(fig,root/'plot.png',formats=['png'])
            old={p.name:digest(p) for p in root.iterdir() if p.is_file()}
            fig,ax=plt.subplots();ax.plot([1,2],[8,9]);save=fig.savefig;calls=[]
            def fail_second(path,**kwargs):
                calls.append(path)
                if len(calls)==2:raise OSError(errno.ENOSPC,'disk full')
                return save(path,**kwargs)
            with patch.object(fig,'savefig',side_effect=fail_second):
                with self.assertRaises(OSError):export_figure(fig,root/'plot.png',formats=['png','pdf'])
            self.assertEqual({p.name:digest(p) for p in root.iterdir() if p.is_file()},old)
            verify_figure_export(root/'plot.figure.json');plt.close(fig)

# ========================================================================
# test_analysis_dataset_semantics

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from analysis import profile_halo_scatter
from core.data import attach_analysis_dataset_units, attach_legacy_analysis_units, write_dataset_contract


class AnalysisDatasetSemanticsTests(unittest.TestCase):
    def test_serialized_profile_scatter_preserves_field_units_and_missing_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'products.h5'
            epoch={'scale_factor':0.25,'redshift':3.,'cosmic_time_gyr':1.5}
            provenance={'source_inventory_sidecar':'analysis_provenance.json'}
            payload={'profiles':[{'halo_catalogue_row':8,
                'center_group_pos_comoving_mpc_h':np.array([0.,2.,3.]),
                'dm_mass_density_proper_msun_h_per_mpc_h3':np.array([4.,np.nan])}],
                'density_profile_scatter':{'median':np.array([4.,np.nan]),'radius_over_r200c':[.1,.2]},
                'velocity_profile_scatter':{'median':[12.,13.]},
                'gas_profiles':{'profile_halo_to_halo_scatter':{'gas_temperature_mass_weighted_K':{'median':[8000.,9000.]}}},
                'unreviewed_scalar':4.}
            with h5py.File(path,'w') as h5:
                _write_product_hdf5(h5,'data',payload)
                attach_analysis_dataset_units(h5['data'],'profiles',epoch=epoch,provenance=provenance)
            # Inspect the reopened deliverable, including nested producer paths.
            with h5py.File(path,'r') as h5:
                data=h5['data']
                self.assertEqual(data['density_profile_scatter/median'].attrs['units'],
                                 '(Msun/h)/(proper Mpc/h)^3')
                self.assertEqual(data['velocity_profile_scatter/median'].attrs['units'],'km/s')
                self.assertEqual(data['gas_profiles/profile_halo_to_halo_scatter/gas_temperature_mass_weighted_K/median'].attrs['units'],'K')
                self.assertTrue(np.isnan(data['profiles/item_0000/dm_mass_density_proper_msun_h_per_mpc_h3'][1]))
                center=data['profiles/item_0000/center_group_pos_comoving_mpc_h']
                self.assertEqual(json.loads(center.attrs['coordinate_conventions_json'])['origin'],
                                 'global simulation Cartesian box origin')
                self.assertEqual(json.loads(center.attrs['epoch_json']),epoch)
                self.assertEqual(json.loads(center.attrs['provenance_json']),provenance)
                self.assertEqual(json.loads(center.attrs['shape_json']),[3])
                self.assertEqual(center.attrs['quality_status'],'NOT_ACCEPTED')
                unresolved=data['unreviewed_scalar']
                self.assertEqual(unresolved.attrs['units_status'],'UNDECLARED')
                self.assertEqual(unresolved.attrs['quantity_definition_status'],'UNDECLARED')

    def test_both_physical_velocity_producers_share_histogram_units(self):
        with tempfile.TemporaryDirectory() as temporary, h5py.File(Path(temporary)/'data.h5','w') as h5:
            _write_product_hdf5(h5,'data',{'particle_velocity':{'bin_edges':[1.,2.],
                'probability_density':[1.]},'halo_velocity':{'bin_edges':[1.,2.],
                'probability_density':[1.]}})
            attach_analysis_dataset_units(h5['data'],'kinematics')
            self.assertEqual(h5['data/particle_velocity/bin_edges'].attrs['units'],'km/s')
            self.assertEqual(h5['data/particle_velocity/probability_density'].attrs['units'],'(km/s)^-1')
            self.assertEqual(h5['data/halo_velocity/bin_edges'].attrs['units'],'km/s')

    def test_exact_specialized_id_contract_survives_context_annotation(self):
        with tempfile.TemporaryDirectory() as temporary, h5py.File(Path(temporary)/'data.h5','w') as h5:
            group=h5.create_group('data')
            ids=group.create_dataset('ids',data=np.array([2**63+1,2**64-1],dtype=np.uint64))
            write_dataset_contract(ids,units='identity',identity_namespace='native:dm:run-1',
                quantity='native particle identity',missing_policy='none')
            attach_analysis_dataset_units(group,'subhalos',epoch={'scale_factor':.5})
            self.assertEqual(ids.dtype,np.dtype('uint64'))
            self.assertEqual(ids[1],np.uint64(2**64-1))
            self.assertEqual(ids.attrs['identity_namespace'],'native:dm:run-1')
            self.assertEqual(ids.attrs['quantity_definition_status'],'PRODUCER_DECLARED')
            self.assertEqual(ids.attrs['missing_policy'],'none')
            self.assertEqual(ids.attrs['epoch_status'],'PRODUCER_DECLARED')

    def test_bias_retains_negative_shot_subtracted_bins(self):
        with tempfile.TemporaryDirectory() as temporary, h5py.File(Path(temporary)/'data.h5','w') as h5:
            _write_product_hdf5(h5,'data',{'halo_power_Mpc_h3':[-2.,3.],
                'bias':[np.nan,1.2],'mode_count':np.array([12,24],dtype=np.uint64)})
            attach_analysis_dataset_units(h5['data'],'bias')
            self.assertEqual(h5['data/halo_power_Mpc_h3'][0],-2.)
            self.assertEqual(h5['data/halo_power_Mpc_h3'].attrs['units'],'(Mpc/h)^3')
            self.assertEqual(h5['data/mode_count'].attrs['units'],'Fourier mode count')

    def test_real_profile_summary_units_match_legacy_aliases(self):
        profiles=[{'r200c_comoving_mpc_h':1.,'radius_comoving_mpc_h':np.array([.1,.2,.4,1.]),
                   'gas_temperature_mass_weighted_K':np.array([100.,200.,400.,1000.])}]
        scatter=profile_halo_scatter(profiles,'gas_temperature_mass_weighted_K',4)
        with tempfile.TemporaryDirectory() as temporary, h5py.File(Path(temporary)/'data.h5','w') as h5:
            _write_product_hdf5(h5,'selected',{'gas_profiles':{'profile_halo_to_halo_scatter':{
                'gas_temperature_mass_weighted_K':scatter}}})
            attach_analysis_dataset_units(h5['selected'],'profiles')
            self.assertEqual(h5['selected/gas_profiles/profile_halo_to_halo_scatter/gas_temperature_mass_weighted_K/median'].attrs['units'],'K')
            legacy=h5.create_group('legacy')
            legacy.create_dataset('xi_r',data=[.1,.2])
            legacy.create_dataset('bias_cross_power_Mpc_h3',data=[1.,2.])
            legacy.create_dataset('dm_speed_bin_edges',data=[0.,1.])
            legacy.create_dataset('host_halo_positions_mpc_h',data=[[1.,2.,3.]])
            attach_legacy_analysis_units(legacy,epoch={'scale_factor':.5})
            self.assertEqual(legacy['xi_r'].attrs['units'],'comoving Mpc/h')
            self.assertEqual(legacy['bias_cross_power_Mpc_h3'].attrs['units'],'(Mpc/h)^3')
            self.assertEqual(legacy['dm_speed_bin_edges'].attrs['units'],'km/s')
            self.assertEqual(legacy['host_halo_positions_mpc_h'].attrs['coordinates_status'],'PRODUCER_DECLARED')
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from figures import review_figure_folder,verify_figure_review

class FigureReviewTests(unittest.TestCase):
    def test_sealed_source_and_distinct_review_generations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);figure,axis=plt.subplots();axis.plot([1,2],[3,4])
            export_figure(figure,root/'line',formats=['png'])
            sealed=verify_figure_export(root/'line.figure.json')
            self.assertIn('plot_data_contract',sealed)
            one=review_figure_folder(root);two=review_figure_folder(root)
            self.assertNotEqual(one['review_bundle'],two['review_bundle'])
            self.assertEqual(one['figures'][0]['contract_status'],'SEALED_ARTIFACTS_VERIFIED')
            verify_figure_review(root/one['review_bundle'])
            verify_figure_review(root/two['review_bundle'])

    def test_changed_source_is_recorded_unreadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);figure,axis=plt.subplots();axis.plot([1,2],[3,4])
            export_figure(figure,root/'line',formats=['png'])
            with (root/'line.png').open('ab') as stream:stream.write(b'changed')
            report=review_figure_folder(root)
            self.assertEqual(report['figures'][0]['status'],'UNREADABLE')
            self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')

    def test_review_manifest_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);report=review_figure_folder(root)
            bundle=root/report['review_bundle'];(bundle/'figure_review.json').write_text('{}')
            with self.assertRaises(ValueError):verify_figure_review(bundle)

test_native_text_writes_CORE = Path(__file__).resolve().parent / 'core'
sys.path.insert(0, str(test_native_text_writes_CORE))
from core.foundation import atomic_text
from evolution import write_output_list, run_prepared_gadget


class NativeTextWriteTests(unittest.TestCase):
    def test_exact_utf8_payload_and_flush_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'nested' / 'param.txt'
            events = []
            fsync, replace = os.fsync, os.replace
            def syncing(fd):
                events.append('fsync'); return fsync(fd)
            def replacing(source, destination):
                events.append('rename'); return replace(source, destination)
            value = '# provenance μ\nTimeBegin 1.000000000000e-02\n'
            with patch('core.foundation.os.fsync', side_effect=syncing), \
                 patch('core.foundation.os.replace', side_effect=replacing):
                atomic_text(path, value)
            self.assertEqual(path.read_bytes(), value.encode('utf-8'))
            self.assertEqual(events, ['fsync', 'rename', 'fsync'])

    def test_disk_full_and_invalid_text_preserve_original(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'Config.sh'
            atomic_text(path, 'PERIODIC\n')
            with patch('core.foundation.os.fsync', side_effect=OSError(errno.ENOSPC, 'full')):
                with self.assertRaises(OSError):
                    atomic_text(path, 'CHANGED\n')
            for value in [b'binary', None, '\ud800']:
                with self.subTest(value=repr(value)):
                    with self.assertRaises((TypeError, UnicodeEncodeError)):
                        atomic_text(path, value)
            self.assertEqual(path.read_bytes(), b'PERIODIC\n')
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_bad_schedule_never_replaces_saved_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'output_times.txt'
            write_output_list(path, [99., 1., 0.])
            original = path.read_bytes()
            for values in [[], [1., 1.], [0., 1.], [True], ['1'], [float('nan')],
                           [float('inf')], [-1.], [-2.], [1., 1.-1e-14]]:
                with self.subTest(values=values):
                    with self.assertRaises(ValueError):
                        write_output_list(path, values)
                    self.assertEqual(path.read_bytes(), original)
            self.assertEqual(original, b'1.000000000000e-02\n5.000000000000e-01\n1.000000000000e+00\n')

    def test_schedule_disk_full_preserves_old_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'output_times.txt'
            write_output_list(path, [1., 0.])
            before = path.read_bytes()
            with patch('core.foundation.os.fsync', side_effect=OSError(errno.ENOSPC, 'full')):
                with self.assertRaises(OSError):
                    write_output_list(path, [99., 1., 0.])
            self.assertEqual(path.read_bytes(), before)

    def test_prepared_schedule_tampering_stops_before_build_or_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = {name: root/name for name in ['config.yaml','param.txt','binary','schedule.txt']}
            for name, p in files.items(): p.write_text(name)
            prepared = {k: str(files[v]) for k,v in [('parameter_path','param.txt'),('binary_path','binary'),('output_schedule_path','schedule.txt')]}
            prepared.update({k: digest(files[v]) for k,v in [('parameter_sha256','param.txt'),('binary_sha256','binary'),('output_schedule_sha256','schedule.txt')]})
            manifest = {'status':'GADGET_PREPARED','config_sha256':digest(files['config.yaml']),'prepared_gadget':prepared}
            path = root/'manifest.json'; path.write_text(json.dumps(manifest))
            original = path.read_bytes()
            files['schedule.txt'].write_text('changed schedule')
            with patch('evolution.build_gadget') as build, patch('evolution.subprocess.run') as executing:
                with self.assertRaisesRegex(ValueError, 'output schedule changed'):
                    run_prepared_gadget(root)
                build.assert_not_called(); executing.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_sigkill_before_recipe_rename_and_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root/'param.txt'
            atomic_text(path, 'old reviewed recipe\n')
            script = """
import os,signal,sys
sys.path.insert(0,sys.argv[1])
import core.foundation as research_acceptance
real=os.fsync
def kill(fd):
    real(fd); os.kill(os.getpid(),signal.SIGKILL)
research_acceptance.os.fsync=kill
research_acceptance.atomic_text(sys.argv[2],'new complete recipe\\n')
"""
            result = subprocess.run([sys.executable,'-c',script,str(test_native_text_writes_CORE),str(path)],capture_output=True,timeout=30)
            self.assertEqual(result.returncode,-9,result.stderr.decode())
            self.assertEqual(path.read_text(),'old reviewed recipe\n')
            orphaned = set(root.iterdir())-{path}
            self.assertEqual(len(orphaned),1)
            self.assertEqual(next(iter(orphaned)).read_text(),'new complete recipe\n')
            atomic_text(path,'new complete recipe\n')
            self.assertEqual(path.read_text(),'new complete recipe\n')
            self.assertTrue(all(p.exists() for p in orphaned))

# ========================================================================
# test_zoom_window

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from zoom import (window_response, model_window_covariance, window_gaussian_check,
                         source_mode_sphere, covariance_source_convergence)


class ZoomWindowTests(unittest.TestCase):
    def test_nested_ic_cannot_pass_a_full_periodic_species_launch_gate(self):
        from evolution import measure_and_validate_ic_power_spectrum
        with tempfile.TemporaryDirectory() as temp:
            ic=Path(temp)/'native.hdf5'
            with h5py.File(ic,'w') as source:
                source.create_group('Header').attrs.update(Music_levelmin=4,Music_levelmax=6)
            with patch('evolution.build_gadget') as build:
                with self.assertRaisesRegex(RuntimeError,'window-aware species power'):
                    measure_and_validate_ic_power_spectrum({},Path(temp),ic,target_pk_path='unused-target.dat')
                build.assert_not_called()

    def setUp(self):
        self.box=10000.
        grid=(np.arange(8)+.5)*self.box/8
        self.q=np.array(np.meshgrid(grid,grid,grid,indexing='ij')).reshape(3,-1).T
        self.modes=np.array([[1,0,0],[0,1,0],[1,1,0]])

    def test_full_periodic_window_recovers_independent_density_modes(self):
        operator=window_response(self.q,np.ones(len(self.q)),self.box,self.modes,self.modes)
        expected=np.zeros((6,6))
        for i in range(3): expected[i,2*i]=1; expected[i+3,2*i+1]=1
        np.testing.assert_allclose(operator,expected,atol=2e-15)
        power=np.array([100.,200.,300.])
        covariance=model_window_covariance(operator,power,self.box)
        np.testing.assert_allclose(covariance,np.diag(np.tile(power/(2*10**3),2)),atol=2e-16)

    def test_cartesian_shortcut_matches_direct_sum_and_rejects_false_cartesian_window(self):
        mass=np.ones(len(self.q))
        fast=window_response(self.q,mass,self.box,self.modes,self.modes)
        # A tiny nonuniform mass triggers the general point-sum path.
        mass[0]+=1e-10
        direct=window_response(self.q,mass,self.box,self.modes,self.modes)
        np.testing.assert_allclose(fast,direct,atol=3e-13)
        duplicate=self.q.copy(); duplicate[0]=duplicate[64]
        result=window_response(duplicate,np.ones(len(duplicate)),self.box,self.modes,self.modes)
        self.assertGreater(np.max(abs(result-fast)),1e-4)

    def test_complete_periodic_shells_stabilize_without_claiming_unbounded_tail(self):
        source=source_mode_sphere(4)
        response=window_response(self.q,np.ones(len(self.q)),self.box,self.modes,source)
        result=covariance_source_convergence(response,source,np.ones(len(source)),self.box,[2,3,4],
                                             maximum_fractional_change=.001)
        self.assertEqual(result['status'],'STABLE_DECLARED_SHELLS')
        self.assertEqual(result['omitted_tail_status'],'NOT_BOUNDED')
        self.assertEqual(result['science_acceptance'],'NOT_ACCEPTED')
        with self.assertRaisesRegex(ValueError,'omits modes'):
            covariance_source_convergence(response[:,2:],source[1:],np.ones(len(source)-1),self.box,[2,3,4],
                                          maximum_fractional_change=.001)

    def test_covariance_convergence_cannot_hide_changes_in_weak_direction(self):
        source=source_mode_sphere(3)
        indices={tuple(row):i for i,row in enumerate(source)}
        response=np.zeros((2,2*len(source)))
        i=indices[(1,0,0)]; j=indices[(2,0,0)]
        response[0,2*i]=np.sqrt(2000.)
        response[1,2*i+1]=np.sqrt(2e-8)
        response[1,2*j]=np.sqrt(6e-8)
        result=covariance_source_convergence(response,source,np.ones(len(source)),self.box,[1,2,3],
                                             maximum_fractional_change=.01)
        self.assertEqual(result['status'],'NOT_STABLE')
        self.assertAlmostEqual(result['records'][1]['maximum_fractional_change'],.75)

    def test_patch_operator_matches_direct_vector_displacement_not_naive_box_power(self):
        q=self.q[self.q[:,0]<.375*self.box]
        mass=1+q[:,1]/self.box
        operator=window_response(q,mass,self.box,self.modes,self.modes)
        rng=np.random.default_rng(481)
        coefficients=rng.normal(size=6)*1e-4
        delta=coefficients[::2]+1j*coefficients[1::2]
        k=self.modes*2*np.pi/self.box
        displacement=np.zeros_like(q)
        for wave,value in zip(k,delta):
            displacement+=2*np.real(np.exp(1j*(q@wave))[:,None]*(1j*wave*value/(wave@wave)))
        measured=np.sum(-1j*(displacement@k.T)*np.exp(-1j*(q@k.T))*mass[:,None],axis=0)/mass.sum()
        expected=operator@coefficients
        np.testing.assert_allclose(np.r_[measured.real,measured.imag],expected,atol=1e-19)
        covariance=model_window_covariance(operator,np.ones(3)*100,self.box)
        self.assertGreater(np.max(np.abs(covariance-np.diag(np.diag(covariance)))),.001)
        self.assertFalse(np.allclose(covariance,np.eye(6)*.05))
        # Real-field negative partners give unequal real/imaginary variances.
        self.assertGreater(np.max(np.abs(np.diag(covariance)[:3]-np.diag(covariance)[3:])),.001)

    def test_correlated_covariance_matches_independent_gaussian_injections(self):
        q=self.q[(self.q[:,0]<.75*self.box)&(self.q[:,1]<.75*self.box)]
        operator=window_response(q,np.ones(len(q)),self.box,self.modes,self.modes)
        power=np.array([100.,200.,300.])
        covariance=model_window_covariance(operator,power,self.box)
        rng=np.random.default_rng(412)
        samples=rng.normal(size=(60000,6))*np.sqrt(np.repeat(power/(2*10**3),2))
        observed=samples@operator.T
        np.testing.assert_allclose(np.cov(observed,rowvar=False),covariance,atol=.0015,rtol=.025)
        eig,vec=np.linalg.eigh(covariance)
        sample=vec@np.sqrt(eig)  # whitened chi-square equals dimension, typical sample
        self.assertEqual(window_gaussian_check(sample[:3]+1j*sample[3:],covariance,
            false_rejection_probability=.001)['status'],'PASS')
        self.assertEqual(window_gaussian_check((sample[:3]+1j*sample[3:])*100,covariance,
            false_rejection_probability=.001)['status'],'FAIL')

    def test_missing_negative_partners_singular_covariance_and_bad_inputs_fail(self):
        with self.assertRaises(ValueError):
            window_response(self.q,np.ones(len(self.q)),self.box,self.modes,np.array([[1,0,0],[-1,0,0]]))
        with self.assertRaises(MemoryError):
            window_response(self.q,np.ones(len(self.q)),self.box,self.modes,self.modes,memory_gib=.001)
        self.assertEqual(window_gaussian_check(np.ones(3,dtype=complex),np.zeros((6,6)),
            false_rejection_probability=.001)['status'],'UNRESOLVED_COVARIANCE')
        with self.assertRaises(ValueError):
            model_window_covariance(np.ones((6,6)),[-1.,1.,1.],self.box)

# ========================================================================
# test_galaxy_publication
sys.path[:0]=[str(Path(__file__).resolve().parent/'core'),str(Path(__file__).resolve().parent)]
from halos import build_galaxy_catalogue,build_galaxy_catalogues,verify_galaxy_catalogue


class GalaxyPublicationTests(unittest.TestCase):
    def fixture(self,root):
        test_stellar_history_snapshot(root,0,.5,[2**63+1],[[1.,2.,3.]],bound_count=1)
        with h5py.File(root/'data/fof_subhalo_tab_000.hdf5','a') as source:
            sub=source['Subhalo']
            for name,data in {'SubhaloPos':[[1.,2.,3.]],'SubhaloVel':[[10.,0.,0.]],
                'SubhaloGroupNr':np.array([0],dtype=np.uint64),'SubhaloRankInGr':np.array([0],dtype=np.uint64),
                'SubhaloMassType':[[0.,0.,0.,0.,1.,0.]],'SubhaloHalfmassRadType':[[0.,0.,0.,0.,1.,0.]]}.items():sub[name]=data

    def test_actual_public_output_has_payload_contracts_and_reuses_completed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            result=build_galaxy_catalogue(root,0,minimum_star_particles=0,minimum_dm_particles=0)
            path=Path(result['path']);before=path.read_bytes()
            verify_hdf5_dataset_payloads(path,require_contract=True)
            record=verify_galaxy_catalogue(path,root)
            self.assertEqual(record['galaxy_count'],1)
            with h5py.File(path) as source:
                self.assertEqual(source['GalaxyID'].dtype.name,'uint64')
                self.assertEqual(source['Position'].attrs['units'],'comoving Mpc/h')
                self.assertIn('epoch_json',source['StellarMass'].attrs)
                self.assertEqual(source['GalaxyID'].attrs['identity_namespace'],source['SubhaloID'].attrs['identity_namespace'])
                self.assertNotEqual(source['GalaxyID'].attrs['identity_namespace'],source['HostFOFID'].attrs['identity_namespace'])
                self.assertEqual(source['SFR'].attrs['missing_policy'].split(';')[0],
                    'NaN when the declared aperture/formation/SFR source is unavailable')
            manifest=build_galaxy_catalogues(root,[0],minimum_star_particles=0,minimum_dm_particles=0)
            self.assertEqual(manifest['status'],'COMPLETE');self.assertEqual(path.read_bytes(),before)
            with h5py.File(path,'r+') as source:source['Position'][0,0]=99.
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):verify_galaxy_catalogue(path,root)

    def test_failed_file_publication_preserves_staged_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            with patch('core.foundation.publish_file',side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):build_galaxy_catalogue(root,0)
            parent=root/'results';self.assertFalse((parent/'galaxies_000.hdf5').exists())
            stage=list(parent.glob('.galaxies_000.hdf5.*.tmp'))[0]
            self.assertTrue(stage.with_name(stage.name+'.failure.json').is_file())
            verify_hdf5_dataset_payloads(stage,require_contract=True)
            build_galaxy_catalogue(root,0)
            self.assertTrue(stage.is_file())

# ========================================================================
# test_snapshot_density_modes

test_snapshot_density_modes_ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(test_snapshot_density_modes_ROOT/'core'))
sys.path.insert(0,str(test_snapshot_density_modes_ROOT));sys.path.insert(0,str(test_snapshot_density_modes_ROOT/'tests'))
from evolution import measure_periodic_density_modes
from core.foundation import MPC_CM


def test_snapshot_density_modes_snapshot(path,positions,masses,*,constant=False,box=2.,shards=1,total=None):
    count=len(positions);counts=np.array([0,count,0,0,0,0],dtype=np.uint64)
    totals=counts.copy();totals[1]=count if total is None else total
    table=np.zeros(6)
    if constant:table[1]=float(masses[0])
    with h5py.File(path,'w') as source:
        source.create_group('Header').attrs.update(Time=.1,BoxSize=box,MassTable=table,
            NumPart_ThisFile=counts,NumPart_Total=totals,NumFilesPerSnapshot=shards)
        source.create_group('Parameters').attrs.update(ComovingIntegrationOn=1,HubbleParam=.7,
            UnitLength_in_cm=MPC_CM,UnitMass_in_g=1e10*MSUN_G)
        group=source.create_group('PartType1');group['Coordinates']=np.asarray(positions,dtype=float)
        group['Coordinates'].attrs.update(to_cgs=MPC_CM,a_scaling=1.,h_scaling=-1.)
        if not constant:
            group['Masses']=np.asarray(masses,dtype=float)
            group['Masses'].attrs.update(to_cgs=1e10*MSUN_G,a_scaling=0.,h_scaling=-1.)
    return path


class SnapshotDensityModesTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()

    def test_weighted_two_particle_power_and_periodic_translation(self):
        path=test_snapshot_density_modes_snapshot(self.root/'snapshot.hdf5',[[0,0,0],[1,0,0]],[1,2])
        report=measure_periodic_density_modes([path],[[1,0,0]],species=(1,))
        self.assertAlmostEqual(report['power_mpc_h3'][0],8/9,places=12)
        self.assertAlmostEqual(report['poisson_reference_mpc_h3'],8*5/9,places=12)
        shifted=test_snapshot_density_modes_snapshot(self.root/'shifted.hdf5',[[.3,0,0],[1.3,0,0]],[1,2])
        translated=measure_periodic_density_modes([shifted],[[1,0,0]],species=(1,))
        np.testing.assert_allclose(report['power_mpc_h3'],translated['power_mpc_h3'],rtol=1e-12)
        self.assertFalse(report['shot_noise_subtracted'])

    def test_uniform_lattice_and_constant_mass_agree_with_variable_mass(self):
        xyz=np.stack(np.meshgrid(*[np.arange(8)*2/8]*3,indexing='ij'),axis=-1).reshape(-1,3)
        modes=[[1,0,0],[1,1,0],[0,0,2]]
        a=test_snapshot_density_modes_snapshot(self.root/'constant.hdf5',xyz,np.ones(len(xyz)),constant=True)
        b=test_snapshot_density_modes_snapshot(self.root/'variable.hdf5',xyz,np.ones(len(xyz)))
        left=measure_periodic_density_modes([a],modes,species=(1,))
        right=measure_periodic_density_modes([b],modes,species=(1,))
        np.testing.assert_allclose(left['power_mpc_h3'],np.zeros(3),atol=1e-28)
        np.testing.assert_allclose(left['power_mpc_h3'],right['power_mpc_h3'],atol=1e-28)

    def test_complete_shards_match_single_file_and_missing_shard_fails(self):
        a=test_snapshot_density_modes_snapshot(self.root/'snapshot.0.hdf5',[[0,0,0]],[1],shards=2,total=2)
        b=test_snapshot_density_modes_snapshot(self.root/'snapshot.1.hdf5',[[1,0,0]],[2],shards=2,total=2)
        report=measure_periodic_density_modes([a,b],[[1,0,0]],species=(1,))
        self.assertAlmostEqual(report['power_mpc_h3'][0],8/9,places=12)
        with self.assertRaisesRegex(ValueError,'missing native shards'):
            measure_periodic_density_modes([a],[[1,0,0]],species=(1,))
        with h5py.File(b,'r+') as f:f['Header'].attrs['BoxSize']=3.
        with self.assertRaisesRegex(ValueError,'shards disagree'):
            measure_periodic_density_modes([a,b],[[1,0,0]],species=(1,))

    def test_invalid_modes_units_and_nonpositive_mass_are_rejected(self):
        path=test_snapshot_density_modes_snapshot(self.root/'snapshot.hdf5',[[0,0,0],[1,0,0]],[1,2])
        for modes in ([[0,0,0]],[[1.,0.,0.]],[[1,0,0],[1,0,0]],[[True,False,False]]):
            with self.subTest(modes=modes),self.assertRaises(ValueError):
                measure_periodic_density_modes([path],modes,species=(1,))
        with h5py.File(path,'r+') as f:f['PartType1/Coordinates'].attrs['a_scaling']=0.
        with self.assertRaisesRegex(ValueError,'comoving coordinate units'):
            measure_periodic_density_modes([path],[[1,0,0]],species=(1,))
        with h5py.File(path,'r+') as f:
            f['PartType1/Coordinates'].attrs['a_scaling']=1.;f['PartType1/Masses'][0]=0.
        with self.assertRaisesRegex(ValueError,'positive mass'):
            measure_periodic_density_modes([path],[[1,0,0]],species=(1,))

    def test_public_campaign_measurement_binds_secondary_shards(self):
        import json
        from unittest.mock import patch
        from verification import campaign_fixture,synthetic_ic_phase_inputs
        from evolution import build_start_epoch_campaign,audit_start_epoch_phases,_read_measurement
        from commands import run_start_epoch_campaign_main as main
        from core.foundation import atomic_json
        spec=campaign_fixture(self.root/'source')
        spec['observables']={'density_modes':{**spec['observables']['shape'],'units':'(Mpc/h)^3'}}
        spec['start_epoch_campaign']['observables']={'density_modes':{
            'units':'(Mpc/h)^3','bin_labels':['1,0,0'],'absolute_tolerance':1.,
            'relative_tolerance':.01,'absolute_floor':.001}}
        plan=build_start_epoch_campaign(spec,[49.,99.,199.],self.root/'campaign')
        campaign=self.root/'campaign/campaign.json'
        phase=self.root/'phases.json'
        audit_start_epoch_phases(campaign,synthetic_ic_phase_inputs(self.root/'ics',plan),phase,modes=[[1,0,0]])
        branch=plan['branches'][0];sets=[]
        for i,a in enumerate((.1,.5)):
            paths=[]
            for shard,(x,mass) in enumerate(((0,1),(5,2))):
                path=test_snapshot_density_modes_snapshot(self.root/f'snapshot_{i:03d}.{shard}.hdf5',[[x,0,0]],[mass],
                    box=10.,shards=2,total=2)
                with h5py.File(path,'r+') as f:
                    f['Header'].attrs['Time']=a
                    f['Parameters'].attrs.update(TimeBegin=branch['start_scale_factor'],OmegaBaryon=.05)
                paths.append(str(path))
            sets.append(paths)
        request=self.root/'request.json';output=self.root/'measurements.json'
        atomic_json(request,{'branch_id':branch['id'],'snapshot_sets':sets,'modes':[[1,0,0]],'ic_phase_audit':str(phase)})
        with patch('sys.argv',['commands.py','measure-density-modes','--campaign',str(campaign),'--request',str(request),'--output',str(output)]):
            self.assertEqual(main(),0)
        report,arrays=_read_measurement(output,plan,campaign)
        np.testing.assert_allclose(arrays['observable_density_modes'],1000/9)
        self.assertIn(sets[1][1],report['sources'])
        with h5py.File(sets[1][1],'r+') as f:f['PartType1/Masses'][0]=3.
        with self.assertRaisesRegex(ValueError,'source changed'):_read_measurement(output,plan,campaign)

# ========================================================================
# test_research_acceptance

test_research_acceptance_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(test_research_acceptance_ROOT), str(test_research_acceptance_ROOT / 'core')]
from core.foundation import validate_spec, acceptance_report, canonical_digest
from core.stream_resolution import stream_feasibility, configured_stream_preflight, saved_stream_cadence


def spec_fixture():
    rule = {'metric': 'maximum_axis_ratio_error', 'units': 'dimensionless',
            'minimum': 0., 'maximum': .05, 'evidence_kind': 'pilot'}
    return {'schema_version': 1, 'experiment_id': 'shape-control',
            'cosmology_role': 'conditioned_on_input_H0', 'target_population': 'declared synthetic halo controls',
            'species': ['dark_matter'], 'resource_budget': {'memory_gib': 2., 'disk_gib': 2., 'wall_hours': 1., 'cpu_slots': 1},
            'excluded_claims': ['H0 inference'], 'stream_resolution': None,
            'observables': {'shape': {'definition': 'mass quadrupole b/a and c/a within declared aperture',
                                    'units': 'dimensionless', 'selection': 'all control haloes',
                                    'required_data': ['particle coordinates and masses'], 'domain': 'synthetic control only',
                                    'tests': {'pilot': rule, 'convergence': dict(rule, evidence_kind='convergence')}}}}


def resolution_fixture():
    return {'minimum_width_proper_kpc': .1, 'minimum_dispersion_km_s': 1.,
            'minimum_progenitor_mass_msun': 1.e5, 'minimum_perturber_mass_msun': 1.e6,
            'minimum_star_members': 100, 'minimum_dm_members': 100,
            'maximum_softening_width_ratio': .1, 'maximum_output_crossing_fraction': .25,
            'maximum_relative_speed_km_s': 100.}


class AcceptanceTests(unittest.TestCase):
    def test_saved_cadence_remeasures_delayed_epochs_and_checks_background(self):
        import numpy as np
        spec = spec_fixture()
        spec['stream_resolution'] = resolution_fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            background = root / 'background.dat'
            background.write_text('hashed control')
            (root / 'manifest.json').write_text(json.dumps({'input_artifacts': {
                'boltzmann_background': {'path': str(background), 'sha256': digest(background)}}}))
            # A delayed saved epoch yields a nominal 1 Myr gap. With no
            # numerical clock bound this remains unresolved, never a PASS.
            with (patch('core.evolution.load_config', return_value={}),
                  patch('core.evolution.collect_gadget_snapshot_schedule', return_value={
                      'status': 'DEVIATED', 'rows': [{'actual_redshift': 1.}, {'actual_redshift': 0.}]}),
                  patch('core.cosmology.class_background_table', return_value=(
                      {'z': 0, 'proper time [Gyr]': 1}, np.array([[1., 1.], [0., 1.001]])))):
                report = saved_stream_cadence(spec, root)
                self.assertEqual(report['status'], 'UNRESOLVED_CLOCK')
                self.assertAlmostEqual(report['actual_intervals_myr'][0], 1.)
                self.assertEqual(report['requested_epoch_status'], 'DEVIATED')
                background.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'unchanged'):
                    saved_stream_cadence(spec, root)

    def test_stream_acceptance_cannot_omit_saved_epoch_evidence(self):
        spec = spec_fixture()
        spec['stream_resolution'] = resolution_fixture()
        with tempfile.TemporaryDirectory() as temp:
            report = acceptance_report({'products': ['shape'], 'research_spec': spec}, {}, Path(temp))
            self.assertEqual(report['context_checks']['saved_stream_cadence']['status'], 'FAIL')
            self.assertEqual(report['status'], 'NOT_ACCEPTED')

    def test_missing_evidence_never_accepts(self):
        manifest = {'products': ['shape'], 'research_spec': spec_fixture()}
        report = acceptance_report(manifest, {}, test_research_acceptance_ROOT)
        self.assertEqual(report['status'], 'NOT_ACCEPTED')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            acceptance_report(dict(manifest, products=[]), {}, test_research_acceptance_ROOT)

    def test_changed_and_added_raw_data_block_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'data').mkdir()
            data = root / 'data' / 'snapshot.hdf5'
            data.write_bytes(b'control')
            config = root / 'config.yaml'
            config.write_text('control configuration')
            manifest = {'products': ['shape'], 'research_spec': spec_fixture(),
                        'config_sha256': digest(config), 'data_sha256': {'data/snapshot.hdf5': digest(data)}}
            report = acceptance_report(manifest, {}, root)
            self.assertEqual(report['context_checks']['input_data']['status'], 'PASS')
            (root / 'data' / 'new_snapshot.hdf5').write_bytes(b'new epoch')
            report = acceptance_report(manifest, {}, root)
            self.assertEqual(report['context_checks']['input_data']['status'], 'FAIL')
            config.write_text('changed cosmology')
            self.assertEqual(acceptance_report(manifest, {}, root)['context_checks']['configuration']['status'], 'FAIL')

    def test_measurements_thresholds_checksums_and_manifest_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'measurements.json'
            data.write_text('{"error":0.03}')
            manifest = {'products': ['shape'], 'research_spec': spec_fixture()}
            record = {'manifest_sha256': canonical_digest(manifest), 'metric': 'maximum_axis_ratio_error',
                      'units': 'dimensionless', 'value': .03, 'files': [{'path': data.name, 'sha256': digest(data)}]}
            evidence = {'shape': {kind: dict(record, evidence_kind=kind) for kind in ('pilot', 'convergence')}}
            self.assertEqual(acceptance_report(manifest, evidence, root)['status'], 'ACCEPTED')
            for key, value in [('value', .06), ('value', float('nan')), ('value', True),
                               ('manifest_sha256', 'stale'), ('evidence_kind', 'render'), ('units', 'kpc')]:
                bad = copy.deepcopy(evidence)
                bad['shape']['pilot'][key] = value
                self.assertEqual(acceptance_report(manifest, bad, root)['status'], 'NOT_ACCEPTED')
            data.write_text('changed')
            self.assertEqual(acceptance_report(manifest, evidence, root)['status'], 'NOT_ACCEPTED')

    def test_science_needs_both_pilot_and_convergence(self):
        spec = spec_fixture()
        del spec['observables']['shape']['tests']['convergence']
        with self.assertRaisesRegex(ValueError, 'pilot and convergence'):
            validate_spec(spec)

    def test_no_h0_recovery_or_undefined_limits(self):
        spec = spec_fixture()
        spec['cosmology_role'] = 'infer_H0'
        with self.assertRaisesRegex(ValueError, 'conditioned'):
            validate_spec(spec)
        spec = spec_fixture()
        spec['observables']['shape']['tests']['pilot'].update(minimum=None, maximum=None)
        with self.assertRaisesRegex(ValueError, 'numerical bound'):
            validate_spec(spec)

    def test_fast_encounter_cadence_not_just_velocity_dispersion(self):
        request = resolution_fixture()
        report = stream_feasibility(request, dm_mass_msun=1e3, tracer_mass_upper_msun=100.,
                                    softening_proper_kpc=.001, output_intervals_myr=[.1])
        self.assertEqual(report['status'], 'PASS')
        # 100 km/s crosses 0.1 kpc in 0.97779 Myr; a 1-Myr cadence fails.
        report = stream_feasibility(request, dm_mass_msun=1e3, tracer_mass_upper_msun=100.,
                                    softening_proper_kpc=.001, output_intervals_myr=[1.])
        self.assertFalse(report['checks']['output_crossing_fraction']['passed'])
        self.assertAlmostEqual(report['checks']['output_crossing_fraction']['measured'], 1.022712165, places=7)

    def test_sampling_and_spline_force_support_cuts(self):
        report = stream_feasibility(resolution_fixture(), dm_mass_msun=1.e5, tracer_mass_upper_msun=1.e4,
                                    softening_proper_kpc=.004, output_intervals_myr=[.1])
        self.assertEqual(report['status'], 'FAIL')
        for name in ('progenitor_star_sampling', 'perturber_dm_sampling', 'force_support_width_ratio'):
            self.assertFalse(report['checks'][name]['passed'])
        with self.assertRaisesRegex(ValueError, 'two science epochs'):
            stream_feasibility(resolution_fixture(), dm_mass_msun=1., tracer_mass_upper_msun=1.,
                               softening_proper_kpc=.001, output_intervals_myr=[])

    def test_numpy_derived_feasibility_can_be_saved_as_json(self):
        report = stream_feasibility(resolution_fixture(), dm_mass_msun=np.float64(1.e3),
                                    tracer_mass_upper_msun=np.float64(100.),
                                    softening_proper_kpc=np.float64(.001),
                                    output_intervals_myr=[np.float64(.1)])
        restored = json.loads(json.dumps(report, allow_nan=False))
        self.assertEqual(restored['status'], 'PASS')
        self.assertIs(type(restored['checks']['force_support_width_ratio']['passed']), bool)

    def test_configured_cadence_uses_class_time_and_physical_h_units(self):
        import numpy as np
        spec = spec_fixture()
        spec['stream_resolution'] = resolution_fixture()
        cfg = {'cosmology': {'H0': 50., 'omega_b': .025, 'omega_cdm': .125},
               'simulation': {'physics': 'hydro', 'box_mpc_h': .001, 'particles_per_dimension': 32,
                              'snapshot_redshift_list': [1., 0.], 'softening_kpc_h': .001,
                              'softening_max_physical_kpc_h': .0005}, 'hydro': {'star_formation': True}}
        with tempfile.TemporaryDirectory() as temp:
            background = Path(temp) / 'bg.dat'
            background.write_text('source')
            with patch('core.cosmology.class_background_table', return_value=(
                {'z': 0, 'proper time [Gyr]': 1}, np.array([[1., 1.], [0., 1.0001]]))):
                report = configured_stream_preflight(spec, cfg, background)
            # The mocked two-row clock provides no numerical error envelope;
            # its nominal interval alone cannot certify the cadence.
            self.assertEqual(report['status'], 'UNRESOLVED_CLOCK')
            self.assertIsNone(report['checks']['output_crossing_fraction']['passed'])
            self.assertAlmostEqual(report['inputs']['maximum_softening_proper_kpc'], .001)
            self.assertAlmostEqual(report['inputs']['output_intervals_myr'][0], .1)
            cfg['simulation']['zoom_softening_mass_ratios'] = [8.]
            with self.assertRaisesRegex(ValueError, 'Zoom'):
                configured_stream_preflight(spec, cfg, background)

    def test_unexpected_exception_preserves_stage_and_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue_fixture(root)
            with patch('commands.analyze_run', side_effect=RuntimeError('deliberate native failure')):
                with self.assertRaisesRegex(RuntimeError, 'Requested products are incomplete'):
                    process_run(root, ['shape'])
            status = json.loads(next((root / 'results').glob('workflow_*.json')).read_text())
            self.assertEqual(status['status'], 'INCOMPLETE')
            self.assertEqual(status['science_status'], 'NOT_ACCEPTED')
            self.assertEqual(status['current_stage'], 'products_finished')
            self.assertIn('deliberate native failure', status['unavailable']['scientific_analysis'])
            self.assertFalse(status['resume']['automatic_resume_safe'])

    def test_keyboard_interrupt_preserves_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue_fixture(root)
            with patch('commands.analyze_run', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    process_run(root, ['shape'])
            status = json.loads(next((root / 'results').glob('workflow_*.json')).read_text())
            self.assertEqual(status['status'], 'INTERRUPTED')
            self.assertEqual(status['product_status'], 'INTERRUPTED')

    def test_generated_products_do_not_imply_scientific_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            catalogue_fixture(root)
            with patch('commands.analyze_run', return_value={'output_directory': 'fixture', 'skipped': {}}):
                result = process_run(root, ['shape'], research_spec=spec_fixture())
            self.assertEqual(result['product_status'], 'COMPLETE')
            self.assertEqual(result['science_status'], 'NOT_ACCEPTED')
            manifest = json.loads(Path(result['artifacts']['research_manifest']).read_text())
            self.assertIn('data/snapshot_000.hdf5', manifest['data_sha256'])
test_saved_dataset_units_ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(test_saved_dataset_units_ROOT/'core'),str(test_saved_dataset_units_ROOT)]
from analysis import analyze_run
from figures import plot_saved_analysis_curves


def saved_curve(path,product='pk'):
    with h5py.File(path,'w') as f:
        snap=f.create_group('snap_007');snap.attrs['epoch_contract_json']=json.dumps(epoch_contract(.5))
        if product=='pk':values={'k_h_Mpc':np.array([.1,.2,.4]),'P_Mpc_h3':np.array([10.,0.,-2.]),
                               'P_gaussian_1sigma_Mpc_h3':np.array([1.,2.,3.]),'mode_count':np.array([4,8,16]),
                               'unreviewed_quantity':np.array([42.])}
        else:values={'mass_msun_h':np.array([1e9,1e10,1e11]),'dn_dlogM':np.array([3.,1.,0.]),
                     'cumulative_n_gt':np.array([.3,.1,.01])}
        group=snap.create_group(product);_write_product_hdf5(group,'data',values)
        attach_analysis_dataset_units(group['data'],product)


class SavedUnitTests(unittest.TestCase):
    def test_power_units_are_from_saved_data_and_nonpositive_signal_is_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'analysis.h5';saved_curve(source)
            figures=plot_saved_analysis_curves(source,root/'plots');self.assertEqual(len(figures),1)
            record=verify_figure_export(root/'plots/pk_saved_snap_007.figure.json')
            self.assertEqual(record['scientific_metadata']['axis_units'],{'x':'h/Mpc','y':'(Mpc/h)^3'})
            self.assertEqual(record['scientific_metadata']['source_hdf5_sha256'],digest(source))
            self.assertEqual(record['scientific_metadata']['nonpositive_ordinate_count'],2)
            with np.load(root/'plots'/record['plot_data'],allow_pickle=False) as f:
                np.testing.assert_array_equal(f['axis_0_line_0_y'],[10.,0.,-2.])
            with h5py.File(source) as f:
                self.assertEqual(f['snap_007/pk/data/unreviewed_quantity'].attrs['units_status'],'UNDECLARED')
                self.assertEqual(f['snap_007/pk/data/unreviewed_quantity'].attrs['units'],'UNDECLARED')

    def test_missing_units_and_incompatible_uncertainty_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'analysis.h5';saved_curve(source)
            with h5py.File(source,'r+') as f:del f['snap_007/pk/data/k_h_Mpc'].attrs['units']
            with self.assertRaisesRegex(ValueError,'producer-declared units'):plot_saved_analysis_curves(source,root/'missing')
            with h5py.File(source,'r+') as f:
                f['snap_007/pk/data/k_h_Mpc'].attrs['units']='h/Mpc'
                f['snap_007/pk/data/P_gaussian_1sigma_Mpc_h3'].attrs['units']='dimensionless'
            with self.assertRaisesRegex(ValueError,'incompatible units'):plot_saved_analysis_curves(source,root/'bad_error')

    def test_cumulative_hmf_is_density_and_labels_follow_saved_declarations(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'analysis.h5';saved_curve(source,'hmf')
            with h5py.File(source,'r+') as f:
                self.assertEqual(f['snap_007/hmf/data/cumulative_n_gt'].attrs['units'],'(Mpc/h)^-3')
                # No values are silently converted or hard-coded label is substituted.
                f['snap_007/hmf/data/mass_msun_h'].attrs['units']='custom declared mass unit'
            plot_saved_analysis_curves(source,root/'plots')
            record=verify_figure_export(root/'plots/hmf_saved_snap_007.figure.json')
            self.assertIn('custom declared mass unit',record['axes'][0]['x_label'])

    def test_full_and_selected_routes_cannot_be_requested_together(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'config.yaml').write_text('analysis: {}\n')
            with self.assertRaisesRegex(ValueError,'selected products or explicit full analysis'):
                analyze_run(root,products=['pk'],full_analysis=True)

    def test_legacy_full_dataset_aliases_and_portable_source_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'analysis.h5'
            with h5py.File(source,'w') as f:
                group=f.create_group('snap_001');group.attrs['epoch_contract_json']=json.dumps(epoch_contract(.8))
                group['hmf_mass']=[1e8,1e9];group['hmf_dn_dlogM']=[3.,.5]
                group['pk_k']=[.2,.4];group['pk_P']=[10.,20.];group['pk_P_gaussian_1sigma']=[1.,2.]
                attach_legacy_analysis_units(group)
            self.assertEqual(len(plot_saved_analysis_curves(source,root)),2)
            record=verify_figure_export(root/'pk_saved_snap_001.figure.json')
            self.assertEqual(record['units_status'],'SAVED_DATASET_DECLARATIONS')
            self.assertEqual(record['scientific_metadata']['source_hdf5'],'analysis.h5')
            self.assertEqual(record['scientific_metadata']['source_hdf5_sha256'],digest(root/record['scientific_metadata']['source_hdf5']))

# ========================================================================
# test_build_cache_receipt
sys.path[:0]=[str(Path(__file__).resolve().parent/'core')]
from evolution import verified_build_cache

class BuildCacheReceiptTests(unittest.TestCase):
    def test_missing_partial_corrupt_or_changed_build_never_hits_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'Gadget4';signature=root/'signature.json';receipt=root/'receipt.json'
            value={'source':'hash','flags':'declared configuration'}
            binary.write_bytes(b'actual control binary payload');atomic_json(signature,value)
            self.assertFalse(verified_build_cache(binary,signature,receipt,value))
            record={'schema':'native-build-cache-v1','status':'COMPLETE','signature_sha256':canonical_digest(value),'binary_sha256':digest(binary)}
            atomic_json(receipt,record);self.assertTrue(verified_build_cache(binary,signature,receipt,value))
            atomic_json(receipt,dict(record,status='PENDING'));self.assertFalse(verified_build_cache(binary,signature,receipt,value))
            atomic_json(receipt,record);binary.write_bytes(b'corrupt');self.assertFalse(verified_build_cache(binary,signature,receipt,value))
            self.assertFalse(verified_build_cache(binary,signature,receipt,dict(value,flags='different')))
            receipt.write_text('{');self.assertFalse(verified_build_cache(binary,signature,receipt,value))

# ========================================================================
# test_resolution_budget
def native_control_fixture(relative):
    configured = os.environ.get('RESEARCH_NATIVE_CONTROL_ROOT')
    root = Path(configured).resolve() if configured else Path(__file__).resolve().parent/'results/native-controls/native'
    path = root/relative
    if configured and not path.exists():
        raise FileNotFoundError('Explicit native control fixture is missing: '+str(path))
    return path

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))
from zoom import (empty_census, accumulate_census, seal_census,
    evaluate_resolution, measure_actual_mass_levels, wdm_fragmentation_mask,
    estimate_then_calibrate_stream_heating, validate_resolution_request)
from zoom import softening_plan, proper_species_softening
from zoom import radial_particle_census


def test_resolution_budget_request():
    return {'schema':'observable-resolution-request-v1', 'radial_edges_proper_kpc':[0,1,2,4],
        'maximum_coarse_mass_fraction':0., 'minimum_relaxation_age_ratio':.01,
        'maximum_timestep_crossing_fraction':.1, 'maximum_force_support_fraction':.1,
        'observables':[{'name':'profile','kind':'halo_profile','minimum_radius_proper_kpc':2.,
            'minimum_members':10,'species':1,'maximum_relative_error':.05,'absolute_error_floor':1.,'units':'Msun'}],
        'refinement':{'maximum_levels':2,'mass_reduction_factor':8.,'softening_reduction_factor':2.,
            'timestep_reduction_factor':2.,'memory_gib':2.,'disk_gib':4.}}


def target():
    return {'age_myr':1., 'maximum_proper_timestep_myr':.001, 'relative_speed_km_s':100.}


def census(masses=None):
    masses=np.ones(100)*1e4 if masses is None else np.asarray(masses)
    c=empty_census([0,1,2,4])
    accumulate_census(c,np.ones(len(masses))*1.5,masses,1,np.ones(len(masses))*.01,coarse=np.zeros(len(masses),dtype=bool))
    return seal_census(c)


class ResolutionBudgetTests(unittest.TestCase):
    def test_counts_do_not_accept_science(self):
        r=evaluate_resolution(target(),census(),test_resolution_budget_request())
        self.assertEqual(r['preflight_status'],'PASS')
        self.assertEqual(r['science_acceptance'],'NOT_ACCEPTED')

    def test_mixed_mass_second_moment(self):
        equal=measure_actual_mass_levels(census(np.ones(100)*1e4))
        mixed=measure_actual_mass_levels(census(np.r_[np.ones(99)*1e4,1e7]))
        self.assertLess(mixed['enclosed_effective_count'][-1],2)
        self.assertEqual(equal['enclosed_effective_count'][-1],100)
        e=estimate_then_calibrate_stream_heating(target(),equal)
        m=estimate_then_calibrate_stream_heating(target(),mixed)
        self.assertGreater(m['variance_rate_km2_s2_per_myr_estimate'][1],e['variance_rate_km2_s2_per_myr_estimate'][1])

    def test_coarse_contamination_is_not_a_count_fraction(self):
        c=empty_census([0,1,2,4])
        accumulate_census(c,[1.5,1.5],[1,99],1,[.01,.01],coarse=np.array([False,True]))
        levels=measure_actual_mass_levels(seal_census(c))
        self.assertEqual(levels['enclosed_coarse_fraction'][1],.99)

    def test_species_softening_serialization(self):
        s={'softening_kpc_h':1.,'softening_max_physical_kpc_h':.5,
           'species_softening_kpc_h':{str(t):{'comoving':float(t+1),'maximum_physical':.5} for t in range(6)},
           'zoom_softening_mass_ratios':[8.,64.]}
        p=softening_plan(s)
        np.testing.assert_allclose(p['comoving'],[1.,2.,3.,4.,5.,6.,4.,8.],rtol=1e-15)
        np.testing.assert_allclose(p['maximum_physical'][-2:],[1.,2.],rtol=1e-15)
        pars={f'SofteningClassOfPartType{t}':p['type_classes'][t] for t in range(6)}
        for k,x in enumerate(p['comoving']):
            pars[f'SofteningComovingClass{k}']=x
            pars[f'SofteningMaxPhysClass{k}']=p['maximum_physical'][k]
        np.testing.assert_allclose(proper_species_softening(pars,2,.1,.5,np.array([6,7])),[.8,1.6],rtol=1e-15)
        with self.assertRaisesRegex(ValueError,'saved'):
            proper_species_softening(pars,2,.1,.5)

    def test_cold_stream_heating_rejects_small_dispersion(self):
        req=test_resolution_budget_request();o=req['observables'][0]
        o.update(kind='cold_stream',width_proper_kpc=1.,dispersion_km_s=.0001,duration_myr=1000.,maximum_heating_variance_fraction=.1)
        r=evaluate_resolution(target(),census(),req)
        self.assertEqual(r['preflight_status'],'FAIL')
        self.assertFalse(r['observables'][0]['checks']['heating_variance_fraction_estimate']['passed'])

    def test_unsupported_and_missing_sidm_are_not_accepted(self):
        for kind in ('first_stars','bh_spike','sidm_core'):
            req=test_resolution_budget_request();req['observables'][0]['kind']=kind
            if kind == 'first_stars':
                req['observables'][0].update(claim_scope='primordial_cooling_environment', species=0)
            elif kind == 'bh_spike':
                req['observables'][0]['claim_scope'] = 'resolved_dm_profile'
            r=evaluate_resolution(target(),census(),req)
            self.assertEqual(r['preflight_status'],'FAIL')
            self.assertEqual(r['science_acceptance'],'NOT_ACCEPTED')

    def test_empty_and_unmeasured_apertures_fail(self):
        r=evaluate_resolution(target(),seal_census(empty_census([0,1,2,4])),test_resolution_budget_request())
        self.assertEqual(r['preflight_status'],'FAIL')
        req=test_resolution_budget_request();req['observables'][0]['minimum_radius_proper_kpc']=.5
        self.assertEqual(evaluate_resolution(target(),census(),req)['preflight_status'],'FAIL')

    def test_wdm_units_scaling_and_equality_mask(self):
        w={'mean_density_msun_comoving_mpc3':1e10,'spacing_comoving_mpc':.01,'k_peak_per_comoving_mpc':10.,'safety_factor':1.}
        r=wdm_fragmentation_mask(w,[1.01e7,1.02e7])
        self.assertAlmostEqual(r['estimated_fragmentation_mass_msun'],1.01e7)
        self.assertEqual(r['masked'],[True,False])
        w['spacing_comoving_mpc']/=2
        self.assertAlmostEqual(wdm_fragmentation_mask(w,[1e7])['estimated_fragmentation_mass_msun'],5.05e6)

    def test_invalid_and_false_mass_census_rejected(self):
        req=test_resolution_budget_request();req['maximum_force_support_fraction']=True
        with self.assertRaises(ValueError):validate_resolution_request(req)
        c=census();c['mass_squared_msun2'][1][1]=1
        with self.assertRaisesRegex(ValueError,'moments'):measure_actual_mass_levels(c)

    def test_native_saved_snapshot_census(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);test_stellar_history_snapshot(root,0,.5,[2**60+1,2**60+2],[[0,0,0],[1,0,0]])
            path=root/'data/snapshot_000.hdf5'
            with h5py.File(path,'r+') as f:
                p=f['Parameters'].attrs
                for t in range(6):
                    p[f'SofteningClassOfPartType{t}']=t
                    p[f'SofteningComovingClass{t}']=.01*(t+1)
                    p[f'SofteningMaxPhysClass{t}']=.01
            c=radial_particle_census([path],0,center_comoving_kpc_h=[0,0,0],radial_edges_proper_kpc=[0,1,2,4],fine_dm_mass_msun=1e10,memory_gib=.001)
            self.assertEqual(sum(c['count'][4]),2)
            self.assertEqual(sum(c['mass_msun'][4]),4e10)
            self.assertEqual(max(c['maximum_softening_proper_kpc'][4]),.02)
            self.assertEqual(c['source_hashes'][str(path.resolve())],file_sha256(path))

    def test_native_override_gravity_is_retained(self):
        low=census();low['gravity_constant_kpc_kms2_msun']=1e-10
        from zoom import estimate_relaxation_and_force_limits
        r=estimate_relaxation_and_force_limits(target(),measure_actual_mass_levels(low))
        normal=estimate_relaxation_and_force_limits(target(),measure_actual_mass_levels(census()))
        self.assertGreater(r['relaxation_age_ratio_estimate'][1],normal['relaxation_age_ratio_estimate'][1])

    def test_synthetic_or_missing_convergence_cannot_accept(self):
        from zoom import compare_resolution_measurements
        r=compare_resolution_measurements(test_resolution_budget_request(),{})
        self.assertEqual(r['profile']['status'],'FAIL')
        self.assertEqual(len(r['profile']['tests']),3)

    def test_refinement_cost_uses_actual_species(self):
        from core.resources import resolution_followup_resources
        cfg={'simulation':{'physics':'dm','particles_per_dimension':32,'pm_grid':32}}
        r=resolution_followup_resources(cfg,[0,100,20,0,0,0],3,test_resolution_budget_request()['refinement'])
        self.assertEqual(r['levels'][1]['estimated_resources']['particles_by_type'],[0,800,20,0,0,0])
        self.assertEqual(r['levels'][2]['estimated_resources']['particles_by_type'][1],6400)

    def test_sidm_limits_detect_collapse_and_boundary_failure(self):
        from zoom import measure_sidm_support
        t=target();t['sidm']={'kernel_length_proper_kpc':1.,'minimum_fine_neighbors':64,
            'coarse_overlap_fraction':.1,'omitted_hazard_bound':.01,'maximum_pair_probability':.2,
            'maximum_total_hazard':.3,'collapse_radius_proper_kpc':.5,'operator_campaign_verified':True}
        limits={'maximum_kernel_collapse_ratio':.1,'minimum_fine_neighbors':32,
            'maximum_coarse_overlap_fraction':0.,'maximum_omitted_hazard_bound':.001,
            'maximum_pair_probability':.1,'maximum_total_hazard':.1}
        r=measure_sidm_support(t,limits)
        self.assertEqual(r['status'],'FAIL_SUPPORT')
        self.assertFalse(r['checks']['kernel_collapse_ratio']['passed'])
        self.assertFalse(r['checks']['coarse_overlap_fraction']['passed'])

    def test_saved_mixed_fluid_range_preserves_fine_particles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);test_stellar_history_snapshot(root,0,.5,[100,101],[[0,0,0],[1,0,0]])
            path=root/'data/snapshot_000.hdf5'
            with h5py.File(path,'r+') as f:
                f.move('PartType4','PartType1');counts=np.array([0,2,0,0,0,0],dtype=np.uint64)
                f['Header'].attrs['NumPart_Total']=counts;f['Header'].attrs['NumPart_ThisFile']=counts
                f['PartType1/Masses'][:]=[1.,1.01]
                p=f['Parameters'].attrs
                for t in range(6):
                    p[f'SofteningClassOfPartType{t}']=t;p[f'SofteningComovingClass{t}']=.01;p[f'SofteningMaxPhysClass{t}']=.01
            c=radial_particle_census([path],0,center_comoving_kpc_h=[0,0,0],radial_edges_proper_kpc=[0,1,2,4],fine_dm_mass_msun=2e10,
                fine_dm_mass_range_msun=[2e10,2.02e10],memory_gib=.001)
            self.assertEqual(sum(c['coarse_mass_msun']),0.)
            self.assertEqual(sum(c['mass_msun'][1]),4.02e10)

    def test_claimed_native_refinement_is_remeasured(self):
        from zoom import compare_resolution_measurements,measure_native_resolution_settings
        from core.data import export_numeric_audit
        from core.foundation import file_sha256
        from core.data import epoch_contract
        source=native_control_fixture('periodic/data/snapshot_003.hdf5')
        if not source.exists():self.skipTest('Executed native particle source unavailable')
        with h5py.File(source) as f:a=float(f['Header'].attrs['Time'])
        sources=[{'path':str(source),'sha256':file_sha256(source)}]
        settings=measure_native_resolution_settings(sources,1,a)
        with tempfile.TemporaryDirectory() as tmp:
            products={}
            for name,step in [('coarse',settings['timestep']),('fine',settings['timestep']/2)]:
                path=Path(tmp)/(name+'.json')
                meta={'origin':'native_simulation','units':'Msun','observable':'profile','selection':'same native control population',
                    'actual_scale_factor':a,'physics_fingerprint':'retained-periodic-control','source_snapshots':sources,
                    'bin_coordinates':[1.,2.],'all_requested_bins_valid':True,'numerical_settings':{**settings,'timestep':step}}
                export_numeric_audit(path,{'resolution_measurement':meta},{'mass':np.array([1.,2.])},
                    {'mass':{'units':'Msun','quantity':'test vector; cannot certify claimed native refinement'}},algorithm=__file__,
                    context={'epoch':epoch_contract(a)},source_hashes={str(source):file_sha256(source)})
                products[name]=str(path)
            decision=compare_resolution_measurements(test_resolution_budget_request(),{'profile':{'timestep':{**products,'dataset':'mass'}}})
            test=next(t for t in decision['profile']['tests'] if t['axis']=='timestep')
            self.assertEqual(test['status'],'FAIL')
            self.assertIn('actual native',test['reason'])

# ========================================================================
# test_observer_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from stars import (ObserverFrame, to_observer, from_observer,
                                  native_local_coordinates, PROPER_MOTION_FACTOR)
from stars import export_stellar_observer_epoch


class ObserverCoordinatesTests(unittest.TestCase):
    def setUp(self):
        self.frame = ObserverFrame([0., 0., 0.], [0., 0., 0.], np.eye(3))

    def test_axis_signs_and_independent_au_per_year_conversion(self):
        data = to_observer([[1., 0., 0.], [0., 2., 0.]],
                           [[12., 4.740470463533349, -9.480940927066698],
                            [-9.480940927066698, -8., 18.961881854133396]], self.frame)
        np.testing.assert_allclose(data[:, :4], [[0., 0., 1., 12.], [np.pi / 2., 0., 2., -8.]], atol=1.e-14)
        # 1 AU/Julian year is 4.7404704635 km/s; at 1 kpc this is 1 mas/year.
        np.testing.assert_allclose(data[:, 4:], [[1., -2.], [1., 2.]], rtol=2.e-14)

    def test_longitude_star_includes_latitude_cosine(self):
        latitude = np.pi / 3.
        data = to_observer([[np.cos(latitude), 0., np.sin(latitude)]],
                           [[0., 1. / PROPER_MOTION_FACTOR, 0.]], self.frame)
        self.assertAlmostEqual(data[0, 4], 1.)
        # Coordinate longitude rate itself is 2 mas/year at latitude 60 degrees.
        self.assertAlmostEqual(data[0, 4] / np.cos(data[0, 1]), 2.)

    def test_nontrivial_rotation_translation_velocity_round_trip(self):
        rotation = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
        frame = ObserverFrame([8., -2., 4.], [25., 240., -7.], rotation)
        rng = np.random.default_rng(9812)
        x, v = rng.normal(size=(300, 3)) * 40., rng.normal(size=(300, 3)) * 170.
        projected = to_observer(x, v, frame)
        xr, vr = from_observer(projected, frame)
        # Trig/rotation round-off scales with the relative vector, including
        # translations that cancel in an individual Cartesian component.
        for actual, expected, origin in ((xr, x, frame.position_kpc), (vr, v, frame.velocity_kms)):
            error = np.linalg.norm(actual - expected, axis=1)
            scale = np.linalg.norm(expected - origin, axis=1) + np.linalg.norm(origin)
            self.assertTrue(np.all(error <= 32. * np.finfo(float).eps * scale))
        self.assertTrue(np.all((projected[:, 0] >= 0.) & (projected[:, 0] < 2 * np.pi)))

    def test_exact_poles_have_explicit_invertible_tangent_convention(self):
        x = np.array([[0., 0., 1.], [0., 0., -2.]])
        v = np.array([[7., -8., 9.], [12., 3., -4.]])
        data = to_observer(x, v, self.frame)
        np.testing.assert_array_equal(data[:, 0], [0., 0.])
        xr, vr = from_observer(data, self.frame)
        np.testing.assert_allclose(xr, x, atol=2.e-16)
        np.testing.assert_allclose(vr, v, atol=2.e-15)

    def test_native_scale_h_velocity_and_periodic_image(self):
        x, v = native_local_coordinates([[99., 1., 40.]], [[2., 4., -6.]],
                                        scale_factor=.25, hubble_param=.5,
                                        box_comoving_kpc_h=100., observer_comoving_kpc_h=[1., 99., 40.])
        np.testing.assert_array_equal(x, [[-1., 1., 0.]])
        np.testing.assert_array_equal(v, [[1., 2., -3.]])
        obs = to_observer(x, v, self.frame)
        xr, vr = from_observer(obs, self.frame)
        np.testing.assert_allclose(xr, x, atol=3.e-16)
        np.testing.assert_allclose(vr, v, atol=3.e-16)

    def test_invalid_geometry_missing_values_and_left_handed_frame_fail(self):
        with self.assertRaisesRegex(ValueError, 'right-handed'):
            ObserverFrame([0.] * 3, [0.] * 3, np.diag([1., 1., -1.]))
        with self.assertRaisesRegex(ValueError, 'right-handed'):
            ObserverFrame([0.] * 3, [0.] * 3, np.diag([1., 1., 2.]))
        with self.assertRaisesRegex(ValueError, 'coincident'):
            to_observer([[0.] * 3], [[0.] * 3], self.frame)
        with self.assertRaisesRegex(ValueError, 'finite'):
            from_observer([[0., 0., 1., np.nan, 0., 0.]], self.frame)
        for index, value in ((0, 2. * np.pi), (1, np.pi), (2, 0.)):
            data = np.array([[0., 0., 1., 0., 0., 0.]])
            data[0, index] = value
            with self.assertRaisesRegex(ValueError, 'invalid'):
                from_observer(data, self.frame)
        with self.assertRaisesRegex(ValueError, 'half-box'):
            native_local_coordinates([[50., 0., 0.]], [[0.] * 3], scale_factor=1.,
                                     hubble_param=.7, box_comoving_kpc_h=100., observer_comoving_kpc_h=[0.] * 3)

    def test_frame_copies_inputs_and_empty_catalogues_are_supported(self):
        position = np.zeros(3)
        frame = ObserverFrame(position, np.zeros(3), np.eye(3))
        position[0] = 8.
        self.assertEqual(frame.position_kpc[0], 0.)
        self.assertFalse(frame.position_kpc.flags.writeable)
        empty = to_observer(np.empty((0, 3)), np.empty((0, 3)), frame)
        self.assertEqual(empty.shape, (0, 6))
        x, v = from_observer(empty, frame)
        self.assertEqual(x.shape, (0, 3))
        self.assertEqual(v.shape, (0, 3))

    def test_tiny_negative_longitude_wrap_remains_in_inverse_domain(self):
        projected = to_observer([[1., -1.e-300, 0.]], [[0.] * 3], self.frame)
        self.assertLess(projected[0, 0], 2. * np.pi)
        x, _ = from_observer(projected, self.frame)
        np.testing.assert_allclose(x, [[1., 0., 0.]], atol=1.e-300)

    def test_saved_stellar_export_preserves_exact_ids_absence_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'snapshot.hdf5'
            source.write_bytes(b'fixture source')
            destination = Path(temporary) / 'result'
            ids = np.array([2**63 + 43, 18], dtype=np.uint64)
            epoch = {'scale_factor': .5, 'h': .7, 'box_comoving_kpc': 100.,
                     'source_files': [str(source)]}
            from core.data import epoch_contract
            epoch['epoch_contract'] = epoch_contract(.5)
            saved = (np.array([[98., 3., 4.], [np.nan] * 3]),
                     np.array([[12., -8., 4.], [np.nan] * 3]),
                     np.array([30., np.nan]), np.array([True, False]), np.array([-1, -2]), epoch)
            with patch('stars._snapshot_files', return_value=[source]), \
                 patch('stars._group_files', return_value=[]), \
                 patch('stars.read_epoch', return_value=saved):
                report = export_stellar_observer_epoch(temporary, 2, ids.tolist(), frame=self.frame,
                         observer_comoving_kpc=[1., 1., 1.], output_directory=destination)
                with np.load(destination / 'truth.npz', allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['particle_ids'], ids)
                    np.testing.assert_array_equal(data['present'], [True, False])
                    self.assertTrue(np.all(np.isnan(data['observables'][1])))
                    x, v = from_observer(data['observables'][:1], self.frame)
                    np.testing.assert_allclose(x, [[-1.5, 1., 1.5]], atol=1.e-15)
                    np.testing.assert_allclose(v, [[12., -8., 4.]], atol=4.e-15)
                self.assertEqual(report['present_count'], 1)
                self.assertEqual(report['science_acceptance'], 'NOT_ACCEPTED')
                self.assertIn(str(source.resolve()), report['source_sha256'])
                with self.assertRaises(FileExistsError):
                    export_stellar_observer_epoch(temporary, 2, ids, frame=self.frame,
                        observer_comoving_kpc=[1., 1., 1.], output_directory=destination)

    def test_changed_stellar_source_cannot_publish_catalogue(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'snapshot.hdf5'
            source.write_bytes(b'original')
            destination = Path(temporary) / 'result'
            def changed(*args):
                source.write_bytes(b'changed')
                return (np.array([[1., 2., 3.]]), np.zeros((1, 3)), np.ones(1),
                        np.ones(1, dtype=bool), np.array([-3]),
                        {'scale_factor': 1., 'h': .7, 'box_comoving_kpc': 100.})
            with patch('stars._snapshot_files', return_value=[source]), \
                 patch('stars._group_files', return_value=[]), \
                 patch('stars.read_epoch', side_effect=changed):
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    export_stellar_observer_epoch(temporary, 2, np.array([43], dtype=np.uint64),
                        frame=self.frame, observer_comoving_kpc=[0.] * 3, output_directory=destination)
            self.assertFalse(destination.exists())

# ========================================================================
# test_radial_shape_workflow

test_radial_shape_workflow_ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(test_radial_shape_workflow_ROOT),str(test_radial_shape_workflow_ROOT/'core')]
from halos import verify_native_shape_export
import verification as test_radial_shape_workflow_fixtures


class RadialShapeWorkflowTests(unittest.TestCase):
    def setUp(self):
        fixture=test_radial_shape_workflow_fixtures.CatalogueTargetTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.root=fixture.root
        with h5py.File(fixture.snap,'r+') as source:
            source['PartType1/Coordinates'][:]=[[99.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
            source['PartType1/ParticleIDs'][:]=np.array([2**63+7,2**63+8,2**63+9],dtype=np.uint64)
        (self.root/'config.yaml').write_text(yaml.safe_dump({'visualization':{}}))
        self.request=dict(snapshot=0,host_row=0,output_directory=str(self.root/'shapes'),memory_gib=1.,
             radial_edges_kpc=[0.,2.],support_radius_kpc=3.,window='spherical_shell',
             estimator='mass_quadrupole',minimum_particles=3,tolerance=1.e-5,maximum_iterations=150,
             minimum_axis_ratio=.1,bootstrap_realizations=0,seed=991,minimum_bootstrap_success_fraction=.9,
             minimum_eigenvalue_gap=.02,maximum_orientation_error_deg=20.)

    def records(self):
        return sorted((self.root/'results').glob('workflow_*.json'))

    def test_requests_fail_before_finders_or_other_analysis(self):
        with patch('halos.run_all') as finder,patch('commands.analyze_run') as analyze:
            for products,request,error in [(['pk'],self.request,ValueError),
                    (['radial_shapes'],self.request|{'memory_gib':.001},MemoryError),
                    (['radial_shapes'],self.request|{'maximum_iterations':False},ValueError)]:
                with self.assertRaises(error):
                    process_run(self.root,products,radial_shapes=request,find_halos=True)
            finder.assert_not_called();analyze.assert_not_called()
        self.assertFalse((self.root/'shapes').exists())

    def test_plot_failure_keeps_measurement_then_resume_reuses_it(self):
        with patch('halos.plot_radial_shapes',side_effect=RuntimeError('renderer unavailable')):
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                process_run(self.root,['radial_shapes'],radial_shapes=self.request)
        previous=self.records()[-1];original=previous.read_bytes()
        status=json.loads(original)
        self.assertEqual(status['artifacts']['radial_shapes']['status'],'MEASURED')
        self.assertIn('radial_shapes_figures',status['unavailable'])
        report=(self.root/'shapes/report.json').read_bytes()
        with patch('halos.export_native_halo_shapes',side_effect=AssertionError('must reuse')):
            result=process_run(self.root,['radial_shapes'],resume_from=previous)
        self.assertEqual(result['product_status'],'COMPLETE')
        self.assertEqual(result['science_status'],'NOT_ACCEPTED')
        self.assertEqual(previous.read_bytes(),original)
        self.assertEqual((self.root/'shapes/report.json').read_bytes(),report)
        self.assertTrue((self.root/'shapes_figures/radial_shapes.pdf').exists())

    def test_interrupt_after_atomic_measurement_is_resumable(self):
        with patch('halos.plot_radial_shapes',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):process_run(self.root,['radial_shapes'],radial_shapes=self.request)
        previous=self.records()[-1]
        self.assertEqual(json.loads(previous.read_text())['status'],'INTERRUPTED')
        result=process_run(self.root,['radial_shapes'],resume_from=previous)
        self.assertEqual(result['product_status'],'COMPLETE')

    def test_resume_rejects_corrupt_parent_report_and_changed_source(self):
        with patch('halos.plot_radial_shapes',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):process_run(self.root,['radial_shapes'],radial_shapes=self.request)
        directory=self.root/'shapes';report=directory/'report.json';original=report.read_bytes()
        report.write_text(original.decode().replace('MEASURED','INCOMPLETE'))
        with self.assertRaisesRegex(ValueError,'checksum changed'):verify_native_shape_export(self.root,self.request)
        report.write_bytes(original)
        parent=directory/'parent.npz';old=parent.read_bytes();parent.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum changed'):verify_native_shape_export(self.root,self.request)
        parent.write_bytes(old)
        with h5py.File(self.root/'data/fof_subhalo_tab_000.hdf5','r+') as source:
            source['Group/Group_R_Crit200'][0]=.8
        with self.assertRaisesRegex(ValueError,'source files changed'):verify_native_shape_export(self.root,self.request)

    def test_nonconverged_bins_remain_incomplete_on_resume(self):
        request=self.request|{'window':'iterative_ellipsoidal_shell','maximum_iterations':1}
        with self.assertRaisesRegex(RuntimeError,'incomplete'):process_run(self.root,['radial_shapes'],radial_shapes=request)
        previous=self.records()[-1]
        with self.assertRaisesRegex(RuntimeError,'incomplete'):process_run(self.root,['radial_shapes'],resume_from=previous)
        report=json.loads((self.root/'shapes/report.json').read_text())
        self.assertEqual(report['shells'][0]['status'],'NOT_CONVERGED')

    def test_top_level_preflight_precedes_build_and_evolution(self):
        import commands as run_gadget
        for graphs,request,error in [(None,self.request,ValueError),(['radial_shapes'],self.request|{'memory_gib':.001},MemoryError)]:
            with patch.object(sys,'argv',['commands.py','--answers','unused.json','--dry-run']),patch('commands.load_json',return_value={'graphs':graphs,'radial_shapes':request}),patch('commands.build_config') as build,patch('commands.prepare_and_run') as evolve:
                with self.assertRaises(error):run_gadget.run_gadget_main()
                build.assert_not_called();evolve.assert_not_called()

    def test_example_separates_global_and_radial_definitions(self):
        from core.foundation import validate_spec
        spec=json.loads((test_radial_shape_workflow_ROOT/'research_spec.example.json').read_text())
        validate_spec(spec,['shape','radial_shapes'])
        self.assertIn('Global',spec['observables']['shape']['definition'])
        self.assertIn('proper',spec['observables']['radial_shapes']['selection'])

    def test_blender_join_preserves_epoch_namespace_units_and_failed_shells(self):
        from halos import export_native_halo_shapes, join_radial_shapes_to_timeline
        import copy
        report=export_native_halo_shapes(self.root,**self.request)
        item={'id':0,'id_source':'catalogue_row','finder':'GADGET-4 FOF','position_mpc_h':[0.,0.,0.]}
        timeline={'box_mpc_h':.1,'snapshots':[{'snapshot':0,'scale_factor':1.,'halo_overlays':{'halos':{'items':[item]}}}]}
        result=join_radial_shapes_to_timeline(report,timeline)
        np.testing.assert_allclose(result['shells'][0]['fitted_semiaxes_comoving_mpc_h'],
             np.asarray(report['shells'][0]['fitted_semiaxes_at_upper_radius_kpc'])*.7/1000.)
        self.assertEqual(result['science_acceptance'],'NOT_ACCEPTED')
        for alteration in ('epoch','finder','duplicate','centre','missing'):
            bad=copy.deepcopy(timeline)
            if alteration=='epoch':bad['snapshots'][0]['scale_factor']=.5
            elif alteration=='finder':bad['snapshots'][0]['halo_overlays']['halos']['items'][0]['finder']='Rockstar 6D'
            elif alteration=='duplicate':bad['snapshots'][0]['halo_overlays']['halos']['items'].append(copy.deepcopy(item))
            elif alteration=='centre':bad['snapshots'][0]['halo_overlays']['halos']['items'][0]['position_mpc_h']=[.01,0.,0.]
            else:bad['snapshots'][0]['halo_overlays']['halos']['items']=[]
            with self.assertRaises(ValueError):join_radial_shapes_to_timeline(report,bad)

    def test_top_level_valid_request_is_forwarded_to_analysis(self):
        import commands as run_gadget
        cfg=yaml.safe_load((test_radial_shape_workflow_ROOT/'core/config.yaml').read_text())
        cfg['spectrum']['file']='control-spectrum'
        cfg['cosmology'].update(model='lcdm',H0=70.,omega_b=.022,omega_cdm=.12)
        cfg['simulation'].update(box_mpc_h=1.,particles_per_dimension=32,physics='dm',max_size_timestep_ln_a=.01)
        cfg['analysis']['density_grid']=32
        cfg['visualization']['filament_grid_size']=32
        cfg['simulation']['snapshot_redshift_list']=[0.]
        answers={'graphs':['radial_shapes'],'radial_shapes':self.request}
        with patch.object(sys,'argv',['commands.py','--answers','unused.json']),patch('commands.load_json',return_value=answers),patch('commands.build_config',return_value=(cfg,self.root)),patch('commands.prepare_and_run',return_value='CONTROL_COMPLETE') as evolve,patch('core.setup_walkthrough.launch_review'),patch('commands.process_run') as analyze:
            run_gadget.run_gadget_main()
        evolve.assert_called_once()
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.kwargs['radial_shapes']['snapshot'],0)
        self.assertEqual(analyze.call_args.kwargs['radial_shapes']['output_directory'],str((self.root/'shapes').resolve()))

    def test_resume_rejects_live_owner(self):
        import os
        with patch('halos.plot_radial_shapes',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):process_run(self.root,['radial_shapes'],radial_shapes=self.request)
        previous=self.records()[-1];record=json.loads(previous.read_text())
        record.update(status='RUNNING',owner_pid=os.getpid());previous.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'still alive'):
            process_run(self.root,['radial_shapes'],resume_from=previous)

# ========================================================================
# test_radial_shape_batch

test_radial_shape_batch_ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(test_radial_shape_batch_ROOT),str(test_radial_shape_batch_ROOT/'core')]
from halos import (export_shape_request,shape_profile_requests,shape_reports,
    validate_shape_request,verify_shape_request,join_radial_shapes_to_timeline)
import verification as workflow_fixtures


class RadialShapeBatchTests(unittest.TestCase):
    def setUp(self):
        fixture=workflow_fixtures.RadialShapeWorkflowTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.root=fixture.root;self.single=fixture.request
        for prefix in ('snapshot','fof_subhalo_tab'):
            source=self.root/f'data/{prefix}_000.hdf5';target=self.root/f'data/{prefix}_001.hdf5'
            shutil.copy2(source,target)
            with h5py.File(target,'r+') as h:
                h['Header'].attrs['Time']=.5
                if 'Redshift' in h['Header'].attrs:h['Header'].attrs['Redshift']=1.
        controls={k:v for k,v in self.single.items() if k not in ('snapshot','host_row','output_directory','memory_gib')}
        controls['parent_selection']='complete_support_sphere'
        self.batch={'schema':'radial-shape-batch-v1','selections':[{'snapshot':0,'host_row':0,'scale_factor':1.},
                    {'snapshot':1,'host_row':0,'scale_factor':.5}],
                    'shape_controls':controls,'output_directory':str(self.root/'batch'),'memory_gib':1.,'maximum_targets':2}

    def export(self):
        return export_shape_request(self.root,self.batch)

    def test_batch_publishes_two_exact_epochs_figures_inspector_and_resume(self):
        result=self.export()
        self.assertEqual(result['execution_status'],'COMPLETE')
        self.assertEqual(result['status'],'MEASURED')
        self.assertEqual(result['science_acceptance'],'NOT_ACCEPTED')
        self.assertEqual(len(result['profiles']),2)
        self.assertEqual(len(result['figure_registry']),4)
        self.assertEqual([r['scale_factor'] for r in result['inspector_references']],[1.,.5])
        self.assertEqual(result,verify_shape_request(self.root,self.batch))
        with patch('halos.radial_density_shapes',side_effect=AssertionError('recomputed')):
            self.assertEqual(result,export_shape_request(self.root,self.batch,resume=True))
        for report in shape_reports(self.root,self.batch):
            self.assertEqual(report['parent_count_in_support'],3)
            self.assertEqual(report['simulation_convergence']['status'],'NOT_EVALUATED')
            shell=report['shells'][0]
            self.assertGreater(shell['mass_msun'],0.)
            self.assertIn('iterations',shell)
            self.assertEqual(shell['sampling_uncertainty']['status'],'NOT_ESTIMATED')
            self.assertEqual(shell['orientation_sampling_resolved'],[False]*3)

    def test_fof_membership_uses_exact_uint64_ids_and_retains_failed_bins(self):
        self.batch['shape_controls']['parent_selection']='fof_members'
        result=self.export()
        self.assertEqual(result['status'],'INCOMPLETE')
        for report in shape_reports(self.root,self.batch):
            self.assertEqual(report['membership_count'],2)
            self.assertEqual(report['parent_count_in_support'],2)
            self.assertEqual(report['shells'][0]['status'],'INSUFFICIENT_PARTICLES')
            self.assertEqual(report['shells'][0]['particle_count'],2)
            self.assertGreater(report['shells'][0]['mass_msun'],0.)
            self.assertTrue(report['shells'][0]['failed_bin'])
        with np.load(Path(shape_profile_requests(self.batch)[0]['output_directory'])/'parent.npz') as data:
            np.testing.assert_array_equal(data['particle_ids'],np.array([2**63+7,2**63+8],dtype=np.uint64))

    def test_invalid_duplicate_resource_and_epoch_requests_fail_before_publication(self):
        invalid=[]
        bad=copy.deepcopy(self.batch);bad['selections'][1]=bad['selections'][0];invalid.append(bad)
        bad=copy.deepcopy(self.batch);bad['maximum_targets']=1;invalid.append(bad)
        bad=copy.deepcopy(self.batch);bad['memory_gib']=.001;invalid.append(bad)
        bad=copy.deepcopy(self.batch);bad['shape_controls']['parent_selection']='guessed_members';invalid.append(bad)
        bad=copy.deepcopy(self.batch);bad['selections'][0]['scale_factor']=True;invalid.append(bad)
        for bad in invalid:
            with self.assertRaises((ValueError,MemoryError)):validate_shape_request(bad)
        self.batch['selections'][1]['scale_factor']=.6
        with patch('halos.radial_density_shapes') as compute:
            with self.assertRaisesRegex(ValueError,'exact saved epoch'):self.export()
            compute.assert_not_called()
        self.assertFalse((self.root/'batch').exists())

    def test_real_standalone_cli_and_resume_forward_complete_batch(self):
        request=self.root/'request.json';request.write_text(json.dumps(self.batch))
        command=[sys.executable,str(test_radial_shape_batch_ROOT/'commands.py'),'measure-halo-shapes', str(self.root),str(request)]
        first=subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(first.returncode,0,first.stderr)
        summary=json.loads(first.stdout)
        self.assertEqual(len(summary['profiles']),2)
        self.assertEqual(summary['status'],'MEASURED')
        second=subprocess.run(command+['--resume'],text=True,capture_output=True)
        self.assertEqual(second.returncode,0,second.stderr)
        self.assertEqual(json.loads(second.stdout),summary)

    def test_real_analysis_cli_registers_batch_artifacts(self):
        request=self.root/'request.json';request.write_text(json.dumps(self.batch))
        command=[sys.executable,str(test_radial_shape_batch_ROOT/'commands.py'),'analysis', str(self.root),'--graphs','radial_shapes','--radial-shapes',str(request)]
        run=subprocess.run(command,text=True,capture_output=True)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        record=json.loads(sorted((self.root/'results').glob('workflow_*.json'))[-1].read_text())
        artifact=record['artifacts']['radial_shapes']
        self.assertEqual(len(artifact['inspector_references']),2)
        self.assertEqual(len(artifact['figure_registry']),4)
        self.assertEqual(record['science_status'],'NOT_ACCEPTED')

    def test_corrupt_batch_source_parent_figure_and_request_are_rejected(self):
        self.export();root=self.root/'batch'
        path=root/'batch.json';old=path.read_bytes();path.write_text('{}')
        with self.assertRaises(ValueError):verify_shape_request(self.root,self.batch)
        path.write_bytes(old)
        req=shape_profile_requests(self.batch)[0]
        for file in [Path(req['output_directory'])/'parent.npz',Path(req['output_directory']).with_name(Path(req['output_directory']).name+'_figures')/'radial_shapes.png']:
            old=file.read_bytes();file.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):verify_shape_request(self.root,self.batch)
            file.write_bytes(old)
        changed=copy.deepcopy(self.batch);changed['shape_controls']['seed']+=1
        with self.assertRaises(ValueError):verify_shape_request(self.root,changed)
        with h5py.File(self.root/'data/snapshot_001.hdf5','r+') as source:source['PartType1/Coordinates'][0,0]=98.
        with self.assertRaises(ValueError):verify_shape_request(self.root,self.batch)

    def test_timeline_consumes_both_epochs_without_row_continuity(self):
        self.export()
        timeline={'box_mpc_h':.1,'snapshots':[{'snapshot':snapshot,'scale_factor':a,'halo_overlays':{'halos':{'items':[
            {'id':0,'id_source':'catalogue_row','finder':'GADGET-4 FOF','position_mpc_h':[0.,0.,0.]}]}}} for snapshot,a in [(0,1.),(1,.5)]]}
        for report in shape_reports(self.root,self.batch):join_radial_shapes_to_timeline(report,timeline)
        for row in timeline['snapshots']:
            shape=row['halo_overlays']['halos']['items'][0]['radial_density_shapes']
            self.assertEqual(shape['scale_factor'],row['scale_factor'])
            self.assertEqual(shape['shells'][0]['simulation_convergence']['status'],'NOT_EVALUATED')

    def test_failure_preserves_unpublished_profiles_and_resume_rejects_changed_inputs(self):
        with patch('halos.plot_radial_shapes',side_effect=RuntimeError('plot failed')):
            with self.assertRaisesRegex(RuntimeError,'plot failed'):self.export()
        self.assertFalse((self.root/'batch').exists())
        retained=list(self.root.glob('.radial-shape-batch-*/export_failure.json'))
        self.assertEqual(len(retained),1)
        self.assertEqual(json.loads(retained[0].read_text())['status'],'FAILED')
        self.assertTrue(list(retained[0].parent.glob('profiles/*/parent.npz')))

    def test_batch_rechecks_earlier_epoch_before_final_publication(self):
        from halos import plot_radial_shapes
        calls=0
        def mutate(shape,figures):
            nonlocal calls
            result=plot_radial_shapes(shape,figures);calls+=1
            if calls==2:
                with h5py.File(self.root/'data/snapshot_000.hdf5','r+') as source:
                    source['PartType1/Coordinates'][0,0]=98.
            return result
        with patch('halos.plot_radial_shapes',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'batch sources changed'):self.export()
        self.assertFalse((self.root/'batch').exists())

    def test_empty_parent_keeps_shell_counts_masses_and_bootstrap_failures(self):
        from halos import radial_density_shapes
        controls=dict(self.batch['shape_controls']);controls.pop('parent_selection');controls['bootstrap_realizations']=32
        result=radial_density_shapes(np.empty((0,3)),np.empty(0),**controls)
        shell=result['shells'][0]
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(shell['particle_count'],0)
        self.assertEqual(shell['mass_msun'],0.)
        self.assertEqual(shell['sampling_uncertainty']['status'],'UNAVAILABLE_FAILED_BIN')
        self.assertEqual(len(result['bootstrap_profiles']),32)

    def test_large_snapshot_small_aperture_uses_bounded_selected_workspace(self):
        path=self.root/'data/snapshot_000.hdf5';n=500_000
        with h5py.File(path,'r+') as source:
            group=source['PartType1']
            positions=group['Coordinates'][:];ids=group['ParticleIDs'][:]
            attributes={name:dict(dataset.attrs) for name,dataset in group.items()}
            del group['Coordinates'];del group['ParticleIDs']
            coordinates=group.create_dataset('Coordinates',(n,3),dtype='f8',chunks=(4096,3),fillvalue=50.)
            coordinates[:3]=positions
            del group['Velocities'];group.create_dataset('Velocities',(n,3),dtype='f8',chunks=(4096,3),fillvalue=0.)
            identities=group.create_dataset('ParticleIDs',(n,),dtype='u8',chunks=(4096,))
            identities[:]=np.arange(n,dtype=np.uint64);identities[:3]=ids
            counts=source['Header'].attrs['NumPart_ThisFile'];counts[1]=n
            source['Header'].attrs['NumPart_ThisFile']=counts;source['Header'].attrs['NumPart_Total']=counts
            if 'Masses' in group:
                mass=group['Masses'][0];del group['Masses'];group.create_dataset('Masses',(n,),dtype='f8',chunks=(4096,),fillvalue=float(mass))
            for name,attrs in attributes.items():
                if name in group:group[name].attrs.update(attrs)
        req=dict(self.single,memory_gib=.1,output_directory=str(self.root/'bounded'))
        report=export_shape_request(self.root,req)
        self.assertEqual(report['parent_count_in_support'],3)
        self.assertLessEqual(report['particle_read_chunk_size'],250_000)

# ========================================================================
# test_start_epoch_campaign
import struct
sys.path[:0]=[str(Path(__file__).resolve().parent/'core'),str(Path(__file__).resolve().parent)]
from evolution import (build_start_epoch_campaign,validate_start_epoch_request,
    require_distinct_valid_epochs,freeze_phase_realization,verify_start_epoch_campaign,
    audit_start_epoch_phases,write_start_epoch_measurements,compare_start_epoch_campaign,
    verify_start_epoch_comparison,stage_start_epoch_branch)
from core.data import verify_numeric_audit
from core.research.generator_controls import solver_bundle
from core.native import run_native_command


def campaign_fixture(root,seeds=(7,19),n=8):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    zs=[49.,99.,199.];bundles=[]
    for z in zs:
        path=solver_bundle(root/('z'+str(int(z))),redshift=z)
        bundles.append({'redshift':z,'manifest':{'path':str(path),'sha256':digest(path)}})
    config={'cosmology':{'model':'lcdm','H0':70.,'omega_b':.05*.7**2,'omega_cdm':.25*.7**2,
                        'A_s':2.1e-9,'n_s':.96,'T_cmb':2.7255,'N_ur':3.046},
            'simulation':{'box_mpc_h':10.,'particles_per_dimension':n,'random_seed':seeds[0],
                          'start_redshift':99.,'end_redshift':0.,'snapshot_redshift_list':[9.,1.,0.],
                          'physics':'hydro','lpt_order_hydro':1,'softening_kpc_h':1.,'err_tol_int_accuracy':.025},
            'spectrum':{'source':'modular','file':bundles[1]['manifest']['path'],'redshift':99.},
            'hydro':{'cooling':False,'star_formation':False}}
    cfg=root/'base.yaml';cfg.write_text(yaml.safe_dump(config))
    spec=spec_fixture();spec['start_epoch_campaign']={'schema_version':1,
        'configuration':{'path':str(cfg),'sha256':digest(cfg)},'redshifts':zs,'reference_redshift':199.,
        'seeds':list(seeds),'common_scale_factors':[.1,.5,1.],'epoch_tolerance':1e-10,'confidence_sigma':2.,
        'observables':{'shape':{'units':'dimensionless','bin_labels':['b/a','c/a'],
            'absolute_tolerance':.01,'relative_tolerance':.01,'absolute_floor':.001}},'linear_bundles':bundles}
    return spec


def synthetic_ic_phase_inputs(root,plan,change_phase=False):
    records=[]
    n=2**plan['grid_level'];q=np.moveaxis(np.indices((n,)*3),0,-1).reshape(-1,3)*10000/n
    for branch in plan['branches']:
        folder=root/branch['id'];folder.mkdir(parents=True)
        ic=folder/'ic.hdf5';noise=folder/'native-noise.bin'
        raw=np.random.Generator(np.random.Philox(branch['seed'])).normal(size=n**3).astype('<f8')
        noise.write_bytes(struct.pack('<3I',n,n,n)+raw.tobytes())
        phase=branch['seed']*.03+(1. if change_phase and branch['start_redshift']==49 else 0.)
        psi=np.sin(q/10000*2*np.pi+phase)*(.03*branch['start_scale_factor'])
        counts=np.array([0,len(q),0,0,0,0],dtype=np.uint64)
        with h5py.File(ic,'w') as f:
            f.create_group('Header').attrs.update(Time=branch['start_scale_factor'],BoxSize=10000.,
                HubbleParam=.7,Omega0=.3,OmegaBaryon=.05,Music_lpt_order=1,Music_lagrangian_schema=1,Music_lagrangian_unitlength_cm=KPC_CM,
                Music_max_abs_displacement_native=np.abs(psi).max(0),NumFilesPerSnapshot=1,
                NumPart_ThisFile=counts,NumPart_Total=counts,MassTable=[0,1.,0,0,0,0])
            g=f.create_group('PartType1');g['ParticleIDs']=np.arange(len(q),dtype=np.uint64)
            g['LagrangianCoordinates']=q;g['Coordinates']=(q+psi)%10000
        records.append({'branch_id':branch['id'],'ic':str(ic),'native_noise':str(noise)})
    return records


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.spec=campaign_fixture(self.root/'source')

    def build(self,spec=None):
        spec=spec or self.spec
        return build_start_epoch_campaign(spec,spec['start_epoch_campaign']['redshifts'],self.root/'campaign')

    def phase(self,plan,change=False):
        inputs=synthetic_ic_phase_inputs(self.root/'ics',plan,change)
        result=audit_start_epoch_phases(self.root/'campaign/campaign.json',inputs,self.root/'phases.json',modes=[[1,0,0],[0,1,0],[0,0,1]])
        return result

    def measurements(self,plan,offset=.001,actual_shift=0.,cost=True):
        self.phase(plan)
        paths=[]
        for branch in plan['branches']:
            snapshots=[]
            for i,a in enumerate((.1,.5,1.)):
                path=self.root/branch['id']/('snapshot_'+str(i)+'.hdf5');path.parent.mkdir(exist_ok=True)
                with h5py.File(path,'w') as f:
                    actual=a+actual_shift if i==0 else a
                    f.create_group('Header').attrs.update(Time=actual,Redshift=1/actual-1)
                    f.create_group('Parameters').attrs.update(TimeBegin=branch['start_scale_factor'],HubbleParam=.7,OmegaBaryon=.05)
                snapshots.append(path)
            delta=0 if branch['start_redshift']==199. else offset*(1+branch['seed']*.01)
            values={'shape':np.full((3,2),.5+delta)}
            invocation=None
            if cost:
                invocation=self.root/branch['id']/'execution.json'
                run_native_command(['/usr/bin/true'],cwd=self.root,receipt_path=invocation,
                    inputs=[branch['configuration']['path']])
            output=self.root/branch['id']/'measurements.json'
            write_start_epoch_measurements(self.root/'campaign/campaign.json',branch['id'],snapshots,values,output,
                ic_phase_audit=self.root/'phases.json',execution_receipt=invocation,execution_kind='evolution')
            paths.append(output)
        return paths

    def test_exact_bundles_frozen_phases_and_unrelated_physics(self):
        plan=self.build();self.assertEqual(len(plan['branches']),6)
        self.assertEqual(len({b['physics_fingerprint'] for b in plan['branches']}),1)
        for seed in (7,19):
            branches=[b for b in plan['branches'] if b['seed']==seed]
            self.assertEqual(len({b['noise']['sha256'] for b in branches}),1)
            for b in branches:
                cfg=yaml.safe_load(Path(b['configuration']['path']).read_text())
                self.assertEqual(cfg['hydro'],{'cooling':False,'star_formation':False})
                self.assertEqual(cfg['simulation']['snapshot_redshift_list'],[9.,1.,0.])
                self.assertEqual(b['bundle']['source_redshift'],b['start_redshift'])
        self.assertEqual(verify_start_epoch_campaign(self.root/'campaign/campaign.json'),plan)

    def test_invalid_epoch_seed_reference_and_actual_schedule(self):
        for zs in ([99],[99,99],[True,99],[float('nan'),99],[0,99],[99,99+1e-10]):
            with self.subTest(zs=zs),self.assertRaises(ValueError):require_distinct_valid_epochs(zs)
        for key,value in (('seeds',[True,2]),('seeds',[1,1]),('reference_redshift',999),
                          ('common_scale_factors',[.01,.5]),('epoch_tolerance',.1)):
            bad=copy.deepcopy(self.spec['start_epoch_campaign']);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_start_epoch_request(bad)

    def test_noise_uint64_reproducible_budget_and_overwrite(self):
        seed=2**63+11
        a=freeze_phase_realization(seed,self.root/'noise-a',grid_size=8,memory_gib=1)
        b=freeze_phase_realization(seed,self.root/'noise-b',grid_size=8,memory_gib=1)
        self.assertEqual(a['noise_sha256'],b['noise_sha256']);self.assertEqual(a['seed'],seed)
        with self.assertRaises(FileExistsError):freeze_phase_realization(seed,self.root/'noise-a',grid_size=8,memory_gib=1)
        with self.assertRaises(MemoryError):freeze_phase_realization(seed,self.root/'big',grid_size=1024,memory_gib=.1)
        self.assertFalse((self.root/'big').exists())

    def test_mismatched_bundle_physics_rejected_before_publication(self):
        entry=self.spec['start_epoch_campaign']['linear_bundles'][0]
        path=Path(entry['manifest']['path']);info=json.loads(path.read_text());info['parameters']['n_s']=.99
        path.write_text(json.dumps(info));entry['manifest']['sha256']=digest(path)
        with self.assertRaises(ValueError):self.build()
        self.assertFalse((self.root/'campaign').exists())

    def test_independent_solver_output_names_do_not_change_physics(self):
        for entry in self.spec['start_epoch_campaign']['linear_bundles']:
            path=Path(entry['manifest']['path'])
            info=json.loads(path.read_text())
            info['parameters']['name']='native-start-'+str(int(entry['redshift']))
            path.write_text(json.dumps(info));entry['manifest']['sha256']=digest(path)
        plan=self.build()
        self.assertEqual(len(plan['branches']),6)
        self.assertEqual(len({branch['physics_fingerprint'] for branch in plan['branches']}),1)

    def test_changed_campaign_config_or_noise_is_rejected(self):
        plan=self.build();Path(plan['branches'][0]['noise']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'product changed'):verify_start_epoch_campaign(self.root/'campaign/campaign.json')

    def test_native_phase_change_is_detected(self):
        plan=self.build();result=self.phase(plan,True)
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(result['checks']['seed-7-start-00']['status'],'FAIL')

    def test_matched_residual_covariance_and_separate_cost_intervals(self):
        plan=self.build();paths=self.measurements(plan)
        result=compare_start_epoch_campaign(self.root/'campaign/campaign.json',paths,self.root/'comparison.json')
        self.assertEqual(result['status'],'PASS');self.assertEqual(result['seed_count'],2)
        contract=result['numeric_data']['array_contract']['datasets']['shape_mean_residual']
        self.assertEqual(contract['epoch_status'],'PRODUCER_DECLARED')
        epoch=json.loads(contract['epoch_json']);self.assertEqual(len(epoch['rows_by_branch']),6)
        with np.load(str(self.root/'comparison.json')+'.data/arrays.npz') as arrays:
            self.assertEqual(arrays['shape_paired_covariance'].shape,(12,12))
            expected=np.cov(np.array([[.00107]*12,[.00119]*12]),rowvar=False,ddof=1)
            np.testing.assert_allclose(arrays['shape_paired_covariance'],expected,atol=1e-18)
            np.testing.assert_allclose(arrays['shape_mean_covariance'],expected/2,atol=1e-18)
        self.assertTrue(all(c['cost']['status']=='MEASURED' for c in result['costs']))
        self.assertNotEqual(result['costs'][0]['integration_intervals_ln_a'],result['costs'][1]['integration_intervals_ln_a'])
        with self.assertRaisesRegex(ValueError,'Synthetic'):verify_start_epoch_comparison({'path':str(self.root/'comparison.json'),'sha256':digest(self.root/'comparison.json')},self.spec['start_epoch_campaign'])

    def test_failure_recommends_refinement_without_changing_physics(self):
        plan=self.build();paths=self.measurements(plan,offset=.2)
        result=compare_start_epoch_campaign(self.root/'campaign/campaign.json',paths,self.root/'comparison.json')
        self.assertEqual(result['status'],'FAIL');self.assertTrue(result['recommended_refinements'])
        self.assertEqual(verify_start_epoch_campaign(self.root/'campaign/campaign.json'),plan)

    def test_single_seed_has_no_invented_covariance(self):
        self.spec=campaign_fixture(self.root/'single-source',seeds=(7,))
        plan=self.build();paths=self.measurements(plan)
        result=compare_start_epoch_campaign(self.root/'campaign/campaign.json',paths,self.root/'comparison.json')
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(result['decisions']['shape']['covariance_status'],'UNAVAILABLE_SINGLE_REALIZATION')

    def test_actual_epoch_mismatch_is_not_interpolated(self):
        plan=self.build();paths=self.measurements(plan,actual_shift=.001)
        with self.assertRaisesRegex(ValueError,'Common actual epoch'):compare_start_epoch_campaign(self.root/'campaign/campaign.json',paths,self.root/'comparison.json')

    def test_duplicate_measurements_and_memory_are_rejected(self):
        plan=self.build();paths=self.measurements(plan)
        with self.assertRaisesRegex(ValueError,'Duplicate'):compare_start_epoch_campaign(self.root/'campaign/campaign.json',[paths[0]]*6,self.root/'comparison.json')
        with self.assertRaises(MemoryError):compare_start_epoch_campaign(self.root/'campaign/campaign.json',paths,self.root/'comparison.json',memory_gib=.001)

    def test_changed_snapshot_rejected_on_comparison(self):
        plan=self.build();paths=self.measurements(plan)
        report=json.loads(paths[0].read_text());snapshot=Path(report['snapshots'][0]['path'])
        with h5py.File(snapshot,'r+') as f:f['Header'].attrs['Time']=.2
        with self.assertRaisesRegex(ValueError,'source changed'):compare_start_epoch_campaign(self.root/'campaign/campaign.json',paths,self.root/'comparison.json')

    def test_acceptance_routing_requires_campaign_even_with_other_evidence(self):
        plan=self.build();manifest={'products':['shape'],'research_spec':self.spec}
        data=self.root/'other-proof.json';data.write_text('{}')
        record={'manifest_sha256':canonical_digest(manifest),'metric':'maximum_axis_ratio_error',
            'units':'dimensionless','value':.01,'files':[{'path':data.name,'sha256':digest(data)}]}
        evidence={'shape':{kind:dict(record,evidence_kind=kind) for kind in ('pilot','convergence')}}
        result=acceptance_report(manifest,evidence,self.root)
        self.assertTrue(all(check['status']=='PASS' for check in result['checks']['shape'].values()))
        self.assertEqual(result['status'],'NOT_ACCEPTED')
        self.assertEqual(result['context_checks']['start_epoch_campaign']['status'],'FAIL')
        validate_spec(self.spec)

    def test_regeneration_requires_authorization_and_records_children(self):
        parent=self.spec['start_epoch_campaign']['linear_bundles'][1]['manifest']
        for item in self.spec['start_epoch_campaign']['linear_bundles']:item['manifest']=copy.deepcopy(parent)
        with self.assertRaisesRegex(ValueError,'explicitly authorized'):self.build()
        calls=[]
        def regenerate(source,z):
            calls.append(z)
            child=solver_bundle(self.root/('regenerated-'+str(int(z))),redshift=z)
            info=json.loads(child.read_text());info['parent_bundle']={'path':str(source),'sha256':digest(source)}
            child.write_text(json.dumps(info));return child
        result=build_start_epoch_campaign(self.spec,self.spec['start_epoch_campaign']['redshifts'],self.root/'campaign',
            regenerate=regenerate,allow_regeneration=True)
        self.assertEqual(calls,[49.,199.]);self.assertEqual(len(result['branches']),6)

    def test_duplicate_external_noise_cannot_claim_independent_seeds(self):
        freeze_phase_realization(7,self.root/'external',grid_size=8,memory_gib=1)
        source=self.root/'external/noise.grafic';entry={'path':str(source),'sha256':digest(source)}
        self.spec['start_epoch_campaign']['noise_by_seed']={'7':entry,'19':entry}
        with self.assertRaisesRegex(ValueError,'Independent seeds'):self.build()
        self.assertFalse((self.root/'campaign').exists())

    def test_reference_is_user_selected_not_automatically_earliest(self):
        self.spec['start_epoch_campaign']['reference_redshift']=49.
        result=self.build();self.assertEqual(result['request']['reference_redshift'],49.)

    def test_unknown_units_and_changed_native_epoch_fail(self):
        bad=copy.deepcopy(self.spec)
        bad['start_epoch_campaign']['observables']['shape']['units']='kpc'
        with self.assertRaisesRegex(ValueError,'units disagree'):validate_spec(bad)
        plan=self.build();inputs=synthetic_ic_phase_inputs(self.root/'ics',plan)
        with h5py.File(inputs[0]['ic'],'r+') as f:f['Header'].attrs['Time']=.5
        with self.assertRaisesRegex(ValueError,'IC epoch'):audit_start_epoch_phases(self.root/'campaign/campaign.json',inputs,
            self.root/'phases.json',modes=[[1,0,0],[0,1,0],[0,0,1]])

    def test_public_cli_build_and_verify(self):
        spec=self.root/'spec.json';spec.write_text(json.dumps(self.spec))
        cli=Path(__file__).resolve().parent/'commands.py'
        first=subprocess.run([sys.executable,str(cli),'start-epoch-campaign','build','--spec',str(spec),'--output',str(self.root/'campaign')],
            capture_output=True,text=True)
        self.assertEqual(first.returncode,0,first.stderr);self.assertEqual(first.stdout.strip(),'PREPARED')
        second=subprocess.run([sys.executable,str(cli),'start-epoch-campaign','verify','--campaign',str(self.root/'campaign/campaign.json')],
            capture_output=True,text=True)
        self.assertEqual(second.returncode,0,second.stderr)

    def test_execution_wrapper_binds_config_noise_and_kind(self):
        from evolution import execute_start_epoch_branch
        plan=self.build();directory=self.root/'execution';directory.mkdir()
        result=execute_start_epoch_branch(self.root/'campaign/campaign.json',plan['branches'][0]['id'],
            ['/usr/bin/true'],directory,execution_kind='numerical_control')
        self.assertEqual(result['returncode'],0)
        self.assertEqual(result['campaign_branch']['execution_kind'],'numerical_control')
        self.assertIn(plan['branches'][0]['configuration']['path'],result['input_hashes'])
        self.assertIn(plan['branches'][0]['noise']['path'],result['input_hashes'])
        with self.assertRaises(ValueError):execute_start_epoch_branch(self.root/'campaign/campaign.json',
            plan['branches'][0]['id'],['/usr/bin/true'],directory,execution_kind='evolution')

    def test_matching_resume_reuses_complete_manifest_and_stale_resume_fails(self):
        plan=self.build();path=self.root/'campaign/campaign.json';before=path.read_bytes()
        result=build_start_epoch_campaign(self.spec,self.spec['start_epoch_campaign']['redshifts'],path.parent,resume=True,
            regenerate=lambda *_:self.fail('Resume must not regenerate'))
        self.assertEqual(result,plan);self.assertEqual(path.read_bytes(),before)
        changed=copy.deepcopy(self.spec);changed['experiment_id']='another-experiment'
        with self.assertRaisesRegex(ValueError,'identical specification'):build_start_epoch_campaign(changed,
            changed['start_epoch_campaign']['redshifts'],path.parent,resume=True)
        Path(plan['branches'][0]['configuration']['path']).write_text('changed')
        with self.assertRaises(ValueError):build_start_epoch_campaign(self.spec,
            self.spec['start_epoch_campaign']['redshifts'],path.parent,resume=True)

    def test_background_must_cover_common_output_clock(self):
        entry=self.spec['start_epoch_campaign']['linear_bundles'][0]
        manifest=Path(entry['manifest']['path']);info=json.loads(manifest.read_text())
        path=manifest.parent/'background.dat';rows=np.loadtxt(path);rows=rows[rows[:,0]>1.]
        np.savetxt(path,rows,header='1:z 2:H [1/Mpc] 3:gr.fac. D 4:gr.fac. f 5:Omega_r(z)')
        info['checksums_sha256']['background.dat']=digest(path);manifest.write_text(json.dumps(info))
        entry['manifest']['sha256']=digest(manifest)
        with self.assertRaisesRegex(ValueError,'output clock range'):self.build()
        self.assertFalse((self.root/'campaign').exists())

    def test_user_epochs_and_seeds_need_not_include_the_original_run(self):
        request=self.spec['start_epoch_campaign'];request['redshifts']=[40.,80.,120.]
        request['reference_redshift']=120.;request['seeds']=[23,19]
        request['linear_bundles']=[]
        for z in request['redshifts']:
            path=solver_bundle(self.root/('selected-'+str(int(z))),redshift=z)
            request['linear_bundles'].append({'redshift':z,'manifest':{'path':str(path),'sha256':digest(path)}})
        plan=self.build()
        self.assertEqual({b['start_redshift'] for b in plan['branches']},{40.,80.,120.})
        self.assertEqual({b['seed'] for b in plan['branches']},{23,19})

    def test_native_staging_preserves_sealed_inputs_and_attaches_noise(self):
        from zoom import verify_staged_zoom_inputs
        self.spec=campaign_fixture(self.root/'native-source',n=32)
        plan=self.build();stage=self.root/'staged'
        stage_start_epoch_branch(self.root/'campaign/campaign.json',plan['branches'][0]['id'],stage)
        verify_staged_zoom_inputs(stage/'native-inputs')
        self.assertIn(plan['branches'][0]['noise']['path'],(stage/'campaign-parent.conf').read_text())

# ========================================================================
# test_data_contract

test_data_contract_ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(test_data_contract_ROOT),str(test_data_contract_ROOT/'core')]
from core.data import (exact_uint64_ids,exact_id_join,native_unit_factor,audit_native_snapshot_contract,coordinate_contract)
from core.cosmology import run_proper_time_clock,validated_proper_time_samples,interpolate_proper_time
from core.foundation import publish_directory, product_provenance, verify_product_provenance
import verification as test_data_contract_fixtures


def clock_fixture(root):
    background=root/'background.dat'
    background.write_text('# 1:z 2:proper time [Gyr] 3:H [1/Mpc] 4:Omega_r(z)\n99 .02 .03 .01\n1 6 .001 .001\n0 13.7 .0002 .0001\n')
    atomic_json(root/'manifest.json',{'input_artifacts':{'boltzmann_background':{'path':str(background),'sha256':digest(background)}}})
    return background


class DataContractTests(unittest.TestCase):
    def test_uint64_extremes_python_lists_and_missing_namespace_joins(self):
        ids=exact_uint64_ids([1,2**63+1,2**64-1])
        self.assertEqual(ids.dtype,np.dtype('uint64'))
        np.testing.assert_array_equal(ids,[np.uint64(1),np.uint64(2**63+1),np.uint64(2**64-1)])
        for values in ([True],[2**64],[-1],np.array([1.,2.]),[1,1]):
            with self.assertRaises(ValueError):exact_uint64_ids(values)
        np.testing.assert_array_equal(exact_id_join(ids,ids[::-1],requested_namespace='dm:x',available_namespace='dm:x'),[2,1,0])
        for namespace in ('gas:x',''):
            with self.assertRaises(ValueError):exact_id_join(ids,ids,requested_namespace='dm:x',available_namespace=namespace)
        with self.assertRaisesRegex(ValueError,'missing'):exact_id_join([2],[1],requested_namespace='dm:x',available_namespace='dm:x')
        np.testing.assert_array_equal(exact_id_join([2],[1],requested_namespace='dm:x',available_namespace='dm:x',require_all=False),[-1])

    def test_roundtrip_units_preserve_nontrivial_scale_and_h(self):
        rng=np.random.default_rng(77);x=rng.normal(size=(19,3));a,h=.173,.683
        expected={'position_proper_kpc':a/h,'position_comoving_kpc':1/h,'mass_msun':1e10/h,
                  'particle_velocity_peculiar_kms':np.sqrt(a),'catalogue_velocity_peculiar_kms':1.}
        for quantity,factor in expected.items():
            value=x*native_unit_factor(quantity,scale_factor=a,hubble_param=h)
            np.testing.assert_allclose(value,x*factor,rtol=0,atol=0)
            np.testing.assert_allclose(value/factor,x,rtol=3e-16,atol=2e-16)
        for a,h in ((0,1),(1,False),(float('nan'),1)):
            with self.assertRaises(ValueError):native_unit_factor('mass_msun',scale_factor=a,hubble_param=h)

    def test_clock_has_requested_actual_a_and_time_without_extrapolation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);clock_fixture(root)
            clock=run_proper_time_clock(root)
            result=epoch_contract(.5,requested_scale_factor=.4,clock=clock)
            self.assertEqual(result['cosmic_time_gyr'],6.)
            self.assertEqual(result['redshift'],1.)
            self.assertEqual(result['requested_scale_factor'],.4)
            self.assertEqual(result['cosmic_time_source']['status'],'VERIFIED')
            with self.assertRaisesRegex(ValueError,'outside'):epoch_contract(.001,clock=clock)
            with self.assertRaises(ValueError):epoch_contract(True,clock=clock)

    def test_clock_corruption_missing_columns_duplicates_and_nonmonotonicity_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);background=clock_fixture(root)
            background.write_text(background.read_text()+'0 14 .001 .001\n')
            with self.assertRaisesRegex(ValueError,'changed'):run_proper_time_clock(root)
            native=json.loads((root/'manifest.json').read_text());native['input_artifacts']['boltzmann_background']['sha256']=digest(background)
            atomic_json(root/'manifest.json',native)
            with self.assertRaisesRegex(ValueError,'increase strictly'):run_proper_time_clock(root)
        for scales,times in (([.1,.1],[1,2]),([.1,.2],[2,1]),([.1,float('nan')],[1,2])):
            with self.assertRaises(ValueError):validated_proper_time_samples(scales,times)
        with self.assertRaisesRegex(ValueError,'outside'):interpolate_proper_time([.1,1.],[1,13.],1.1)

    def test_atomic_json_disk_failure_retains_old_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'status.json';atomic_json(path,{'status':'retained'})
            old=path.read_bytes()
            with patch('core.foundation.os.replace',side_effect=OSError(errno.ENOSPC,'disk full')):
                with self.assertRaises(OSError):atomic_json(path,{'status':'new'})
            self.assertEqual(path.read_bytes(),old)
            self.assertEqual(list(path.parent.iterdir()),[path])

    def test_publication_fsync_failure_never_publishes_partial_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);stage=root/'stage';stage.mkdir();(stage/'data').write_bytes(b'payload')
            with patch('core.foundation.os.fsync',side_effect=OSError(errno.ENOSPC,'disk full')):
                with self.assertRaises(OSError):publish_directory(stage,root/'final')
            self.assertFalse((root/'final').exists());self.assertTrue((stage/'data').exists())
            publish_directory(stage,root/'final')
            self.assertEqual((root/'final/data').read_bytes(),b'payload')

    def test_actual_hdf5_fletcher_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'product.hdf5'
            with h5py.File(path,'w') as f:
                ids=f.create_dataset('ParticleIDs',data=np.array([2**63+1,2**64-1],dtype=np.uint64),chunks=(2,),fletcher32=True)
                write_dataset_contract(ids,units='identity',identity_namespace='star_population:control')
                offset=ids.id.get_chunk_info_by_coord((0,)).byte_offset
            with path.open('r+b') as raw:
                raw.seek(offset);byte=raw.read(1);raw.seek(offset);raw.write(bytes([byte[0]^1]))
            with h5py.File(path) as f:
                with self.assertRaises(OSError):f['ParticleIDs'][:]

    def test_full_saved_snapshot_audit_rejects_narrow_ids_duplicate_and_payload_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_data_contract_fixtures.test_stellar_history_snapshot(root,0,.5,[2**63+1,2**63+2],[[1,2,3],[4,5,6]])
            result=audit_native_snapshot_contract(root,0)
            self.assertEqual(result['particle_count'],2)
            contract=result['native_dataset_contracts'][0]['contract']
            ids=contract['datasets']['PartType4/ParticleIDs']
            self.assertEqual(ids['dtype'],'uint64')
            self.assertIn('star_population',ids['identity_namespace'])
            position=contract['datasets']['PartType4/Coordinates']
            self.assertEqual(json.loads(position['provenance_json'])['field_unit_conversion']['target_units'],'proper cm')
            self.assertEqual(json.loads(position['epoch_json']),result['epoch'])
            self.assertEqual(result['epoch']['cosmic_time_source']['status'],'UNAVAILABLE')
            path=root/'data/snapshot_000.hdf5'
            with h5py.File(path,'r+') as f:f['PartType4/ParticleIDs'][1]=2**63+1
            with self.assertRaisesRegex(ValueError,'duplicate'):audit_native_snapshot_contract(root,0)
            with h5py.File(path,'r+') as f:
                del f['PartType4/ParticleIDs'];f['PartType4/ParticleIDs']=np.array([1,2],dtype=np.uint32)
            with self.assertRaisesRegex(ValueError,'uint64'):audit_native_snapshot_contract(root,0)
            with h5py.File(path,'r+') as f:
                del f['PartType4/ParticleIDs'];f['PartType4/ParticleIDs']=np.array([1,2],dtype=np.uint64)
                f['PartType4/Coordinates'][0,0]=np.nan
            with self.assertRaisesRegex(ValueError,'Coordinates contains NaN or infinity'):audit_native_snapshot_contract(root,0)

    def test_native_chunk_protection_legacy_strict_and_corruption(self):
        from core.data import seal_hdf5_numeric_datasets
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            test_data_contract_fixtures.test_stellar_history_snapshot(root,0,.5,[2**63+1,2**63+2],[[1,2,3],[4,5,6]])
            result=audit_native_snapshot_contract(root,0)
            self.assertTrue(any(f['chunk_integrity']['status']=='LEGACY_UNPROTECTED' for f in result['fields']))
            self.assertTrue(all(len(f['payload_sha256'])==64 for f in result['fields']))
            with self.assertRaisesRegex(ValueError,'Fletcher32'):
                audit_native_snapshot_contract(root,0,require_chunk_checksums=True)
            path=root/'data/snapshot_000.hdf5'
            with h5py.File(path,'r+') as source:
                seal_hdf5_numeric_datasets(source['PartType4'])
            result=audit_native_snapshot_contract(root,0,require_chunk_checksums=True)
            self.assertTrue(all(f['chunk_integrity']['status']=='PROTECTED' for f in result['fields']))
            with h5py.File(path) as source:
                offset=source['PartType4/ParticleIDs'].id.get_chunk_info_by_coord((0,)).byte_offset
            with path.open('r+b') as raw:
                raw.seek(offset);original=raw.read(1);raw.seek(offset);raw.write(bytes([original[0]^1]))
            with self.assertRaises(OSError):
                audit_native_snapshot_contract(root,0,require_chunk_checksums=True)

    def test_native_partial_and_previous_shards_do_not_enter_discovery(self):
        from halos import _snapshot_files
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_data_contract_fixtures.test_stellar_history_snapshot(root,0,.5,[2**63+1,2**63+2],[[1,2,3],[4,5,6]])
            native=root/'data/snapshot_000.hdf5'
            for suffix in ('.partial.ABCDEF','.previous.ABCDEF'):
                native.with_name(native.name+suffix).write_bytes(b'retained failed or previous native bytes')
            self.assertEqual([p.resolve() for p in _snapshot_files(root,0)],[native.resolve()])
            self.assertEqual(audit_native_snapshot_contract(root,0)['particle_count'],2)

    def test_restart_epoch_and_identity_metadata_equal_despite_status_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);clock_fixture(root);ic=root/'ic.hdf5';ic.write_bytes(b'exact IC identity')
            m=json.loads((root/'manifest.json').read_text());m['gadget_handoff']={'initial_conditions':[{'path':str(ic),'sha256':digest(ic)}]}
            m.update(cosmology_parameters={'H0':67.},simulation_parameters={'seed':8},status='INTERRUPTED')
            atomic_json(root/'manifest.json',m)
            coordinates=coordinate_contract(position_units='proper kpc',velocity_convention='peculiar km/s',origin='global box',periodic_wrapping='wrapped')
            before=product_provenance(root,algorithm=__file__,selection='control',source_files=[ic],coordinates=coordinates)
            epoch_before=epoch_contract(.5,clock=run_proper_time_clock(root))
            m.update(status='SIMULATION_COMPLETE',resume_attempts=[{'status':'COMPLETE'}]);atomic_json(root/'manifest.json',m)
            after=product_provenance(root,algorithm=__file__,selection='control',source_files=[ic],coordinates=coordinates)
            self.assertEqual(before['identity_namespaces'],after['identity_namespaces'])
            self.assertEqual(epoch_before,epoch_contract(.5,clock=run_proper_time_clock(root)))
            self.assertNotEqual(before['run_manifest_sha256'],after['run_manifest_sha256'])

    def test_product_publication_rejects_changed_external_algorithm_and_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'snapshot';source.write_bytes(b'measured input')
            algorithm=root/'external.py';algorithm.write_text('# measured implementation\n')
            contract=product_provenance(root,algorithm=algorithm,selection='control',source_files=[source],coordinates={})
            verify_product_provenance(root,contract)
            algorithm.write_text('# changed implementation\n')
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                verify_product_provenance(root,contract)
            algorithm.write_text('# measured implementation\n');source.write_bytes(b'changed input')
            with self.assertRaisesRegex(ValueError,'source changed'):
                verify_product_provenance(root,contract)
test_blender_data_contract_ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(test_blender_data_contract_ROOT/'core'),str(test_blender_data_contract_ROOT)]
from visualization import make_blender_timeline_loader,_periodic_trajectory_copies


def test_blender_data_contract_fixture(root):
    for n,a,ids,xyz in [(0,.5,[2**63+1,2**64-1],[[99,5,5],[30,5,5]]),
                         (1,1.,[2**64-1,2**63+1,2**63+3],[[31,5,5],[1,5,5],[40,5,5]])]:
        test_stellar_history_snapshot(root,n,a,ids,xyz)
        with h5py.File(root/'data'/f'snapshot_{n:03d}.hdf5','r+') as f:
            f.move('PartType4','PartType1')
            for name in ['NumPart_Total','NumPart_ThisFile']:
                counts=f['Header'].attrs[name];counts[1]=counts[4];counts[4]=0;f['Header'].attrs[name]=counts
    cfg={'cosmology':{'model':'lcdm'},'simulation':{'box_mpc_h':.1,'physics':'dm'},
         'analysis':{'hmf_mass_definition':'fof','hmf_min_particles':30},'blender':{'particle_mode':'sampled'},
         'visualization':{'blender_render':False,'blender_sequence_max_points':100,'blender_galaxy_overlays':False}}
    (root/'config.yaml').write_text(yaml.safe_dump(cfg))
    clock_fixture(root)
    return cfg


class BlenderContractTests(unittest.TestCase):
    def test_saved_ids_periodic_copies_time_and_portable_checksums(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_blender_data_contract_fixture(root)
            loader=make_blender_timeline_loader(root,max_points=10)
            manifest_path=loader.parent/'blender_dm_timeline.json';m=verify_timeline_assets(manifest_path)
            self.assertEqual([v['epoch_contract']['cosmic_time_gyr'] for v in m['snapshots']],[6.,13.7])
            self.assertEqual(m['data_contract']['coordinates']['position_units'],'comoving Mpc/h')
            with np.load(loader.parent/m['intervals'][0]['identity_asset'],allow_pickle=False) as f:
                self.assertEqual(f['matched_vertex_ids'].dtype,np.dtype('uint64'))
                self.assertEqual(list(map(int,f['matched_vertex_ids'])),[2**63+1,2**63+1,2**64-1])
                self.assertEqual(list(map(int,f['right_only_ids'])),[2**63+3])
            before=digest(manifest_path)
            second=make_blender_timeline_loader(root,max_points=10)
            self.assertNotEqual(loader.parent,second.parent);self.assertEqual(digest(manifest_path),before)
            # A copied sealed bundle needs no original development path to verify.
            import shutil
            copied=root/'portable';shutil.copytree(loader.parent,copied);verify_timeline_assets(copied/manifest_path.name)
            asset=copied/m['intervals'][0]['identity_asset'];asset.write_bytes(asset.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'asset missing or changed'):verify_timeline_assets(copied/manifest_path.name)

    def test_changed_clock_and_duplicate_unsampled_id_fail_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_blender_data_contract_fixture(root);background=root/'background.dat';original=background.read_bytes();background.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError,'changed'):make_blender_timeline_loader(root,max_points=1)
            self.assertFalse((root/'results').exists());background.write_bytes(original)
            with h5py.File(root/'data/snapshot_001.hdf5','r+') as f:f['PartType1/ParticleIDs'][2]=2**64-1
            with self.assertRaisesRegex(ValueError,'duplicate'):make_blender_timeline_loader(root,max_points=1)
            self.assertFalse((root/'results').exists())

    def test_native_time_and_unit_mismatch_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_blender_data_contract_fixture(root);path=root/'data/snapshot_001.hdf5'
            with h5py.File(path,'r+') as f:f['Header'].attrs['Time']=np.nan
            with self.assertRaises(ValueError):make_blender_timeline_loader(root,max_points=10)
            self.assertFalse((root/'results').exists())
            with h5py.File(path,'r+') as f:
                f['Header'].attrs['Time']=1.
                f['Parameters'].attrs['UnitLength_in_cm']=1.
            with self.assertRaises(ValueError):make_blender_timeline_loader(root,max_points=10)
            self.assertFalse((root/'results').exists())

    def test_failed_bundle_is_retained_and_never_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_blender_data_contract_fixture(root);destination=root/'bundle'
            with patch('core.foundation.publish_directory',side_effect=OSError(errno.ENOSPC,'disk full')):
                with self.assertRaises(OSError):make_blender_timeline_loader(root,max_points=10,output_directory=destination)
            self.assertFalse(destination.exists())
            stages=list(root.glob('.blender-timeline-*'));self.assertEqual(len(stages),1)
            failure=json.loads((stages[0]/'export_failure.json').read_text());self.assertEqual(failure['status'],'FAIL')
            self.assertTrue((stages[0]/'blender_dm_timeline.json').is_file())

    def test_changed_input_during_measurement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_blender_data_contract_fixture(root)
            import visualization as visualize
            original=visualize._export_blender_timeline
            def mutate(*args,**kwargs):
                result=original(*args,**kwargs)
                with h5py.File(root/'data/snapshot_001.hdf5','r+') as f:f['PartType1/Coordinates'][0,0]=32.
                return result
            with patch('visualization._export_blender_timeline',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'source changed'):make_blender_timeline_loader(root,max_points=10)
            self.assertFalse(list((root/'results/3d').glob('timeline_*')))

    def test_manifest_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_blender_data_contract_fixture(root);loader=make_blender_timeline_loader(root,max_points=10)
            manifest=loader.parent/'blender_dm_timeline.json';m=json.loads(manifest.read_text());m['box_mpc_h']=.2;atomic_json(manifest,m)
            with self.assertRaisesRegex(ValueError,'manifest checksum'):verify_timeline_assets(manifest)

    def test_truncated_generated_ply_is_rejected_before_sealing(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_blender_data_contract_fixture(root)
            import visualization as visualize
            original=visualize._write_blender_particle_ply
            def truncate(*args,**kwargs):
                path=original(*args,**kwargs);path.write_bytes(path.read_bytes()[:-4]);return path
            with patch('visualization._write_blender_particle_ply',side_effect=truncate):
                with self.assertRaisesRegex(ValueError,'PLY payload'):make_blender_timeline_loader(root,max_points=10)
            self.assertFalse(list((root/'results/3d').glob('timeline_*')))

    @patch('visualization._available_ram_bytes', return_value=4*1024**3)
    def test_exact_snapshot_sample_and_full_exports_retain_row_identities(self, available_ram):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=test_blender_data_contract_fixture(root);cfg['visualization']['blender_interpolate_motion']=False
            for mode in ['sampled','full']:
                cfg['blender']['particle_mode']=mode
                loader=make_blender_timeline_loader(root,max_points=10,config=cfg)
                m=verify_timeline_assets(loader.parent/'blender_dm_timeline.json')
                for entry in m['snapshots']:
                    with h5py.File(loader.parent/entry['identity_asset']) as f:
                        self.assertEqual(len(f['ParticleIDs']),entry['particle_count'])
                        self.assertEqual(f['ParticleIDs'].dtype,np.dtype('uint64'))
                        self.assertEqual(f['ParticleIDs'].attrs['identity_namespace'],m['identity_namespace'])
                        self.assertTrue(f['ParticleIDs'].fletcher32)

    def test_combined_public_export_seals_metadata_and_preserves_prior_bundle(self):
        from commands import export
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);test_blender_data_contract_fixture(root)
            first=export(root,10,'preview',12);before=digest(first)
            m=json.loads(first.read_text());self.assertEqual(m['science_acceptance'],'NOT_ACCEPTED')
            self.assertEqual(digest(first),first.with_name(first.name+'.sha256').read_text().strip())
            for kind,name in m['species_manifests'].items():
                verify_timeline_assets(first.parent/name)
                self.assertEqual(digest(first.parent/name),m['species_manifest_sha256'][kind])
            second=export(root,10,'preview',12)
            self.assertNotEqual(first.parent,second.parent);self.assertEqual(digest(first),before)

# ========================================================================
# test_blender_single_publication

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from visualization import make_blender_loader
from blender import verify_single_scene_assets


class BlenderSinglePublicationTests(unittest.TestCase):
    @patch('visualization._available_ram_bytes', return_value=4*1024**3)
    def test_public_sample_and_full_exports_relocate_with_exact_ids(self, available_ram):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'run';root.mkdir();test_blender_data_contract_fixture(root)
            native=root/'data/snapshot_001.hdf5';before=digest(native)
            for mode in ('sampled','full'):
                loader=make_blender_loader(root,snap=1,max_points=10,sequence=False,
                    _particle_mode=mode,_allow_render=False)
                manifest=verify_single_scene_assets(loader.parent/'blender_single_contract.json')
                self.assertEqual(manifest['science_acceptance'],'NOT_ACCEPTED')
                self.assertEqual(manifest['epoch']['actual_scale_factor'],1.)
                status=json.loads((loader.parent/'blender_status_dm.json').read_text())
                with h5py.File(loader.parent/status['identity_asset'],'r') as identities:
                    np.testing.assert_array_equal(identities['ParticleIDs'][:],np.array([2**64-1,2**63+1,2**63+3],dtype=np.uint64))
                    self.assertEqual(identities['ParticleIDs'].attrs['identity_namespace'],manifest['identity_namespace'])
                    self.assertEqual(identities['ParticleIDs'].attrs['epoch_status'],'PRODUCER_DECLARED')
                moved=Path(temporary)/mode
                shutil.move(str(loader.parent),moved)
                verify_single_scene_assets(moved/'blender_single_contract.json')
                self.assertIn('verify_single_scene_assets(base /',(moved/loader.name).read_text())
            self.assertEqual(digest(native),before)
            shutil.rmtree(root)
            for mode in ('sampled','full'):verify_single_scene_assets(Path(temporary)/mode/'blender_single_contract.json')

    def test_failed_publication_preserves_prior_export_and_staged_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);test_blender_data_contract_fixture(root)
            prior=make_blender_loader(root,snap=1,sequence=False,_allow_render=False)
            before=digest(prior)
            with patch('core.foundation.publish_directory',side_effect=OSError('injected rename failure')):
                with self.assertRaisesRegex(OSError,'injected'):
                    make_blender_loader(root,snap=1,sequence=False,_allow_render=False)
            self.assertEqual(digest(prior),before)
            stage=next((root/'results/3d').glob('.blender-single-*'))
            self.assertTrue((stage/'export_failure.json').is_file())
            verify_single_scene_assets(stage/'blender_single_contract.json')

    def test_truncated_ply_and_changed_native_source_are_rejected(self):
        import visualization as visualize
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);test_blender_data_contract_fixture(root)
            writer=visualize._write_blender_particle_ply
            def truncated(*args,**kwargs):
                result=writer(*args,**kwargs)
                path=Path(args[0]);path.write_bytes(path.read_bytes()[:-4])
                return result
            with patch('visualization._write_blender_particle_ply',side_effect=truncated):
                with self.assertRaisesRegex(ValueError,'PLY payload'):
                    make_blender_loader(root,snap=1,sequence=False,_allow_render=False)
            original=visualize._export_blender_single
            def mutated(*args,**kwargs):
                loader=original(*args,**kwargs)
                with h5py.File(root/'data/snapshot_001.hdf5','r+') as source:
                    source['PartType1/Coordinates'][0,0]=11.
                return loader
            with patch('visualization._export_blender_single',side_effect=mutated):
                with self.assertRaisesRegex(ValueError,'source changed'):
                    make_blender_loader(root,snap=1,sequence=False,_allow_render=False)
            self.assertFalse(list((root/'results/3d').glob('snapshot_001_dm_*')))

    def test_manifest_and_geometry_tampering_is_detected_by_consumer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);test_blender_data_contract_fixture(root)
            loader=make_blender_loader(root,snap=1,sequence=False,_allow_render=False)
            manifest_path=loader.parent/'blender_single_contract.json'
            manifest=verify_single_scene_assets(manifest_path)
            ply=loader.parent/next(item['path'] for item in manifest['assets'] if item['path'].endswith('.ply'))
            with ply.open('r+b') as stream:stream.seek(-4,2);stream.write(b'xxxx')
            with self.assertRaisesRegex(ValueError,'asset missing or changed'):verify_single_scene_assets(manifest_path)

# ========================================================================
# test_blender_scene_routes
sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
import blender as blender_scene
from blender import verify_scene_publication


class BlenderSceneRouteTests(unittest.TestCase):
    def fixture(self, root, skip_render=False):
        manifest = root / 'timeline.json'
        manifest.write_text(json.dumps({'schema': 'transport-fixture', 'skip_render': skip_render}))
        return manifest, root / 'movie.mp4'

    def scene_record(self, blend):
        publication = verify_scene_publication(blend)
        return json.loads((blend.parent / publication["archive"] / "scene_manifest.json").read_text())

    def timeline_builder(self, source, movie):
        source, movie = Path(source), Path(movie)
        movie.with_name(source.stem + '.blend').write_bytes(b'fake editable transport fixture')
        if not json.loads(source.read_text()).get('skip_render'):
            movie.write_bytes(b'fake movie transport fixture')

    def test_timeline_preserves_return_and_archives_scene_and_movie(self):
        with tempfile.TemporaryDirectory() as directory:
            source, movie = self.fixture(Path(directory).resolve())
            with patch.object(blender_scene, '_build_timeline_scene', self.timeline_builder):
                self.assertEqual(blender_scene.build_timeline_scene(source, movie), movie)
            record = self.scene_record(movie.with_name(source.stem + '.blend'))
            self.assertEqual(record['science_acceptance'], 'NOT_ACCEPTED')
            self.assertEqual(record['render_artifacts'], [movie.name])
            self.assertEqual(record['source_integrity_status'], 'LEGACY_UNSEALED')

    def test_skip_render_publishes_only_scene_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, movie = self.fixture(Path(directory).resolve(), skip_render=True)
            with patch.object(blender_scene, '_build_timeline_scene', self.timeline_builder):
                blend = blender_scene.build_timeline_scene(source, movie)
                self.assertFalse(movie.exists())
                self.assertEqual(self.scene_record(blend)['render_artifacts'], [])
                with self.assertRaises(FileExistsError):
                    blender_scene.build_timeline_scene(source, movie)

    def test_missing_requested_movie_cannot_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            source, movie = self.fixture(Path(directory).resolve())
            def incomplete(source, movie):
                Path(movie).with_name(Path(source).stem + '.blend').write_bytes(b'fake scene')
            with patch.object(blender_scene, '_build_timeline_scene', incomplete):
                with self.assertRaises(FileNotFoundError):
                    blender_scene.build_timeline_scene(source, movie)
            self.assertFalse(movie.exists())
            self.assertFalse(source.with_suffix('.blend').exists())

    def test_stellar_scene_references_final_directory_and_publishes_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); source = root / 'stars.json'
            source.write_text(json.dumps({'schema': 'darkmattersim-stellar-candidate-blender-v1'}))
            blend = root / 'stars.blend'; calls = []
            def builder(source, staged, render, *, public_directory):
                calls.append(public_directory)
                staged.write_bytes(b'fake stellar scene')
                if render: staged.with_suffix('.png').write_bytes(b'fake preview')
            with patch.object(blender_scene, '_build_stellar_candidate_scene', builder):
                self.assertEqual(blender_scene.build_stellar_candidate_scene(source, blend, True), blend)
            self.assertEqual(calls, [root])
            self.assertEqual(self.scene_record(blend)['render_artifacts'], ['stars.png'])


    def ply_fixture(self, root):
        path = root / 'particles.ply'
        path.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n' + b'\0' * 12)
        return path

    def single_builder(self, ply, movie, *args, source_record=None):
        movie.with_suffix('.blend').write_bytes(b'fake single-snapshot scene')
        movie.write_bytes(b'fake single-snapshot movie')

    def test_raw_single_route_retains_legacy_status_without_invented_science(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); ply = self.ply_fixture(root); movie = root / 'single.mp4'
            with patch.object(blender_scene, '_build_scene', self.single_builder):
                self.assertEqual(blender_scene.build_scene(ply, movie, 10.), movie)
            record = self.scene_record(movie.with_suffix('.blend'))
            self.assertEqual(record['source_integrity_status'], 'LEGACY_UNSEALED')
            source = json.loads((movie.parent / verify_scene_publication(movie.with_suffix('.blend'))['archive'] / 'source_manifest.json').read_text())
            self.assertIsNone(source['scientific_contract'])

    def test_single_route_rejects_source_mutation_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); ply = self.ply_fixture(root); movie = root / 'single.mp4'
            def mutate(ply, movie, *args, source_record=None):
                self.single_builder(ply, movie, *args, source_record=source_record)
                ply.write_bytes(ply.read_bytes()[:-1] + b'x')
            with patch.object(blender_scene, '_build_scene', mutate):
                with self.assertRaises(ValueError): blender_scene.build_scene(ply, movie, 10.)
            self.assertFalse(movie.exists())
            self.assertFalse(movie.with_suffix('.blend').exists())

    def test_explicit_missing_single_contract_cannot_fall_back_to_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); ply = self.ply_fixture(root)
            with patch.object(blender_scene, '_build_scene') as builder:
                with self.assertRaises(ValueError):
                    blender_scene.build_scene(ply, root / 'single.mp4', 10., manifest_path=root / 'missing.json')
                builder.assert_not_called()

# ========================================================================
# test_finder_comparison
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from halos import match_by_exact_membership_and_phase_space,publish_finder_disagreements
from halos import detect_disappearance_swap_jump_fragmentation


def obj(i,x,members):return {'id':i,'position_proper_kpc':[x,0.,0.],'velocity_peculiar_kms':[0.,0.,0.],'mass_msun':10.,'mass_definition':'bound','radius_proper_kpc':1.,'vmax_kms':2.,'central':True,'host_id':None,'members':members,'quality':[]}
def cat(ns,objects):return {'finder':ns,'namespace':ns,'particle_namespace':'run:dm','epoch':epoch_contract(.5),'box_proper_kpc':100.,'objects':objects,'source_hashes':{str(Path(__file__).resolve()):file_sha256(__file__)}}
def test_finder_comparison_request():return {'minimum_shared_particles':1,'minimum_overlap_fraction':.5,'maximum_distance_proper_kpc':3.,'maximum_velocity_kms':5.,'tie_merit_tolerance':0.,'maximum_candidate_pairs':100,'allow_phase_only':False}


class FinderComparisonTests(unittest.TestCase):
    def test_periodic_exact_ids_and_reordering(self):
        left=cat('fof',[obj(0,99.,[2**64-1,2**63+1]),obj(1,50.,[7])]);right=cat('rockstar',[obj(42,1.,[2**63+1,2**64-1])])
        result=match_by_exact_membership_and_phase_space(left,right,test_finder_comparison_request());r=result['candidates'][0]
        self.assertEqual(r['status'],'RECIPROCAL_UNIQUE_MATCH');self.assertEqual(r['distance_proper_kpc'],2.)
        self.assertEqual(result['unmatched_or_ambiguous_left'],[1])
        left['objects'].reverse();self.assertEqual(match_by_exact_membership_and_phase_space(left,right,test_finder_comparison_request())['candidates'],result['candidates'])

    def test_overlapping_memberships_preserve_ties_and_mass_definitions(self):
        left=cat('fof',[obj(0,0.,[1,2])]);right=cat('rockstar',[obj(8,0.,[1,2]),obj(9,1.,[1,2])])
        right['objects'][0]['mass_definition']='m200c';right['objects'][0]['mass_msun']=20.
        result=match_by_exact_membership_and_phase_space(left,right,test_finder_comparison_request())
        self.assertTrue(all(r['status']=='AMBIGUOUS_OR_NONRECIPROCAL' for r in result['candidates']))
        self.assertIsNone(result['candidates'][0]['mass_difference_msun']);self.assertEqual(left['objects'][0]['mass_msun'],10.)

    def test_namespaces_epochs_duplicate_ids_and_pair_budget_rejected(self):
        left=cat('a',[obj(0,0,[1])]);right=cat('b',[obj(0,0,[1])])
        bad=copy.deepcopy(right);bad['particle_namespace']='another-run:dm'
        with self.assertRaisesRegex(ValueError,'namespaces'):match_by_exact_membership_and_phase_space(left,bad,test_finder_comparison_request())
        bad=copy.deepcopy(right);bad['objects'][0]['members']=[1,1]
        with self.assertRaises(ValueError):match_by_exact_membership_and_phase_space(left,bad,test_finder_comparison_request())
        bad=copy.deepcopy(right);bad['epoch']=epoch_contract(.6)
        with self.assertRaisesRegex(ValueError,'epoch'):match_by_exact_membership_and_phase_space(left,bad,test_finder_comparison_request())
        right['objects'].append(obj(1,0,[2]))
        with self.assertRaises(MemoryError):match_by_exact_membership_and_phase_space(left,right,{**test_finder_comparison_request(),'maximum_candidate_pairs':1})

    def test_missing_membership_requires_explicit_phase_policy(self):
        left=cat('a',[obj(0,0,None)]);right=cat('b',[obj(0,0,[1])])
        self.assertEqual(match_by_exact_membership_and_phase_space(left,right,test_finder_comparison_request())['candidates'],[])
        result=match_by_exact_membership_and_phase_space(left,right,{**test_finder_comparison_request(),'allow_phase_only':True})
        self.assertEqual(result['candidates'][0]['status'],'PHASE_ONLY_CANDIDATE')
        self.assertEqual(result['unmatched_or_ambiguous_left'],[0])

    def test_numeric_publication_corruption(self):
        from core.data import verify_numeric_audit
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'comparison.json'
            publish_finder_disagreements(path,cat('a',[obj(0,0,[1])]),cat('b',[obj(9,0,[1])]),test_finder_comparison_request())
            self.assertEqual(verify_numeric_audit(path)['candidates'][0]['right_id'],9)
            path.write_text('{}')
            with self.assertRaises(ValueError):verify_numeric_audit(path)

    def test_tree_disappearance_gap_flip_fragmentation_and_broken_links(self):
        nodes=[{'namespace':'native-tree','id':i,'epoch':e,'mass_msun':m,'position_proper_kpc':[x,0.,0.],'central':c,'host_id':0} for i,e,m,x,c in [(1,0,10.,99.,True),(2,2,100.,1.,False),(3,2,1.,20.,False),(4,1,1.,0.,False)]]
        links=[{'source':['native-tree',1,0],'target':['native-tree',i,2],'origin':'native_tree'} for i in (2,3,999)]
        result=detect_disappearance_swap_jump_fragmentation(nodes,links,epochs=[0,1,2],maximum_mass_ratio=3.,maximum_position_jump_proper_kpc=5.,box_proper_kpc=100.,transient_maximum_detections=1)
        kinds={f['kind'] for f in result['flags']}
        self.assertTrue({'MISSING_DETECTION_GAP','CENTRAL_SATELLITE_FLIP','MASS_JUMP','FRAGMENTATION','BROKEN_LINK','TRANSIENT_DETECTION'}<=kinds)
        self.assertEqual(result['original_links'],links);self.assertEqual(nodes[0]['mass_msun'],10.)
        self.assertFalse(any(f['kind']=='POSITION_JUMP' and f['detail'].get('minimum_image_distance_proper_kpc')==2. for f in result['flags']))

    def test_tree_exact_links_periodic_flags_and_pair_budget(self):
        nodes=[{'namespace':'tree','id':i,'epoch':e,'mass_msun':1.,'position_proper_kpc':[x,0.,0.],
            'core_particle_id':core,'phantom':i==3} for i,e,x,core in [(1,0,101.,10),(2,0,20.,20),(3,1,1.,20),(4,1,20.,10)]]
        links=[{'source':['tree',i,0],'target':['tree',i+2,1],'origin':'native_tree'} for i in (1,2)]
        opts=dict(epochs=[0,1],maximum_mass_ratio=3.,maximum_position_jump_proper_kpc=5.,box_proper_kpc=100.,transient_maximum_detections=1)
        kinds={r['kind'] for r in detect_disappearance_swap_jump_fragmentation(nodes,links,**opts)['flags']}
        self.assertTrue({'RECIPROCAL_CORE_SWAP','PHANTOM_PREDICTION','PERIODIC_COORDINATE_OUTSIDE_BOX'}<=kinds)
        bad=copy.deepcopy(links);bad[0]['source'][1]=1.
        with self.assertRaises(ValueError):detect_disappearance_swap_jump_fragmentation(nodes,bad,**opts)
        with self.assertRaises(MemoryError):detect_disappearance_swap_jump_fragmentation(nodes,links,**{**opts,'maximum_nodes':1})

    def test_separate_repair_proposal_keeps_original_links(self):
        from halos import publish_tree_repair_proposal, verify_tree_repair_proposal
        from core.data import export_numeric_audit,verify_numeric_audit
        from core.foundation import file_sha256
        nodes=[{'namespace':'tree','id':i,'epoch':e,'mass_msun':1.,'position_proper_kpc':[0.,0.,0.]} for i,e in [(1,0),(2,1)]]
        opts=dict(epochs=[0,1],maximum_mass_ratio=3.,maximum_position_jump_proper_kpc=5.,box_proper_kpc=100.,transient_maximum_detections=1)
        with tempfile.TemporaryDirectory() as tmp:
            original=Path(tmp)/'original.json';proposal=Path(tmp)/'proposal.json'
            report=detect_disappearance_swap_jump_fragmentation(nodes,[],**opts)
            export_numeric_audit(original,report,{'node_ids':np.array([1,2],dtype=np.uint64)},
                {'node_ids':{'units':'identity','quantity':'native IDs','identity_namespace':'tree'}},algorithm=__file__,context={'epoch':epoch_contract(.5)},source_hashes={str(Path(__file__).resolve()):file_sha256(__file__)})
            before=original.read_bytes();links=[{'source':['tree',1,0],'target':['tree',2,1],'origin':'proposed_repair'}]
            result=publish_tree_repair_proposal(proposal,original,links,rationale='Review-only candidate',thresholds=opts)
            self.assertEqual(original.read_bytes(),before);self.assertEqual(result['original_links'],[])
            self.assertEqual(verify_numeric_audit(proposal)['proposed_links'],links)
            self.assertEqual(verify_tree_repair_proposal(proposal)['proposed_links'],links)
            request=Path(tmp)/'request.json';public=Path(tmp)/'public.json'
            atomic_json(request,{'original_audit':str(original),'proposed_links':links,
                'rationale':'Review-only public candidate','thresholds':opts})
            for extra in ([],['--verify']):
                completed=subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'commands.py'),
                    'propose-tree-repair',str(request),str(public),*extra],capture_output=True,text=True)
                self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertEqual(original.read_bytes(),before)
            request.write_text('{}')
            with self.assertRaisesRegex(ValueError,'source changed'):verify_tree_repair_proposal(public)
            arrays=original.with_name(original.name+'.data')/'arrays.npz'
            with arrays.open('ab') as stream:stream.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'source changed'):verify_tree_repair_proposal(proposal)

# ========================================================================
# test_exact_output_timeline

test_exact_output_timeline_ROOT = Path(__file__).resolve().parent


class ExactOutputTimelineTests(unittest.TestCase):
    def test_native_dyadic_steps_reach_every_even_boundary_without_crossing(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('Native C++ compiler unavailable')
        source = r'''
#include "exact_output_timeline.h"
#include <cassert>
#include <limits>

template <typename I> void check(I end) {
  for (I boundary = 2; boundary < 1024; boundary += 2) {
    // Every legal active clock before the output; test varying physical limits.
    for (I current = 0; current < boundary; current += 2) {
      for (I proposed = 2; proposed <= 2048; proposed *= 2) {
        I step = research_output_step_limit(proposed, current, boundary, end);
        assert(step >= 2 && step <= proposed);
        assert((step & (step - 1)) == 0);
        assert(current + step <= boundary);
      }
    }
    I current = 0;
    int count = 0;
    while (current < boundary) {
      I step = research_output_step_limit(I(128), current, boundary, end);
      // Native alignment disallows increasing into an unsynchronized bin.
      while (current % step) step /= 2;
      assert(step >= 2);
      current += step;
      assert(++count < 100);
    }
    assert(current == boundary);
  }
  // Disabled, exhausted and beyond-run output requests do not alter dynamics.
  assert(research_output_step_limit(I(128), I(20), I(-1), end) == 128);
  assert(research_output_step_limit(I(128), I(20), I(20), end) == 128);
  assert(research_output_step_limit(I(128), I(20), I(2)*end, end) == 128);
  // Large clocks must not overflow addition or select a step beyond the boundary.
  I now = end - 1024;
  assert(research_output_step_limit(end / 2, now, end - 2, end) == 512);
}
int main() { check<int>(1 << 29); check<long long>(1LL << 60); }
'''
        with tempfile.TemporaryDirectory() as temporary:
            source_file = Path(temporary) / 'control.cc'
            source_file.write_text(source)
            binary = Path(temporary) / 'control'
            subprocess.run([compiler, '-std=c++11', '-O2', '-I',
                str(test_exact_output_timeline_ROOT / 'core/software/gadget4/src/time_integration'),
                str(source_file), '-o', str(binary)], check=True, capture_output=True)
            subprocess.run([str(binary)], check=True, capture_output=True)

# ========================================================================
# test_native_catalogue_bundles
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.data import staged_native_catalogues,rockstar_directory,rebase_recipe_paths
from core.native import native_owned_fds


@staged_native_catalogues
def fixture_native(run_dir,*,fail=False,dry_run=False):
    root=Path(run_dir);output=root/'data/rockstar';output.mkdir(parents=True,exist_ok=True)
    (output/'out_0.list').write_text('#ID Mvir\n17 2.0\n')
    (root/'input/finder.log').write_text('synthetic transport; no native finder executed')
    if (root/'input/nested/recipe.txt').exists():(root/'input/nested/recipe.txt').write_text('staged recipe')
    if not native_owned_fds():raise AssertionError('Native wrapper lacks inherited ownership')
    if fail:raise KeyboardInterrupt()
    return {'status':'COMPLETE','artifact':str(output/'out_0.list')}


class NativeCatalogueBundleTests(unittest.TestCase):
    def test_recipe_rebase_is_single_pass_and_respects_path_boundaries(self):
        root = '/workspace/run'
        previous = root + '/results/native_catalogues/previous'
        stage = root + '/results/native_catalogues/.next'
        original = f'INBASE="{previous}/data"\nSOURCE={root}/input\nOTHER={root}2/input\n'
        rebased = rebase_recipe_paths(original, {root:stage, previous:stage})
        self.assertEqual(rebased, f'INBASE="{stage}/data"\nSOURCE={stage}/input\nOTHER={root}2/input\n')

    def test_public_wrapper_preserves_legacy_inputs_and_publishes_verified_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'data/rockstar').mkdir(parents=True);(root/'input').mkdir()
            prior=root/'data/rockstar/out_0.list';prior.write_bytes(b'legacy completed bytes')
            first=fixture_native(root);current=rockstar_directory(root)
            self.assertEqual((current/'out_0.list').read_text(),'#ID Mvir\n17 2.0\n')
            self.assertEqual(prior.read_bytes(),b'legacy completed bytes')
            self.assertTrue(Path(first['artifact']).is_file())
            self.assertEqual(json.loads((Path(first['bundle_directory'])/'bundle.json').read_text())['science_acceptance'],'NOT_ACCEPTED')
            second=fixture_native(root)
            self.assertNotEqual(first['bundle_directory'],second['bundle_directory'])
            self.assertTrue(Path(first['bundle_directory']).is_dir())
            (rockstar_directory(root)/'out_0.list').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'asset checksum'):rockstar_directory(root)

    def test_nested_input_writes_do_not_mutate_prior_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'data').mkdir();(root/'input/nested').mkdir(parents=True)
            source=root/'input/nested/recipe.txt';source.write_text('original recipe')
            result=fixture_native(root)
            self.assertEqual(source.read_text(),'original recipe')
            bundle=Path(result['bundle_directory']);nested=bundle/'input/nested/recipe.txt'
            self.assertEqual(nested.read_text(),'staged recipe');self.assertFalse(nested.parent.is_symlink())
            manifest=json.loads((bundle/'bundle.json').read_text())
            self.assertIn('nested/recipe.txt',[item['path'] for item in manifest['input_assets']])
            nested.write_text('changed')
            with self.assertRaisesRegex(ValueError,'recipe/log checksum'):rockstar_directory(root)

    def test_failure_keeps_prior_index_and_retains_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'data').mkdir();(root/'input').mkdir()
            first=fixture_native(root);index=root/'input/native_catalogue_index.json';before=index.read_bytes()
            with self.assertRaises(KeyboardInterrupt):fixture_native(root,fail=True)
            self.assertEqual(index.read_bytes(),before)
            stage=list((root/'results/native_catalogues').glob('.fixture_native-*'))[0]
            self.assertTrue((stage/'data/rockstar/out_0.list').is_file())
            self.assertTrue((stage/'failure.json').is_file())
            self.assertEqual(rockstar_directory(root),Path(first['bundle_directory'])/'data/rockstar')

# ========================================================================
# test_inference_figures

test_inference_figures_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(test_inference_figures_ROOT), str(test_inference_figures_ROOT / 'core')]
from figures import weighted_quantiles, normalized_weights, corner_figure, predictive_figure, export_diagnostics
from figures import apply_plot_style


class FigureTests(unittest.TestCase):
    def setUp(self):
        apply_plot_style({'figure_formats': ['png'], 'figure_dpi': 80})

    def test_weighted_empirical_cdf_ignores_zero_weights_and_extreme_scales(self):
        np.testing.assert_equal(weighted_quantiles([1., 2., 100.], [0., .5, 1.], [1., 3., 0.]), [1., 2., 2.])
        np.testing.assert_allclose(normalized_weights([1e308, 1e308], 2), [.5, .5])
        with self.assertRaises(ValueError):
            normalized_weights([1., -1.], 2)
        with self.assertRaises(ValueError):
            weighted_quantiles([1., float('nan')], [.5], None)

    def test_catalogue_is_never_labelled_posterior(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'corner'
            samples = np.array([[1., 4.], [2., 4.], [3., 4.]])
            provenance = {'source': 'analytic fixture', 'selection': 'all three points'}
            with self.assertRaisesRegex(ValueError, 'sampler'):
                corner_figure(samples, ['mass', 'shape'], ['Msun', 'dimensionless'], target,
                              sample_kind='posterior', provenance=provenance)
            corner_figure(samples, ['mass', 'shape'], ['Msun', 'dimensionless'], target,
                          sample_kind='population', provenance=provenance)
            record = json.loads(target.with_suffix('.figure.json').read_text())
            self.assertEqual(record['scientific_metadata']['kind'], 'population')
            self.assertTrue(target.with_suffix('.png').is_file())

    def test_residual_sign_scale_and_envelope_values(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'predictions'
            predictive_figure([0., 1.], [1., 3.], [2., 1.], [[3., 2.]] * 3, target,
                coordinate_label='radius [kpc]', observable_label='speed [km/s]',
                provenance={'source': 'analytic control', 'selection': 'both bins',
                            'model': 'constant predictions', 'uncertainty_definition': 'known Gaussian sigma'})
            record = json.loads(target.with_suffix('.figure.json').read_text())['scientific_metadata']
            np.testing.assert_equal(record['normalized_residual_quantiles_025_16_50_84_975'], [[1., -1.]] * 5)
            with self.assertRaises(ValueError):
                predictive_figure([1., 0.], [1., 3.], [2., 0.], [[3., 2.]] * 3, target,
                                  coordinate_label='x', observable_label='y', provenance={})

    def test_npz_export_retains_source_hash_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez(root / 'samples.npz', samples=np.arange(10.).reshape(5, 2))
            (root / 'metadata.json').write_text(json.dumps({'labels': ['x', 'y'], 'units': ['kpc', 'km/s'],
                'sample_kind': 'population', 'provenance': {'selection': 'all control points'}}))
            request = {'samples_npz': str(root / 'samples.npz'), 'metadata_json': str(root / 'metadata.json'),
                       'output_directory': str(root / 'figures')}
            record = export_diagnostics(request)
            self.assertEqual(len(record['provenance']['source_sha256']), 64)
            self.assertEqual(record['science_status'], 'NOT_ACCEPTED')
            self.assertTrue(Path(record['figures']['corner'][0]).is_file())
            with self.assertRaises(FileExistsError):
                export_diagnostics(request)

# ========================================================================
# test_zoom_resolution

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from zoom import contamination_epoch, audit_zoom_regions


class ZoomResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        test_stellar_history_snapshot(self.root, 0, 0.5, [1], [[0, 0, 0]])
        self.path = self.root / "data/snapshot_000.hdf5"
        with h5py.File(self.path, "r+") as f:
            f.move("PartType4", "PartType1")
            counts = np.array([0, 1, 2, 0, 0, 1], dtype=np.uint64)
            f["Header"].attrs["NumPart_Total"] = counts
            f["Header"].attrs["NumPart_ThisFile"] = counts
            for t, xyz, masses, ids in [
                (2, [[99, 0, 0], [50, 0, 0]], [8.0, 8.0], [2, 3]),
                (5, [[0, 0, 0]], [1.0], [4]),
            ]:
                p = f.create_group(f"PartType{t}")
                p["Coordinates"] = np.asarray(xyz, dtype=float)
                p["Velocities"] = np.zeros((len(ids), 3))
                p["Masses"] = masses
                p["ParticleIDs"] = np.asarray(ids, dtype=np.uint64)
                for field in ("Coordinates", "Velocities", "Masses"):
                    p[field].attrs.update(dict(f["PartType1"][field].attrs))

    def kwargs(self):
        return dict(
            center_comoving_kpc_h=[0, 0, 0],
            radius_proper_kpc=2.0,
            fine_dm_mass_msun=2e10,
            maximum_coarse_mass_fraction=0.0,
        )

    def test_periodic_boundary_coarse_mass_and_bh_exclusion(self):
        row = contamination_epoch([self.path], 0, **self.kwargs())
        self.assertEqual(row["status"], "FAIL")
        self.assertEqual(row["coarse_count_inside"], 1)
        self.assertEqual(row["nearest_coarse_distance_proper_kpc"], 1.0)
        self.assertAlmostEqual(row["coarse_mass_fraction"], 0.8)
        self.assertEqual(row["fine_dm_count_inside"], 1)
        self.assertEqual(row["stellar_count_inside"], 0)

    def test_wrong_mass_and_oversized_region_rejected(self):
        kwargs = self.kwargs()
        kwargs["fine_dm_mass_msun"] = 3e10
        with self.assertRaisesRegex(ValueError, "smaller"):
            contamination_epoch([self.path], 0, **kwargs)
        kwargs = self.kwargs()
        kwargs["radius_proper_kpc"] = 51.0
        with self.assertRaisesRegex(ValueError, "half"):
            contamination_epoch([self.path], 0, **kwargs)

    def test_failed_gate_preserves_report_and_figures(self):
        result = audit_zoom_regions(
            self.root,
            regions=[
                dict(snapshot=0, center_comoving_kpc_h=[0, 0, 0], radius_proper_kpc=2.0)
            ],
            fine_dm_mass_msun=2e10,
            maximum_coarse_mass_fraction=0.0,
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(Path(result["figure"]).is_file())
        self.assertTrue((Path(result["directory"]) / "provenance.json").is_file())

# ========================================================================
# test_numpy_contract_integrity
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))


class NumpyContractIntegrityTests(unittest.TestCase):
    def fixture(self,path):
        np.savez_compressed(path,ids=np.array([2**63+3,2**64-1],dtype=np.uint64),
                            positions=np.array([[1.,2.,3.],[np.nan]*3]),present=[True,False])
        return numpy_archive_contract(path,{
            'ids':{'units':'identity','identity_namespace':'fixture:star_population'},
            'positions':{'units':'proper kpc','quantity':'synthetic observer offsets','missing_policy':'NaN for absent IDs'},
            'present':{'units':'boolean availability mask','quantity':'requested particle availability'}},epoch=epoch_contract(.5))

    def test_exact_ids_actual_shape_missingness_and_unaccepted_quality(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'truth.npz';contract=self.fixture(path)
            verify_numpy_archive_contract(path,contract)
            self.assertEqual(json.loads(contract['datasets']['positions']['shape_json']),[2,3])
            self.assertEqual(contract['datasets']['ids']['dtype'],'uint64')
            self.assertEqual(contract['datasets']['ids']['identity_namespace'],'fixture:star_population')
            self.assertEqual(contract['datasets']['positions']['quality_status'],'NOT_ACCEPTED')
            with np.load(path,allow_pickle=False) as source:
                self.assertEqual(int(source['ids'][1]),2**64-1)
                self.assertTrue(np.isnan(source['positions'][1]).all())

    def test_valid_archive_rewrite_and_false_payload_digest_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'truth.npz';contract=self.fixture(path)
            changed=copy.deepcopy(contract);changed['datasets']['ids']['payload_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'payload checksum'):verify_numpy_archive_contract(path,changed)
            with np.load(path,allow_pickle=False) as source:arrays={name:source[name] for name in source.files}
            arrays['positions'][0,0]=99.
            np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):verify_numpy_archive_contract(path,contract)

    def test_member_inventory_identity_dtype_and_declared_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'truth.npz';contract=self.fixture(path)
            with self.assertRaises(MemoryError):verify_numpy_archive_contract(path,contract,memory_bytes=1)
            with self.assertRaisesRegex(ValueError,'every array'):numpy_archive_contract(path,{})
            np.savez(path,ids=np.array([3.]))
            with self.assertRaisesRegex(ValueError,'must be uint64'):
                numpy_archive_contract(path,{'ids':{'units':'identity','identity_namespace':'fixture:dm'}})
            np.savez(path,ids=np.array([3],dtype=np.uint64))
            with zipfile.ZipFile(path,'a') as archive:
                archive.writestr('ids.npy',archive.read('ids.npy'))
            with self.assertRaisesRegex(ValueError,'duplicate member'):
                numpy_archive_contract(path,{'ids':{'units':'identity','identity_namespace':'fixture:dm'}})

# ========================================================================
# test_native_identity_domains
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.data import canonical_native_links,validate_signed_native_particle_ids
from analysis import rockstar_table_from_path


class NativeIdentityDomainTests(unittest.TestCase):
    def test_missing_link_keeps_zero_available_distinct_from_zero_placeholder(self):
        ids,available=canonical_native_links(np.array([-1,0,2**63-1],dtype=np.int64))
        self.assertEqual(ids.dtype,np.dtype('uint64'))
        np.testing.assert_array_equal(ids,np.array([0,0,2**63-1],dtype=np.uint64))
        np.testing.assert_array_equal(available,[False,True,True])
        with self.assertRaises(ValueError):canonical_native_links(np.array([-2],dtype=np.int64))
        with self.assertRaises(ValueError):canonical_native_links(np.array([1.],dtype=float))

    def test_unsigned_scientific_identity_stays_exact_across_entire_domain(self):
        expected=np.array([0,2**63+1,2**64-1],dtype=np.uint64)
        ids,available=canonical_native_links(expected)
        np.testing.assert_array_equal(ids,expected);self.assertTrue(available.all())

    def test_pinned_signed_reader_preflight_rejects_wide_ids_without_casting(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'snapshot.hdf5'
            with h5py.File(path,'w') as source:
                source['PartType1/ParticleIDs']=np.array([0,2**63-1],dtype=np.uint64)
            self.assertEqual(validate_signed_native_particle_ids([path],chunk_rows=1)['checked_particle_ids'],2)
            with h5py.File(path,'r+') as source:source['PartType1/ParticleIDs'][1]=np.uint64(2**63+1)
            with self.assertRaisesRegex(ValueError,'signed-int64'):validate_signed_native_particle_ids([path],chunk_rows=1)
            with h5py.File(path) as source:self.assertEqual(int(source['PartType1/ParticleIDs'][1]),2**63+1)

    def test_real_ascii_reader_exposes_exact_links_and_availability(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'out_0.list';path.write_text('#ID PID DescID Mvir\n0 -1 9223372036854775807 2.0\n')
            table=rockstar_table_from_path(path)
            self.assertEqual(int(table['_canonical_identity_links']['descid']['values'][0]),2**63-1)
            self.assertFalse(table['_canonical_identity_links']['pid']['available'][0])
            self.assertTrue(table['_canonical_identity_links']['id']['available'][0])

# ========================================================================
# test_native_execution_receipts
import inspect,signal
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.native import stop_owned_process_groups, invocation_record, verify_invocation_inputs
from halos import run_rockstar

class NativeExecutionTests(unittest.TestCase):
    def test_input_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);binary=root/'binary';binary.write_bytes(b'not executed');cfg=root/'native.cfg';cfg.write_text('input')
            record=invocation_record([binary,cfg],root);cfg.write_text('changed')
            with self.assertRaises(ValueError):verify_invocation_inputs(record)
    def test_spawn_failure_retains_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);binary=root/'binary';binary.write_bytes(b'not executed');receipt=root/'invocation.json'
            with patch('core.native.subprocess.Popen',side_effect=OSError('spawn failed')):
                with self.assertRaises(OSError):run_native_command([binary],cwd=root,receipt_path=receipt)
            record=json.loads(receipt.read_text());self.assertEqual(record['status'],'FAILED')
            self.assertEqual(record['science_acceptance'],'NOT_ACCEPTED')
    def test_cleanup_escalates_only_declared_new_session_groups(self):
        process=Mock(pid=43210)
        with patch('core.foundation.os.killpg') as kill:
            stop_owned_process_groups([process,None],grace_seconds=0.)
        self.assertEqual(kill.call_args_list[0].args,(43210,signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args,(43210,signal.SIGKILL))
        process.wait.assert_called_once()
    def test_server_early_exit_runs_cleanup_and_retains_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);software=root/'software/rockstar';software.mkdir(parents=True)
            (software/'rockstar').write_bytes(b'not executed');(root/'input').mkdir();conf=root/'input/rockstar.cfg';conf.write_text('config')
            process=Mock(pid=43210);process.poll.return_value=1
            with patch('halos.ROOT',root),patch('halos.native_engine_directory',return_value=software),patch('halos.rockstar_worker_count',return_value=8),patch('halos.rockstar_config',return_value=conf),patch('halos.subprocess.Popen',return_value=process),patch('halos.stop_owned_process_groups') as cleanup:
                with self.assertRaisesRegex(RuntimeError,'before creating'):inspect.unwrap(run_rockstar)(root,{})
            cleanup.assert_called_once_with([None,process])
            self.assertEqual(json.loads((root/'input/rockstar_invocation.json').read_text())['status'],'FAILED')

    def test_timeout_records_failure_and_cleans_owned_group(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);binary=root/'binary';binary.write_bytes(b'not executed');receipt=root/'invocation.json'
            process=Mock(pid=43210);process.wait.side_effect=subprocess.TimeoutExpired('native',.25)
            with patch('core.native.subprocess.Popen',return_value=process) as spawn,patch('core.native.stop_owned_process_groups') as cleanup:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_native_command([binary],cwd=root,receipt_path=receipt,timeout=.25)
            self.assertNotIn('timeout',spawn.call_args.kwargs)
            process.wait.assert_called_once_with(timeout=.25)
            cleanup.assert_called_once_with([process])
            self.assertEqual(json.loads(receipt.read_text())['failure']['type'],'TimeoutExpired')

# ========================================================================
# test_native_source_manifests
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.cosmology import native_source_manifest,verify_native_source_manifest

class NativeSourceManifestTests(unittest.TestCase):
    def fixture(self,root):
        (root/'solver.c').write_text('int main(void){return 0;}')
        (root/'Makefile').write_text('CC=cc\n')
        (root/'helper.pl').write_text('print "native helper";')
        (root/'build').mkdir();(root/'build/generated.c').write_text('ignored generated build tree')
    def test_recipe_and_perl_source_bytes_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            with patch('core.cosmology._pinned_software_revision',return_value={'status':'DECLARED','commit':'fixture'}):
                record=native_source_manifest('fixture',root);verify_native_source_manifest(record)
            names={item['path'] for item in record['source_tree']['files']}
            self.assertEqual(names,{'solver.c','Makefile','helper.pl'})
            self.assertIn('UNDECLARED',record['build_equivalence_status'])
            self.assertEqual(record['science_acceptance'],'NOT_ACCEPTED')
    def test_changed_or_added_native_source_is_rejected(self):
        for change in ('changed','added'):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);self.fixture(root)
                with patch('core.cosmology._pinned_software_revision',return_value={'status':'DECLARED'}):
                    record=native_source_manifest('fixture',root)
                    (root/('solver.c' if change=='changed' else 'extra.h')).write_text('new bytes')
                    with self.assertRaises(ValueError):verify_native_source_manifest(record)

# ========================================================================
# test_sealed_blender_invocation
sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
import visualization as visualize


class SealedBlenderInvocationTests(unittest.TestCase):
    def fixture(self, root):
        manifest = root / 'blender_single_contract.json'
        loader = root / 'loader.py'; loader.write_text('# mock transport loader\n')
        status = root / 'status.json'; atomic_json(status, {'movie': 'single.mp4', 'render_status': 'NOT_REQUESTED'})
        assets = [{'path': p.name, 'sha256': digest(p), 'size_bytes': p.stat().st_size} for p in (loader, status)]
        atomic_json(manifest, {'schema': 'blender-single-publication-v1', 'assets': assets})
        atomic_text(manifest.with_name(manifest.name + '.sha256'), digest(manifest) + '\n')
        return manifest, loader, status

    def config(self):
        return {'visualization': {'blender_render': True}}

    def test_invocation_sees_sealed_inputs_and_retains_resealed_alias_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); manifest, loader, status = self.fixture(root)
            consumed = []
            def runner(command, *, cwd, receipt_path, inputs, **kwargs):
                self.assertEqual(manifest.with_name(manifest.name + '.sha256').read_text().strip(), digest(manifest))
                consumed.extend(inputs)
                atomic_json(receipt_path, {'input_hashes': {str(p.resolve()): digest(p) for p in inputs}, 'status': 'RETURNED'})
                (root / 'single.mp4').write_bytes(b'fake movie')
                (root / 'single.blend').write_bytes(b'fake blend')
            with patch.object(visualize, '_find_blender_executable', return_value='/mock/blender'), \
                 patch('core.native.run_native_command', runner), \
                 patch('blender.verify_scene_publication', return_value={'state': 'COMPLETE'}):
                visualize._render_sealed_blender_bundle(root, manifest, loader, status, self.config())
            self.assertIn(manifest, consumed)
            record = json.loads((root / 'blender_invocation.json').read_text())
            verify_invocation_inputs(record)
            self.assertNotEqual(record['consumed_input_hashes'][str(manifest)], digest(manifest))
            self.assertEqual(json.loads(status.read_text())['render_status'], 'COMPLETE')
            for item in json.loads(manifest.read_text())['assets']:
                self.assertEqual(item['sha256'], digest(root / item['path']))
            self.assertEqual(record['input_hashes'][str(root / '.render_inputs' / manifest.name)], record['consumed_input_hashes'][str(manifest)])

    def test_missing_blender_preserves_export_with_explicit_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); manifest, loader, status = self.fixture(root)
            with patch.object(visualize, '_find_blender_executable', return_value=None), \
                 patch('core.native.run_native_command') as runner:
                visualize._render_sealed_blender_bundle(root, manifest, loader, status, self.config())
                runner.assert_not_called()
            self.assertEqual(json.loads(status.read_text())['render_status'], 'SKIPPED')
            self.assertEqual(manifest.with_name(manifest.name + '.sha256').read_text().strip(), digest(manifest))

    def test_disabled_render_does_not_change_sealed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(); manifest, loader, status = self.fixture(root); before = manifest.read_bytes()
            with patch('core.native.run_native_command') as runner:
                visualize._render_sealed_blender_bundle(root, manifest, loader, status, self.config(), enabled=False)
                runner.assert_not_called()
            self.assertEqual(manifest.read_bytes(), before)

# ========================================================================
# test_filament_publication

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from visualization import _load_disperse_skeleton,_DISPERSE_SKELETON_CACHE


class FilamentPublicationTests(unittest.TestCase):
    def fixture(self,root):
        config=test_blender_data_contract_fixture(root)
        executable=root/'mse-control';executable.write_text('deferred executable transport fixture\n')
        converter=root/'skelconv-control';converter.write_text('deferred converter transport fixture\n')
        config['visualization'].update(disperse_output_coordinate_representation='comoving_mpc_h',
            disperse_executable=str(executable),skelconv_executable=str(converter),disperse_persistence_cut=.35)
        (root/'config.yaml').write_text(yaml.safe_dump(config))

    def geometry(self,*args,output_directory,**kwargs):
        output=Path(output_directory);(output/'raw-control.dat').write_bytes(b'deferred geometry transport fixture')
        result = {'points_mpc_h':np.array([[.01,.02,.03],[.04,.05,.06]]),'lines':np.array([2,0,1],dtype=np.int64),
                  'metadata':{'coordinate_conversion':'comoving_mpc_h', 'point_connectivity_file':'geometry.npz'}}
        np.savez(output/'geometry.npz', points_mpc_h=result['points_mpc_h'], lines=result['lines'],
                 metadata_json=np.array(json.dumps(result['metadata'])))
        return result

    def test_public_bundle_contains_canonical_numeric_geometry_and_hash_verified_cache(self):
        _DISPERSE_SKELETON_CACHE.clear()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            with patch('visualization._compute_disperse_skeleton',side_effect=self.geometry) as kernel:
                result=_load_disperse_skeleton(root,1,8,.35)
                again=_load_disperse_skeleton(root,1,8,.35)
                self.assertEqual(kernel.call_count,1)
            self.assertIs(result,again)
            bundle=Path(result['metadata']['bundle_directory'])
            manifest=json.loads((bundle/'filament_contract.json').read_text())
            self.assertEqual(manifest['science_acceptance'],'NOT_ACCEPTED')
            self.assertEqual(manifest['epoch']['actual_scale_factor'],1.)
            for item in manifest['assets']:self.assertEqual(digest(bundle/item['path']),item['sha256'])
            with h5py.File(bundle/'skeleton.hdf5','r') as numeric:
                self.assertEqual(numeric['points_comoving_mpc_h'].attrs['units'],'comoving Mpc/h')
                self.assertTrue(numeric['points_comoving_mpc_h'].fletcher32)
            (bundle/'raw-control.dat').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'asset changed'):_load_disperse_skeleton(root,1,8,.35)

    def test_publication_failure_retains_native_work_and_source_mutation_rejects_commit(self):
        _DISPERSE_SKELETON_CACHE.clear()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            with patch('visualization._compute_disperse_skeleton',side_effect=self.geometry),patch(
                'core.foundation.publish_directory',side_effect=OSError('injected skeleton publication failure')):
                with self.assertRaisesRegex(OSError,'injected'):_load_disperse_skeleton(root,1,8,.35)
            def mutate(*args,**kwargs):
                result=self.geometry(*args,**kwargs)
                with h5py.File(root/'data/snapshot_001.hdf5','r+') as source:source['PartType1/Coordinates'][0,0]=13.
                return result
            with patch('visualization._compute_disperse_skeleton',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'source changed'):_load_disperse_skeleton(root,1,8,.35)
            stages=list((root/'results/filaments').glob('.filaments-*'))
            self.assertEqual(len(stages),2)
            self.assertTrue(all((stage/'raw-control.dat').is_file() for stage in stages))
            self.assertTrue(all((stage/'export_failure.json').is_file() for stage in stages))

    def test_undeclared_output_units_cannot_be_inferred_from_coordinate_magnitudes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config=test_blender_data_contract_fixture(root)
            with self.assertRaisesRegex(ValueError,'explicitly declare'):_load_disperse_skeleton(root,1,8,.35)
from zoom import (publish_resolution_budget,
    publish_resolution_convergence, verify_resolution_convergence, stage_resolution_refinement)
from evolution import analyze_event, predict_events, publish_history_candidates, plan_event_followups

test_block07_08_integration_ROOT=Path(__file__).resolve().parent


def sealed(path, data):
    atomic_json(path,data)
    atomic_text(path.with_name(path.name+'.sha256'),file_sha256(path)+'\n')


def trajectory(times=(0.,1.,2.), positions=((-2.,0,0),(0.,0,0),(2.,0,0))):
    return {'object_id':'black_hole:9223372036854775809','times_myr':list(times),
        'relative_positions_proper_kpc':[list(p) for p in positions],
        'valid':[True]*len(times),'encounter_radius_proper_kpc':1.,'transitions':[]}


class BlockIntegrationTests(unittest.TestCase):
    def test_saved_minimum_has_conservative_time_bracket_and_clock_bound(self):
        h=trajectory();event=predict_events([h])[0]
        measured=analyze_event(event,[h],1.1,clock_error_myr=.1)
        self.assertEqual(measured['status'],'MEASURED_EVENT')
        self.assertEqual(measured['measured_bracket_myr'],[0.,2.])
        self.assertEqual(measured['time_error_bound_myr'],1.1)
        self.assertEqual(measured['science_acceptance'],'NOT_ACCEPTED')
        self.assertEqual(analyze_event(event,[h],.9)['status'],'UNRESOLVED_EVENT')

    def test_interpolated_encounter_and_transition_are_not_measured_events(self):
        h=trajectory((0.,2.),((-2.,0,0),(2.,0,0)))
        event=predict_events([h])[0]
        self.assertEqual(analyze_event(event,[h],10.)['status'],'UNRESOLVED_EVENT')
        h['transitions']=[{'kind':'bh_merger','left_index':0,'right_index':1}]
        transition=next(e for e in predict_events([h]) if e['kind']=='bh_merger')
        self.assertEqual(analyze_event(transition,[h],10.)['status'],'UNRESOLVED_EVENT')
        with self.assertRaises(ValueError):analyze_event(event,[h],1.,clock_error_myr=-1.)

    def test_resolution_arrays_roundtrip_and_corruption(self):
        from core.data import epoch_contract, coordinate_contract
        from core.data import verify_numeric_audit
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'budget.json'
            budget=evaluate_resolution(target(),census(),test_resolution_budget_request())
            result=publish_resolution_budget(path,budget,epoch=epoch_contract(.5),
                coordinates=coordinate_contract(position_units='proper kpc',origin='test aperture',
                    periodic_wrapping='minimum image',velocity_convention='none'))
            self.assertEqual(verify_numeric_audit(path)['fingerprint'],budget['fingerprint'])
            archive=path.parent/result['numeric_data']['directory']/'arrays.npz'
            with np.load(archive) as arrays:
                self.assertEqual(arrays['species_count'].dtype,np.dtype('int64'))
                self.assertEqual(arrays['species_mass_msun'][1,1],1e6)
            with archive.open('ab') as f:f.write(b'corruption')
            with self.assertRaises(ValueError):verify_numeric_audit(path)

    def test_count_only_convergence_receipt_cannot_grant_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp).resolve()/'convergence.json'
            budget=evaluate_resolution(target(),census(),test_resolution_budget_request())
            result=publish_resolution_convergence(path,budget['fingerprint'],test_resolution_budget_request(),{})
            self.assertEqual(result['status'],'FAIL')
            with self.assertRaisesRegex(ValueError,'retained measurements'):
                verify_resolution_convergence(path,budget['fingerprint'])
            result['comparisons']={'profile':{}}
            result['observables']['profile']['status']='PASS'
            sealed(path,result)
            with self.assertRaisesRegex(ValueError,'decisions differ'):
                verify_resolution_convergence(path,budget['fingerprint'])

    def test_actual_music_refinement_staging_and_caps(self):
        from zoom import verify_staged_zoom_inputs
        from zoom import verify_generated_zoom
        from zoom import refinement_generation_limits
        configured=os.environ.get('RESEARCH_NATIVE_ZOOM_BRANCH')
        base=Path(configured).resolve() if configured else test_block07_08_integration_ROOT/'results/start-epoch-controls/seed-9223372036854775819-start-00'
        parent=base/'staged/native-inputs';generation=base/'generated/generation.json'
        if configured and not generation.exists():raise FileNotFoundError('Explicit native MUSIC generation is missing')
        if not generation.exists():self.skipTest('Retained native MUSIC generation unavailable')
        original=verify_staged_zoom_inputs(parent);generated=verify_generated_zoom(generation)
        base_request={key:original[key] for key in ('box_mpc_h','parent_level','refinement_level','seeds_by_level','phase_noise_by_level')}
        base_request.update(spectrum_manifest=original['solver_manifest']['path'],padding_cells=2,poisson_accuracy=1e-14,
            lpt_order=original['perturbation_order'],lpt_seed_kind=original['lpt_seed_kind'],lpt_seed=original['lpt_seed'])
        next_level=original['refinement_level']+1
        inherited_level=str(original['refinement_level'])
        new_seeds={str(next_level):17}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();budget=evaluate_resolution(target(),census(),test_resolution_budget_request())
            budget['refinement']['native_generation_source']={'path':str(generation),'sha256':file_sha256(generation)}
            receipt=root/'provenance.json';atomic_json(receipt,{'records':[{'resolution_budget':budget}]})
            atomic_text(root/'provenance.sha256',file_sha256(receipt)+'\n')
            result=stage_resolution_refinement(root/'child',parent_staging_directory=parent,budget_receipt=receipt,
                base_request=base_request,refinement_levels=1,new_seeds_by_level=new_seeds,padding_cells=3)
            child=verify_staged_zoom_inputs(root/'child')
            self.assertEqual(child['refinement_level'],next_level)
            for level,seed in original['seeds_by_level'].items():
                self.assertEqual(child['seeds_by_level'][level],seed)
            self.assertEqual(child['phase_noise_by_level'],original['phase_noise_by_level'])
            self.assertEqual(result['lineage']['fine_mass_reduction'],8)
            memory,disk,_=refinement_generation_limits(root/'child/zoom_template.conf',10.,20.)
            self.assertEqual((memory,disk),(2.,4.))
            with self.assertRaisesRegex(ValueError,'follow-up limit'):
                stage_resolution_refinement(root/'too-many',parent_staging_directory=parent,budget_receipt=receipt,
                    base_request=base_request,refinement_levels=3,new_seeds_by_level={},padding_cells=3)
            changed=copy.deepcopy(base_request);changed['seeds_by_level'][inherited_level]=original['seeds_by_level'][inherited_level]+1
            with self.assertRaisesRegex(ValueError,'parent seeds'):
                stage_resolution_refinement(root/'wrong-phase',parent_staging_directory=parent,budget_receipt=receipt,
                    base_request=changed,refinement_levels=1,new_seeds_by_level=new_seeds,padding_cells=3)
            receipt.write_text('{}')
            with self.assertRaisesRegex(ValueError,'budget changed'):
                refinement_generation_limits(root/'child/zoom_template.conf',10.,20.)

    def test_actual_bh_history_adapter_preserves_uint64_and_missing_population(self):
        from verification import test_black_hole_analysis_fixture as fixture
        from verification import clock_fixture
        from black_holes import analyze_black_holes
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();fixture(root);clock_fixture(root)
            history=analyze_black_holes(root)
            result=publish_history_candidates(root,root/'events.json',history_path=history['catalogue'],population='black_hole',
                selected_object_ids=[2**63+1,17],host_centers_comoving_kpc={'0':[10.,12.,14.],'1':[10.,12.,14.]},
                periodic_box_comoving_kpc=200.,encounter_radius_proper_kpc=1.,transitions_by_object={},memory_gib=.1)
            self.assertEqual(result['histories'][0]['object_id'],'black_hole:'+str(2**63+1))
            self.assertEqual(result['histories'][0]['valid'],[True,True])
            self.assertEqual(result['histories'][1]['valid'],[False,False])
            self.assertFalse(any(e['kind']=='bh_merger' for e in result['events']))

    def test_followup_planner_clock_limits_and_restart_decisions(self):
        # Test the inverse-clock orchestration with a declared analytic clock;
        # actual native checkpoint execution is a separate retained campaign.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'input').mkdir()
            (root/'input/param.txt').write_text('TimeBegin 0.5\nTimeMax 1.0\nMaxSizeTimestep 0.1\nComovingIntegrationOn 1\n')
            (root/'input/Config.sh').write_text('SYNCHRONIZED_TIMESTEPS\n')
            h=trajectory((500.,750.,1000.))
            h['transitions']=[{'kind':'bh_pair','left_index':0,'right_index':2}]
            product=root/'history.json'
            sealed(product,{'schema':'event-history-candidates-v1','run_dir':str(root),
                'histories':[h],'events':predict_events([h]),'actual_output_times_myr':h['times_myr'],'source_hashes':{}})
            args=dict(history_product=product,required_time_error_myr=20.,checkpoint_directories=[],ranks=1,
                maximum_branches=1,maximum_outputs=100,maximum_timestep=.001,wall_hours=.1,memory_gib=1.,disk_gib=1.,omp_threads=1)
            clock=(np.array([.5,1.]),np.array([.5,1.]),{'status':'analytic test control'})
            with patch('core.cosmology.run_proper_time_clock',return_value=clock),patch('core.data.proper_time_error_at_epoch',return_value={'absolute_bound_gyr':0.}):
                unavailable=plan_event_followups(root,root/'unavailable',**args)
                self.assertTrue(all(e['status']=='REQUIRES_EARLIER_CHECKPOINT_OR_IC_RERUN' for e in unavailable['events']))
                with patch('evolution.choose_restart',return_value={'status':'AVAILABLE','directory':'analytic-checkpoint-control'}):
                    result=plan_event_followups(root,root/'queued',**args)
                    self.assertGreater(result['branch_requests'],0)
                    self.assertEqual(result['events'][0]['status'],'QUEUED_CHECKPOINT_BRANCH')
                    self.assertEqual(result['branch_requests'],1)
                    self.assertEqual(result['events'][1]['status'],'SHARED_QUEUED_BRANCH')
                    limit=plan_event_followups(root,root/'limit',**{**args,'maximum_branches':0})
                    self.assertTrue(all(e['status']=='FOLLOWUP_LIMIT_REACHED' for e in limit['events']))
                    cap=plan_event_followups(root,root/'cap',**{**args,'maximum_outputs':3})
                    self.assertTrue(all(e['status']=='OUTPUT_LIMIT_REACHED' for e in cap['events']))
                measured=plan_event_followups(root,root/'measured',**{**args,'required_time_error_myr':260.})
                self.assertEqual(measured['branch_requests'],0)
                self.assertTrue(all(e['status']=='MEASURED_EVENT' for e in measured['events']))
            with patch('core.cosmology.run_proper_time_clock',return_value=clock),patch('core.data.proper_time_error_at_epoch',return_value={'absolute_bound_gyr':None}):
                result=plan_event_followups(root,root/'unbounded',**args)
                self.assertTrue(all(e['status']=='UNBOUNDED_BACKGROUND_CLOCK' for e in result['events']))

    def test_public_zoom_audit_publishes_budget_and_acceptance_refuses_missing_convergence(self):
        from verification import ZoomResolutionTests
        from zoom import audit_zoom_regions
        from core.data import verify_numeric_audit
        from verification import spec_fixture
        from core.foundation import acceptance_report
        fixture=ZoomResolutionTests();fixture.setUp()
        try:
            import h5py
            with h5py.File(fixture.path,'r+') as f:
                p=f['Parameters'].attrs
                for t in range(6):
                    p[f'SofteningClassOfPartType{t}']=t
                    p[f'SofteningComovingClass{t}']=.01
                    p[f'SofteningMaxPhysClass{t}']=.01
            req=test_resolution_budget_request();req['radial_edges_proper_kpc']=[0,1,2]
            result=audit_zoom_regions(fixture.root,regions=[dict(snapshot=0,center_comoving_kpc_h=[0,0,0],radius_proper_kpc=2.)],
                fine_dm_mass_msun=2e10,maximum_coarse_mass_fraction=0.,
                resolution_budget={'request':req,'targets_by_snapshot':{'0':target()},'memory_gib':.1,'convergence_by_snapshot':{}})
            provenance=Path(result['directory'])/'provenance.json';data=json.loads(provenance.read_text())
            row=data['records'][0]
            self.assertEqual(verify_numeric_audit(provenance.parent/row['resolution_numeric_audit']['path'])['census']['measurement_origin'],'native_saved_snapshot')
            spec=spec_fixture();spec['resolution_budget']=req
            report=acceptance_report({'products':['shape'],'research_spec':spec},
                {'resolution_budget':str(provenance.relative_to(fixture.root))},fixture.root)
            self.assertEqual(report['context_checks']['resolution_budget']['status'],'FAIL')
            self.assertIn('convergence is missing',report['context_checks']['resolution_budget']['reason'])
        finally:
            fixture.doCleanups()

    def test_paired_comparison_logic_and_receipt_faults(self):
        # Numeric comparator control with isolated mocked settings readback.
        # This checks tolerance/axis logic; it is not physical convergence evidence.
        from zoom import compare_resolution_measurements
        from core.data import export_numeric_audit
        from core.data import epoch_contract
        source=native_control_fixture('periodic/data/snapshot_003.hdf5')
        if not source.exists():self.skipTest('Native source fixture unavailable')
        import h5py
        with h5py.File(source) as f:a=float(f['Header'].attrs['Time'])
        sources=[{'path':str(source),'sha256':file_sha256(source)}]
        with tempfile.TemporaryDirectory() as tmp:
            comparisons={'profile':{}};settings=[]
            for axis in ('mass','softening','timestep'):
                pair={'dataset':'profile'}
                for level in ('coarse','fine'):
                    values={'mass':8.,'softening':2.,'timestep':.01}
                    if level=='fine':values[axis]/=2
                    settings.append(values)
                    path=Path(tmp).resolve()/(axis+'-'+level+'.json')
                    meta={'origin':'native_simulation','units':'Msun','observable':'profile','selection':'comparator control',
                        'actual_scale_factor':a,'physics_fingerprint':'control','source_snapshots':sources,
                        'bin_coordinates':[1.,2.],'all_requested_bins_valid':True,'numerical_settings':values}
                    export_numeric_audit(path,{'resolution_measurement':meta},{'profile':np.array([0.,1.])},
                        {'profile':{'units':'Msun','quantity':'comparator control vector'}},algorithm=__file__,
                        context={'epoch':epoch_contract(a)},source_hashes={str(source):file_sha256(source)})
                    pair[level]=str(path)
                comparisons['profile'][axis]=pair
            with patch('zoom.measure_native_resolution_settings',side_effect=copy.deepcopy(settings)):
                self.assertEqual(compare_resolution_measurements(test_resolution_budget_request(),comparisons)['profile']['status'],'PASS')
            path=Path(tmp).resolve()/'receipt.json'
            with patch('zoom.measure_native_resolution_settings',side_effect=copy.deepcopy(settings)):
                publish_resolution_convergence(path,'control-fingerprint',test_resolution_budget_request(),comparisons)
            with patch('zoom.measure_native_resolution_settings',side_effect=copy.deepcopy(settings)):
                self.assertEqual(verify_resolution_convergence(path,'control-fingerprint')['status'],'PASS')
            with self.assertRaisesRegex(ValueError,'target/config'):
                verify_resolution_convergence(path,'wrong-fingerprint')
            coarse=Path(comparisons['profile']['mass']['coarse']);coarse.write_text('{}')
            with self.assertRaisesRegex(ValueError,'source changed'):
                verify_resolution_convergence(path,'control-fingerprint')

    def test_backbone_only_obeys_output_limit_and_extended_clock(self):
        from evolution import merge_and_quantize_native_output_times
        clock={'time_begin':.5,'time_max':1.,'maximum_timestep':.001,'timebins':60,'comoving':True}
        with self.assertRaisesRegex(ValueError,'output budget'):
            merge_and_quantize_native_output_times([.5,.6,.7,1.],[],clock,maximum_outputs=3)
        result=merge_and_quantize_native_output_times([.5,1.],[],clock)
        self.assertEqual(result['rows'][-1]['native_integer_tick'],1<<60)

# ========================================================================
# test_zoom_phases

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from zoom import linear_density_modes, compare_density_phases


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ic = self.root / "ic.hdf5"
        q = np.array(list(np.ndindex(16, 16, 16)), dtype=float) / 16
        psi = np.zeros_like(q)
        psi[:, 0] = 0.001 * np.sin(2 * np.pi * q[:, 0])
        with h5py.File(self.ic, "w") as f:
            h = f.create_group("Header").attrs
            for key, value in dict(
                BoxSize=1.0,
                Time=0.01,
                HubbleParam=0.7,
                Omega0=0.3,
                OmegaBaryon=0.05,
                OmegaLambda=0.7,
                Music_levelmin=4,
                Music_shift_back=1,
                Music_lpt_order=1,
                Music_lagrangian_schema=1,
                Music_lagrangian_unitlength_cm=KPC_CM,
                Music_max_abs_displacement_native=[0.001, 0, 0],
                Music_output_coordinate_shift_box_fraction=[0.0, 0.0, 0.0],
                NumFilesPerSnapshot=1,
                NumPart_Total=[0, len(q), 0, 0, 0, 0],
                NumPart_ThisFile=[0, len(q), 0, 0, 0, 0],
                MassTable=[0, 1.0, 0, 0, 0, 0],
            ).items():
                h[key] = value
            g = f.create_group("PartType1")
            g["Coordinates"] = (q + psi) % 1
            g["LagrangianCoordinates"] = q
            g["ParticleIDs"] = np.arange(len(q), dtype=np.uint64)
        self.noise = self.root / "noise.bin"
        with self.noise.open("wb") as f:
            np.array([16] * 3, dtype="=u4").tofile(f)
            np.arange(16**3, dtype="=f8").tofile(f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_analytic_plane_wave(self):
        actual = linear_density_modes(self.ic, [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(actual, [-np.pi * 0.001, 0], atol=1e-15)

    def test_species_window_uses_its_own_mass_normalization(self):
        with h5py.File(self.ic,'r+') as source:
            q=source['PartType1/LagrangianCoordinates'][:]
            psi=(source['PartType1/Coordinates'][:]-q+.5)%1-.5
            gas=source.create_group('PartType0')
            gas['Coordinates']=(q+2*psi)%1
            gas['LagrangianCoordinates']=q
            gas['ParticleIDs']=np.arange(len(q),dtype=np.uint64)+len(q)
            for name in ('NumPart_Total','NumPart_ThisFile'):
                source['Header'].attrs[name]=[len(q),len(q),0,0,0,0]
            source['Header'].attrs['MassTable']=[1.,1.,0,0,0,0]
            source['Header'].attrs['Music_max_abs_displacement_native']=[.002,0,0]
        np.testing.assert_allclose(linear_density_modes(self.ic,[[1,0,0]],particle_types=(0,)),[-2*np.pi*.001],atol=1e-15)
        np.testing.assert_allclose(linear_density_modes(self.ic,[[1,0,0]]),[-1.5*np.pi*.001],atol=1e-15)
        with self.assertRaises(ValueError):
            linear_density_modes(self.ic,[[1,0,0]],particle_types=(1,1))

    def test_phase_reversal_fails_and_retains_evidence(self):
        import shutil

        zoom = self.root / "zoom.hdf5"
        shutil.copyfile(self.ic, zoom)
        with h5py.File(zoom, "r+") as f:
            g = f["PartType1"]
            g["Coordinates"][:] = (
                2 * g["LagrangianCoordinates"][:] - g["Coordinates"][:]
            ) % 1
        report = compare_density_phases(
            self.ic,
            zoom,
            self.noise,
            self.noise,
            self.root / "fail.json",
            minimum_unique_modes=12,
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["coarse_noise_bitwise_equal"])
        self.assertAlmostEqual(report["weighted_phase_rms_rad"], np.pi)
        self.assertTrue((self.root / "fail.json").exists())

    def test_default_requires_more_than_small_benchmark(self):
        with self.assertRaisesRegex(ValueError, "Too few"):
            compare_density_phases(
                self.ic, self.ic, self.noise, self.noise, self.root / "report.json"
            )

    def test_native_metadata_fail_closed(self):
        for key, value in [
            ("Music_lpt_order", 2),
            ("Music_max_abs_displacement_native", [0.3, 0, 0]),
            ("NumFilesPerSnapshot", 2),
            ("NumPart_Total", [0, 4095, 0, 0, 0, 0]),
        ]:
            with self.subTest(key=key):
                with h5py.File(self.ic, "r+") as f:
                    old = f["Header"].attrs[key]
                    f["Header"].attrs[key] = value
                with self.assertRaises(ValueError):
                    linear_density_modes(self.ic, [[1, 0, 0]])
                with h5py.File(self.ic, "r+") as f:
                    f["Header"].attrs[key] = old

    def test_memory_gate_before_file_open(self):
        with self.assertRaises(MemoryError):
            linear_density_modes(self.root / "missing", [[1, 0, 0]], memory_gib=0.001)

# ========================================================================
# test_ic_validation
from initial_conditions import (validate_linear_bundle, resolve_start_epoch, measure_ic_modes,
                           validate_ic_realization, compare_ic_generators, validate_refinement_continuity)
import verification as staging_fixtures
import verification as phase_fixtures


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = staging_fixtures.StagingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f = self.fixture

    def test_exact_epoch_species_and_finest_mesh(self):
        result = validate_linear_bundle(self.f.manifest,redshift=99,box_mpc_h=10,refinement_level=7)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['mode_coverage']['status'],'PASS')
        self.assertEqual(result['species'],['baryon','cdm'])
        self.assertEqual(resolve_start_epoch(self.f.manifest,99)['source_redshift'],99)

    def test_no_growth_rescaling_or_implicit_regeneration(self):
        with self.assertRaisesRegex(ValueError,'exact requested epoch'):
            validate_linear_bundle(self.f.manifest,redshift=49)
        with self.assertRaisesRegex(ValueError,'explicitly authorized'):
            resolve_start_epoch(self.f.manifest,49,regenerate=lambda *_:self.fail('unauthorized solver'))

    def test_gauge_hash_input_and_coverage_rejected(self):
        self.f.info['class_parameters']['start']['nbody_gauge_transfer_functions']='no'
        self.f.save_solver_input();self.f.save_manifest()
        with self.assertRaisesRegex(ValueError,'N-body gauge'):
            validate_linear_bundle(self.f.manifest)
        self.f.info['class_parameters']['start']['nbody_gauge_transfer_functions']='yes'
        self.f.save_solver_input();self.f.save_manifest()
        with self.assertRaisesRegex(ValueError,'finest mesh'):
            validate_linear_bundle(self.f.manifest,box_mpc_h=.001,refinement_level=20)
        (self.f.root/'input.ini').write_text('H0 = 80\n')
        with self.assertRaisesRegex(ValueError,'Changed'):
            validate_linear_bundle(self.f.manifest)

    def test_distinct_authorized_child_requires_parent_hash_and_exact_epoch(self):
        root = self.f.root/'child';root.mkdir()
        for name in ('transfer.dat','background.dat','input.ini'):
            shutil.copyfile(self.f.root/name,root/name)
        info = json.loads(json.dumps(self.f.info))
        info['parameters']['z_start']=49
        info['class_parameters']['start']['z_pk']=49
        transfer = root/'transfer.dat'
        transfer.write_text(transfer.read_text().replace('z = 99','z = 49'))
        ini = root/'input.ini';ini.write_text(ini.read_text().replace('z_pk = 99.0','z_pk = 49.0'))
        info['checksums_sha256']={name:file_sha256(root/name) for name in ('transfer.dat','background.dat','input.ini')}
        child = root/'spectrum.json'
        child.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError,'parent bundle lineage'):
            resolve_start_epoch(self.f.manifest,49,allow_regeneration=True,regenerate=lambda *_:child)
        info['parent_bundle']={'path':str(self.f.manifest),'sha256':file_sha256(self.f.manifest)}
        child.write_text(json.dumps(info))
        result = resolve_start_epoch(self.f.manifest,49,allow_regeneration=True,regenerate=lambda *_:child)
        self.assertEqual(result['source_redshift'],49)


class RealizationTests(unittest.TestCase):
    def setUp(self):
        self.f = audit_fixtures.ZoomAuditTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        with h5py.File(self.f.ic,'r+') as source:
            h=source['Header'].attrs
            h['Music_lagrangian_schema']=1
            from core.foundation import KPC_CM
            h['Music_lagrangian_unitlength_cm']=KPC_CM
            h['Music_max_abs_displacement_native']=[0.,0.,0.]
            for t in (0,1,2):
                group=source['PartType'+str(t)]
                group['LagrangianCoordinates']=group['Coordinates'][:]

    def test_no_evidence_is_incomplete_not_pass(self):
        report = validate_ic_realization(self.f.ic,self.f.cfg)
        self.assertEqual(report['status'],'INCOMPLETE')
        for name in ('mass','exact_ids','momentum','barycenter'):
            self.assertEqual(report['checks'][name]['status'],'PASS')
        self.assertEqual(report['checks']['density_modes']['status'],'PENDING')

    def test_expected_zero_velocity_and_density_amplitudes(self):
        report = validate_ic_realization(self.f.ic,self.f.cfg,modes=[[1,0,0]],
                    expected_density=[1.+0j],expected_velocity=[0j])
        self.assertEqual(report['checks']['density_modes']['status'],'PASS')
        self.assertEqual(report['checks']['velocity_modes']['status'],'PASS')
        bad = validate_ic_realization(self.f.ic,self.f.cfg,modes=[[1,0,0]],expected_density=[-1.+0j])
        self.assertEqual(bad['status'],'FAIL')
        self.assertAlmostEqual(bad['checks']['density_modes']['maximum_phase_error_radians'],np.pi)

    def test_total_momentum_includes_thermal_and_units(self):
        with h5py.File(self.f.ic,'r+') as source:
            source['PartType1/Velocities'][:]=[10,0,0]
        report=validate_ic_realization(self.f.ic,self.f.cfg)
        self.assertEqual(report['status'],'FAIL')
        self.assertEqual(report['checks']['momentum']['status'],'FAIL')
        self.assertGreater(report['checks']['momentum']['bulk_velocity_km_s'][0],0)

    def test_boundary_positions_checked_and_jump_measured(self):
        q=np.array([[0.,.1,.2],[.9,.2,.3]])
        good=validate_refinement_continuity(q,np.ones_like(q),q+1,np.ones_like(q),box=1,tolerance=1e-8)
        self.assertEqual(good['status'],'PASS')
        bad=validate_refinement_continuity(q,np.ones_like(q),q,np.ones_like(q)*2,box=1,tolerance=1e-8)
        self.assertEqual(bad['status'],'FAIL')
        with self.assertRaisesRegex(ValueError,'different Lagrangian'):
            validate_refinement_continuity(q,np.ones_like(q),q+.1,np.ones_like(q),box=1,tolerance=1e-8)


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.f=phase_fixtures.PhaseTests();self.f.setUp();self.addCleanup(self.f.tearDown)
        with h5py.File(self.f.ic,'r+') as source:
            q=source['PartType1/LagrangianCoordinates'][:]
            v=np.zeros_like(q);v[:,0]=.01*np.sin(2*np.pi*q[:,0])
            source['PartType1/Velocities']=v
        self.second=self.f.root/'second.hdf5';shutil.copyfile(self.f.ic,self.second)

    def test_identical_generators_pass_and_reversed_phase_fails(self):
        result=compare_ic_generators(self.f.ic,self.second,[[1,0,0],[0,1,0]])
        self.assertEqual(result['status'],'PASS')
        with h5py.File(self.second,'r+') as source:
            q=source['PartType1/LagrangianCoordinates'][:]
            psi=(source['PartType1/Coordinates'][:]-q+.5)%1-.5
            source['PartType1/Coordinates'][:]=(q-psi)%1
            source['PartType1/Velocities'][:]*=-1
        result=compare_ic_generators(self.f.ic,self.second,[[1,0,0]])
        self.assertEqual(result['status'],'FAIL')

    def test_epochs_and_identity_corruption_rejected(self):
        with h5py.File(self.second,'r+') as source:
            source['Header'].attrs['Time']=.02
        with self.assertRaisesRegex(ValueError,'epoch'):
            compare_ic_generators(self.f.ic,self.second,[[1,0,0]])
        with h5py.File(self.second,'r+') as source:
            source['Header'].attrs['Time']=.01
            source['PartType1/ParticleIDs'][0]=source['PartType1/ParticleIDs'][1]
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            compare_ic_generators(self.f.ic,self.second,[[1,0,0]])

    def test_bulk_fourier_subtracts_thermal_component(self):
        before=measure_ic_modes(self.f.ic,[[1,0,0]])
        with h5py.File(self.f.ic,'r+') as source:
            g=source['PartType1'];q=g['LagrangianCoordinates'][:]
            thermal=np.zeros_like(q);thermal[:,0]=2*np.sin(2*np.pi*q[:,0])
            g['ThermalVelocities']=thermal
            g['Velocities'][:]+=thermal
        after=measure_ic_modes(self.f.ic,[[1,0,0]])
        np.testing.assert_allclose(before['velocity_divergence'],after['velocity_divergence'],rtol=1e-12)

class PublicAuditTests(unittest.TestCase):
    def test_complex_fourier_prediction_archive_is_verified_and_immutable(self):
        import commands as audit_ic
        from core.data import verify_numeric_audit
        f=ZoomAuditTestsForExport()
        self.addCleanup(f.cleanup)
        result=validate_ic_realization(f.ic,f.cfg,f.root/'audit.json',modes=[[1,0,0]],
            expected_density=audit_ic._complex_prediction({'real':[1.],'imag':[0.]}),expected_velocity=[0j])
        self.assertEqual(verify_numeric_audit(f.root/'audit.json'),result)
        self.assertIn('numeric_data',result)
        with self.assertRaises(FileExistsError):
            validate_ic_realization(f.ic,f.cfg,f.root/'audit.json',modes=[[1,0,0]])

    def test_public_cli_incomplete_is_nonzero_and_bundle_pass_is_zero(self):
        import os,subprocess,sys
        f=ZoomAuditTestsForExport();self.addCleanup(f.cleanup)
        root=Path(__file__).resolve().parent
        request=f.root/'request.json'
        request.write_text(json.dumps({'ic':str(f.ic),'configuration':str(f.cfg)}))
        result=subprocess.run([sys.executable,str(root/'commands.py'),'audit-ic', 'realization',str(request)],
                    capture_output=True,text=True,cwd=root)
        self.assertEqual(result.returncode,2,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'INCOMPLETE')
        request.write_text(json.dumps({'ic':str(f.ic),'configuration':str(f.cfg),'modes':[[0,0,0]]}))
        result=subprocess.run([sys.executable,str(root/'commands.py'),'audit-ic', 'realization',str(request)],
                    capture_output=True,text=True,cwd=root)
        self.assertEqual(result.returncode,1)
        self.assertIn('nonzero integer',result.stderr)


class ZoomAuditTestsForExport:
    def __init__(self):
        self.fixture=audit_fixtures.ZoomAuditTests();self.fixture.setUp()
        self.ic,self.cfg = self.fixture.ic,self.fixture.cfg
        self.root=self.ic.parent
        from core.foundation import KPC_CM
        with h5py.File(self.ic,'r+') as source:
            source['Header'].attrs.update(Music_lagrangian_schema=1,
                Music_lagrangian_unitlength_cm=KPC_CM,Music_max_abs_displacement_native=[0.,0.,0.])
            for t in (0,1,2):
                group=source['PartType'+str(t)];group['LagrangianCoordinates']=group['Coordinates'][:]
    def cleanup(self):
        self.fixture.doCleanups()

# ========================================================================
# test_snapshot_intake_publication
sys.path[:0] = [str(Path(__file__).resolve().parent / 'core'), str(Path(__file__).resolve().parent)]
from analysis import prepare_hdf5_folder, verify_snapshot_intake


class SnapshotIntakeTests(unittest.TestCase):
    def source(self, root):
        root.mkdir()
        for index, a in enumerate((.5, 1.)):
            test_stellar_history_snapshot(root, index, a, [2**63 + index + 1], [[1, 2, 3]])
            path = root / 'data' / ('snapshot_%03d.hdf5' % index)
            with h5py.File(path, 'r+') as source:
                source['Header'].attrs['Research2Generation'] = 'fixture-%d' % index
                source['Header'].attrs['Research2ShardCount'] = 1
            atomic_json(root / 'data' / ('snapshot_%03d.publication.json' % index),
                {'schema':'native-hdf5-set-publication-v1','state':'COMPLETE',
                 'generation':'fixture-%d' % index,'shard_count':1})
        return root / 'data'

    def test_complete_intake_preserves_markers_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = self.source(root / 'original')
            output = root / 'imported'
            original = digest(source / 'snapshot_000.hdf5')
            prepare_hdf5_folder(source, output_dir=output, products=['profiles'])
            self.assertEqual(verify_snapshot_intake(output)['status'], 'INTEGRITY_VERIFIED')
            self.assertTrue((output / 'data/snapshot_000.hdf5').is_symlink())
            self.assertEqual((output / 'data/snapshot_000.publication.json').read_bytes(),
                             (source / 'snapshot_000.publication.json').read_bytes())
            self.assertEqual(digest(source / 'snapshot_000.hdf5'), original)
            record = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(record['epochs']['0']['actual_scale_factor'], .5)
            with h5py.File(source / 'snapshot_000.hdf5', 'r+') as saved:
                saved['PartType4/Coordinates'][0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'source changed'): verify_snapshot_intake(output)

    def test_interrupted_publication_preserves_stage_and_originals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = self.source(root / 'original')
            output = root / 'imported'
            before = digest(source / 'snapshot_000.hdf5')
            with patch('core.foundation.publish_directory', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    prepare_hdf5_folder(source, output_dir=output, products=['profiles'])
            self.assertFalse(output.exists())
            self.assertTrue(list(root.glob('.snapshot-intake-*/manifest.json')))
            self.assertTrue(list(root.glob('.snapshot-intake-*/failure.json')))
            self.assertEqual(digest(source / 'snapshot_000.hdf5'), before)

# ========================================================================
# test_paired_report_publication
test_paired_report_publication_ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(test_paired_report_publication_ROOT/'core'))
from analysis import create_paired_analysis_report,verify_paired_analysis_report

class PdfFixture:
    def __init__(self,path):self.path=Path(path)
    def __enter__(self):self.path.write_bytes(b'explicit mocked PDF transport fixture');return self
    def __exit__(self,*args):return False
    def infodict(self):return {}
    def savefig(self,*args,**kwargs):return None

class PairedReportPublicationTests(unittest.TestCase):
    def fixture(self,root):
        comparison=root/'comparison';comparison.mkdir()
        atomic_json(comparison/'comparison_summary.json',{'pairs':[]})
        return comparison

    def test_retained_input_generation_and_corrupt_pdf_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);comparison=self.fixture(root)
            with patch('matplotlib.backends.backend_pdf.PdfPages',PdfFixture):
                output=create_paired_analysis_report(root/'A',root/'B',comparison)
            record=verify_paired_analysis_report(comparison)
            self.assertEqual(record['science_acceptance'],'NOT_ACCEPTED')
            self.assertIn('inputs/comparison_summary.json',record['artifacts'])
            with self.assertRaises(FileExistsError):create_paired_analysis_report(root/'A',root/'B',comparison)
            output.write_bytes(b'corrupt public copy')
            with self.assertRaisesRegex(ValueError,'Public paired'):verify_paired_analysis_report(comparison)

    def test_report_failure_keeps_staged_pdf_and_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);comparison=self.fixture(root)
            with patch('matplotlib.backends.backend_pdf.PdfPages',PdfFixture),patch('core.foundation.publish_directory',side_effect=OSError('simulated interruption')):
                with self.assertRaisesRegex(OSError,'interruption'):
                    create_paired_analysis_report(root/'A',root/'B',comparison)
            self.assertFalse((comparison/'paired_analysis_report.pdf').exists())
            self.assertTrue(list(comparison.glob('.paired-report-stage-*/failure.json')))
            self.assertTrue(list(comparison.glob('.paired-report-stage-*/paired_analysis_report.pdf')))

    def test_pending_publication_blocks_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);comparison=self.fixture(root)
            atomic_json(comparison/'paired_report_publication.json',{'state':'PENDING','generation':'paired_reports/failed'})
            with self.assertRaisesRegex(ValueError,'pending'):verify_paired_analysis_report(comparison)

# ========================================================================
# test_analysis_finder_contracts
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.data import attach_analysis_identity_contracts

class AnalysisFinderContractTests(unittest.TestCase):
    def test_exact_ids_missing_links_and_distinct_native_domains(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'finder.h5'
            with h5py.File(path,'w') as source:
                _write_product_hdf5(source,'data',{'per_halo_rockstar_id':np.array([0,-1,12],dtype=np.int64),
                    'per_halo_gadget_index':np.array([0,1,2],dtype=np.int64),
                    'per_halo_rockstar_index':np.array([0,1,2],dtype=np.int64),
                    'per_halo_mass_definition_pairs':{'mvir':{'rockstar_halo_id':np.array([2**63+7,2**64-1],dtype=np.uint64)}}})
                group=source['data'];epoch={'actual_scale_factor':.5}
                attach_analysis_identity_contracts(group,'finder',snapshot=7,identity_namespaces={'halo':'run:halo','subhalo':'run:subhalo'},epoch=epoch,provenance={})
                attach_analysis_dataset_units(group,'finder',epoch=epoch,provenance={})
                ids=group['per_halo_rockstar_id']
                self.assertEqual(ids.dtype,np.dtype('uint64'));np.testing.assert_array_equal(ids[:],[0,0,12])
                np.testing.assert_array_equal(group['per_halo_rockstar_id_available'][:],[True,False,True])
                self.assertNotEqual(ids.attrs['identity_namespace'],group['per_halo_rockstar_index'].attrs['identity_namespace'])
                self.assertNotEqual(ids.attrs['identity_namespace'],group['per_halo_gadget_index'].attrs['identity_namespace'])
                self.assertEqual(group['per_halo_mass_definition_pairs/mvir/rockstar_halo_id'][1],np.uint64(2**64-1))
                self.assertIn('not cross-finder identity',json.loads(ids.attrs['ancestry_json'])['semantics'])
            verify_hdf5_dataset_payloads(path,require_contract=True)
    def test_nested_hmf_estimator_units(self):
        with tempfile.TemporaryDirectory() as temporary,h5py.File(Path(temporary)/'cosmology.h5','w') as source:
            _write_product_hdf5(source,'data',{'spatial_jackknife':{'sigma_dn_dM_h4_per_msun_mpc3':[1.],
                'sigma_dn_dlog10M_per_mpc_h3':[2.],'regions':8},
                'rare_halos':{'1000000000000.0':3,'maximum_mass_msun_h':2e12},
                'analytic_sheth_tormen':{'sigma_linear_top_hat':[.5],'dn_dlog10M_mpc_h3':[1.]}})
            group=source['data'];attach_analysis_dataset_units(group,'hmf')
            self.assertEqual(group['spatial_jackknife/sigma_dn_dM_h4_per_msun_mpc3'].attrs['units'],'(Msun/h)^-1 (Mpc/h)^-3')
            self.assertEqual(group['rare_halos/1000000000000.0'].attrs['units'],'halo count')
            self.assertEqual(group['analytic_sheth_tormen/sigma_linear_top_hat'].attrs['units'],'dimensionless')
    def test_full_route_identity_and_scalar_profile_row(self):
        with tempfile.TemporaryDirectory() as temporary,h5py.File(Path(temporary)/'full.h5','w') as source:
            group=source.create_group('snapshot')
            group.create_dataset('concentration_halo_id',data=np.array([12],dtype=np.int64))
            group.create_dataset('finder_match_gadget_index',data=np.array([0],dtype=np.int64))
            group.create_group('halo_profiles').create_dataset('halo_catalogue_row',data=np.int64(0))
            attach_analysis_identity_contracts(group,'legacy',snapshot=1,identity_namespaces={'halo':'run:halo','subhalo':'run:subhalo'},epoch={},provenance={})
            attach_legacy_analysis_units(group)
            self.assertEqual(group['concentration_halo_id'].dtype,np.dtype('uint64'))
            self.assertEqual(group['halo_profiles/halo_catalogue_row'].dtype,np.dtype('uint64'))
            self.assertTrue(group['halo_profiles/halo_catalogue_row_available'][()])
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))

class CheckpointTests(unittest.TestCase):
    def rank(self,root,rank,ranks=2,a=.2,generation=123):
        path=root/f'restart.{rank}';path.write_bytes(bytes(range(256))*16)
        receipt={'schema':'gadget-rank-checkpoint-v1','rank':rank,'ranks':ranks,'a':a,'generation':generation,'size_bytes':path.stat().st_size,'crc32':f'{zlib.crc32(path.read_bytes()):08x}'}
        Path(str(path)+'.checksum.json').write_text(json.dumps(receipt));return path
    def test_complete_rank_checksums_and_exact_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rank in range(2):self.rank(root,rank)
            result=verify_native_checkpoints(root,2,require_receipts=True)
            self.assertEqual(result['status'],'VERIFIED_NATIVE_CHECKSUM');self.assertEqual(result['a'],.2)
            self.assertEqual([row['rank'] for row in result['files']],[0,1])
    def test_modified_and_truncated_payload_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.rank(root,0,ranks=1)
            data=path.read_bytes();path.write_bytes(bytes([data[0]^1])+data[1:])
            with self.assertRaisesRegex(ValueError,'checksum'):verify_native_checkpoints(root,1)
            path.write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError,'size'):verify_native_checkpoints(root,1)
    def test_mixed_generation_and_epoch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.rank(root,0);self.rank(root,1,generation=124)
            with self.assertRaisesRegex(ValueError,'generations'):verify_native_checkpoints(root,2)
            self.rank(root,1,a=.3)
            with self.assertRaisesRegex(ValueError,'epochs'):verify_native_checkpoints(root,2)
    def test_missing_receipt_and_legacy_scope_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.rank(root,0);path=self.rank(root,1)
            Path(str(path)+'.checksum.json').unlink()
            with self.assertRaisesRegex(ValueError,'Incomplete'):verify_native_checkpoints(root,2)
            Path(str(root/'restart.0')+'.checksum.json').unlink()
            self.assertEqual(verify_native_checkpoints(root,2)['status'],'LEGACY_UNVERIFIED')
            with self.assertRaisesRegex(ValueError,'missing'):verify_native_checkpoints(root,2,require_receipts=True)
    def test_rank_filename_alias_is_rejected_even_with_canonical_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.rank(root,0,ranks=1)
            (root/'restart.00').write_bytes((root/'restart.0').read_bytes())
            with self.assertRaisesRegex(ValueError,'Noncanonical'):verify_native_checkpoints(root,1)
    def test_malformed_rank_time_generation_and_inventory_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.rank(root,0,ranks=1);receipt=Path(str(path)+'.checksum.json')
            original=json.loads(receipt.read_text())
            for key,value in [('rank',True),('ranks',2),('a',float('nan')),('a',True),('generation',True),('size_bytes',True),('schema','wrong')]:
                altered=dict(original);altered[key]=value;receipt.write_text(json.dumps(altered))
                with self.subTest(key=key,value=value):
                    with self.assertRaises(ValueError):verify_native_checkpoints(root,1)
            with self.assertRaises(ValueError):verify_native_checkpoints(root,2)
            receipt.write_text('{')
            with self.assertRaises(json.JSONDecodeError):verify_native_checkpoints(root,1)

# ========================================================================
# test_identity_graph
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from halos import IdentityGraph,IdentityQueries,verify_identity_graph
test_identity_graph_ROOT=Path(__file__).resolve().parent


def graph():
    g=IdentityGraph(maximum_nodes=100,maximum_edges=100)
    g.add_epoch(9,epoch_contract(.5));g.add_epoch(2,epoch_contract(1.))
    return g


class IdentityGraphTests(unittest.TestCase):
    def test_birth_mass_return_bh_merger_stripping_and_reordered_epochs(self):
        g=graph();gas=g.add_node('gas',2**64-1,9,kind='gas');star=g.add_node('stars',2**63+1,9,kind='ssp')
        sink=g.add_node('gas',5,2,kind='gas');bh=g.add_node('bh',2**63+2,9,kind='bh');merged=g.add_node('bh',8,2,kind='bh')
        stream=g.add_node('stream',3,2,kind='stream')
        for a,b,kind in [(gas,star,'birth_parent'),(star,sink,'mass_return'),(bh,merged,'bh_merger'),(star,stream,'stripping')]:
            g.add_edge(a,b,kind,evidence={'origin':'native_ancestry','source':'lifecycle control'})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'graph';g.publish(out,source_hashes={str(Path(__file__).resolve()):file_sha256(__file__)})
            q=IdentityQueries(out);self.assertEqual(q.birth_parent(star)['parents'],[gas]);self.assertEqual(q.descendants(bh),[merged])
            self.assertEqual(q.descendants(star),sorted([sink,stream]));self.assertEqual(q.birth_parent(merged)['status'],'UNAVAILABLE')
            with np.load(out/'nodes.npz') as data:self.assertIn(np.uint64(2**64-1),data['object_ids'])

    def test_bound_owner_ambiguity_and_membership_reversible(self):
        g=graph();a=g.add_node('subfind',0,9,kind='subhalo');b=g.add_node('rockstar',0,9,kind='subhalo')
        evidence={'origin':'native_membership'}
        g.add_membership(a,'dm',[2**64-1,2**63+1],bound=True,evidence=evidence)
        g.add_membership(b,'dm',[2**64-1],bound=True,evidence=evidence)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'graph';g.publish(out,source_hashes={str(Path(__file__).resolve()):file_sha256(__file__)})
            q=IdentityQueries(out)
            self.assertEqual(q.members_at_epoch(a,particle_namespace='dm').tolist(),[2**63+1,2**64-1])
            self.assertEqual(q.bound_owner(('dm',2**64-1,9))['status'],'AMBIGUOUS')
            self.assertEqual(q.bound_owner(('dm',2**63+1,9))['owners'],[a])
            self.assertEqual(q.members_at_epoch(a,particle_namespace='gas').size,0)

    def test_no_float_bool_overflow_duplicate_or_cross_epoch_joins(self):
        g=graph()
        for value in (1.,True,-1,2**64):
            with self.assertRaises(ValueError):g.add_node('dm',value,9,kind='particle')
        a=g.add_node('dm',1,9,kind='particle');b=g.add_node('halos',1,2,kind='halo')
        with self.assertRaisesRegex(ValueError,'Duplicate'):g.add_node('dm',1,9,kind='particle')
        with self.assertRaisesRegex(ValueError,'same saved epoch'):g.add_edge(a,b,'bound_owner',evidence={'origin':'native'})
        with self.assertRaisesRegex(ValueError,'native evidence'):g.add_edge(a,b,'descendant',evidence={'origin':'row_match'})
        with self.assertRaisesRegex(ValueError,'physical time'):g.add_edge(b,a,'descendant',evidence={'origin':'native_ancestry'})

    def test_cycle_missing_and_parent_zoom_evidence(self):
        g=graph();a=g.add_node('gas',1,9,kind='gas');b=g.add_node('star',1,9,kind='ssp')
        g.add_edge(a,b,'birth_parent',evidence={'origin':'native_ancestry'});g.add_edge(b,a,'mass_return',evidence={'origin':'native_ancestry'})
        with self.assertRaisesRegex(ValueError,'cycle'):g.validate()
        g=graph();a=g.add_node('parent',1,9,kind='halo');b=g.add_node('zoom',9,2,kind='halo')
        with self.assertRaisesRegex(ValueError,'Lagrangian'):g.add_edge(a,b,'parent_zoom',evidence={'origin':'same_row'})
        g.add_edge(a,b,'parent_zoom',evidence={'origin':'lagrangian_identity','shared_particle_ids':[7,8]})
        missing=g.add_node('star',2,2,kind='ssp',status='MISSING')
        with self.assertRaisesRegex(ValueError,'Missing'):g.add_edge(a,missing,'matching_confidence',evidence={'origin':'derived'})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'graph';g.publish(out,source_hashes={str(Path(__file__).resolve()):file_sha256(__file__)})
            self.assertEqual(IdentityQueries(out).parent_zoom_match(b)['status'],'MATCHED')

    def test_public_cli_reuse_change_and_index_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);req=root/'request.json';out=root/'graph'
            data={'limits':{'maximum_nodes':10,'maximum_edges':10},'epochs':[{'storage_index':0,'epoch_contract':epoch_contract(.5)}],
                'nodes':[{'namespace':'dm','id':2**64-1,'epoch':0,'kind':'particle'}],'edges':[],
                'source_hashes':{str(Path(__file__).resolve()):file_sha256(__file__)}}
            req.write_text(json.dumps(data));cmd=[sys.executable,str(test_identity_graph_ROOT/'commands.py'),'build-identity-graph', str(req),str(out)]
            subprocess.run(cmd,check=True,capture_output=True);subprocess.run(cmd+['--reuse-completed'],check=True,capture_output=True)
            original=(out/'index.sqlite').read_bytes();(out/'index.sqlite').write_bytes(original+b'bad')
            with self.assertRaisesRegex(ValueError,'index changed'):verify_identity_graph(out)

    def test_node_and_edge_limits(self):
        g=IdentityGraph(maximum_nodes=1,maximum_edges=1);g.add_epoch(0,epoch_contract(.5));g.add_node('dm',1,0,kind='particle')
        with self.assertRaises(MemoryError):g.add_node('dm',2,0,kind='particle')

# ========================================================================
# test_event_cadence
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from evolution import (quantize_native_output_time, merge_and_quantize_native_output_times,
    validate_user_schedule, event_window, saved_outputs_resolve, choose_restart, history_cadence_support)
from evolution import snapshot_redshifts


def clock():
    return {'time_begin':.5,'time_max':1.,'maximum_timestep':.1,'timebins':29,'comoving':True}


class EventCadenceTests(unittest.TestCase):
    def test_native_power_two_clock_rounding(self):
        c=clock();r=quantize_native_output_time(.7,c)
        # GADGET max dt=.1 maps to 2^26 ticks; .7 lands at five
        # synchronized subdivisions of a full log(2) simulation interval.
        self.assertEqual(r['native_integer_tick'],4*(1<<26))
        self.assertAlmostEqual(r['planned_quantized_time'],.5*2**(4/8))
        self.assertIsNone(r['actual_output_time'])

    def test_exact_synchronized_clock_retains_legal_tick_precision(self):
        for bins in (29, 60):
            c={**clock(), 'timebins':bins, 'output_mode':'exact_synchronized'}
            for requested in (.5, .7, 1.):
                r=quantize_native_output_time(requested,c)
                self.assertEqual(r['native_integer_tick'] % 2, 0)
                interval=__import__('math').log(2)/(1 << bins)
                self.assertLessEqual(abs(r['planned_quantized_time']-requested),
                                     2*interval+1e-15)
                self.assertIsNone(r['actual_output_time'])
            self.assertEqual(quantize_native_output_time(1.,c)['native_integer_tick'],1 << bins)
        c={'time_begin':0.,'time_max':1.,'maximum_timestep':.125,
           'timebins':29,'comoving':False,'output_mode':'exact_synchronized'}
        self.assertLess(abs(quantize_native_output_time(.19,c)['planned_quantized_time']-.19),4e-9)
        with self.assertRaises(ValueError):
            quantize_native_output_time(.7,{**clock(),'output_mode':'unknown'})

    def test_isolated_clock_zero_is_not_a_redshift(self):
        c={'time_begin':0.,'time_max':1.,'maximum_timestep':.125,'timebins':29,'comoving':False}
        self.assertEqual(quantize_native_output_time(.19,c)['planned_quantized_time'],.25)
        self.assertEqual(quantize_native_output_time(0.,c)['native_integer_tick'],0)

    def test_repeated_windows_and_collisions_are_retained(self):
        w={'event_id':'pericenter-1','start':.6,'end':.8,'maximum_interval':.02}
        a=merge_and_quantize_native_output_times([.5,1.],[w],clock())
        b=merge_and_quantize_native_output_times([.5,1.],[w,w],clock())
        self.assertEqual(a,b)
        self.assertTrue(a['collisions'])
        self.assertEqual(a['requested_times'][0],.5)
        self.assertEqual(a['requested_times'][-1],1.)

    def test_encounter_between_coarse_outputs_remains_candidate(self):
        h={'object_id':'exact-uint64:1152921504606846977','times_myr':[0.,100.],
            'relative_positions_proper_kpc':[[-10.,0.,0.],[10.,0.,0.]],'valid':[True,True],
            'encounter_radius_proper_kpc':1.,'transitions':[]}
        e=predict_events([h])[0]
        self.assertEqual(e['kind'],'encounter')
        self.assertEqual(e['predicted_time_myr'],50.)
        self.assertEqual(e['interpolation_uncertainty_myr'],50.)
        w=event_window(e,1.)
        self.assertEqual(saved_outputs_resolve(w,[0.,100.])['status'],'UNRESOLVED_CADENCE')
        self.assertEqual(saved_outputs_resolve(w,list(np.arange(0,101,2)))['status'],'RESOLVED_CADENCE')
        self.assertEqual(e['status'],'CANDIDATE')

    def test_invalid_missing_trajectories_do_not_predict(self):
        h={'object_id':'host:7','times_myr':[0.,1.], 'relative_positions_proper_kpc':[[-1,0,0],[np.nan]*3],
           'valid':[True,False],'encounter_radius_proper_kpc':1.,'transitions':[]}
        self.assertEqual(predict_events([h]),[])
        h['transitions']=[{'kind':'stripping','left_index':0,'right_index':1}]
        with self.assertRaisesRegex(ValueError,'missing'):predict_events([h])

    def test_all_declared_event_types_preserve_brackets(self):
        kinds=['stripping','major_merger','bh_pair','bh_merger','bh_recoil','starburst']
        h={'object_id':'host:7','times_myr':[0.,1.], 'relative_positions_proper_kpc':[[1,0,0],[1,0,0]],
           'valid':[True,True],'encounter_radius_proper_kpc':1.,'transitions':[{'kind':k,'left_index':0,'right_index':1} for k in kinds]}
        events=predict_events([h]);self.assertEqual({e['kind'] for e in events},set(kinds))
        self.assertTrue(all(e['input_bracket_myr']==[0.,1.] for e in events))

    def test_restart_unavailable_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            r=choose_restart([d],1,.6)
            self.assertEqual(r['status'],'NO_AVAILABLE_RESTART')
            self.assertTrue(r['failures'])

    def test_actual_retained_interior_restart(self):
        d=native_control_fixture('periodic-restart/interrupted-restart')
        if not d.exists():self.skipTest('Executed native restart fixture unavailable')
        r=choose_restart([d],1,.5003)
        self.assertEqual(r['status'],'AVAILABLE')
        self.assertEqual(r['checkpoint']['clock_representation'],'scale_factor')
        self.assertEqual(choose_restart([d],1,.5)['status'],'NO_AVAILABLE_RESTART')

    def test_public_native_configuration_merges_windows(self):
        cfg={'simulation':{'start_redshift':1.,'end_redshift':0.,'snapshot_redshift_list':[1.,0.],
            'max_size_timestep_ln_a':.01,'event_cadence':{'maximum_outputs':100,'windows':[
                {'event_id':'encounter','start':.6,'end':.7,'maximum_interval':.02}]}}}
        values=snapshot_redshifts(cfg)
        self.assertGreater(len(values),2)
        self.assertEqual(values[-1],0.)

    def test_explicit_epochs_and_output_limits(self):
        self.assertEqual(validate_user_schedule({'z_start':10.,'z_end':0.,'backbone_redshifts':[5.,0.]}),[1/6,1.])
        with self.assertRaises(ValueError):validate_user_schedule({'z_start':10.,'z_end':0.,'backbone_redshifts':[0.,5.]})
        with self.assertRaisesRegex(ValueError,'budget'):
            merge_and_quantize_native_output_times([.5,1.],[{'event_id':'x','start':.6,'end':.7,'maximum_interval':1e-5}],clock(),maximum_outputs=20)
        c=clock();c['maximum_timestep']=1e-20
        with self.assertRaises(ValueError):quantize_native_output_time(.7,c)

    def test_history_actual_times_and_clock_bounds(self):
        epochs=[{'snapshot':0,'scale_factor':.5,'epoch_contract':{'cosmic_time_gyr':2.,
                  'cosmic_time_numerical_error_gyr':{'status':'MODEL_CONDITIONAL_BOUND','absolute_bound_gyr':.0001}}},
                {'snapshot':3,'scale_factor':.6,'epoch_contract':{'cosmic_time_gyr':3.,
                  'cosmic_time_numerical_error_gyr':{'status':'UNBOUNDED'}}}]
        r=history_cadence_support(epochs)
        self.assertEqual(r['intervals_myr'],[1000.])
        self.assertEqual(r['outputs'][0]['clock_error_myr'],.1)
        self.assertIsNone(r['outputs'][1]['clock_error_myr'])
        self.assertEqual(r['event_resolution'],'NOT_MEASURED')

    def test_existing_stellar_history_producer_feeds_event_candidates(self):
        from verification import StellarHistoryTests
        from verification import clock_fixture
        from stars import trace_stellar_candidates
        from evolution import publish_history_candidates
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);clock_fixture(root)
            catalogue,_=StellarHistoryTests().fixture(root)
            history=trace_stellar_candidates(root,catalogue=catalogue,snapshots=[0,1],candidate_ids=[1])
            out=root/'events.json'
            report=publish_history_candidates(root,out,history_path=history['catalogue'],population='stellar_candidate',
                selected_object_ids=[1],host_centers_comoving_kpc={'0':[0.,10.,10.],'1':[0.,10.,10.]},
                periodic_box_comoving_kpc=200.,encounter_radius_proper_kpc=1.,transitions_by_object={},memory_gib=.1)
            self.assertEqual(len(report['histories']),1)
            self.assertEqual(report['histories'][0]['valid'],[True,True])
            self.assertEqual(report['events'][0]['kind'],'encounter')
            self.assertEqual(report['events'][0]['status'],'CANDIDATE')
            self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')
            with self.assertRaises(FileExistsError):
                publish_history_candidates(root,out,history_path=history['catalogue'],population='stellar_candidate',
                    selected_object_ids=[1],host_centers_comoving_kpc={'0':[0.,10.,10.],'1':[0.,10.,10.]},
                    periodic_box_comoving_kpc=200.,encounter_radius_proper_kpc=1.,transitions_by_object={},memory_gib=.1)

# ========================================================================
# test_external_hdf5_contract
test_external_hdf5_contract_ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(test_external_hdf5_contract_ROOT/'core'))
from core.data import external_hdf5_contract,verify_external_hdf5_contract


class ExternalHdf5ContractTests(unittest.TestCase):
    def fixture(self,root):
        path=root/'native.hdf5'
        with h5py.File(path,'w') as source:
            source.create_dataset('PartType1/ParticleIDs',data=np.array([2**63+1,2**64-1],dtype=np.uint64),fletcher32=True)
            source.create_dataset('PartType1/Coordinates',data=np.array([[1.,2.,3.],[4.,5.,6.]]),fletcher32=True)
        specs={'PartType1/ParticleIDs':{'units':'identity','identity_namespace':'native_ic:fixture',
            'quantity':'native particle ID','ancestry':{'relation':'generated test namespace'}},
            'PartType1/Coordinates':{'units':'comoving kpc/h','quantity':'native particle position'}}
        return path,specs

    def test_readonly_uint64_semantics_and_exact_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,specs=self.fixture(root);before=digest(path)
            contract=external_hdf5_contract(path,specs,epoch=epoch_contract(.5))
            self.assertEqual(digest(path),before)
            self.assertEqual(verify_external_hdf5_contract(path,contract),contract)
            self.assertEqual(contract['datasets']['PartType1/ParticleIDs']['dtype'],'uint64')
            self.assertEqual(contract['datasets']['PartType1/Coordinates']['units'],'comoving kpc/h')
            with h5py.File(path) as source:
                self.assertNotIn('data_contract_schema',source['PartType1/Coordinates'].attrs)

    def test_missing_declaration_and_signed_identity_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,specs=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'every external'):
                external_hdf5_contract(path,{'PartType1/Coordinates':specs['PartType1/Coordinates']})
            with h5py.File(path,'r+') as source:
                del source['PartType1/ParticleIDs'];source['PartType1/ParticleIDs']=np.array([1,2],dtype=np.int64)
            with self.assertRaisesRegex(ValueError,'uint64'):external_hdf5_contract(path,specs)

    def test_valid_hdf5_value_rewrite_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,specs=self.fixture(root)
            contract=external_hdf5_contract(path,specs)
            with h5py.File(path,'r+') as source:source['PartType1/Coordinates'][0,0]=9.
            with self.assertRaisesRegex(ValueError,'checksum'):verify_external_hdf5_contract(path,contract)

    def test_bad_payload_digest_rejected_even_with_matching_file_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,specs=self.fixture(root)
            contract=external_hdf5_contract(path,specs)
            contract['datasets']['PartType1/Coordinates']['payload_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'numeric payload'):verify_external_hdf5_contract(path,contract)

# ========================================================================
# test_cosmology_product_publication
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.cosmology import write_class_cosmology_products
from core.data import verify_hdf5_product,recover_hdf5_product

class CosmologyProductTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'pk.dat';source.write_text('# redshift z=1\n0.01 2\n0.1 3\n1 4\n')
        return source,root/'cosmology.h5'
    def write(self,root):
        source,target=self.fixture(root)
        write_class_cosmology_products('lcdm',1.,None,None,source,target)
        return source,target
    def test_canonical_actual_epoch_and_chunk_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source,target=self.write(root)
            verify_hdf5_product(target);verify_hdf5_dataset_payloads(target,require_contract=True)
            with h5py.File(target) as product:
                dataset=product['linear_power_spectrum/P_mpc_h3']
                self.assertTrue(dataset.fletcher32)
                self.assertEqual(json.loads(dataset.attrs['epoch_json'])['actual_scale_factor'],.5)
                self.assertIn('payload_sha256',dataset.attrs)
    def test_corrupt_numeric_payload_and_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source,target=self.write(root);record=verify_hdf5_product(target)
            with h5py.File(target,'r+') as product:product['linear_power_spectrum/P_mpc_h3'][0]=99
            with self.assertRaises(ValueError):verify_hdf5_product(target)
            self.assertEqual(recover_hdf5_product(target,record['generation']),record)
    def test_publication_failure_retains_scientific_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source,target=self.fixture(root)
            with patch('core.data.publish_hdf5_product',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):write_class_cosmology_products('lcdm',1.,None,None,source,target)
            self.assertFalse(target.exists())
            self.assertTrue(list(root.glob('.cosmology-product-*/cosmology.h5')))
            self.assertTrue(list(root.glob('.cosmology-product-*/failure.json')))
    def test_interrupted_initial_publication_is_pending_until_recovery(self):
        from core.foundation import atomic_copy
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source,target=self.fixture(root)
            def interrupt(original,destination):
                result=atomic_copy(original,destination)
                if Path(destination)==target:raise KeyboardInterrupt()
                return result
            with patch('core.data.atomic_copy',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):write_class_cosmology_products('lcdm',1.,None,None,source,target)
            with self.assertRaisesRegex(ValueError,'pending'):verify_hdf5_product(target)
            index=json.loads(Path(str(target)+'.publication.json').read_text())
            recover_hdf5_product(target,index['generation'])
            verify_hdf5_product(target)

    def test_second_export_keeps_completed_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();source,target=self.write(root);first=verify_hdf5_product(target)
            source.write_text('# redshift z=1\n0.01 5\n0.1 6\n1 7\n')
            write_class_cosmology_products('lcdm',1.,None,None,source,target)
            second=verify_hdf5_product(target)
            self.assertNotEqual(first['generation'],second['generation'])
            self.assertTrue((root/first['generation']/target.name).is_file())

# ========================================================================
# test_hdf5_contract_integrity

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.data import (seal_hdf5_numeric_datasets,hdf5_numeric_payload_sha256)


class Hdf5ContractIntegrityTests(unittest.TestCase):
    def test_signed_or_floating_identity_cannot_receive_a_canonical_namespace(self):
        with tempfile.TemporaryDirectory() as temporary,h5py.File(Path(temporary)/'ids.h5','w') as source:
            for kind in ('int64','float64'):
                dataset=source.create_dataset(kind,data=np.array([1,2],dtype=kind))
                with self.assertRaisesRegex(ValueError,'uint64'):
                    write_dataset_contract(dataset,units='identity',identity_namespace='fixture:dm')
                self.assertNotIn('data_contract_schema',dataset.attrs)

    def fixture(self,path):
        with h5py.File(path,'w') as h5:
            group=h5.create_group('snap_000/pk/data')
            for key,values in [('k_h_Mpc',[.1,.2,.3]),('P_Mpc_h3',[2.,0.,-3.])]:
                group.create_dataset(key,data=values)
            group['P_Mpc_h3'].attrs['retained_attribute']='existing producer attribute'
            attach_analysis_dataset_units(group,'pk')

    def test_real_contiguous_output_migrates_without_changing_values_or_attributes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data.h5';self.fixture(path)
            with h5py.File(path,'r') as source:
                data=source['snap_000/pk/data/P_Mpc_h3']
                self.assertTrue(data.fletcher32)
                self.assertEqual(data.attrs['retained_attribute'],'existing producer attribute')
                np.testing.assert_array_equal(data[:],[2.,0.,-3.])
            receipt=verify_hdf5_dataset_payloads(path,require_contract=True)
            self.assertEqual(len(receipt['checked']),2)
            self.assertEqual(receipt['legacy_unsealed'],[])

    def test_valid_hdf5_rewrite_cannot_silently_change_sealed_scientific_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data.h5';self.fixture(path)
            with h5py.File(path,'r+') as source:source['snap_000/pk/data/P_Mpc_h3'][0]=99.
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):verify_hdf5_dataset_payloads(path)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):plot_saved_analysis_curves(path,Path(temporary)/'figures')

    def test_scalar_uint64_and_empty_payloads_have_verifiable_digests(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'ids.h5'
            with h5py.File(path,'w') as source:
                for key,values in [('id',np.uint64(2**64-1)),('empty',np.empty(0,dtype=np.uint64))]:
                    data=source.create_dataset(key,data=values)
                    write_dataset_contract(data,units='identity',identity_namespace='native:dm')
            self.assertEqual(len(verify_hdf5_dataset_payloads(path,require_contract=True)['checked']),2)
            with h5py.File(path,'r+') as source:source['id'][()]=np.uint64(2**64-2)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):verify_hdf5_dataset_payloads(path)

    def test_oversized_row_migration_streams_with_bounded_workspace(self):
        with tempfile.TemporaryDirectory() as temporary,h5py.File(Path(temporary)/'data.h5','w') as source:
            data=source.create_dataset('large',shape=(2,1024),dtype='float64')
            data[0,500]=3.5
            before=hdf5_numeric_payload_sha256(data,memory_bytes=64)
            seal_hdf5_numeric_datasets(source,memory_bytes=64)
            self.assertTrue(source['large'].fletcher32)
            self.assertEqual(hdf5_numeric_payload_sha256(source['large'],memory_bytes=64),before)
            self.assertEqual(source['large'][0,500],3.5)

# ========================================================================
# test_zoom_generation

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from zoom import generate_zoom, verify_generated_zoom


class ZoomGenerationTests(unittest.TestCase):
    def setUp(self):
        planner = patch('zoom.plan_native_music', return_value={'status': 'PASS'})
        self.planner = planner.start()
        self.addCleanup(planner.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cfg = self.root / "source.conf"
        self.cfg.write_text("""[setup]
levelmin=5
levelmax=7
baryons=yes
use_2LPT=no
[cosmology]
transfer=DMS_file
input_model=lcdm
transfer_file=source.txt
Omega_m=0.3
H0=70
[output]
format=arepo
filename=old.hdf5
arepo_num_files=1
arepo_coarsetype=2
arepo_doubleprec=yes
arepo_longids=yes
""")
        (self.root / "source.txt").write_text("Synthetic fixture input")
        self.binary = self.root / "MUSIC"
        self.binary.write_text("Placeholder: never executed in unit tests")
        self.destination = self.root / "new"
        self.proof = self.root / "native-transfer-suite.json"
        rejections = (
            "reject-second-order",
            "reject-backscaling",
            "reject-gauge",
            "reject-model",
            "reject-redshift",
            "reject-mode-coverage",
            "reject-poisson",
        )
        self.proof.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "nested": {"status": "PASS"},
                    "periodic": {"status": "PASS"},
                    "regression_driver_sha256": file_sha256(
                        Path(__file__).parent / 'core/research/generator_controls.py'
                    ),
                    "execution": {
                        "binary_sha256": file_sha256(self.binary),
                        "rejections": {key: {"exit_code": 1} for key in rejections},
                    },
                }
            )
        )

    def test_native_failure_retains_partial_files_and_failure_ledger(self):
        def failed_run(command, *, cwd, **kwargs):
            self.assertEqual(command, [str(self.binary.resolve()), "music.conf"])
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "2")
            staged = (cwd / "music.conf").read_text()
            self.assertIn("Omega_m = 0.3", staged)
            self.assertIn("H0 = 70", staged)
            self.assertIn("use_2LPT = no", staged)
            (cwd / "ics_gadget.hdf5").write_text("partial invalid native output")
            return subprocess.CompletedProcess(command, 1)

        with (
            patch('zoom.run_native_command', side_effect=failed_run),
            patch('zoom.audit_music_ic') as audit,
        ):
            with self.assertRaisesRegex(RuntimeError, "partial files are not accepted"):
                generate_zoom(
                    self.cfg,
                    self.binary,
                    self.destination,
                    native_validation=self.proof,
                )
            audit.assert_not_called()
        report = json.loads((self.destination / "generation.json").read_text())
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["science_acceptance"], "NOT_ACCEPTED")
        self.assertEqual(report["native_exit_code"], 1)
        self.assertTrue((self.destination / "ics_gadget.hdf5").is_file())
        self.assertNotIn("structural_audit", report)

    def test_native_budget_failure_never_launches_generator(self):
        self.planner.side_effect = MemoryError('native estimate exceeds memory budget')
        with patch('zoom.run_native_command') as native:
            with self.assertRaisesRegex(MemoryError, 'exceeds memory budget'):
                generate_zoom(self.cfg, self.binary, self.destination, native_validation=self.proof)
            native.assert_not_called()
        report = json.loads((self.destination/'generation.json').read_text())
        self.assertEqual(report['status'], 'FAILED')
        self.assertNotIn('native_exit_code', report)

    def test_native_timeout_retains_failure_and_elapsed_measurement(self):
        with patch('zoom.run_native_command', side_effect=subprocess.TimeoutExpired('MUSIC',36)) as native:
            with self.assertRaises(subprocess.TimeoutExpired):
                generate_zoom(self.cfg,self.binary,self.destination,native_validation=self.proof,native_wall_hours=.01)
            self.assertEqual(native.call_args.kwargs['timeout'],36.)
        report=json.loads((self.destination/'generation.json').read_text())
        self.assertEqual(report['status'],'FAILED')
        self.assertGreaterEqual(report['native_execution_wall_seconds'],0.)
        self.assertNotIn('native_exit_code',report)

    def test_native_success_cannot_bypass_structural_audit(self):
        with (
            patch(
                'zoom.run_native_command',
                return_value=subprocess.CompletedProcess([], 0),
            ),
            patch(
                'zoom.audit_music_ic',
                side_effect=ValueError("mass closure failed"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "mass closure failed"):
                generate_zoom(
                    self.cfg,
                    self.binary,
                    self.destination,
                    native_validation=self.proof,
                )
        report = json.loads((self.destination / "generation.json").read_text())
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["native_exit_code"], 0)
        self.assertNotIn("artifacts", report)

    def test_changed_binary_cannot_reuse_native_validation(self):
        self.binary.write_text("changed binary")
        with self.assertRaisesRegex(ValueError, "current native binary"):
            generate_zoom(
                self.cfg, self.binary, self.destination, native_validation=self.proof
            )
        self.assertFalse(self.destination.exists())

    def test_bad_species_and_existing_products_rejected_before_execution(self):
        self.cfg.write_text(
            self.cfg.read_text().replace("arepo_coarsetype=2", "arepo_coarsetype=5")
        )
        with patch('zoom.run_native_command') as native:
            with self.assertRaisesRegex(ValueError, "reserved"):
                generate_zoom(
                    self.cfg,
                    self.binary,
                    self.destination,
                    native_validation=self.proof,
                )
            native.assert_not_called()
        self.assertFalse(self.destination.exists())
        self.destination.mkdir()
        preserved = self.destination / "science-data"
        preserved.write_text("keep")
        with self.assertRaises(FileExistsError):
            generate_zoom(
                self.cfg, self.binary, self.destination, native_validation=self.proof
            )
        self.assertEqual(preserved.read_text(), "keep")

    def test_failed_velocity_never_returns_a_generated_handoff(self):
        with (patch('zoom.run_native_command', return_value=subprocess.CompletedProcess([], 0)),
              patch('zoom.audit_music_ic', return_value={'status':'STRUCTURAL_PASS'}),
              patch('zoom.audit_zoom_velocity', return_value={'status':'FAIL'})):
            with self.assertRaisesRegex(ValueError, 'velocity audit failed'):
                generate_zoom(self.cfg, self.binary, self.destination, native_validation=self.proof)
        report = json.loads((self.destination / 'generation.json').read_text())
        self.assertEqual(report['status'], 'FAILED')
        self.assertEqual(report['velocity_audit']['status'], 'FAIL')
        self.assertNotIn('artifacts', report)

    def test_failed_parent_phase_retains_both_audits(self):
        parent, noise = self.root/'parent.hdf5', self.root/'wnoise.bin'
        parent.write_bytes(b'fixture')
        noise.write_bytes(b'fixture-noise')
        with (patch('zoom.run_native_command', return_value=subprocess.CompletedProcess([], 0)),
              patch('zoom.audit_music_ic', return_value={'status':'STRUCTURAL_PASS'}),
              patch('zoom.audit_zoom_velocity', return_value={'status':'PASS'}),
              patch('zoom.compare_density_phases', return_value={'status':'FAIL'}) as phase):
            with self.assertRaisesRegex(ValueError, 'parent phases failed'):
                generate_zoom(self.cfg, self.binary, self.destination, native_validation=self.proof,
                              parent_ic=parent, parent_noise=noise)
            self.assertEqual(phase.call_args.args[:4], (parent.resolve(), self.destination.resolve()/'ics_gadget.hdf5',
                                                       noise.resolve(), self.destination.resolve()/'wnoise_0005.bin'))
        report = json.loads((self.destination / 'generation.json').read_text())
        self.assertEqual(report['velocity_audit']['status'], 'PASS')
        self.assertEqual(report['phase_audit']['status'], 'FAIL')
        self.assertEqual(report['science_acceptance'], 'NOT_ACCEPTED')

    def test_parent_files_required_together_before_native_launch(self):
        with patch('zoom.run_native_command') as native:
            with self.assertRaisesRegex(ValueError, 'both parent'):
                generate_zoom(self.cfg, self.binary, self.destination, native_validation=self.proof,
                              parent_ic=self.root/'missing')
            native.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_automatic_zoom_template_cannot_invent_a_target(self):
        self.cfg.write_text(self.cfg.read_text()+'\n[darkmattersim]\nrequires_traced_region=yes\n')
        with patch('zoom.run_native_command') as native:
            with self.assertRaisesRegex(ValueError,'actual traced'):
                generate_zoom(self.cfg,self.binary,self.destination,native_validation=self.proof)
            native.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_traced_coverage_failure_blocks_api_handoff_and_is_retained(self):
        region=self.root/'region.json'
        region.write_text('{}')
        def native_run(command,*,cwd,**kwargs):
            (cwd/'ics_gadget.hdf5').write_bytes(b'fixture')
            return subprocess.CompletedProcess(command,0)
        def coverage_check(structural,region,destination,**kwargs):
            self.assertEqual(json.loads(structural.read_text())['status'],'GENERATED_STRUCTURAL_PASS')
            result={'status':'FAIL','coarse_intersections':1}
            destination.write_text(json.dumps(result))
            return result
        with (patch('zoom.apply_traced_region',return_value={'path':str(region)}),
              patch('zoom.run_native_command',side_effect=native_run),
              patch('zoom.audit_music_ic',return_value={'status':'STRUCTURAL_PASS'}),
              patch('zoom.audit_zoom_velocity',return_value={'status':'PASS'}),
              patch('zoom.audit_target_coverage',side_effect=coverage_check) as coverage):
            with self.assertRaisesRegex(ValueError,'intersect traced target'):
                generate_zoom(self.cfg,self.binary,self.destination,native_validation=self.proof,traced_region=region)
            coverage.assert_called_once()
        report=json.loads((self.destination/'generation.json').read_text())
        self.assertEqual(report['status'],'FAILED')
        self.assertEqual(report['static_target_coverage']['status'],'FAIL')
        self.assertIn('target-coverage.json',report['artifacts'])
        self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')


class GeneratedZoomReaderTests(unittest.TestCase):
    def fixture(self, root):
        from core.foundation import atomic_json, atomic_text
        for name in ('music.conf','transfers.txt','ics_gadget.hdf5','generation.log'):
            (root/name).write_text('explicit mocked transport fixture')
        report={'status':'GENERATED_STRUCTURAL_PASS','publication_contract':'native-ic-generation-v1',
            'science_acceptance':'NOT_ACCEPTED','artifacts':{name:file_sha256(root/name)
                for name in ('music.conf','transfers.txt','ics_gadget.hdf5','generation.log')}}
        path=root/'generation.json'
        atomic_json(path,report);atomic_text(root/'generation.sha256',file_sha256(path)+'\n')
        return path,report

    def test_current_seal_and_exact_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,report=self.fixture(root)
            self.assertEqual(verify_generated_zoom(path),report)
            (root/'extra.conf').write_text('new recipe')
            with self.assertRaisesRegex(ValueError,'inventory changed'):verify_generated_zoom(path)

    def test_changed_artifact_and_missing_report_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,report=self.fixture(root)
            ic=root/'ics_gadget.hdf5';original=ic.read_bytes();ic.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):verify_generated_zoom(path)
            ic.write_bytes(original);(root/'generation.sha256').unlink()
            with self.assertRaisesRegex(ValueError,'seal'):verify_generated_zoom(path)

    def test_failure_status_never_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,report=self.fixture(root)
            report['status']='FAILED';path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'incomplete or failed'):verify_generated_zoom(path)

# ========================================================================
# test_atomic_native_copy

test_atomic_native_copy_CORE=Path(__file__).resolve().parent/'core'
sys.path.insert(0,str(test_atomic_native_copy_CORE))
import core.foundation as acceptance


class AtomicNativeCopyTests(unittest.TestCase):
    def test_real_executable_bytes_mode_and_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'compiled-control'; destination=Path(temp)/'Gadget4-copy-control'
            subprocess.run(['cc','-x','c','-','-o',str(source)],input=b'#include <stdio.h>\nint main(int argc,char **argv){if(argc!=2)return 1;puts(argv[1]);return 0;}\n',capture_output=True,check=True)
            checksum=acceptance.atomic_copy(source,destination)
            self.assertEqual(checksum,acceptance.file_sha256(source))
            self.assertEqual(acceptance.file_sha256(destination),checksum)
            self.assertEqual(destination.stat().st_mode & 0o777, source.stat().st_mode & 0o777)
            self.assertEqual(destination.stat().st_mtime_ns,source.stat().st_mtime_ns)
            result=subprocess.run([str(destination),'verified native executable copy'],capture_output=True,check=True)
            self.assertEqual(result.stdout,b'verified native executable copy\n')

    def test_disk_full_preserves_destination_and_cleans_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source'; destination=root/'destination'
            source.write_bytes(b'new complete recipe');destination.write_bytes(b'old reviewed recipe')
            with patch('core.foundation.os.fsync',side_effect=OSError(errno.ENOSPC,'full')):
                with self.assertRaises(OSError):acceptance.atomic_copy(source,destination)
            self.assertEqual(destination.read_bytes(),b'old reviewed recipe')
            self.assertEqual(set(root.iterdir()),{source,destination})

    def test_source_change_during_copy_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';destination=root/'destination'
            source.write_bytes(b'input');destination.write_bytes(b'old')
            copying=acceptance.shutil.copyfileobj
            def change(original,copied,**kwargs):
                copying(original,copied,**kwargs);source.write_bytes(b'mutated input')
            with patch('core.foundation.shutil.copyfileobj',side_effect=change):
                with self.assertRaisesRegex(ValueError,'source changed'):acceptance.atomic_copy(source,destination)
            self.assertEqual(destination.read_bytes(),b'old')
            self.assertEqual(set(root.iterdir()),{source,destination})

    def test_staged_copy_corruption_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';destination=root/'destination'
            source.write_bytes(b'input');destination.write_bytes(b'old')
            def wrong(original,copied,**kwargs):copied.write(b'wrong')
            with patch('core.foundation.shutil.copyfileobj',side_effect=wrong):
                with self.assertRaisesRegex(ValueError,'staged bytes differ'):acceptance.atomic_copy(source,destination)
            self.assertEqual(destination.read_bytes(),b'old')

    def test_same_file_missing_source_and_symlink_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';target=root/'target';destination=root/'destination'
            source.write_bytes(b'input');target.write_bytes(b'preserved external target')
            with self.assertRaises(ValueError):acceptance.atomic_copy(source,source)
            with self.assertRaises(FileNotFoundError):acceptance.atomic_copy(root/'missing',target)
            destination.symlink_to(target)
            acceptance.atomic_copy(source,destination)
            self.assertFalse(destination.is_symlink())
            self.assertEqual(target.read_bytes(),b'preserved external target')
            self.assertEqual(source.read_bytes(),destination.read_bytes())

    def test_sigkill_after_copy_flush_preserves_old_executable_and_replays(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'source';destination=root/'Gadget4'
            source.write_bytes(b'new complete executable');source.chmod(0o755);destination.write_bytes(b'old executable')
            script="""
import os,sys,signal
sys.path.insert(0,sys.argv[1])
import core.foundation as research_acceptance
real=os.fsync
def kill(fd):
    real(fd);os.kill(os.getpid(),signal.SIGKILL)
research_acceptance.os.fsync=kill
research_acceptance.atomic_copy(sys.argv[2],sys.argv[3])
"""
            result=subprocess.run([sys.executable,'-c',script,str(test_atomic_native_copy_CORE),str(source),str(destination)],capture_output=True,timeout=30)
            self.assertEqual(result.returncode,-9,result.stderr.decode())
            self.assertEqual(destination.read_bytes(),b'old executable')
            orphaned=set(root.iterdir())-{source,destination}
            self.assertEqual(len(orphaned),1)
            self.assertEqual(next(iter(orphaned)).read_bytes(),source.read_bytes())
            acceptance.atomic_copy(source,destination)
            self.assertEqual(destination.read_bytes(),source.read_bytes())
            self.assertTrue(all(p.exists() for p in orphaned))

# ========================================================================
# test_solver_route_contracts
import concurrent.futures
sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
import core.cosmology as cosmology


class SolverRouteTests(unittest.TestCase):
    def test_solver_routes_reject_conflicting_owner_before_native_work(self):
        routes = [cosmology.run_boltzmann, cosmology.run_modular_spectrum,
                  cosmology.run_linear_spectrum, cosmology.select_start_redshift]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with native_run_lock(root), concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                for function in routes:
                    args = ({}, root, 'lcdm') if function is cosmology.select_start_redshift else ({}, root, 'lcdm', 63.)
                    with self.assertRaisesRegex(RuntimeError, 'owned'):
                        pool.submit(function, *args).result(timeout=10)
            self.assertFalse((root / 'input').exists())

    def test_epoch_diagnostic_arrays_source_binding_and_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pk = root / 'spectrum.fixture'; pk.write_bytes(b'original spectrum bytes')
            trials = [{'z':63., 'rms_displacement_fraction':.1,
                       'source_pk':str(pk),'source_pk_sha256':digest(pk)}]
            cosmology._publish_start_epoch_selection(root,
                {'criterion':'synthetic displacement transport fixture','history':trials}, trials)
            pointer = root / 'input/start_redshift_validation.json'
            record = cosmology.verify_start_epoch_selection(pointer)
            audit = root / record['numeric_audit']['path']
            measured = verify_numeric_audit(audit)
            self.assertEqual(measured['numeric_data']['array_contract']['datasets']['scale_factor']['units'], 'dimensionless')
            with np.load(audit.with_name(audit.name + '.data') / 'arrays.npz', allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays['scale_factor'], [1/64])
            pk.write_bytes(b'changed spectrum')
            with self.assertRaisesRegex(ValueError, 'input spectrum changed'):
                cosmology.verify_start_epoch_selection(pointer)

    def test_epoch_diagnostic_missing_seal_is_not_reusable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pk = root / 'spectrum.fixture'; pk.write_bytes(b'original')
            trial = {'z':99.,'rms_displacement_fraction':.08,
                     'source_pk':str(pk),'source_pk_sha256':digest(pk)}
            cosmology._publish_start_epoch_selection(root, {'criterion':'transport fixture'}, [trial])
            pointer = root / 'input/start_redshift_validation.json'
            pointer.with_name(pointer.name + '.sha256').unlink()
            with self.assertRaisesRegex(ValueError, 'seal missing'):
                cosmology.verify_start_epoch_selection(pointer)

# ========================================================================
# test_native_numerical_controls
test_native_numerical_controls_ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(test_native_numerical_controls_ROOT/'core'))
from core.research.controls import analytic_field, analytic_orbits, stencil_fixture, stencil_tides, spline_direct_field, residual_summary, isolated_invariants, cosmological_energy_balance
from core.research.controls import validate_control_request, fixture as test_native_numerical_controls_fixture, run_control_campaign, verify_control_campaign, _force_test
from core.research.controls import control_config_text


def control_request():
    p=dict(kind='plummer',gm=1.,scale=1.,axis_y=1.3,axis_z=.7)
    profile=dict(mode='analytic',pm_grid=32,multipole_order=3,diagnostic_variant=False)
    tol=dict(force_relative=2e-8,tide_relative=2e-5,orbit_relative=5e-4,energy_relative=5e-5,angular_relative=5e-5,cosmological_balance_relative=.01,reference_floor=1e-7)
    case=dict(id='plummer',mode='analytic',potential=p,mpi_ranks=1,profile=profile,times=[0.,.05,.1,.2,.4],timestep=.002,softening=.01,opening_angle=.3,force_accuracy=.001,integration_accuracy=.01,stencil_step=.001,particles=28,seed=123,restart=False,tolerances=tol)
    return dict(schema_version=1,resources=dict(cpu_slots=4,build_jobs=4,memory_gib=3.,wall_seconds=180.,disk_margin_gib=.1),cases=[case],comparisons=[])


class AnalyticControlTests(unittest.TestCase):
    def test_potential_gradient_and_tidal_sign_independent_differences(self):
        x=np.array([[.7,.2,.3],[-.4,.8,.2]])
        for kind in ('plummer','hernquist','nfw','harmonic'):
            p={**control_request()['cases'][0]['potential'],'kind':kind};phi,a,t=analytic_field(x,p);step=1e-5
            for j in range(3):
                offset=np.eye(3)[j]*step
                np.testing.assert_allclose(-(analytic_field(x+offset,p)[0]-analytic_field(x-offset,p)[0])/(2*step),a[:,j],atol=2e-10,rtol=2e-8)
                np.testing.assert_allclose((analytic_field(x+offset,p)[1]-analytic_field(x-offset,p)[1])/(2*step),t[:,:,j],atol=2e-10,rtol=2e-8)
            np.testing.assert_allclose(t,np.swapaxes(t,-1,-2),atol=1e-15)
    def test_nfw_small_radius_stable_and_cusp_rejected(self):
        p={**control_request()['cases'][0]['potential'],'kind':'nfw'}
        _,a,_=analytic_field(np.array([[1e-7,0.,0.]]),p)
        self.assertAlmostEqual(a[0,0],-.5,places=6)
        with self.assertRaisesRegex(ValueError,'cusp'):analytic_field(np.zeros((1,3)),p)
    def test_harmonic_orbit_matches_closed_form_and_torque(self):
        p={**control_request()['cases'][0]['potential'],'kind':'harmonic'}
        x=np.array([[1.,.5,.3]]);v=np.array([[.2,-.4,.1]]);times=np.linspace(0,2,17);w=np.array([1,1.3,.7])
        xx,vv=analytic_orbits(x,v,times,p)
        truth=x[None]*np.cos(times[:,None,None]*w)+v[None]/w*np.sin(times[:,None,None]*w)
        np.testing.assert_allclose(xx,truth,atol=3e-12)
        energy,angular=isolated_invariants(xx,vv,p)
        np.testing.assert_allclose(energy,np.broadcast_to(energy[0],energy.shape),atol=1e-11)
        self.assertGreater(np.max(np.abs(angular-angular[0])),.1)
    def test_circular_hernquist_orbit_conserves_energy_and_angular(self):
        p={**control_request()['cases'][0]['potential'],'kind':'hernquist'}
        x=np.array([[1.,0.,0.]]);v=np.array([[0.,.5,0.]]);times=np.linspace(0,4,41)
        xx,vv=analytic_orbits(x,v,times,p);e,l=isolated_invariants(xx,vv,p)
        np.testing.assert_allclose(np.linalg.norm(xx,axis=-1),1.,atol=1e-10)
        np.testing.assert_allclose(e,np.broadcast_to(e[0],e.shape),atol=1e-11)
        np.testing.assert_allclose(l,np.broadcast_to(l[0],l.shape),atol=1e-11)
    def test_stencil_second_order_refinement(self):
        p=control_request()['cases'][0]['potential'];errors=[]
        for step in (.02,.01,.005):
            x,_=stencil_fixture(p,step);_,a,t=analytic_field(x,p)
            errors.append(np.linalg.norm(stencil_tides(a,step)-t[::7]))
        self.assertGreater(errors[0]/errors[1],3.9);self.assertGreater(errors[1]/errors[2],3.9)
    def test_spline_force_newtonian_and_momentum(self):
        x=np.array([[0.,0.,0.],[2.,0.,0.],[0.,3.,0.]]);m=np.array([1.,2.,3.])
        a=spline_direct_field(x,m,.1,block_size=1)
        np.testing.assert_allclose(a[0],[.5,1/3,0.],atol=1e-15)
        np.testing.assert_allclose(np.sum(m[:,None]*a,axis=0),0.,atol=1e-15)
    def test_spline_coincident_finite_and_join_continuity(self):
        x=np.array([[0.,0.,0.],[0.,0.,0.]])
        np.testing.assert_equal(spline_direct_field(x,np.ones(2),1.),0.)
        for r in (.5,1.):
            a=spline_direct_field(np.array([[0.,0.,0.],[r-1e-8,0.,0.]]),np.ones(2),1.)
            b=spline_direct_field(np.array([[0.,0.,0.],[r+1e-8,0.,0.]]),np.ones(2),1.)
            np.testing.assert_allclose(a,b,rtol=2e-7,atol=2e-7)
    def test_cosmological_free_streaming_balance(self):
        a=np.geomspace(.1,1,10001);k=1/a**2;w=np.zeros_like(a)
        residual=cosmological_energy_balance(a,k,w)
        self.assertLess(np.max(np.abs(residual))/k[0],2e-8)
        self.assertGreater(abs(k[-1]-k[0]),90)
    def test_zero_reference_dimensional_floor_and_worst_case(self):
        r=np.array([[0.,0.,0.],[.03,0.,.04]])
        m=residual_summary(r,np.zeros_like(r),.1)
        self.assertEqual(m['worst_index'],1);self.assertAlmostEqual(m['relative_max'],.5)
        with self.assertRaises(ValueError):residual_summary(r*np.nan,r,.1)
    def test_invalid_physics_inputs(self):
        p=control_request()['cases'][0]['potential']
        for value in (0.,-1.,float('nan'),True):
            with self.assertRaises(ValueError):analytic_field(np.ones((2,3)),{**p,'gm':value})


class NativeHarnessContractTests(unittest.TestCase):
    def test_closed_build_profiles_cannot_inject_options(self):
        p=control_request()['cases'][0]['profile'];s=control_config_text(p)
        self.assertIn('EXTERNALGRAVITY_CONTROL\n',s);self.assertNotIn('SELFGRAVITY\n',s);self.assertNotIn('PERIODIC\n',s)
        with self.assertRaises(ValueError):control_config_text({**p,'injected':'SIDM'})
    def test_declared_resources_and_tolerances_preflight(self):
        request=control_request();self.assertEqual(validate_control_request(request),request)
        for update in ({'mpi_ranks':5},{'particles':100},{'id':'../unsafe'},{'times':[0.,1.,1.]},{'timestep':True}):
            bad=copy.deepcopy(request);bad['cases'][0].update(update)
            with self.assertRaises(ValueError):validate_control_request(bad)
        bad=copy.deepcopy(request);bad['cases'][0]['tolerances'].pop('force_relative')
        with self.assertRaises(ValueError):validate_control_request(bad)
    def test_comparison_cannot_change_unrelated_physics(self):
        request=control_request();right=copy.deepcopy(request['cases'][0]);right.update(id='rank2',mpi_ranks=2);request['cases'].append(right)
        request['comparisons']=[dict(left='plummer',right='rank2',kind='rank',relative_tolerance=1e-7,absolute_floor=.1)]
        validate_control_request(request);right['potential']['gm']=2.
        with self.assertRaisesRegex(ValueError,'Unrelated'):validate_control_request(request)
    def test_sidm_disabled_equivalence_requires_distinct_diagnostic_build(self):
        request=control_request();right=copy.deepcopy(request['cases'][0]);right['id']='diagnostics';request['cases'].append(right)
        request['comparisons']=[dict(left='plummer',right='diagnostics',kind='sidm_disabled',relative_tolerance=1e-7,absolute_floor=.1)]
        with self.assertRaises(ValueError):validate_control_request(request)
        right['profile']['diagnostic_variant']=True;validate_control_request(request)
        right['profile']['pm_grid']=64
        with self.assertRaises(ValueError):validate_control_request(request)
    def test_fixture_ids_above_float_precision_and_reproducible_seed(self):
        case=control_request()['cases'][0]
        x,v,m,ids,box=test_native_numerical_controls_fixture(case)
        self.assertEqual(ids.dtype,np.dtype('uint64'));self.assertTrue(np.all(ids>2**53));self.assertEqual(len(ids),28)
        other=copy.deepcopy(case);other.update(mode='isolated_tree',particles=128);other['profile']['mode']='isolated_tree'
        a=test_native_numerical_controls_fixture(other);b=test_native_numerical_controls_fixture(other);np.testing.assert_array_equal(a[0],b[0]);other['seed']+=1
        self.assertFalse(np.array_equal(a[0],test_native_numerical_controls_fixture(other)[0]))
    def test_retains_failed_native_execution_without_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            dest=Path(root)/'failed'
            with patch('core.evolution.build_gadget',side_effect=RuntimeError('controlled compile failure')):
                with self.assertRaisesRegex(RuntimeError,'compile failure'):run_control_campaign(control_request(),dest)
            self.assertEqual(json.loads((dest/'execution.json').read_text())['status'],'FAILED')
            self.assertFalse((dest/'campaign.json').exists())
            with self.assertRaises(FileExistsError):run_control_campaign(control_request(),dest)
    def test_native_signed_diagnostic_ids_recovered_exactly(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'data').mkdir()
            (root/'data/forcetest.txt').write_text('1 0 -9223372036854775685 .5 1 2 3 0 4 5 6 4.1 5.1 6.1 -1 -1\n')
            ids,time,a,b,_=_force_test(root,control_request()['cases'][0])
            self.assertEqual(int(ids[0]),2**63+123);np.testing.assert_equal(a,[[4.1,5.1,6.1]]);np.testing.assert_equal(b,[[4.,5.,6.]])
    def test_public_cli_rejects_mixed_workflow_flags(self):
        result=subprocess.run([sys.executable,str(test_native_numerical_controls_ROOT/'commands.py'),'gadget', '--numerical-controls','x','--run-dir','y','--dry-run'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('cannot be mixed',result.stderr)
    def test_audit_cli_requires_actual_arguments(self):
        result=subprocess.run([sys.executable,str(test_native_numerical_controls_ROOT/'commands.py'), 'audit-native-handoff'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2);self.assertIn('requires',result.stderr)



class IsolatedCheckpointClockTests(unittest.TestCase):
    def make(self,root,rank=0,ranks=1,time=0.,generation=123):
        from verification import CheckpointTests
        p=CheckpointTests().rank(root,rank,ranks=ranks,a=.2,generation=generation)
        receipt=Path(str(p)+'.checksum.json');r=json.loads(receipt.read_text());r.update(schema='gadget-rank-checkpoint-v2',native_time=time);r.pop('a');receipt.write_text(json.dumps(r))
        epoch=dict(schema='native-checkpoint-epoch-v2',rank=rank,generation=generation,native_time=time,clock_representation='native_time_L_over_V',cosmic_time_status='NOT_APPLICABLE',cosmic_time_gyr=None)
        Path(str(p)+'.epoch.json').write_text(json.dumps(epoch));return p
    def test_native_time_zero_has_no_fabricated_redshift(self):
        from core.data import verify_native_checkpoints
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make(root)
            r=verify_native_checkpoints(root,1,require_receipts=True)
            self.assertIsNone(r['a']);self.assertIsNone(r['z']);self.assertEqual(r['clock_value'],0.);self.assertEqual(r['clock_representation'],'native_time_L_over_V')
    def test_mixed_native_and_cosmological_clocks_rejected(self):
        from core.data import verify_native_checkpoints
        from verification import CheckpointTests
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.make(root,ranks=2,time=.2);CheckpointTests().rank(root,1,ranks=2,a=.2)
            with self.assertRaisesRegex(ValueError,'epochs'):verify_native_checkpoints(root,2)
    def test_ambiguous_or_false_cosmological_sidecar_rejected(self):
        from core.data import verify_native_checkpoints
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=self.make(root)
            receipt=Path(str(p)+'.checksum.json');r=json.loads(receipt.read_text());r['a']=0.;receipt.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'Ambiguous'):verify_native_checkpoints(root,1)
            self.make(root);ep=Path(str(p)+'.epoch.json');r=json.loads(ep.read_text());r['z']=0.;ep.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'cosmological'):verify_native_checkpoints(root,1)
    def test_recover_isolated_backup_preserves_clock(self):
        from core.data import recover_native_checkpoints
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);p=self.make(root,time=.4)
            for suffix in ('','.checksum.json','.epoch.json'):
                __import__('shutil').copy2(str(p)+suffix,root/('bak-restart.0'+suffix))
            p.write_bytes(b'corrupted')
            r=recover_native_checkpoints(root,1)
            self.assertEqual(r['status'],'COMPLETE');self.assertEqual(r['checkpoint']['clock_value'],.4);self.assertIsNone(r['checkpoint']['a'])


class ExecutedNativeCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        name=os.environ.get('RESEARCH_NATIVE_CONTROL_RECEIPT')
        if not name:raise unittest.SkipTest('Execute the native campaign to supply measured receipt')
        cls.path=Path(name).resolve();cls.report=verify_control_campaign(cls.path)
    def test_every_native_case_and_declared_variant_passed(self):
        self.assertEqual(self.report['status'],'PASS')
        self.assertGreaterEqual(len(self.report['cases']),18)
        self.assertTrue(all(c['status']=='PASS' for c in self.report['cases']+self.report['comparisons']))
        self.assertEqual({c['kind'] for c in self.report['comparisons']},{'rank','restart','sidm_disabled','settings'})
    def test_resume_keeps_exact_native_payloads(self):
        from core.foundation import file_sha256 as digest
        before={c['report']:digest(c['report']) for c in self.report['cases']}
        result=run_control_campaign(self.report['request'],self.path.parent,resume=True)
        self.assertEqual(result['fingerprint'],self.report['fingerprint'])
        self.assertEqual(before,{p:digest(p) for p in before})
    def test_changed_tolerance_cannot_resume(self):
        request=copy.deepcopy(self.report['request'])
        for case in request['cases']:case['tolerances']['force_relative']*=2
        with self.assertRaisesRegex(ValueError,'request/code changed'):run_control_campaign(request,self.path.parent,resume=True)
    def test_every_restart_paused_at_evolved_interior_time(self):
        from core.data import verify_native_checkpoints
        for case in self.report['request']['cases']:
            if not case['restart']:continue
            root=self.path.parent/case['id'];row=verify_native_checkpoints(root/'interrupted-restart',case['mpi_ranks'],require_receipts=True)
            self.assertGreater(row['clock_value'],case['times'][0]);self.assertLess(row['clock_value'],case['times'][-1])
            if case['mode']=='analytic':self.assertIsNone(row['a'])
    def test_numeric_payload_mutation_rejected_on_retained_copy(self):
        import shutil
        from core.data import verify_numeric_audit
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=Path(self.report['cases'][0]['report']);target=root/source.name
            shutil.copy2(source,target);shutil.copy2(str(source)+'.sha256',str(target)+'.sha256')
            shutil.copytree(str(source)+'.data',str(target)+'.data')
            payload=Path(str(target)+'.data')/'arrays.npz';payload.write_bytes(payload.read_bytes()[:-32])
            with self.assertRaises((ValueError,OSError)):verify_numeric_audit(target)
    def test_public_audit_verifies_current_native_products(self):
        result=subprocess.run([sys.executable,str(test_native_numerical_controls_ROOT/'commands.py'),'audit-native-handoff', '--numerical-control-receipt',str(self.path)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'PASS')
    def test_canonical_epoch_rows_match_native_frames(self):
        from core.data import verify_numeric_audit
        for row in self.report['cases']:
            report=verify_numeric_audit(row['report'])
            bundle=Path(row['report']+'.data');contract=json.loads((bundle/'contract.json').read_text())
            dataset=contract['datasets']['position'];epochs=json.loads(dataset['epoch_json'])['rows']
            self.assertEqual(len(epochs),len(report['case']['times']))
            if report['case']['mode']=='analytic':
                self.assertTrue(all(e['actual_scale_factor'] is None and e['redshift'] is None for e in epochs))
                self.assertEqual(epochs[0]['native_time_code'],0.)
            elif report['case']['mode']=='periodic_treepm':self.assertTrue(all(e['actual_scale_factor']>0 for e in epochs))

# ========================================================================
# test_science_types
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.foundation import ProductRef, StageRequest, StageResult, PotentialState


class ScienceTypesTests(unittest.TestCase):
    def test_immutable_parameters_and_changed_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json';path.write_text('{}')
            product=ProductRef('input',path,file_sha256(path),1);parameters={'selection':[1,2]}
            request=StageRequest('finder',parameters,(product,),('catalogue',));request.verify_inputs()
            parameters['selection'].append(3);self.assertEqual(request.parameters['selection'],(1,2))
            with self.assertRaises(TypeError):request.parameters['other']=1
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'changed'):request.verify_inputs()

    def test_execution_status_and_product_identity_contracts(self):
        with self.assertRaises(ValueError):StageResult('ACCEPTED',(),(),())
        with self.assertRaises(ValueError):ProductRef('x','x','bad',1)
        with self.assertRaises(ValueError):StageRequest('finder',{},(),('x','x'))

# ========================================================================
# test_checkpoint_clock

test_checkpoint_clock_ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(test_checkpoint_clock_ROOT/'core')]
from core.cosmology import proper_time_numerical_envelope
from core.foundation import C_SI, SECONDS_PER_YEAR


class ClockEnvelopeTests(unittest.TestCase):
    def test_constant_H_exact_log_clock_is_inside_declared_envelope(self):
        a=np.geomspace(.01,1.,32)
        H=np.full(len(a),.0003)
        factor=MPC_CM/(C_SI*100*SECONDS_PER_YEAR*1e9*H[0])
        t=1.+factor*np.log(a/a[0])
        envelope=proper_time_numerical_envelope(a,t,H)
        for i in range(len(a)-1):
            x=(a[i]+a[i+1])/2
            exact=1.+factor*np.log(x/a[0])
            self.assertLessEqual(abs(interpolate_proper_time(a,t,x)-exact),envelope['interval_absolute_bound_gyr'][i])
        contract=epoch_contract(.1,clock=(a,t,{'numerical_error_envelope':envelope}))
        self.assertEqual(contract['cosmic_time_numerical_error_gyr']['status'],'MODEL_CONDITIONAL_BOUND')

    def test_unsorted_rows_and_source_time_discrepancy_are_retained(self):
        a=np.array([.1,.3,.6]);t=np.array([1.,4.,8.]);H=np.array([.002,.001,.0005])
        expected=proper_time_numerical_envelope(a,t,H)
        reversed_result=proper_time_numerical_envelope(a[::-1],t[::-1],H[::-1])
        np.testing.assert_allclose(expected['interval_absolute_bound_gyr'],reversed_result['interval_absolute_bound_gyr'])
        self.assertGreater(max(expected['source_node_residual_gyr']),0)
        for bad in ([.002,0,.0005],[.002,float('nan'),.0005],[.001]):
            with self.assertRaises(ValueError):proper_time_numerical_envelope(a,t,bad)


class NativePeriodicEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        source=cls.root/'writer.cc';cls.binary=cls.root/'writer'
        source.write_text('#include "checkpoint_epoch.h"\nint main(int n,char **v){char stage[4096];if(n!=4)return 2;return native_checkpoint_epoch_stage(v[1],v[2],0,1234,atof(v[3]),stage,sizeof(stage))?0:1;}\n')
        subprocess.run(['c++','-std=c++11','-I',str(test_checkpoint_clock_ROOT/'core/software/gadget4/src/io'),str(source),'-lz','-o',str(cls.binary)],capture_output=True,check=True)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def run_writer(self,root,a):
        return subprocess.run([str(self.binary),str(root),str(root/'restart.0'),str(a)],capture_output=True,timeout=30)
    def table(self,root,payload):
        (root/'canonical-clock.tsv').write_text('# native-canonical-clock-v1 '+('a'*64)+f' {zlib.crc32(payload.encode()):08x}\n'+payload)
    def test_actual_writer_requested_actual_epoch_and_explicit_absence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.table(root,'.1 1\n.5 6\n1 13\n')
            self.assertEqual(self.run_writer(root,.3).returncode,0)
            record=json.loads(next(root.glob('*.epoch.json.tmp.*')).read_text())
            self.assertEqual(record['generation'],1234);self.assertAlmostEqual(record['cosmic_time_gyr'],3.5)
            self.assertEqual(record['a'],.3);self.assertEqual(record['z'],1/.3-1)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.assertEqual(self.run_writer(root,.3).returncode,0)
            record=json.loads(next(root.glob('*.epoch.json.tmp.*')).read_text())
            self.assertIsNone(record['cosmic_time_gyr']);self.assertEqual(record['cosmic_time_status'],'UNAVAILABLE')
    def test_corrupt_nonmonotonic_or_outside_domain_never_publishes_epoch(self):
        for payload,a in [('.1 1\n.1 2\n',.1),('.1 2\n.5 1\n',.3),('.1 1\n.5 3\n',.6),('.1 1\nnan 2\n',.3)]:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);self.table(root,payload)
                self.assertEqual(self.run_writer(root,a).returncode,1)
                self.assertEqual(list(root.glob('*.epoch.json.tmp.*')),[])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.table(root,'.1 1\n.5 3\n')
            path=root/'canonical-clock.tsv';path.write_text(path.read_text().replace('.5 3','.5 4'))
            self.assertEqual(self.run_writer(root,.3).returncode,1)
from zoom import velocity_divergence_modes


class MixedContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.ic=self.root/'ic.hdf5';self.cfg=self.root/'music.conf'
        self.cfg.write_text('''[setup]
boxlength=1
zstart=99
levelmin=2
levelmax=2
baryons=yes
use_2LPT=yes
[cosmology]
H0=70
Omega_m=0.3
Omega_b=0.05
m_nu1=0
m_nu2=0
m_nu3=0
transfer=DMS_file
mixed_2lpt=yes
lpt2_displacement=0.44
lpt2_velocity=0.855
[output]
format=arepo
arepo_coarsetype=2
arepo_doubleprec=yes
arepo_longids=yes
arepo_num_files=1
''')
        q=np.array(list(np.ndindex(4,4,4)),dtype=float)*250
        rho=3*100**2/(8*np.pi*G_MPC_KMS2_MSUN)/1e10
        mean=np.array([.05*rho/64,.25*rho/64,0,0,0,0])
        with h5py.File(self.ic,'w') as source:
            source.create_group('Header').attrs.update(BoxSize=1000.,Time=.01,HubbleParam=.7,Omega0=.3,
                OmegaBaryon=.05,NumFilesPerSnapshot=1,NumPart_ThisFile=[64,64,0,0,0,0],
                NumPart_Total=np.array([64,64,0,0,0,0],dtype=np.uint32),NumPart_Total_HighWord=np.zeros(6,dtype=np.uint32),
                MassTable=np.zeros(6),Music_lpt_order=2,Music_mixed_fluid_schema=1,
                Music_mean_species_masses_native=mean,Music_lagrangian_schema=1,
                Music_lagrangian_unitlength_cm=KPC_CM,Music_max_abs_displacement_native=[1,0,0],
                Music_2lpt_displacement_coefficient=.44,Music_2lpt_velocity_coefficient=.855)
            for t in (0,1):
                g=source.create_group('PartType'+str(t));g['ParticleIDs']=np.arange(64,dtype=np.uint64)+np.uint64(t*64)
                g['LagrangianCoordinates']=q;g['Coordinates']=q
                g['Masses']=mean[t]*np.ones(64)
                g['Velocities']=np.zeros_like(q);g['FirstOrderDisplacements']=np.zeros_like(q)
                g['FirstOrderVelocities']=np.zeros_like(q)
        self.q=q;self.mean=mean

    def test_parent_variable_mass_schema_and_units_are_accepted(self):
        report=audit_music_ic(self.ic,self.cfg,input_kind='parent')
        self.assertEqual(report['status'],'STRUCTURAL_PASS')
        self.assertEqual(report['species']['1']['mass_levels_native'],[self.mean[1]])
        self.assertAlmostEqual(report['species']['1']['actual_total_mass_native'],64*self.mean[1],places=12)

    def test_compensated_density_enters_linear_modes_without_displacement(self):
        wave=.01*np.cos(2*np.pi*self.q[:,0]/1000)
        with h5py.File(self.ic,'r+') as f:
            f['PartType1/Masses'][:]=self.mean[1]*(1-wave/6)
            f['PartType0/Masses'][:]=self.mean[0]*(1+5*wave/6)
        self.assertAlmostEqual(linear_density_modes(self.ic,[[1,0,0]],particle_types=(1,))[0].real,-.01/12)
        self.assertAlmostEqual(linear_density_modes(self.ic,[[1,0,0]],particle_types=(0,))[0].real,5*.01/12)
        np.testing.assert_allclose(linear_density_modes(self.ic,[[1,0,0]]),0.,atol=1e-15)

    def test_velocity_modes_use_linear_component_and_reference_mass(self):
        with h5py.File(self.ic,'r+') as f:
            v=np.zeros_like(self.q);v[:,0]=np.sin(2*np.pi*self.q[:,0]/1000)
            f['PartType1/FirstOrderVelocities'][:]=v
            f['PartType1/Velocities'][:]=v*20
        expected=.1*(2*np.pi/1000)/2*(5/6)
        self.assertAlmostEqual(velocity_divergence_modes(self.ic,[[1,0,0]])[0].real,expected,places=14)

    def test_wrong_kernel_order_and_missing_components_rejected(self):
        with h5py.File(self.ic,'r+') as f:f['Header'].attrs['Music_2lpt_velocity_coefficient']=6/7
        with self.assertRaisesRegex(ValueError,'kernel differs'):
            audit_music_ic(self.ic,self.cfg,input_kind='parent')
        with h5py.File(self.ic,'r+') as f:
            f['Header'].attrs['Music_2lpt_velocity_coefficient']=.855
            del f['PartType1/FirstOrderDisplacements']
        with self.assertRaises(KeyError):audit_music_ic(self.ic,self.cfg,input_kind='parent')

# ========================================================================
# test_native_hdf5_sets
sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from core.data import verify_native_hdf5_set, recover_native_hdf5_set, native_publication_copy_members


class NativeSetTests(unittest.TestCase):
    def fixture(self, root, generations=('epoch-a', 'epoch-a')):
        files = []
        for i, generation in enumerate(generations):
            path = root / ('snapshot_001.%d.hdf5' % i)
            with h5py.File(path, 'w') as source:
                header = source.create_group('Header')
                header.attrs['Research2Generation'] = generation
                header.attrs['Research2ShardCount'] = 2
            files.append(path)
        marker = root / 'snapshot_001.publication.json'
        marker.write_text(json.dumps({'schema': 'native-hdf5-set-publication-v1',
                                     'generation': 'epoch-a', 'state': 'COMPLETE', 'shard_count': 2}))
        return files, marker

    def test_complete_and_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            files, marker = self.fixture(Path(temp).resolve())
            self.assertEqual(verify_native_hdf5_set(files, require_publication=True)['status'], 'COMPLETE')
            record = json.loads(marker.read_text()); record['state'] = 'PENDING'
            marker.write_text(json.dumps(record))
            with self.assertRaises(ValueError): verify_native_hdf5_set(files)

    def test_mixed_missing_and_copied(self):
        with tempfile.TemporaryDirectory() as temp:
            files, marker = self.fixture(Path(temp).resolve(), ('epoch-a', 'epoch-b'))
            with self.assertRaises(ValueError): verify_native_hdf5_set(files)
            files, marker = self.fixture(Path(temp).resolve())
            with self.assertRaises(ValueError): verify_native_hdf5_set(files[:1])
            marker.unlink()
            self.assertEqual(verify_native_hdf5_set(files)['status'], 'COPIED_GENERATION_UNSEALED')
            with self.assertRaises(ValueError): verify_native_hdf5_set(files, require_publication=True)

    def test_explicit_recovery_preserves_displaced_and_rejects_uncommitted(self):
        from core.foundation import file_sha256 as digest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            files, marker = self.fixture(root)
            original = [digest(p) for p in files]
            marker.rename(marker.with_name(marker.name + '.previous.old'))
            files[0].rename(files[0].with_name(files[0].name + '.previous.old'))
            with h5py.File(files[0], 'w') as source:
                header = source.create_group('Header')
                header.attrs['Research2Generation'] = 'epoch-b'
                header.attrs['Research2ShardCount'] = 2
            displaced = digest(files[0])
            marker.write_text(json.dumps({'schema': 'native-hdf5-set-publication-v1',
                'state': 'PENDING', 'generation': 'epoch-b', 'shard_count': 2}))
            with self.assertRaises(ValueError):
                recover_native_hdf5_set(root / 'snapshot_001', 'epoch-b')
            result = recover_native_hdf5_set(root / 'snapshot_001', 'epoch-a')
            self.assertEqual([digest(p) for p in files], original)
            self.assertEqual(digest(Path(result['recovery_archive']) / 'displaced' / files[0].name), displaced)
            self.assertEqual(result['status'], 'COMPLETE')

    def test_copy_preserves_marker_on_base_rename_and_rejects_partial_set(self):
        with tempfile.TemporaryDirectory() as temp:
            files, marker = self.fixture(Path(temp).resolve())
            outputs = [(source, Path('grouped_snapshot_001.%d.hdf5' % i))
                       for i, source in enumerate(files)]
            self.assertEqual(native_publication_copy_members(outputs),
                             [(marker, Path('grouped_snapshot_001.publication.json'))])
            with self.assertRaises(ValueError): native_publication_copy_members(outputs[:1])
            bad = [(files[0], Path('grouped_snapshot_001.1.hdf5')), outputs[1]]
            with self.assertRaises(ValueError): native_publication_copy_members(bad)

    def test_catalogue_publication_carries_native_marker(self):
        from core.data import publish_catalogue_set, assert_catalogue_publication_ready
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / 'data').mkdir()
            source = root / 'stage'; source.mkdir()
            files, marker = self.fixture(source)
            outputs = [(path, Path('grouped_snapshot_001.%d.hdf5' % i))
                       for i, path in enumerate(files)]
            publish_catalogue_set(root, 1, outputs, [],
                                  epoch={'scope': 'transport fixture'}, provenance={'scope': 'transport fixture'})
            assert_catalogue_publication_ready(root, 1)
            copies = [root / 'data' / relative for _, relative in outputs]
            self.assertEqual(verify_native_hdf5_set(copies, require_publication=True)['status'], 'COMPLETE')
            self.assertEqual((root / 'data/grouped_snapshot_001.publication.json').read_bytes(), marker.read_bytes())

    def test_discovery_excludes_recovery_history_and_rejects_pending(self):
        from analysis import find_hdf5_files, discover_snapshot_sets
        from core.foundation import atomic_copy
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            files, marker = self.fixture(root)
            hidden = root / '.native_set_history/archive/displaced'
            hidden.mkdir(parents=True)
            atomic_copy(files[0], hidden / files[0].name)
            self.assertEqual(find_hdf5_files(root, 'snapshot_001'), files)
            self.assertEqual(discover_snapshot_sets(root)[1], files)
            record = json.loads(marker.read_text()); record['state'] = 'PENDING'
            marker.write_text(json.dumps(record))
            with self.assertRaises(ValueError): find_hdf5_files(root, 'snapshot_001')
            with self.assertRaises(ValueError): discover_snapshot_sets(root)

    def test_native_metadata_rejects_fractional_counts_and_wrong_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            files, _ = self.fixture(root)
            with h5py.File(files[0], 'r+') as source:
                del source['Header'].attrs['Research2ShardCount']
                source['Header'].attrs['Research2ShardCount'] = 2.5
            with self.assertRaisesRegex(ValueError, 'exact positive'):
                verify_native_hdf5_set(files)
            files, _ = self.fixture(root)
            misplaced = root / 'snapshot_001.2.hdf5'
            files[1].rename(misplaced)
            with self.assertRaisesRegex(ValueError, 'indices'):
                verify_native_hdf5_set([files[0], misplaced])

    def test_snapshot_numbers_ignore_hidden_recovery_generations(self):
        from analysis import snapshot_ids
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            data = root / 'data'; data.mkdir()
            self.fixture(data)
            hidden = data / '.native_set_history/old'
            hidden.mkdir(parents=True)
            (hidden / 'snapshot_999.hdf5').write_bytes(b'archived transport bytes')
            self.assertEqual(snapshot_ids(root), [1])

# ========================================================================
# test_stellar_finder_publication
sys.path[:0]=[str(Path(__file__).resolve().parent/'core'),str(Path(__file__).resolve().parent)]
import halos as galaxies


class StellarFinderPublicationTests(unittest.TestCase):
    def controls(self):
        return dict(minimum_particles=3,velocity_neighbors=4,physical_neighbors=8,cell_fraction=.125,
            outlier_threshold=2.,significance_level=1.,velocity_ratio=2.,opening_angle_pi=.5,linking_length_kpc=1.)

    def fixture(self,root):
        n=256;ids=np.arange(n,dtype=np.uint64)+np.uint64(2**63+11)
        test_stellar_history_snapshot(root,0,.5,ids,np.column_stack((np.linspace(1.,2.,n),np.ones(n),np.ones(n))),bound_count=n)
        with h5py.File(root/'data/fof_subhalo_tab_000.hdf5','a') as source:
            source['Header'].attrs.update(NumFiles=1,Ngroups_Total=1,Ngroups_ThisFile=1)
            group=source.create_group('Group')
            group['GroupLenType']=np.array([[0,0,0,0,n,0]],dtype=np.uint64)
            group['GroupOffsetType']=np.zeros((1,6),dtype=np.uint64)
            group['GroupPos']=[[1.,1.,1.]]
        return ids

    def native(self,command,**kwargs):
        # Native process transport only. No finder algorithm is executed.
        self.assertTrue(kwargs.get('pass_fds'))
        kwargs['stdout'].write('STF_EMPTY_SELECTION\n')

    def patches(self):
        from contextlib import ExitStack
        stack=ExitStack();binary=Path(galaxies.__file__).parent/'software/velociraptor/build/stf'
        actual_is_file=Path.is_file;actual_digest=galaxies.file_sha256
        stack.enter_context(patch.object(Path,'is_file',lambda path:True if path==binary else actual_is_file(path)))
        stack.enter_context(patch('halos.file_sha256',side_effect=lambda path:'f'*64 if Path(path)==binary else actual_digest(path)))
        stack.enter_context(patch('subprocess.run',side_effect=self.native))
        stack.enter_context(patch('halos._stellar_candidate_figure'))
        stack.enter_context(patch('halos._stellar_candidate_diagnostics'))
        stack.enter_context(patch('halos._stellar_candidate_blender_assets'))
        return stack

    def test_exact_dense_mapping_original_epoch_and_completed_portable_native_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);ids=self.fixture(root)
            with self.patches():result=galaxies.find_stellar_substructures(root,0,0,controls=self.controls(),threads=1)
            directory=Path(result['directory']);verify_hdf5_dataset_payloads(directory/'candidates.hdf5',require_contract=True)
            verify_hdf5_dataset_payloads(directory/'native_work/stars.hdf5',require_contract=True)
            with h5py.File(directory/'native_work/stars.hdf5') as source:
                np.testing.assert_array_equal(source['DenseIDMap/OriginalParticleIDs'][:],ids)
                np.testing.assert_array_equal(source['PartType4/ParticleIDs'][:],np.arange(1,len(ids)+1,dtype=np.uint64))
                self.assertEqual(float(source['Header'].attrs['Time']),1.)
                self.assertIn('epoch_json',source['PartType4/Coordinates'].attrs)
            report=json.loads((directory/'provenance.json').read_text())
            self.assertEqual(report['scale_factor'],.5)
            self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')
            self.assertTrue((directory/'native_work/invocation.json').is_file())
            self.assertTrue((directory/'native_work/completion.json').is_file())

    def test_failed_publication_retains_native_and_exact_candidate_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            with self.patches(),patch('core.foundation.publish_directory',side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):galaxies.find_stellar_substructures(root,0,0,controls=self.controls(),threads=1)
            parent=root/'results/stellar_substructures';stage=list(parent.glob('.stellar-stf-*'))[0]
            self.assertFalse((parent/'snapshot_000_host_0').exists())
            for name in ['native_work/stars.hdf5','native_work/finder.log','candidates.hdf5','provenance.json','export_failure.json']:
                self.assertTrue((stage/name).is_file())

# ========================================================================
# test_black_hole_model
import math

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from black_holes import (ThermalModel, accretion_rates, salpeter_time_years,
                              constant_environment_growth, energy_mass_budget, native_parameters)


class BlackHoleReferenceTests(unittest.TestCase):
    def setUp(self):
        self.model = ThermalModel(.1, .05, 1., 1.)

    def test_eddington_and_salpeter_scales(self):
        rates = accretion_rates(1e8, 1e-18, 1e6, 0., self.model)
        self.assertAlmostEqual(rates["eddington_inflow_msun_year"], 2.2198, delta=.001)
        self.assertAlmostEqual(salpeter_time_years(self.model)/1e6, 50.05, delta=.1)
        self.assertEqual(rates["inflow_msun_year"], rates["eddington_inflow_msun_year"])

    def test_bondi_mass_density_and_velocity_scaling(self):
        base = accretion_rates(1e4, 1e-26, 1e6, 0., self.model)
        double_mass = accretion_rates(2e4, 1e-26, 1e6, 0., self.model)
        double_sound = accretion_rates(1e4, 1e-26, 2e6, 0., self.model)
        self.assertAlmostEqual(double_mass["inflow_msun_year"]/base["inflow_msun_year"], 4.)
        self.assertAlmostEqual(double_sound["inflow_msun_year"]/base["inflow_msun_year"], .125)

    def test_exact_growth_switch_and_subdivision(self):
        # Low density initially gives Bondi-limited growth, then crosses the cap.
        rho, cs, mass = 1e-23, 1e6, 1e5
        t = salpeter_time_years(self.model)
        rates = accretion_rates(mass, rho, cs, 0., self.model)
        self.assertLess(rates["bondi_inflow_msun_year"], rates["eddington_inflow_msun_year"])
        total = constant_environment_growth(mass, 10*t, rho, cs, 0., self.model)
        first = constant_environment_growth(mass, 2*t, rho, cs, 0., self.model)
        split = constant_environment_growth(first, 8*t, rho, cs, 0., self.model)
        final_rates = accretion_rates(total, rho, cs, 0., self.model)
        self.assertGreater(final_rates["bondi_inflow_msun_year"], final_rates["eddington_inflow_msun_year"])
        self.assertAlmostEqual(total/split, 1., places=12)
        edd = constant_environment_growth(1e8, t, 1e-18, cs, 0., self.model)
        self.assertAlmostEqual(edd/1e8, math.e, places=12)

    def test_mass_and_energy_accounting(self):
        budget = energy_mass_budget(100., self.model)
        self.assertEqual(budget["gas_mass_removed_msun"], budget["bh_mass_gained_msun"] + budget["radiated_mass_equivalent_msun"])
        self.assertAlmostEqual((budget["escaping_radiation_energy_erg"] + budget["coupled_thermal_energy_erg"])/budget["radiated_energy_erg"], 1.)

    def test_vacuum_and_invalid_inputs(self):
        self.assertEqual(constant_environment_growth(1e6, 1e8, 0., 1e6, 0., self.model), 1e6)
        for args in ((0., .1, 1., 1.), (.1, 1.1, 1., 1.), (.1, .1, float("nan"), 1.)):
            with self.assertRaises(ValueError):
                ThermalModel(*args)
        with self.assertRaises(ValueError):
            accretion_rates(1e5, 1e-20, 0., 0., self.model)

    def test_native_settings_are_explicit(self):
        settings = dict(enabled=True, radiative_efficiency=.1, feedback_coupling=.05,
                        bondi_boost=1., eddington_factor=1., kernel_radius_proper_kpc=1.,
                        seed_mass_msun=1e5, source_fraction=.05, maximum_count=1000,
                        seeding_mode="ic_particles", seed_density_hydrogen_cm3=0., seed_exclusion_proper_kpc=0.)
        self.assertEqual(native_parameters(settings)["BHSeedMassMsun"], 1e5)
        self.assertEqual(native_parameters({"enabled": False}), {})
        for invalid in ({"enabled": True}, dict(settings, maximum_count=True),
                        dict(settings, source_fraction=.5), dict(settings, typo=1)):
            with self.assertRaises(ValueError):
                native_parameters(invalid)

# ========================================================================
# test_ic_species_reference

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from analysis import ic_species_density_reference


class SpeciesReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path=Path(temporary.name)/'transfers.txt'
        self.cfg={'cosmology':{'model':'lcdm','omega_b':.02,'omega_cdm':.1}}

    def test_generator_matter_density_uses_cosmic_mass_fractions_and_never_theta(self):
        rows=np.array([[.1,-2,-1,500,700],[1,-4,-3,600,800]])
        header='DarkMatterSim monofonIC transfer table v1\nModel: lcdm; redshift: 99; gauge: N-body'
        np.savetxt(self.path,rows,header=header)
        for species,column in [('baryon',2),('cdm',1)]:
            k,density,matter=ic_species_density_reference(self.path,self.cfg,species)
            np.testing.assert_array_equal(k,rows[:,0])
            np.testing.assert_array_equal(density,rows[:,column])
            np.testing.assert_allclose(matter,(5*rows[:,1]+rows[:,2])/6)
        rows[:,3:]*=1e6
        np.savetxt(self.path,rows,header=header)
        np.testing.assert_allclose(ic_species_density_reference(self.path,self.cfg,'baryon')[2],matter)

    def test_validation_table_preserves_explicit_matter_and_ignores_absent_species_nan(self):
        rows=np.array([[.1,-2,-1,-1.8,np.nan,500,700,np.nan],[1,-4,-3,-3.8,np.nan,600,800,np.nan]])
        np.savetxt(self.path,rows,header='k_h_per_Mpc delta_cdm delta_b delta_m delta_wdm theta_cdm_Mpc^-1 theta_b_Mpc^-1 theta_wdm_Mpc^-1')
        np.testing.assert_array_equal(ic_species_density_reference(self.path,self.cfg,'cdm')[2],rows[:,3])
        with self.assertRaisesRegex(ValueError,'invalid k/density'):
            ic_species_density_reference(self.path,self.cfg,'wdm')

    def test_unlabeled_columns_or_wrong_model_cannot_be_guessed(self):
        rows=np.array([[.1,-2,-1,500,700],[1,-4,-3,600,800]])
        np.savetxt(self.path,rows)
        with self.assertRaisesRegex(ValueError,'Unrecognized'):
            ic_species_density_reference(self.path,self.cfg,'cdm')
        np.savetxt(self.path,rows,header='DarkMatterSim monofonIC transfer table v1\nModel: ede; redshift: 99; gauge: N-body')
        with self.assertRaisesRegex(ValueError,'model/schema mismatch'):
            ic_species_density_reference(self.path,self.cfg,'cdm')

# ========================================================================
# test_model_exports
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.data import export_model_arrays, verify_model_arrays

class ModelExportTests(unittest.TestCase):
    def write(self,root):
        return export_model_arrays(root/'figure',{'covariance':np.eye(2)},
            {'covariance':{'units':'dimensionless','quantity':'density covariance'}},algorithm=__file__)

    def test_numeric_roundtrip_and_immutable_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);record=self.write(root)
            contract=verify_model_arrays(record['directory'])
            self.assertEqual(contract,record['array_contract'])
            with self.assertRaises(FileExistsError):self.write(root)

    def test_corrupt_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);record=self.write(root);bundle=Path(record['directory'])
            np.savez(bundle/'arrays.npz',covariance=np.zeros((2,2)))
            with self.assertRaises(ValueError):verify_model_arrays(bundle)

    def test_contract_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);record=self.write(root);bundle=Path(record['directory'])
            (bundle/'contract.json').write_text('{}')
            with self.assertRaises(ValueError):verify_model_arrays(bundle)

    def test_failed_publication_retains_staged_numeric_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('core.data.publish_directory',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):self.write(root)
            self.assertFalse((root/'figure.data').exists())
            self.assertTrue(list(root.glob('.figure-data-*/arrays.npz')))
            self.assertTrue(list(root.glob('.figure-data-*/failure.json')))


class NumericAuditTests(unittest.TestCase):
    def test_measured_complex_arrays_and_report_integrity(self):
        from core.data import export_numeric_audit, verify_numeric_audit
        from core.foundation import file_sha256 as digest
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input';source.write_bytes(b'input physics')
            target=root/'audit.json'
            arrays={'density':np.array([1+2j,3-4j])}
            specifications={'density':{'units':'dimensionless','quantity':'complex density Fourier mode'}}
            report=export_numeric_audit(target,{'status':'PASS'},arrays,specifications,
                algorithm=__file__,context={'epoch':{'scale_factor':.5}},source_hashes={str(source):digest(source)})
            self.assertEqual(verify_numeric_audit(target),report)
            self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')
            target.write_text(target.read_text().replace('PASS','FAIL'))
            with self.assertRaisesRegex(ValueError,'checksum'):verify_numeric_audit(target)

    def test_changed_sources_refuse_audit_publication(self):
        from core.data import export_numeric_audit
        from core.foundation import file_sha256 as digest
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input';source.write_bytes(b'before');expected=digest(source)
            source.write_bytes(b'after')
            with self.assertRaisesRegex(RuntimeError,'source changed'):
                export_numeric_audit(root/'audit.json',{'status':'PASS'},{'x':np.array([1.])},
                    {'x':{'units':'dimensionless'}},algorithm=__file__,context={},source_hashes={str(source):expected})
            self.assertFalse((root/'audit.json').exists())

    def test_numeric_audit_valid_archive_rewrite_is_rejected(self):
        from core.data import export_numeric_audit, verify_numeric_audit
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);target=root/'audit.json'
            export_numeric_audit(target,{'status':'PASS'},{'density':np.array([1+2j])},
                {'density':{'units':'dimensionless','quantity':'complex density mode'}},
                algorithm=__file__,context={},source_hashes={})
            np.savez_compressed(root/'audit.json.data/arrays.npz',density=np.array([2+3j]))
            with self.assertRaises(ValueError):verify_numeric_audit(target)

    def test_numeric_audit_survives_staged_directory_publication(self):
        from core.data import export_numeric_audit, verify_numeric_audit
        from core.foundation import publish_directory
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage = root / 'stage'; stage.mkdir()
            report = export_numeric_audit(stage / 'audit.json', {'status':'MEASURED'},
                {'modes':np.array([1+2j])}, {'modes':{'units':'dimensionless','quantity':'Fourier mode'}},
                algorithm=__file__, context={}, source_hashes={})
            self.assertEqual(report['numeric_data']['directory'], 'audit.json.data')
            final = root / 'published'
            publish_directory(stage, final)
            self.assertEqual(verify_numeric_audit(final / 'audit.json'), report)

# ========================================================================
# test_identity_history
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
import verification as test_stellar_history
from halos import add_stellar_history


class IdentityHistoryTests(unittest.TestCase):
    def test_saved_history_missing_unbound_and_identity_continuity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);catalogue,ids=test_stellar_history.StellarHistoryTests().fixture(root)
            history=trace_stellar_candidates(root,catalogue=catalogue,snapshots=[0,1],candidate_ids=[1])
            graph=IdentityGraph(maximum_nodes=100,maximum_edges=100)
            sources=add_stellar_history(graph,history['catalogue'],particle_namespace='run:ssp',owner_namespace='run:subfind',maximum_bytes=128*1024**2)
            graph.publish(root/'graph',source_hashes=sources);q=IdentityQueries(root/'graph')
            self.assertEqual(q.bound_owner(('run:ssp',int(ids[1]),0))['status'],'UNDETECTED')
            self.assertEqual(q.bound_owner(('run:ssp',int(ids[0]),1))['status'],'UNBOUND')
            self.assertEqual(q.descendants(('run:ssp',int(ids[0]),0)),[('run:ssp',int(ids[0]),1)])
            self.assertEqual(q.birth_parent(('run:ssp',int(ids[0]),1))['status'],'UNAVAILABLE')

# ========================================================================
# test_native_run_lock
import threading,time
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.native import native_run_lock_fds

test_native_run_lock_CORE=Path(__file__).resolve().parent/'core'
PROBE="""
import sys
sys.path.insert(0,sys.argv[1])
from core.native import native_run_lock
with native_run_lock(sys.argv[2]):print('ACQUIRED',flush=True)
"""

class NativeRunLockTests(unittest.TestCase):
    def test_nested_owner_and_empty_run_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            with native_run_lock(root):
                fd=native_run_lock_fds(root)
                self.assertEqual(len(fd),1)
                with native_run_lock(root):self.assertEqual(native_run_lock_fds(root),fd)
                self.assertFalse(root.exists())
            with native_run_lock(root):pass
            self.assertTrue(list(root.parent.glob('.native-run-locks/*.lock')))
    def test_competing_process_and_symlink_alias_fail_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();data=root/'manifest.json';data.write_text('unchanged')
            alias=Path(tmp)/'alias';alias.symlink_to(root,target_is_directory=True)
            with native_run_lock(root):
                result=subprocess.run([sys.executable,'-c',PROBE,str(test_native_run_lock_CORE),str(alias)],capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0);self.assertIn('already owned',result.stderr)
                self.assertEqual(data.read_text(),'unchanged')
    def test_other_thread_cannot_reenter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';errors=[]
            def competing():
                try:
                    with native_run_lock(root):errors.append('incorrect acquisition')
                except RuntimeError as exc:errors.append(str(exc))
            with native_run_lock(root):
                thread=threading.Thread(target=competing);thread.start();thread.join(timeout=10)
                self.assertFalse(thread.is_alive());self.assertIn('another thread',errors[0])
    def test_exception_releases_kernel_owner_without_deleting_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            try:
                with native_run_lock(root):raise KeyboardInterrupt()
            except KeyboardInterrupt:pass
            lock=next(root.parent.glob('.native-run-locks/*.lock'));inode=lock.stat().st_ino
            with native_run_lock(root):self.assertEqual(lock.stat().st_ino,inode)
    def test_sigkill_wrapper_child_descriptor_retains_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            code="""
import sys,subprocess,os,signal
sys.path.insert(0,sys.argv[1])
from core.native import native_run_lock,native_run_lock_fds
with native_run_lock(sys.argv[2]):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],pass_fds=native_run_lock_fds(sys.argv[2]))
    print(child.pid,flush=True)
    os.kill(os.getpid(),signal.SIGKILL)
"""
            parent=subprocess.Popen([sys.executable,'-c',code,str(test_native_run_lock_CORE),str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            child=int(parent.stdout.readline());parent.wait(timeout=30)
            try:
                self.assertEqual(parent.returncode,-9)
                result=subprocess.run([sys.executable,'-c',PROBE,str(test_native_run_lock_CORE),str(root)],capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0);self.assertIn('already owned',result.stderr)
            finally:
                os.kill(child,signal.SIGKILL)
                parent.stdout.close();parent.stderr.close()
            deadline=time.monotonic()+10
            while True:
                try:
                    with native_run_lock(root):break
                except RuntimeError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.01)
    def test_owner_receipt_disk_full_releases_lock_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';entered=False
            with patch('core.foundation.os.fsync',side_effect=OSError(errno.ENOSPC,'full')):
                with self.assertRaises(OSError):
                    with native_run_lock(root):entered=True
            self.assertFalse(entered)
            with native_run_lock(root):pass
    def test_public_resume_competitor_stops_before_manifest_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir();manifest=root/'manifest.json';manifest.write_text('{malformed preserved')
            with native_run_lock(root):
                result=subprocess.run([sys.executable,str(test_native_run_lock_CORE.parent/'commands.py'),'gadget','--resume','--run-dir',str(root)],capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0);self.assertIn('already owned',result.stderr)
                self.assertEqual(manifest.read_text(),'{malformed preserved')
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from halos import gadget_catalogue
import verification as test_zoom_catalogue_target
test_finder_adapters_ROOT=Path(__file__).resolve().parent


class FinderAdapterTests(unittest.TestCase):
 def fixture(self):
  case=test_zoom_catalogue_target.CatalogueTargetTests();case.setUp();self.addCleanup(case.doCleanups)
  with h5py.File(case.catalogue,'r+') as f:
   f['Group/GroupMass']=np.array([2.,1.]);f['Group/GroupVel']=np.array([[0.,0.,0.],[0.,0.,0.]])
  return case

 def test_exact_fof_units_public_graph_and_reuse(self):
  case=self.fixture();root=case.root
  catalogue=gadget_catalogue(root,0,finder='fof',particle_type=1,particle_namespace='actual:dm',memory_gib=.1,selected_rows=[0,1])
  self.assertEqual(catalogue['objects'][0]['members'],case.fixture.ids[:2].tolist())
  self.assertAlmostEqual(catalogue['objects'][0]['radius_proper_kpc'],1.)
  self.assertEqual(catalogue['objects'][0]['radius_definition'],'R200c')
  request=root/'graph-request.json';out=root/'graph'
  request.write_text(json.dumps({'run_dir':str(root),'selections':[{'snapshot':0,'finder':'fof','particle_type':1,'particle_namespace':'actual:dm','memory_gib':.1,'selected_rows':[0,1]}],'maximum_nodes':100,'maximum_edges':100}))
  command=[sys.executable,str(test_finder_adapters_ROOT/'commands.py'),'build-identity-graph', str(request),str(out),'--from-native']
  subprocess.run(command,check=True);subprocess.run(command+['--reuse-completed'],check=True)
  q=IdentityQueries(out);self.assertEqual(q.members_at_epoch((catalogue['namespace'],0,0),particle_namespace='actual:dm').tolist(),sorted(case.fixture.ids[:2].tolist()))
  self.assertEqual(q.bound_owner(('actual:dm',int(case.fixture.ids[0]),0))['status'],'OWNERSHIP_UNAVAILABLE')

 def test_incomplete_wrong_epoch_and_workspace_refused(self):
  case=self.fixture();opts={'finder':'fof','particle_type':1,'particle_namespace':'actual:dm','memory_gib':.1,'selected_rows':[0]}
  with self.assertRaises(MemoryError):gadget_catalogue(case.root,0,**{**opts,'memory_gib':1e-8})
  with h5py.File(case.catalogue,'r+') as f:f['Header'].attrs['Time']=.5
  with self.assertRaisesRegex(ValueError,'frame mismatch'):gadget_catalogue(case.root,0,**opts)

 def test_native_stellar_members_keep_ssp_kind(self):
  from halos import graph_from_catalogues
  case=self.fixture();case.add_galaxy()
  catalogue=gadget_catalogue(case.root,0,finder='fof',particle_type=4,particle_namespace='run:ssp',memory_gib=.1,selected_rows=[0])
  graph,sources=graph_from_catalogues([catalogue],maximum_nodes=10,maximum_edges=10)
  self.assertEqual(catalogue['particle_kind'],'ssp')
  self.assertEqual(graph.nodes[('run:ssp',2**63+50,0)]['kind'],'ssp')

 def test_native_consistent_trees_header_number_suffix_and_cli(self):
  from halos import consistent_tree_series
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);paths=[]
   for i,a in enumerate((.5,1.)):
    from verification import test_stellar_history_snapshot as snapshot
    snapshot(root,0 if i==0 else 7,a,[100],[[0,0,0]])
    p=root/f'hlist-{i}.list'
    p.write_text('#scale(0) id(1) desc_scale(2) desc_id(3) pid(4) mvir(5) x(6) y(7) z(8) phantom(9)\n'+f'{a} {7+i} 1.0 {8 if i==0 else -1} -1 {10+100*i} .099 0 0 0\n');paths.append(str(p))
   imported=consistent_tree_series(root,catalogues=paths,epochs_by_scale_factor={'0.5':0,'1':7},hubble_param=.5,box_comoving_kpc_h=100.,mass_column='mvir',memory_gib=.1)
   self.assertEqual(imported['links'][0]['target'][1:], [8,7])
   self.assertEqual(imported['nodes'][0]['mass_msun'],20.)
   request=root/'request.json';out=root/'audit.json'
   from core.data import epoch_contract
   request.write_text(json.dumps({'run_dir':str(root),'native_intake':{'catalogues':paths,'epochs_by_scale_factor':{'0.5':0,'1':7},'hubble_param':.5,'box_comoving_kpc_h':100.,'mass_column':'mvir','memory_gib':.1},
    'thresholds':{'epochs':[0,7],'maximum_mass_ratio':3.,'maximum_position_jump_proper_kpc':5.,'box_proper_kpc':200.,'transient_maximum_detections':1},'epoch_contract':epoch_contract(1.)}))
   subprocess.run([sys.executable,str(test_finder_adapters_ROOT/'commands.py'),'audit-tree-integrity', str(request),str(out),'--from-consistent-trees'],check=True,capture_output=True)
   result=json.loads(out.read_text());self.assertTrue(any(f['kind']=='MASS_JUMP' for f in result['flags']))

 def test_rockstar_missing_and_truncated_membership_not_measured_empty(self):
  from halos import rockstar_catalogue
  from verification import test_stellar_history_snapshot as snapshot
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);snapshot(root,0,.5,[1],[[0,0,0]]);base=root/'rockstar';base.mkdir()
   table={key:np.array([value]) for key,value in {'pid':-1,'mvir':10.,'x':.01,'y':0.,'z':0.,'vx':0.,'vy':0.,'vz':0.}.items()}
   table['_canonical_identity_links']={'id':{'values':np.array([7],dtype=np.uint64),'available':np.array([True])}}
   options=dict(particle_namespace='control:dm',memory_gib=.1,selected_ids=[7],mass_definition='mvir')
   with patch('core.data.rockstar_directory',return_value=base),patch('analysis.rockstar_catalogue_index',return_value=0),patch('analysis.rockstar_host_table',return_value=table):
    self.assertIsNone(rockstar_catalogue(root,0,**options)['objects'][0]['members'])
    dump=base/'halos_0.0.particles'
    dump.write_text('#Halo table begins here:\n#7 0 1 '+'0 '*17+'\n#Particle table begins here:\n')
    with self.assertRaisesRegex(ValueError,'Truncated'):rockstar_catalogue(root,0,**options)
    dump.write_text(dump.read_text()+'0 0 0 0 0 0 1 0 0 7\n')
    self.assertEqual(rockstar_catalogue(root,0,**options)['objects'][0]['members'],[1])
    catalogue=rockstar_catalogue(root,0,**options)
    self.assertIn('NO_NATIVE_UNBINDING',catalogue['objects'][0]['quality'])
    from halos import graph_from_catalogues,IdentityQueries
    graph,sources=graph_from_catalogues([catalogue],maximum_nodes=8,maximum_edges=8)
    graph.publish(root/'assigned-graph',source_hashes=sources)
    query=IdentityQueries(root/'assigned-graph')
    self.assertEqual(query.bound_owner(('control:dm',1,0))['status'],'OWNERSHIP_UNAVAILABLE')
    self.assertEqual(query.members_at_epoch((catalogue['namespace'],7,0),particle_namespace='control:dm').tolist(),[1])

# ========================================================================
# test_scene_product_publication
sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
import core.foundation as durable
from blender import publish_scene_product, recover_scene_publication


class ScenePublicationTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'; source.mkdir()
        payload = source / 'positions.fixture'; payload.write_bytes(b'synthetic transport fixture')
        manifest = source / 'manifest.json'
        durable.atomic_json(manifest, {'schema': 'synthetic-scene-input-v1',
            'publication_contract': 'synthetic-transport-fixture',
            'artifacts': [{'path': payload.name, 'sha256': durable.file_sha256(payload)}]})
        durable.atomic_text(manifest.with_name(manifest.name + '.sha256'), durable.file_sha256(manifest) + '\n')
        return manifest, root / 'viewer.blend'

    def builder(self, source, target, render):
        target.write_bytes(b'fake blend transport bytes; not a valid Blender scene')
        if render: target.with_suffix('.png').write_bytes(b'fake preview transport bytes; not a PNG')

    def publish(self, source, target, builder=None, render=True):
        return publish_scene_product(source, target, builder or self.builder,
            input_schema='synthetic-scene-input-v1', algorithm=__file__, render=render)

    def test_complete_immutable_and_corrupt_public_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp).resolve())
            self.publish(source, target)
            self.assertEqual(verify_scene_publication(target)['state'], 'COMPLETE')
            with self.assertRaises(FileExistsError): self.publish(source, target)
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'alias changed'): verify_scene_publication(target)

    def test_interrupted_alias_copy_replays_without_builder(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp).resolve())
            original = durable.atomic_copy
            calls = []
            def builder(*args):
                calls.append(args)
                self.builder(*args)
            def interrupted(src, dst):
                if Path(dst) == target: raise KeyboardInterrupt
                return original(src, dst)
            with patch('core.foundation.atomic_copy', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt): self.publish(source, target, builder)
            with self.assertRaisesRegex(ValueError, 'incomplete'): verify_scene_publication(target)
            pointer = target.with_name(target.name + '.publication.json')
            archive = target.parent / json.loads(pointer.read_text())['archive']
            self.assertTrue(archive.is_dir())
            self.assertFalse((archive / 'failure.json').exists())
            result = recover_scene_publication(target, archive)
            self.assertEqual(result['state'], 'COMPLETE')
            self.assertEqual(len(calls), 1)
            verify_scene_publication(target)

    def test_source_mutation_retains_stage_and_rejects_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source, target = self.fixture(root)
            def changed(src, dst, render):
                self.builder(src, dst, render)
                (source.parent / 'positions.fixture').write_bytes(b'changed source')
            with self.assertRaises(ValueError): self.publish(source, target, changed)
            self.assertFalse(target.exists())
            self.assertTrue(list(root.glob('.scene-stage-*/viewer.blend')))
            self.assertTrue(list(root.glob('.scene-stage-*/failure.json')))

    def test_multiple_nested_render_products_publish_and_verify(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp).resolve())
            def builder(src, dst, render):
                self.builder(src, dst, False)
                (dst.parent / 'movie.mp4').write_bytes(b'fake movie transport bytes')
                previews = dst.parent / 'section_previews'; previews.mkdir()
                (previews / 'section_001.png').write_bytes(b'fake section preview')
            publish_scene_product(source, target, builder, input_schema='synthetic-scene-input-v1',
                algorithm=__file__, render=True, render_artifacts=['movie.mp4', 'section_previews/section_001.png'])
            record = verify_scene_publication(target)
            self.assertIn('section_previews/section_001.png', record['aliases'])
            self.assertTrue((target.parent / 'movie.mp4').is_file())
            (target.parent / 'section_previews/section_001.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'alias changed'): verify_scene_publication(target)

    def test_missing_requested_movie_never_publishes(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = self.fixture(Path(temp).resolve())
            with self.assertRaises(FileNotFoundError):
                publish_scene_product(source, target, self.builder,
                    input_schema='synthetic-scene-input-v1', algorithm=__file__,
                    render=True, render_artifacts=['missing.mp4'])
            self.assertFalse(target.exists())

# ========================================================================
# test_zoom_target_coverage

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from zoom import coarse_intersections, audit_target_coverage


class CoverageTests(unittest.TestCase):
    def test_periodic_cube_footprints_and_variable_widths(self):
        # A corner overlap crosses the periodic boundary on every axis. Its
        # Euclidean separation exceeds the half-width sum, so a sphere would
        # miss a real cube intersection.
        points = np.array([[0.01, 0.01, 0.01]])
        centers = np.array([[0.96, 0.96, 0.96], [0.5, 0.5, 0.5], [0.2, 0.01, 0.01]])
        touched = coarse_intersections(
            points, centers, np.array([0.1, 0.1, 0.4]), box=1, guard=0.01
        )
        np.testing.assert_array_equal(touched, [True, False, True])

    def test_invalid_widths_and_unwrapped_q_rejected(self):
        points = np.array([[0.1, 0.1, 0.1]])
        with self.assertRaises(ValueError):
            coarse_intersections(points, points, np.array([-0.1]), box=1, guard=0.01)
        with self.assertRaises(ValueError):
            coarse_intersections(points + 1, points, np.array([0.1]), box=1, guard=0.01)

    def test_bound_galaxy_metadata_is_not_mistaken_for_traced_q(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            particles=root/'selected_particles.hdf5'
            with h5py.File(particles,'w') as source:
                source.create_group('PartType1')['LagrangianCoordinates']=np.array([[.01,.01,.01]],dtype=np.float64)
                source.create_group('SelectedGalaxyBoundMembers').create_group('PartType4')['ParticleIDs']=np.array([99],dtype=np.uint64)
            region=root/'region.json'
            region.write_text(json.dumps({'status':'TRACED_PROPOSAL',
                'selected_particles_sha256':file_sha256(particles),'selected_counts':{'1':1,'2':0},
                'guard':{'parent_cell_comoving_kpc_h':.01},'box_comoving_kpc_h':1.}))
            ic=root/'ics_gadget.hdf5'
            with h5py.File(ic,'w') as source:
                source.create_group('Header').attrs.update(BoxSize=1.,Music_lagrangian_schema=1,
                    Music_lagrangian_unitlength_cm=KPC_CM,Music_levelmax=2,Music_levelmin=1,
                    MassTable=np.array([.2,.8,8.,0,0,0]),OmegaBaryon=.06,Omega0=.3)
                coarse=source.create_group('PartType2')
                # Undoing COM staggering puts this coarse cell at .5 on
                # each axis; it cannot overlap the selected .01 guard cube.
                coarse['LagrangianCoordinates']=np.array([[.525,.525,.525]],dtype=np.float64)
                coarse['ParticleIDs']=np.array([2**63+1],dtype=np.uint64)
            for name in ('music.conf','transfers.txt','generation.log'):
                (root/name).write_text('transport fixture')
            generation=root/'generation-structural.json'
            generation.write_text(json.dumps({'status':'GENERATED_STRUCTURAL_PASS',
                'traced_region':{'sha256':file_sha256(region)},'artifacts':{name:file_sha256(root/name) for name in ('ics_gadget.hdf5','music.conf','transfers.txt','generation.log')}}))
            report=audit_target_coverage(generation,region,root/'coverage.json')
            self.assertEqual(report['status'],'PASS')
            self.assertEqual(report['selected_parent_count'],1)
            with h5py.File(ic,'r+') as source:
                source['Header'].attrs.update(Music_mixed_fluid_schema=1,
                    Music_mean_species_masses_native=[.2,.8,0,0,0,0],
                    MassTable=[0,0,8.,0,0,0])
            record=json.loads(generation.read_text())
            record['artifacts']['ics_gadget.hdf5']=file_sha256(ic)
            generation.write_text(json.dumps(record))
            self.assertEqual(audit_target_coverage(generation,region,root/'mixed-coverage.json')['status'],'PASS')

# ========================================================================
# test_subhalo_publication
sys.path[:0]=[str(Path(__file__).resolve().parent/'core'),str(Path(__file__).resolve().parent)]
from halos import verify_subhalo_export


class SubhaloPublicationTests(unittest.TestCase):
    def test_both_representations_preserve_uint64_and_optional_missingness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);catalogue_fixture(root)
            result=export_subhalo_catalogue(root,0)
            directory=(root/'results'/result['artifact']).parent
            report=verify_subhalo_export(directory)
            with (root/'results'/result['artifact']).open() as source:rows=list(csv.DictReader(source))
            self.assertEqual(rows[0]['most_bound_particle_id'],str(2**63+1))
            self.assertEqual(rows[0]['sfr_msun_year'],'')
            with np.load(directory/'columns.npz',allow_pickle=False) as columns:
                self.assertEqual(int(columns['most_bound_particle_id'][0]),2**63+1)
                self.assertFalse(columns['sfr_msun_year_available'].any())
                self.assertTrue(np.isnan(columns['sfr_msun_year']).all())
            self.assertEqual(report['array_contract']['datasets']['half_mass_radius_proper_kpc']['units'],'proper kpc')
            self.assertIn(':snapshot:0',report['array_contract']['datasets']['subhalo_row']['identity_namespace'])
            with self.assertRaises(FileExistsError):export_subhalo_catalogue(root,0)
            with (directory/Path(result['artifact']).name).open('a') as source:source.write('changed')
            with self.assertRaisesRegex(ValueError,'CSV checksum'):verify_subhalo_export(directory)

    def test_absent_most_bound_id_retains_uint64_with_false_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,_=catalogue_fixture(root)
            with h5py.File(path,'r+') as source:del source['Subhalo/SubhaloIDMostbound']
            result=export_subhalo_catalogue(root,0)
            with np.load(root/'results'/result['numeric_artifact'],allow_pickle=False) as columns:
                self.assertEqual(columns['most_bound_particle_id'].dtype,np.dtype('uint64'))
                self.assertFalse(columns['most_bound_particle_id_available'].any())

    def test_interrupted_publication_retains_complete_stage_and_prior_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);catalogue_fixture(root)
            output=root/'results';output.mkdir();prior=output/'subhalos_000.csv';prior.write_bytes(b'legacy completed export')
            with patch('core.foundation.publish_directory',side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):export_subhalo_catalogue(root,0)
            self.assertEqual(prior.read_bytes(),b'legacy completed export')
            self.assertFalse((output/'subhalos_000').exists())
            stage=list(output.glob('.subhalos-*'))[0]
            self.assertTrue((stage/'export_failure.json').is_file());verify_subhalo_export(stage)
            export_subhalo_catalogue(root,0)
            self.assertTrue(stage.is_dir());self.assertEqual(prior.read_bytes(),b'legacy completed export')
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.data import recover_native_checkpoints

class RecoveryTests(unittest.TestCase):
    def candidate(self,root,rank,generation,prefix='restart'):
        p=root/f'{prefix}.{rank}';p.write_bytes(f'{generation}:{rank}:payload'.encode()*1024)
        Path(str(p)+'.checksum.json').write_text(json.dumps({'schema':'gadget-rank-checkpoint-v1','rank':rank,'ranks':2,'generation':generation,'a':.2,'size_bytes':p.stat().st_size,'crc32':f'{zlib.crc32(p.read_bytes()):08x}'}))
    def mixed(self,root):
        self.candidate(root,0,20);self.candidate(root,0,10,'bak-restart')
        self.candidate(root,1,10);self.candidate(root,1,9,'bak-restart')
    def test_select_coherent_generation_preserve_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.mixed(root);before={p.name:p.read_bytes() for p in root.iterdir()}
            result=recover_native_checkpoints(root,2)
            self.assertEqual(result['generation'],10);self.assertEqual(result['status'],'COMPLETE')
            for name,data in before.items():self.assertEqual((root/result['archive']/'original'/name).read_bytes(),data)
            self.assertEqual(verify_native_checkpoints(root,2)['files'][0]['generation'],10)
            self.assertEqual(recover_native_checkpoints(root,2)['status'],'ALREADY_VERIFIED')
    def test_no_coherent_generation_preserves_all_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.candidate(root,0,10);self.candidate(root,1,20)
            before={p.name:p.read_bytes() for p in root.iterdir()}
            with self.assertRaisesRegex(ValueError,'No complete coherent'):recover_native_checkpoints(root,2)
            self.assertEqual({p.name:p.read_bytes() for p in root.iterdir()},before)
    def test_restore_interruption_replays_immutable_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.mixed(root);original=acceptance.atomic_copy
            def interrupted(source,destination):
                result=original(source,destination)
                if Path(destination)==root/'restart.0':raise KeyboardInterrupt('after first rank payload')
                return result
            with patch('core.foundation.atomic_copy',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):recover_native_checkpoints(root,2)
            journal=json.loads((root/'checkpoint_recovery.json').read_text());self.assertEqual(journal['status'],'PENDING')
            result=recover_native_checkpoints(root,2);self.assertEqual(result['archive'],journal['archive']);self.assertEqual(result['status'],'COMPLETE')
    def test_pending_archive_corruption_fails_before_further_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.mixed(root);original=acceptance.atomic_copy
            def interrupted(source,destination):
                if Path(destination)==root/'restart.0':raise KeyboardInterrupt()
                return original(source,destination)
            with patch('core.foundation.atomic_copy',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):recover_native_checkpoints(root,2)
            record=json.loads((root/'checkpoint_recovery.json').read_text());(root/record['archive']/'selected/restart.0').write_bytes(b'corrupt archive')
            before=(root/'restart.0').read_bytes()
            with self.assertRaisesRegex(ValueError,'archived bytes changed'):recover_native_checkpoints(root,2)
            self.assertEqual((root/'restart.0').read_bytes(),before)
    def test_real_sigkill_then_pending_transaction_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.mixed(root)
            script="""
import sys,os,signal
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import core.foundation as research_acceptance
from core.data import recover_native_checkpoints
original=research_acceptance.atomic_copy
root=Path(sys.argv[2])
def copying(source,destination):
    result=original(source,destination)
    if Path(destination)==root/'restart.0':os.kill(os.getpid(),signal.SIGKILL)
    return result
research_acceptance.atomic_copy=copying
recover_native_checkpoints(root,2)
"""
            result=subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve().parent/'core'),str(root)],capture_output=True,timeout=30)
            self.assertEqual(result.returncode,-9,result.stderr.decode())
            self.assertEqual(json.loads((root/'checkpoint_recovery.json').read_text())['status'],'PENDING')
            self.assertEqual(recover_native_checkpoints(root,2)['status'],'COMPLETE')
    def test_disk_full_archive_does_not_replace_current_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.mixed(root);before=(root/'restart.0').read_bytes();original=os.fsync;calls=0
            def syncing(fd):
                nonlocal calls
                calls+=1
                if calls==1:raise OSError(errno.ENOSPC,'full')
                return original(fd)
            # Acquire the owner before injecting disk-full into the archive write.
            from core.native import native_run_lock
            with native_run_lock(root), patch('core.foundation.os.fsync',side_effect=syncing):
                with self.assertRaises(OSError):recover_native_checkpoints(root,2)
            self.assertEqual((root/'restart.0').read_bytes(),before)
            self.assertTrue(list(root.glob('.checkpoint-recovery-*/recovery_failure.json')))

# ========================================================================
# test_zoom_resources

sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))
from zoom import estimate_mesh_resources, plan_native_music


def test_zoom_resources_fixture():
    return {'schema':'darkmattersim-music-mesh-v1','scalar_bytes':8,'boundary_cells':4,
            'convolution_margin':4,'hierarchies':{
                'poisson':[{'level':4,'shape':[16]*3,'offset':[0]*3},
                           {'level':5,'shape':[8]*3,'offset':[12]*3}],
                'convolution':[{'level':4,'shape':[16]*3,'offset':[0]*3},
                               {'level':5,'shape':[16]*3,'offset':[8]*3}]}}


class ZoomResourceTests(unittest.TestCase):
    def test_seed_stream_ranges_include_cube_indices_and_32bit_limit(self):
        from zoom import _music_seed_stream_ranges
        self.assertEqual(_music_seed_stream_ranges({'5':421,'6':10421,'7':20421})[-1],
            {'level':7,'first_seed':20421,'last_seed':20484,'raw_cube_count':64})
        with self.assertRaisesRegex(ValueError,'reuse raw Gaussian'):
            _music_seed_stream_ranges({'6':422,'7':423})
        with self.assertRaisesRegex(ValueError,'32-bit'):
            _music_seed_stream_ranges({'16':1})
        self.assertEqual(_music_seed_stream_ranges({'6':422,'7':423},noise_levels={6,7}),[])

    def test_frozen_full_grid_budget_checked_before_native_or_file_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg=root/'music.conf'
            cfg.write_text('[setup]\nlevelmin=5\nlevelmax=13\n[random]\n'+
                           ''.join(f'seed[{level}]=missing{level}.grafic\n' for level in range(5,14)))
            with patch('zoom.run_native_command') as native:
                with self.assertRaisesRegex(MemoryError,'Frozen full noise hierarchy'):
                    plan_native_music(cfg,root/'missing-binary',root/'plan',memory_gib=96.)
                native.assert_not_called()
            self.assertFalse((root/'plan').exists())

    def test_fft_padding_and_multigrid_storage_are_counted(self):
        result = estimate_mesh_resources(test_zoom_resources_fixture())
        self.assertEqual(result['single_convolution_field_bytes'], 8*(16*16*18 + 24*24*26))
        expected = 8*(24**3 + 16**3 + sum((2**i+8)**3 for i in range(4)))
        self.assertEqual(result['single_poisson_field_bytes'], expected)
        self.assertEqual(result['particle_count_upper_bound'], 2*(16**3+8**3))
        self.assertIsNone(result['estimated_wall_seconds'])
        self.assertEqual(result['time_status'], 'NOT_CALIBRATED')
        doubled = copy.deepcopy(test_zoom_resources_fixture())
        doubled['convolution_margin'] = -1
        self.assertGreater(estimate_mesh_resources(doubled)['single_convolution_field_bytes'],
                           result['single_convolution_field_bytes'])

    def test_invalid_or_incomplete_native_geometry_rejected(self):
        for mutate in (lambda p:p.update(scalar_bytes=7),
                       lambda p:p['hierarchies']['poisson'][0].update(shape=[8]*3),
                       lambda p:p['hierarchies']['poisson'][1].update(level=4),
                       lambda p:p['hierarchies']['convolution'][1].update(shape=[True]*3)):
            plan=test_zoom_resources_fixture(); mutate(plan)
            with self.assertRaises(ValueError): estimate_mesh_resources(plan)

    def test_memory_disk_and_unsupported_binary_fail_before_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cfg,binary=root/'music.conf',root/'MUSIC'
            cfg.write_text('[setup]\nlevelmin=4\nlevelmax=5\nbaryons=yes\n[output]\nformat=arepo\n[random]\nseed[4]=421\nseed[5]=10421\n'); binary.write_bytes(b'not executed')
            def planner(command,*,cwd,**kwargs):
                self.assertEqual(command,[str(binary.resolve()),'planner.conf','--plan-mesh','native-mesh.json'])
                (cwd/'native-mesh.json').write_text(json.dumps(test_zoom_resources_fixture()))
                return subprocess.CompletedProcess(command,0)
            with patch('zoom.run_native_command',side_effect=planner):
                for name,memory,disk in [('memory',.001,None),('disk',1.,.001)]:
                    with self.assertRaises(MemoryError):
                        plan_native_music(cfg,binary,root/name,memory_gib=memory,disk_gib=disk)
                    report=json.loads((root/name/'resource-plan.json').read_text())
                    self.assertEqual(report['status'],'FAIL')
                    self.assertIn(name,report['reason'])
                result=plan_native_music(cfg,binary,root/'valid',memory_gib=1.)
                self.assertEqual(result['status'],'PASS')
                with self.assertRaises(FileExistsError):
                    plan_native_music(cfg,binary,root/'valid',memory_gib=1.)
            with patch('zoom.run_native_command',return_value=subprocess.CompletedProcess([],0)):
                with self.assertRaisesRegex(RuntimeError,'cannot provide'):
                    plan_native_music(cfg,binary,root/'unsupported',memory_gib=1.)

# ========================================================================
# test_black_hole_analysis

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from black_holes import analyze_black_holes, measured_epoch, NoBlackHoleData


def test_black_hole_analysis_fixture(root):
    settings = dict(
        enabled=True,
        radiative_efficiency=0.1,
        feedback_coupling=0.05,
        bondi_boost=1.0,
        eddington_factor=1.0,
        kernel_radius_proper_kpc=1.0,
        seed_mass_msun=1e5,
        source_fraction=0.05,
        maximum_count=100,
        seeding_mode="ic_particles",
        seed_density_hydrogen_cm3=0.0,
        seed_exclusion_proper_kpc=0.0,
    )
    (root / "config.yaml").write_text(yaml.safe_dump({"black_holes": settings}))
    for number, a in enumerate((0.5, 1.0)):
        test_stellar_history_snapshot(root, number, a, [2**63 + 1], [[5.0, 6.0, 7.0]])
        with h5py.File(root / "data" / f"snapshot_{number:03d}.hdf5", "a") as f:
            f.move("PartType4", "PartType5")
            for name in ("NumPart_Total", "NumPart_ThisFile"):
                f["Header"].attrs[name] = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint64)
            f["Parameters"].attrs.update(native_parameters(settings))
            bh = f["PartType5"]
            for name, value, cgs, hs in (
                ("BH_Mass", 0.5 + 0.1 * number, 1e10 * MSUN_G, -1.0),
                ("BH_InflowRate", 0.01, 1e10 * MSUN_G / (KPC_CM / 1e5), 0.0),
                ("BH_RadiatedMass", 0.01 + 0.01 * number, 1e10 * MSUN_G, -1.0),
                ("BH_FeedbackEnergy", 1e-6 * (number + 1), 1e10 * MSUN_G * 1e10, -1.0),
            ):
                bh[name] = [value]
                bh[name].attrs.update(to_cgs=cgs, a_scaling=0.0, h_scaling=hs)
    return settings


class BlackHoleAnalysisTests(unittest.TestCase):
    def test_no_formed_bhs_is_distinct_from_invalid_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test_black_hole_analysis_fixture(root)
            for path in (root / "data").glob("snapshot_*.hdf5"):
                with h5py.File(path, "r+") as f:
                    for field in ("NumPart_Total", "NumPart_ThisFile"):
                        counts = f["Header"].attrs[field]
                        counts[5] = 0
                        f["Header"].attrs[field] = counts
                    del f["PartType5"]
            with self.assertRaises(NoBlackHoleData):
                analyze_black_holes(root)
            self.assertFalse((root / "results/black_holes").exists())

    def test_physical_masses_luminosity_and_exact_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test_black_hole_analysis_fixture(root)
            result = analyze_black_holes(root)
            record = json.loads(
                (Path(result["directory"]) / "provenance.json").read_text()
            )
            first = record["history"][0]
            self.assertEqual(first["particle_id"], 2**63 + 1)
            self.assertEqual(first["subgrid_mass_msun"], 1e10)
            self.assertEqual(first["gravity_mass_msun"], 2e10)
            expected = 0.1 * 0.01 * 1e10 * MSUN_G / (KPC_CM / 1e5) * (C_SI * 100) ** 2
            self.assertAlmostEqual(
                first["bolometric_luminosity_erg_s"] / expected, 1.0, places=12
            )
            self.assertTrue(Path(result["figure"]).is_file())
            with h5py.File(result["catalogue"]) as output:
                metadata = json.loads(output["ProvenanceJSON"][()].decode())
                self.assertEqual(
                    metadata["schema"], "darkmattersim-native-bh-history-v1"
                )
                self.assertNotIn("history", metadata)
            with self.assertRaises(FileExistsError):
                analyze_black_holes(root)

    def test_history_budget_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test_black_hole_analysis_fixture(root)
            with self.assertRaises(MemoryError):
                analyze_black_holes(root, memory_gib=0.01)
            self.assertFalse((root / "results/black_holes").exists())

    def test_boundary_markers_and_wrong_runtime_model_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = test_black_hole_analysis_fixture(root)
            path = root / "data/snapshot_000.hdf5"
            with h5py.File(path, "a") as f:
                del f["PartType5/BH_Mass"]
            with self.assertRaisesRegex(ValueError, "native field"):
                measured_epoch([path], 0, settings)
            with h5py.File(path, "a") as f:
                f["Parameters"].attrs["BHBondiBoost"] = 10.0
            with self.assertRaisesRegex(ValueError, "runtime choices"):
                measured_epoch([path], 0, settings)

    def test_units_and_unexplained_mass_loss_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = test_black_hole_analysis_fixture(root)
            path = root / "data/snapshot_001.hdf5"
            with h5py.File(path, "a") as f:
                f["PartType5/BH_InflowRate"].attrs["h_scaling"] = -1.0
            with self.assertRaisesRegex(ValueError, "unit convention"):
                measured_epoch([path], 1, settings)
            with h5py.File(path, "a") as f:
                f["PartType5/BH_InflowRate"].attrs["h_scaling"] = 0.0
                f["PartType5/BH_Mass"][0] = 0.4
            with self.assertRaisesRegex(ValueError, "decreases"):
                analyze_black_holes(root)
            self.assertFalse((root / "results/black_holes").exists())

# ========================================================================
# test_history_publication
sys.path[:0]=[str(Path(__file__).resolve().parent/'core'),str(Path(__file__).resolve().parent)]


class HistoryPublicationTests(unittest.TestCase):
    def test_stellar_failure_retains_sealed_numeric_stage_and_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);catalogue,ids=StellarHistoryTests().fixture(root)
            args=dict(catalogue=catalogue,snapshots=[0,1],candidate_ids=[1])
            with patch('stars.history_figure',side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):trace_stellar_candidates(root,**args)
            parent=root/'results';stages=list(parent.glob('.stellar-history-*'))
            self.assertEqual(len(stages),1)
            self.assertFalse((parent/'stellar_histories').exists())
            self.assertTrue((stages[0]/'export_failure.json').is_file())
            verify_hdf5_dataset_payloads(stages[0]/'history.hdf5',require_contract=True)
            with h5py.File(stages[0]/'history.hdf5') as source:
                self.assertEqual(source['ParticleIDs'].dtype.name,'uint64')
                self.assertEqual(source['Epochs/000/Coordinates'].attrs['units'],'comoving kpc')
                self.assertIn('epoch_json',source['Epochs/000/Coordinates'].attrs)
                for mask in source['Transitions/000_001'].values():
                    self.assertTrue(mask.fletcher32)
                    self.assertEqual(mask.attrs['quality_status'],'NOT_ACCEPTED')
            with patch('stars.history_figure'),patch('stars.plot_stream_diagnostics'):
                result=trace_stellar_candidates(root,**args)
            self.assertTrue(Path(result['catalogue']).is_file());self.assertTrue(stages[0].is_dir())

    def test_black_hole_failure_retains_stage_with_exact_identity_and_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);test_black_hole_analysis_fixture(root)
            with patch('black_holes.bh_figure',side_effect=RuntimeError('renderer unavailable')):
                with self.assertRaisesRegex(RuntimeError,'renderer unavailable'):analyze_black_holes(root)
            parent=root/'results';stage=list(parent.glob('.bh-analysis-*'))[0]
            self.assertFalse((parent/'black_holes').exists())
            self.assertTrue((stage/'export_failure.json').is_file())
            verify_hdf5_dataset_payloads(stage/'history.hdf5',require_contract=True)
            with h5py.File(stage/'history.hdf5') as source:
                self.assertEqual(int(source['Epochs/000/ParticleIDs'][0]),2**63+1)
                self.assertIn('identity_namespace',source['Epochs/000/ParticleIDs'].attrs)
                self.assertIn('epoch_json',source['Epochs/000/BH_Mass'].attrs)
                self.assertEqual(source['Epochs/000/Coordinates'].attrs['coordinates_status'],'PRODUCER_DECLARED')
            report=json.loads((stage/'provenance.json').read_text())
            self.assertIn('history_sha256',report)
            with patch('black_holes.bh_figure'):analyze_black_holes(root)
            self.assertTrue(stage.is_dir())

# ========================================================================
# test_zoom_handoff
from core.data import export_numeric_audit

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
import verification as test_zoom_handoff_fixtures
from zoom import verify_native_handoff, _configuration
from evolution import validate_ic_hdf5


class HandoffTests(unittest.TestCase):
    def setUp(self):
        conservation = patch('initial_conditions.validate_ic_realization',return_value={'status':'INCOMPLETE'})
        self.conservation = conservation.start()
        self.addCleanup(conservation.stop)
        fixture=test_zoom_handoff_fixtures.StagingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.stage()
        self.fixture=fixture
        self.root=fixture.root
        self.native=self.root/'native'
        self.native.mkdir()
        self.destination=self.root/'audit'
        source=fixture.destination/'parent.conf'
        native_cfg=_configuration(source)
        native_cfg.set('cosmology','transfer_file','transfers.txt')
        with (self.native/'music.conf').open('w') as stream:
            native_cfg.write(stream)
        shutil.copy2(fixture.destination/'transfers.txt',self.native/'transfers.txt')
        self.ic=self.native/'ics_gadget.hdf5'
        self.ic.write_bytes(b'IC payload mocked only in these source/recipe boundary tests')
        (self.native/'generation.log').write_text('fixture')
        binary=self.root/'MUSIC'
        binary.write_bytes(b'not executed in unit tests')
        proof=self.root/'proof.json'
        controls={'reject-second-order','reject-backscaling','reject-gauge','reject-model',
                  'reject-redshift','reject-mode-coverage','reject-poisson'}
        proof.write_text(json.dumps({'status':'PASS','nested':{'status':'PASS'},'periodic':{'status':'PASS'},
            'regression_driver_sha256':file_sha256(Path(__file__).parent / 'core/research/generator_controls.py'),
            'execution':{'binary_sha256':file_sha256(binary),'rejections':{key:{'exit_code':1} for key in controls}}}))
        record=lambda path:{'path':str(path),'sha256':file_sha256(path)}
        self.generation=self.native/'generation.json'
        self.info={'status':'GENERATED_STRUCTURAL_PASS','source_configuration':record(source),
            'source_transfer_table':record(fixture.destination/'transfers.txt'),'binary':record(binary),
            'native_validation':record(proof),'traced_region':None,
            'artifacts':{name:file_sha256(self.native/name) for name in ('music.conf','transfers.txt','ics_gadget.hdf5','generation.log')}}
        self.save()
        self.cfg={'spectrum':{'source':'modular','file':str(fixture.manifest)},
            'cosmology':{'model':'lcdm','H0':70.,'omega_b':.05*.7**2,'omega_cdm':.25*.7**2,
                        'A_s':2.1e-9,'n_s':.96,'T_cmb':2.7255,'N_ur':3.046},
            'simulation':{'physics':'hydro','lpt_order_hydro':1,'start_redshift':99.,
                          'particles_per_dimension':32,'box_mpc_h':10.}}

    def save(self):
        self.generation.write_text(json.dumps(self.info))

    def audit(self):
        return verify_native_handoff(self.ic,self.generation,self.cfg,self.destination)

    def mock_velocity_measurement(self, status):
        def measure(ic, table, destination, **kwargs):
            return export_numeric_audit(destination,{'status':status},
                {'fixture':np.array([1.])},{'fixture':{'units':'dimensionless','quantity':'mocked transport measurement'}},
                algorithm=__file__,context={'measurement_scope':'MOCKED_NOT_NATIVE'},source_hashes={})
        return measure

    def test_retained_pass_does_not_skip_delivered_measurements(self):
        with (patch('zoom.audit_music_ic',return_value={'status':'STRUCTURAL_PASS'}) as structural,
              patch('zoom.initial_condition_census',return_value={'particles_by_type':[32**3,32**3,0,0,0,0]}) as census,
              patch('zoom.audit_zoom_velocity',side_effect=self.mock_velocity_measurement('PASS')) as velocity):
            result=self.audit()
        structural.assert_called_once()
        self.conservation.assert_called_once()
        census.assert_called_once()
        velocity.assert_called_once()
        self.assertEqual(result['status'],'TECHNICAL_PASS')
        self.assertEqual(result['science_acceptance'],'NOT_ACCEPTED')
        self.assertEqual(json.loads((self.destination/'handoff.json').read_text())['status'],'TECHNICAL_PASS')
        with self.assertRaises(FileExistsError):
            self.audit()

    def test_wrong_requested_cosmology_or_changed_ic_stops_before_measurement(self):
        self.cfg['cosmology']['A_s']=3e-9
        with patch('zoom.audit_music_ic') as structural:
            with self.assertRaisesRegex(ValueError,'mismatch: A_s'):
                self.audit()
            self.assertFalse(self.destination.exists())
            self.cfg['cosmology']['A_s']=2.1e-9
            self.ic.write_bytes(b'changed IC')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                self.audit()
            structural.assert_not_called()

    def test_consistently_rehashed_different_seed_is_rejected_by_reference_recipe(self):
        path=self.native/'music.conf'
        path.write_text(path.read_text().replace('seed[5] = 150','seed[5] = 151'))
        self.info['artifacts']['music.conf']=file_sha256(path)
        self.save()
        with self.assertRaisesRegex(ValueError,'differs from regenerated'):
            self.audit()
        self.assertEqual(json.loads((self.destination/'handoff.json').read_text())['status'],'FAILED')

    def test_failed_remeasurement_cannot_reuse_saved_generation_pass(self):
        with (patch('zoom.audit_music_ic',return_value={'status':'STRUCTURAL_PASS'}),
              patch('zoom.initial_condition_census',return_value={'particles_by_type':[32**3,32**3,0,0,0,0]}),
              patch('zoom.audit_zoom_velocity',side_effect=self.mock_velocity_measurement('FAIL'))):
            with self.assertRaisesRegex(ValueError,'velocity audit failed'):
                self.audit()
        self.assertEqual(json.loads((self.destination/'handoff.json').read_text())['status'],'FAILED')

    def test_failed_conservation_cannot_be_hidden_by_structural_pass(self):
        self.conservation.return_value={'status':'FAIL'}
        with patch('zoom.audit_music_ic',return_value={'status':'STRUCTURAL_PASS'}):
            with self.assertRaisesRegex(ValueError,'conservation audit failed'):
                self.audit()
        self.assertEqual(json.loads((self.destination/'handoff.json').read_text())['status'],'FAILED')

    def test_simulation_dispatches_native_ic_to_retained_validation(self):
        self.cfg['simulation'].update(ic_engine_hydro='native_music',native_generation_report=str(self.generation))
        report={'resource_census':{'particles_by_type':[2,2,3,0,0,0]},'velocity_audit':{'status':'PASS'},
                'structural_audit':{'status':'STRUCTURAL_PASS'}}
        self.destination.mkdir()
        (self.destination/'handoff.json').write_text(json.dumps(report))
        with (patch('evolution.initial_condition_files',return_value=[self.ic]),
              patch('zoom.verify_native_handoff',return_value=report) as verifier):
            result=validate_ic_hdf5(self.ic,self.cfg,native_validation_dir=self.destination)
        verifier.assert_called_once()
        self.assertEqual(result['particles_by_type'][2],3)
        self.assertEqual(result['science_acceptance'],'NOT_ACCEPTED')
        with patch('evolution.initial_condition_files',return_value=[self.ic]):
            with self.assertRaisesRegex(ValueError,'retained audit destination'):
                validate_ic_hdf5(self.ic,self.cfg)

# ========================================================================
# test_map_publication

test_map_publication_ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(test_map_publication_ROOT/'core'))
from visualization import make_maps


def map_fixture(root):
    test_blender_data_contract_fixture(root)
    config=yaml.safe_load((test_map_publication_ROOT/'core/config.yaml').read_text())
    config['simulation'].update(box_mpc_h=.1,particles_per_dimension=2,physics='dm')
    config['cosmology'].update(H0=50.,omega_b=.0125,omega_cdm=.0625,model='lcdm')
    config['analysis']['density_grid']=8
    config['visualization'].update(structure_map_overlays=[],structure_story_figure=False,
        slice_mode='full_box',slice_thickness_mpc_h=None,slice_center_mpc_h=None,
        figure_dpi=60,map_window_mpc_h=None,filament_grid_size=8)
    (root/'config.yaml').write_text(yaml.safe_dump(config))


class MapPublicationTests(unittest.TestCase):
    def test_public_numeric_map_conserves_native_mass_with_correct_axes_and_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);map_fixture(root)
            native=root/'data/snapshot_001.hdf5';before=digest(native)
            made=make_maps(root,['dm'],['xy'],story_figure=False)
            latest=next(path for path in made if path.name.endswith('_001.png'))
            with h5py.File(latest.with_suffix('.hdf5'),'r') as numeric:
                measured=numeric['measured'][:]
                self.assertAlmostEqual(float(measured.sum())*(.1/8)**2,3.e10,delta=.02)
                self.assertEqual(numeric['measured'].attrs['units'],'h Msun / Mpc^2')
                self.assertEqual(numeric['measured'].attrs['quality_status'],'NOT_ACCEPTED')
                epoch=json.loads(numeric['measured'].attrs['epoch_json'])
                self.assertEqual(epoch['actual_scale_factor'],1.)
                axes=json.loads(numeric.attrs['pixel_axes_json'])
                self.assertEqual((axes['rows'],axes['columns']),(1,0))
                positive=measured>0
                np.testing.assert_allclose(numeric['log10_display'][:][positive],np.log10(measured[positive]))
                self.assertTrue(np.isnan(numeric['log10_display'][:][~positive]).all())
            self.assertEqual(digest(native),before)
            manifest=json.loads((latest.parent/'map_contract.json').read_text())
            for item in manifest['artifacts']:self.assertEqual(digest(latest.parent/item['path']),item['sha256'])
            second=make_maps(root,['dm'],['xy'],story_figure=False)
            self.assertNotEqual(second[0].parent,made[0].parent)
            self.assertTrue(latest.is_file())

    def test_failed_map_publication_retains_measurements_and_previous_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);map_fixture(root)
            prior=make_maps(root,['dm'],['xy'],story_figure=False);before=digest(prior[0])
            with patch('core.foundation.publish_directory',side_effect=OSError('injected map publication failure')):
                with self.assertRaisesRegex(OSError,'injected'):
                    make_maps(root,['dm'],['xy'],story_figure=False)
            self.assertEqual(digest(prior[0]),before)
            stage=next((root/'results/structure_maps').glob('.maps-*'))
            self.assertTrue((stage/'export_failure.json').is_file())
            self.assertTrue(list(stage.glob('*.hdf5')))

# ========================================================================
# test_comparison_map_publication

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from visualization import make_comparison_maps


class ComparisonMapPublicationTests(unittest.TestCase):
    def fixtures(self,root):
        first=root/'a';second=root/'b';first.mkdir();second.mkdir()
        map_fixture(first);map_fixture(second)
        return first,second

    def test_identical_runs_retain_equal_measurements_zero_ratio_and_both_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);first,second=self.fixtures(root)
            bundle=make_comparison_maps(first,second,fields=['dm'],projections=['xy'])
            self.assertTrue((bundle/'comparison_contract.json').is_file())
            for path in bundle.glob('compare_*.hdf5'):
                with h5py.File(path,'r') as data:
                    np.testing.assert_array_equal(data['measured_a'][:],data['measured_b'][:])
                    support=data['common_positive_support'][:]
                    self.assertTrue(support.any())
                    np.testing.assert_array_equal(data['log10_ratio_b_over_a'][:][support],0.)
                    self.assertTrue(np.isnan(data['log10_ratio_b_over_a'][:][~support]).all())
                    self.assertEqual(data['measured_a'].attrs['units'],'h Msun / Mpc^2')
                    epochs=json.loads(data['log10_ratio_b_over_a'].attrs['epoch_json'])
                    self.assertEqual(epochs['run_a']['actual_scale_factor'],epochs['run_b']['actual_scale_factor'])
            receipt=json.loads((bundle/'comparison_contract.json').read_text())
            for item in receipt['artifacts']:self.assertEqual(digest(bundle/item['path']),item['sha256'])
            again=make_comparison_maps(first,second,fields=['dm'],projections=['xy'])
            self.assertNotEqual(bundle,again)

    def test_failed_publication_and_native_source_mutation_preserve_previous_bundles(self):
        import visualization as visualize
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);first,second=self.fixtures(root)
            prior=make_comparison_maps(first,second,fields=['dm'],projections=['xy'])
            before=digest(prior/'comparison_contract.json')
            with patch('core.foundation.publish_directory',side_effect=OSError('injected comparison failure')):
                with self.assertRaisesRegex(OSError,'injected'):
                    make_comparison_maps(first,second,fields=['dm'],projections=['xy'])
            self.assertEqual(digest(prior/'comparison_contract.json'),before)
            original=visualize._write_comparison_maps
            def mutate(*args,**kwargs):
                result=original(*args,**kwargs)
                with h5py.File(first/'data/snapshot_001.hdf5','r+') as source:source['PartType1/Coordinates'][0,0]=2.
                return result
            with patch('visualization._write_comparison_maps',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'source changed'):
                    make_comparison_maps(first,second,fields=['dm'],projections=['xy'])
            stages=list((root/'comparison_maps').glob('.comparison-maps-*'))
            self.assertEqual(len(stages),2)
            self.assertTrue(all((stage/'export_failure.json').is_file() for stage in stages))

# ========================================================================
# test_resource_limits
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.resources import _cgroup_limit_bytes, cgroup_available_memory, cpu_count, memory_limit_gb


class ResourceLimitTests(unittest.TestCase):
    def test_slurm_allocation_caps_unrestricted_affinity_and_memory(self):
        import os
        with patch.dict(os.environ,{'SLURM_CPUS_ON_NODE':'2','SLURM_MEM_PER_NODE':'1024','SLURM_MEM_PER_CPU':'256'}):
            with patch('core.foundation.os.sched_getaffinity',return_value=set(range(64)),create=True):
                self.assertLessEqual(cpu_count(),2)
                self.assertLessEqual(memory_limit_gb({'hardware':{'memory_limit_gb':4}}),.5)
        with patch.dict(os.environ,{'SLURM_CPUS_ON_NODE':'invalid'}):
            self.assertRaisesRegex(ValueError,'SLURM_CPUS_ON_NODE',cpu_count)

    def test_headroom_counts_only_reclaimable_file_cache_and_clamps_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);limit=root/'limit';used=root/'used';stat=root/'stat'
            limit.write_text('6000');used.write_text('4500')
            for raw, expected in [('inactive_file 4000',5500), ('inactive_file 9000',6000),
                                  ('inactive_file -1',1500), ('inactive_file invalid',1500),
                                  ('active_file 4000',1500)]:
                stat.write_text(raw)
                self.assertEqual(cgroup_available_memory(limit,used,stat),expected)
            stat.unlink()
            self.assertEqual(cgroup_available_memory(limit,used,stat),1500)
            limit.write_text('max')
            self.assertIsNone(cgroup_available_memory(limit,used,stat))

    def test_cgroup_reads_close_descriptors_including_unlimited_and_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i,value in enumerate(('max','not-a-limit','4096','8192',str(1<<63))):
                path=Path(tmp)/str(i);path.write_text(value);paths.append(path)
            streams=[];real_open=open
            def tracked_open(*args,**kwargs):
                stream=real_open(*args,**kwargs);streams.append(stream);return stream
            with patch('builtins.open',side_effect=tracked_open):self.assertEqual(_cgroup_limit_bytes(paths),4096)
            self.assertTrue(all(stream.closed for stream in streams))

# ========================================================================
# test_native_portability_patches

test_native_portability_patches_ROOT = Path(__file__).resolve().parent


@unittest.skipUnless(shutil.which('git'), 'git is required for patch replay')
class NativePortabilityPatchTests(unittest.TestCase):
    def test_reverse_and_reapply_preserves_source_bytes(self):
        cases = (
            ('gadget4', 'src/data/idstorage.h', 'gadget4-idstorage-math-header.patch'),
            ('rockstar', 'io/io_arepo.c', 'rockstar-gadget4-cosmology-metadata.patch'),
            ('rockstar', 'scripts/gen_merger_cfg.pl', 'rockstar-merger-config-argument.patch'),
        )
        for engine, relative, name in cases:
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as temporary:
                source = test_native_portability_patches_ROOT/'core/software'/engine/relative
                target = Path(temporary)/relative
                target.parent.mkdir(parents=True)
                shutil.copy2(source, target)
                expected = hashlib.sha256(target.read_bytes()).hexdigest()
                patch = test_native_portability_patches_ROOT/'core/patches'/name
                for options in (['--reverse'], []):
                    subprocess.run(['git', 'apply', *options, str(patch)],
                        cwd=temporary, check=True, capture_output=True)
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), expected)

    def test_exact_synchronized_output_patch_replays_all_native_files(self):
        relatives = ('src/main/run.cc', 'src/time_integration/timestep.cc',
                     'src/time_integration/exact_output_timeline.h', 'Template-Config.sh',
                     'buildsystem/option_registry.py', 'defines_extra')
        patch = test_native_portability_patches_ROOT/'core/patches/gadget4-exact-synchronized-outputs.patch'
        with tempfile.TemporaryDirectory() as temporary:
            expected = {}
            for relative in relatives:
                target = Path(temporary)/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(test_native_portability_patches_ROOT/'core/software/gadget4'/relative, target)
                expected[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
            for options in (['--reverse'], []):
                subprocess.run(['git', 'apply', *options, str(patch)],
                               cwd=temporary, check=True, capture_output=True)
            self.assertEqual({relative: hashlib.sha256((Path(temporary)/relative).read_bytes()).hexdigest()
                              for relative in relatives}, expected)

# ========================================================================
# test_catalogue_kinematics

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from analysis import (plot_halo_velocity_distribution,plot_spin_distribution)
from core.data import convert_native_catalogue_kinematics


class CatalogueKinematicsTests(unittest.TestCase):
    def fixture(self,root,a):
        test_stellar_history_snapshot(root,0,a,[1,2],[[1.,2.,3.],[4.,5.,6.]])
        native=root/'data/snapshot_000.hdf5'
        with h5py.File(native) as source:parameters=dict(source['Parameters'].attrs)
        (root/'results').mkdir()
        catalogue=root/'data/fof_subhalo_tab_000.hdf5'
        with h5py.File(catalogue,'w') as source:
            source.create_group('Header').attrs.update(Time=a,Ngroups_ThisFile=2,Nsubgroups_ThisFile=2)
            source.create_group('Parameters').attrs.update(parameters)
            source.create_group('Group')['GroupVel']=np.array([[3.,4.,0.],[0.,0.,10.]])*a
            sub=source.create_group('Subhalo')
            sub['SubhaloVel']=np.array([[3.,4.,0.],[0.,0.,10.]])
            sub['SubhaloSpin']=np.array([[1.,0.,0.],[0.,2.,0.]])
            sub['SubhaloMass']=np.ones(2)
        return catalogue

    def test_actual_fof_and_subfind_producers_agree_at_nontrivial_epochs(self):
        for a in (.25,.75):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);catalogue=self.fixture(root,a)
                fof=plot_halo_velocity_distribution(root,0)
                self.assertEqual(fof['source_field'],'GroupVel')
                with h5py.File(catalogue,'r+') as source:del source['Group/GroupVel']
                subfind=plot_halo_velocity_distribution(root,0)
                self.assertEqual(subfind['source_field'],'SubhaloVel')
                np.testing.assert_allclose(fof['bin_edges'],subfind['bin_edges'],rtol=1.e-15,atol=0.)
                np.testing.assert_array_equal(fof['probability_density'],subfind['probability_density'])
                np.testing.assert_allclose(plot_spin_distribution(root,0),[2.,4.])

    def test_specific_angular_momentum_uses_actual_length_velocity_and_h(self):
        vector=np.array([[1.,0.,0.]])
        result=convert_native_catalogue_kinematics(vector,'SubhaloSpin',scale_factor=.17,hubble_param=.5,
            unit_length_cm=3*KPC_CM,unit_velocity_cm_s=2.e5)
        np.testing.assert_array_equal(result,[[12.,0.,0.]])
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            convert_native_catalogue_kinematics(vector,'GroupSpin',scale_factor=.17,hubble_param=.5,
                unit_length_cm=3*KPC_CM,unit_velocity_cm_s=2.e5)

    def test_mixed_catalogue_epoch_and_unit_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);catalogue=self.fixture(root,.25)
            with h5py.File(catalogue,'r+') as source:source['Header'].attrs['Time']=.5
            with self.assertRaisesRegex(ValueError,'epochs disagree'):plot_halo_velocity_distribution(root,0)
            with h5py.File(catalogue,'r+') as source:
                source['Header'].attrs['Time']=.25
                source['Parameters'].attrs['UnitVelocity_in_cm_per_s']=2.e5
            with self.assertRaisesRegex(ValueError,'units disagree'):plot_halo_velocity_distribution(root,0)

    def test_fractional_native_catalogue_identity_is_not_truncated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'out_0.list'
            path.write_text('# id mvir desc_id\n1.5 3e10 -1\n')
            self.assertIsNone(rockstar_table_from_path(path))
            path.write_text('# id mvir desc_id\n9007199254740993 3e10 -1\n')
            table=rockstar_table_from_path(path)
            self.assertEqual(int(table['id'][0]),9007199254740993)

# ========================================================================
# test_isolated_merger_publication
import importlib.util
test_isolated_merger_publication_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(test_isolated_merger_publication_ROOT / 'core'))
from blender import verify_export_manifest
from core.data import isolated_epoch_contract
test_isolated_merger_publication_spec = importlib.util.spec_from_file_location('isolated_merger_export', test_isolated_merger_publication_ROOT / 'core/visualization.py')
exporter = importlib.util.module_from_spec(test_isolated_merger_publication_spec)
test_isolated_merger_publication_spec.loader.exec_module(exporter)


class IsolatedMergerTests(unittest.TestCase):
    def source(self, root):
        root.mkdir()
        for index in range(2):
            with h5py.File(root / ('snapshot_%03d.hdf5' % index), 'w') as source:
                header = source.create_group('Header')
                header.attrs['Time'] = float(index)
                header.attrs['NumFilesPerSnapshot'] = 1
                header.attrs['NumPart_ThisFile'] = [0, 2, 2, 0, 0, 0]
                header.attrs['NumPart_Total'] = [0, 2, 2, 0, 0, 0]
                parameters = source.create_group('Parameters')
                parameters.attrs['ComovingIntegrationOn'] = 0
                parameters.attrs['PeriodicBoundariesOn'] = 0
                parameters.attrs['UnitLength_in_cm'] = exporter.CM_PER_KPC
                parameters.attrs['UnitVelocity_in_cm_per_s'] = 1e5
                for species in (1, 2):
                    group = source.create_group('PartType%d' % species)
                    group['ParticleIDs'] = np.array([2**63 + species * 2, 2**63 + species * 2 + 1], dtype=np.uint64)
                    group['Coordinates'] = np.array([[index, 0., 0.], [index + 1., 1., 0.]])
        return root

    def export(self, source, destination):
        return exporter.export(source, destination, max_particles=2, dark_type=1,
            stellar_type=2, gas_type=0, black_hole_type=5, length_unit_kpc=None,
            assume_noncomoving=False, allow_periodic=False)

    def test_exact_ids_isolated_time_and_immutable_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root / 'snapshots')
            manifest = self.export(source, root / 'view')
            record = verify_export_manifest(manifest, schema='darkmattersim-isolated-gadget-merger-v1')
            epoch = record['outputs'][0]['epoch']
            self.assertIsNone(epoch['actual_scale_factor'])
            self.assertEqual(epoch['native_time_code'], 0.)
            product = record['outputs'][0]['numeric_products']['dm']
            verify_model_arrays(manifest.parent / product['directory'])
            with np.load(manifest.parent / product['directory'] / 'arrays.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['particle_ids'].dtype, np.dtype('uint64'))
                self.assertTrue(np.all(arrays['particle_ids'] > 2**63))
            with self.assertRaises(FileExistsError): self.export(source, root / 'view')
            manifest.with_name(manifest.name + '.sha256').unlink()
            with self.assertRaises(ValueError): verify_export_manifest(manifest, schema=record['schema'])

    def test_isolated_time_never_becomes_scale_factor(self):
        epoch = isolated_epoch_contract(120., time_unit_gyr=.01)
        self.assertEqual(epoch['elapsed_time_gyr'], 1.2)
        self.assertIsNone(epoch['redshift'])
        with self.assertRaises(ValueError): isolated_epoch_contract(1., time_unit_gyr=-1.)

    def test_changed_numeric_asset_and_failed_stage_are_retained(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root / 'snapshots')
            with patch.object(exporter, '_write_position_ply', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt): self.export(source, root / 'failed')
            self.assertFalse((root / 'failed').exists())
            self.assertTrue(list(root.glob('.merger-export-*/failure.json')))
            manifest = self.export(source, root / 'view')
            record = json.loads(manifest.read_text())
            asset = manifest.parent / record['outputs'][0]['numeric_products']['dm']['directory'] / 'arrays.npz'
            asset.write_bytes(asset.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact missing or changed'):
                verify_export_manifest(manifest, schema=record['schema'])

# ========================================================================
# test_catalogue_publication
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
import core.data as publication
from analysis import group_files,group_ordered_snapshot_files


class CataloguePublicationTests(unittest.TestCase):
    def fixture(self,root):
        data=root/'data';data.mkdir()
        old=data/'fof_subhalo_tab_000.0.hdf5';old.write_bytes(b'old catalogue shard zero')
        stale=data/'fof_subhalo_tab_000.1.hdf5';stale.write_bytes(b'old stale shard one')
        ordered=data/'grouped_snapshot_000.hdf5';ordered.write_bytes(b'old particle order')
        sources=root/'native_stage';sources.mkdir()
        catalogue=sources/'catalogue.hdf5';catalogue.write_bytes(b'new complete single catalogue')
        particle=sources/'particles.hdf5';particle.write_bytes(b'new particle order')
        return [(catalogue,Path('fof_subhalo_tab_000.hdf5')),(particle,Path('grouped_snapshot_000.hdf5'))],[old,stale,ordered]

    def publish(self,root,outputs,previous):
        return publication.publish_catalogue_set(root,0,outputs,previous,
            epoch={'scale_factor':.5,'scope':'synthetic transport fixture'},provenance={'scope':'synthetic transport fixture'})

    def test_full_set_replaces_stale_layout_and_preserves_both_generations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();outputs,previous=self.fixture(root)
            result=self.publish(root,outputs,previous)
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual([p.name for p in group_files(root,0)],['fof_subhalo_tab_000.hdf5'])
            self.assertEqual(group_ordered_snapshot_files(root,0)[0].read_bytes(),b'new particle order')
            archive=root/result['archive']
            self.assertEqual((archive/'previous/fof_subhalo_tab_000.1.hdf5').read_bytes(),b'old stale shard one')
            self.assertEqual((archive/'next/fof_subhalo_tab_000.hdf5').read_bytes(),b'new complete single catalogue')
            self.assertEqual(publication.replay_catalogue_publication(root,0)['status'],'COMPLETE')
            analysis=root/'analysis_workspace';analysis.mkdir();(analysis/'data').symlink_to(root/'data',target_is_directory=True)
            self.assertEqual(len(group_files(analysis,0)),1)
            self.assertEqual(group_ordered_snapshot_files(analysis,0)[0].read_bytes(),b'new particle order')

    def test_interrupted_first_target_copy_closes_both_public_readers_then_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();outputs,previous=self.fixture(root)
            real_copy=publication.atomic_copy;count=0
            def interrupted(source,destination):
                nonlocal count
                value=real_copy(source,destination)
                if Path(destination).parent==root/'data':
                    count+=1
                    if count==1:raise KeyboardInterrupt()
                return value
            with patch('core.data.atomic_copy',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):self.publish(root,outputs,previous)
            for reader in (group_files,group_ordered_snapshot_files):
                with self.assertRaisesRegex(ValueError,'explicit replay'):reader(root,0)
            publication.replay_catalogue_publication(root,0)
            self.assertEqual(len(group_files(root,0)),1)
            self.assertTrue(next((root/'results/catalogue_publications').glob('snapshot_*/failure.json')).is_file())

    def test_corrupt_archive_or_changed_published_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();outputs,previous=self.fixture(root);record=self.publish(root,outputs,previous)
            target=root/'data/fof_subhalo_tab_000.hdf5';before=target.read_bytes();target.write_bytes(b'changed valid-container bytes')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):group_files(root,0)
            with self.assertRaises(ValueError):publication.replay_catalogue_publication(root,0)
            target.write_bytes(before)
            (root/record['archive']/'next/fof_subhalo_tab_000.hdf5').write_bytes(b'corrupt immutable archive')
            with self.assertRaisesRegex(ValueError,'archived bytes'):publication.replay_catalogue_publication(root,0)

    def test_plan_destination_cannot_escape_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();outputs,previous=self.fixture(root)
            outputs[0]=(outputs[0][0],Path('../outside.hdf5'))
            with self.assertRaisesRegex(ValueError,'escapes'):self.publish(root,outputs,previous)
            self.assertFalse((root/'outside.hdf5').exists())
            self.assertFalse((root/'data/.catalogue_publications/000.json').exists())
            self.assertEqual(previous[0].read_bytes(),b'old catalogue shard zero')
sys.path.insert(0, str(Path(__file__).resolve().parent/'core'))
from core.data import (publish_scientific_dataset, read_scientific_product)
from core.data import publish_hdf5_product


def test_scientific_product_spec(a=.5):
    return {'units': 'identity', 'quantity': 'exact native particle identity', 'missing_policy': 'none',
        'identity_namespace': 'control:dark_matter', 'dtype': 'uint64',
        'coordinates': coordinate_contract(position_units='comoving kpc/h', origin='box origin',
            periodic_wrapping='periodic box', velocity_convention='not applicable to IDs'),
        'epoch': epoch_contract(a), 'provenance': {'scope': 'transport control; no science acceptance'}}


class ScientificProductTests(unittest.TestCase):
    def stage(self, root, name='stage.hdf5', a=.5, values=None):
        path = root/name
        with h5py.File(path, 'w') as out:
            d = out.create_dataset('ParticleIDs', data=np.array([0, 2**63+1, 2**64-1] if values is None else values, dtype=np.uint64))
            publish_scientific_dataset(d, test_scientific_product_spec(a))
        return path

    def test_public_writer_reader_preserves_uint64_and_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=self.stage(root);target=root/'published.hdf5'
            publish_hdf5_product(stage,target)
            reader=read_scientific_product(target,memory_bytes=8)
            chunks=list(reader.iter_dataset('ParticleIDs'))
            self.assertEqual([int(c[0]) for c in chunks],[0,2**63+1,2**64-1])
            self.assertTrue(all(c.nbytes<=8 for c in chunks))
            with h5py.File(target) as out:self.assertTrue(out['ParticleIDs'].fletcher32)

    def test_explicit_units_frame_epoch_required_before_modification(self):
        with tempfile.TemporaryDirectory() as tmp,h5py.File(Path(tmp)/'stage.h5','w') as out:
            d=out.create_dataset('ids',data=np.array([1],dtype=np.uint64))
            for key in ('units','coordinates','epoch','quantity','provenance','missing_policy'):
                declaration=test_scientific_product_spec();declaration.pop(key)
                with self.subTest(key=key),self.assertRaises(ValueError):publish_scientific_dataset(d,declaration)
                self.assertEqual(len(d.attrs),0)

    def test_identity_overflow_and_signed_storage_never_cast(self):
        from core.data import exact_uint64_ids
        for bad in ([2**64],[-1],[1.0],[True]):
            with self.assertRaises((ValueError,OverflowError,TypeError)):exact_uint64_ids(bad)
        with tempfile.TemporaryDirectory() as tmp,h5py.File(Path(tmp)/'stage.h5','w') as out:
            d=out.create_dataset('ids',data=np.array([1],dtype=np.int64))
            declaration=test_scientific_product_spec();declaration.pop('dtype')
            with self.assertRaisesRegex(ValueError,'uint64'):publish_scientific_dataset(d,declaration)
            self.assertEqual(len(d.attrs),0)

    def test_semantic_and_valid_numeric_rewrites_rejected_before_republication(self):
        from core.data import verify_hdf5_dataset_payloads
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for field in ('units','payload'):
                path=self.stage(root,name=field+'.hdf5')
                with h5py.File(path,'r+') as out:
                    if field=='units':out['ParticleIDs'].attrs['units']='Msun'
                    else:out['ParticleIDs'][0]=8
                with self.assertRaisesRegex(ValueError,'checksum'):verify_hdf5_dataset_payloads(path,require_contract=True)
                with self.assertRaises(ValueError):publish_hdf5_product(path,root/(field+'-published.hdf5'))

    def test_interruption_replay_retains_previous_and_rejects_pending(self):
        import core.data as publication
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'published.hdf5';stage=self.stage(root)
            first=publish_hdf5_product(stage,target);before=digest(target)
            next_stage=self.stage(root,name='next.hdf5',values=[5])
            original=publication.atomic_copy
            def stop(source,destination):
                if Path(destination).resolve()==target.resolve():raise KeyboardInterrupt('injected copy interruption')
                return original(source,destination)
            with patch('core.data.atomic_copy',side_effect=stop),self.assertRaises(KeyboardInterrupt):
                publish_hdf5_product(next_stage,target)
            self.assertEqual(digest(target),before)
            with self.assertRaisesRegex(ValueError,'pending'):read_scientific_product(target)
            recover_hdf5_product(target,first['generation'])
            self.assertEqual(int(next(read_scientific_product(target).iter_dataset('ParticleIDs'))[0]),0)
            self.assertEqual(len(list((root/'.hdf5_product_history').iterdir())),2)

    def shards(self,root):
        paths=[]
        for i in range(2):
            path=self.stage(root,name='snapshot_001.%d.hdf5'%i,values=[i])
            with h5py.File(path,'r+') as out:
                header=out.create_group('Header');header.attrs.update(Time=.5,Research2Generation='g1',Research2ShardCount=2)
            paths.append(path)
        marker=root/'snapshot_001.publication.json'
        marker.write_text(json.dumps({'schema':'native-hdf5-set-publication-v1','state':'COMPLETE','generation':'g1','shard_count':2}))
        return paths,marker

    def test_missing_mixed_generation_and_epoch_shards_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths,marker=self.shards(root)
            reader=read_scientific_product({'shards':list(reversed(paths))},memory_bytes=8)
            self.assertEqual([int(c[0]) for c in reader.iter_dataset('ParticleIDs')],[0,1])
            with self.assertRaises(ValueError):read_scientific_product({'shards':paths[:1]})
            for key,value in [('Research2Generation','g2'),('Time',.7)]:
                with h5py.File(paths[1],'r+') as out:out['Header'].attrs[key]=value
                with self.assertRaises(ValueError):read_scientific_product({'shards':paths})
                with h5py.File(paths[1],'r+') as out:out['Header'].attrs[key]='g1' if key=='Research2Generation' else .5
            with h5py.File(paths[1],'r+') as out:publish_scientific_dataset(out['ParticleIDs'],test_scientific_product_spec(.7))
            with self.assertRaisesRegex(ValueError,'epoch|semantics mismatch'):read_scientific_product({'shards':paths})

    def test_changed_shard_during_bounded_consumption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths,_=self.shards(Path(tmp));reader=read_scientific_product({'shards':paths})
            with h5py.File(paths[1],'r+') as out:out['ParticleIDs'][0]=99
            with self.assertRaises(ValueError):list(reader.iter_dataset('ParticleIDs'))

    def test_concurrent_public_writer_cannot_enter_owned_destination(self):
        from core.native import native_run_lock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=self.stage(root);target=root/'published.hdf5';failures=[]
            def writer():
                try:publish_hdf5_product(stage,target)
                except Exception as exc:failures.append(exc)
            with native_run_lock(target,operation='control owner'):
                worker=threading.Thread(target=writer);worker.start();worker.join(5)
                self.assertFalse(worker.is_alive())
            self.assertEqual(len(failures),1);self.assertIsInstance(failures[0],RuntimeError)
            self.assertFalse(target.exists())

    def test_isolated_epoch_and_scalar_empty_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'stage.h5'
            with h5py.File(stage,'w') as out:
                declaration=test_scientific_product_spec();declaration['epoch']=isolated_epoch_contract(0.)
                for name,value in [('scalar',np.uint64(2**64-1)),('empty',np.empty(0,dtype=np.uint64))]:
                    publish_scientific_dataset(out.create_dataset(name,data=value),declaration)
            target=root/'published.hdf5';publish_hdf5_product(stage,target)
            reader=read_scientific_product(target,memory_bytes=8)
            self.assertEqual(int(next(reader.iter_dataset('scalar'))),2**64-1)
            self.assertEqual(list(reader.iter_dataset('empty')),[])

    def test_existing_publication_helper_requires_complete_semantic_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stage=root/'stage.hdf5'
            with h5py.File(stage,'w') as out:out.create_dataset('ids',data=np.array([1],dtype=np.uint64))
            with self.assertRaises(ValueError):publish_hdf5_product(stage,root/'bad.hdf5',semantic_specs={})
            publish_hdf5_product(stage,root/'good.hdf5',semantic_specs={'ids':test_scientific_product_spec()})
            self.assertEqual(int(next(read_scientific_product(root/'good.hdf5').iter_dataset('ids'))[0]),1)

    def test_native_reserved_identity_bit_is_rejected_without_changing_input(self):
        from core.data import validate_gadget_native_particle_ids
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.hdf5'
            with h5py.File(path,'w') as out:
                out['PartType1/ParticleIDs']=np.array([0,2**63-1],dtype=np.uint64)
                out['PartType4/ParticleIDs']=np.array([2**63+1],dtype=np.uint64)
            before=digest(path)
            with self.assertRaisesRegex(ValueError,'reserved'):validate_gadget_native_particle_ids([path],chunk_rows=1)
            self.assertEqual(digest(path),before)

    def test_public_cli_strict_semantics_routes_to_bounded_reader(self):
        import commands as audit_data_contract
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'product.hdf5'
            publish_hdf5_product(self.stage(root),target)
            output=root/'audit.json'
            with patch('sys.argv',['commands.py',str(root),'--product',str(target),
                '--product-kind','hdf5','--require-scientific-semantics','--memory-gib','.001','--output',str(output)]):
                audit_data_contract.audit_data_contract_main()
            self.assertEqual(json.loads(output.read_text())['record']['semantics_status'],'VERIFIED')

# ========================================================================
# test_zoom_softening

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from zoom import validate_coarse_softening_masses


class ZoomSofteningTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(
            softening_kpc_h=2.0,
            softening_max_physical_kpc_h=0.5,
            zoom_softening_mass_ratios=[9.6, 76.8],
        )

    def test_combined_coarse_baryon_mass_and_physical_cap(self):
        plan = softening_plan(self.cfg)
        np.testing.assert_allclose(plan["comoving"][6:], 2 * np.cbrt([9.6, 76.8]))
        np.testing.assert_allclose(
            plan["maximum_physical"], np.array(plan["comoving"]) / 4
        )
        self.assertEqual(plan["type_classes"], [0, 1, 6, 3, 4, 5])

    def test_invalid_ratios(self):
        for ratios in ([], [1], [True], [float("nan")], [8, 8], [64, 8]):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                softening_plan({**self.cfg, "zoom_softening_mass_ratios": ratios})

    def test_actual_variable_coarse_masses_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ic.hdf5"
            with h5py.File(p, "w") as f:
                f.create_group('Header')
                for t, masses in [(1, [3.0, 3.0]), (2, [28.8, 230.4, 28.8])]:
                    g = f.create_group(f"PartType{t}")
                    g["Coordinates"] = np.zeros((len(masses), 3))
                    g["Masses"] = masses
            self.assertEqual(
                validate_coarse_softening_masses([p], self.cfg)["status"], "PASS"
            )
            with h5py.File(p, "r+") as f:
                f["PartType2/Masses"][0] = 40
            with self.assertRaisesRegex(ValueError, "no unique"):
                validate_coarse_softening_masses([p], self.cfg)

    def test_compensated_mixed_mass_uses_verified_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ic.hdf5'
            with h5py.File(p,'w') as f:
                f.create_group('Header').attrs.update(Music_mixed_fluid_schema=1,
                    Music_mean_species_masses_native=[.6,3.,0,0,0,0])
                for t,masses in ((1,[2.8,3.2]),(2,[28.8,230.4])):
                    group=f.create_group('PartType'+str(t))
                    group['Coordinates']=np.zeros((len(masses),3))
                    group['Masses']=masses
            report=validate_coarse_softening_masses([p],self.cfg)
            self.assertTrue(report['mixed_fluid_mean_mass'])
            self.assertEqual(report['fine_dm_mass_native'],3.)
            with h5py.File(p,'r+') as f:
                f['Header'].attrs['Music_mean_species_masses_native']=[.6,4.,0,0,0,0]
            with self.assertRaisesRegex(ValueError,'verified mixed-fluid mean'):
                validate_coarse_softening_masses([p],self.cfg)

    def test_native_options_and_parameter_classes_connected(self):
        import yaml
        from evolution import gadget_config_text, gadget_param_text

        base = Path(__file__).resolve().parent
        cfg = yaml.safe_load((base / "core/config.yaml").read_text())
        cfg["cosmology"].update(
            model="lcdm", H0=70.0, omega_b=0.0245, omega_cdm=0.1225, omega_radiation=0.0
        )
        cfg["simulation"].update(
            self.cfg,
            particles_per_dimension=16,
            pm_grid=32,
            physics="dm",
            box_mpc_h=10,
            asmth=1.25,
            multipole_order=3,
            start_redshift=99,
            snapshot_redshift_list=[99.0, 0.0],
            end_redshift=0.0,
            time_limit_cpu_seconds=600.0,
            restart_every_seconds=60,
        )
        cfg["hardware"].update(memory_limit_gb=2, mpi_ranks=1, omp_threads=1)
        cfg["simulation"].update(
            num_files_per_snapshot=1,
            err_tol_theta=0.5,
            err_tol_force_acc=0.005,
            err_tol_int_accuracy=0.025,
            max_size_timestep_ln_a=0.01,
        )
        options = gadget_config_text(cfg)
        self.assertIn("NSOFTCLASSES=8", options)
        self.assertIn("INDIVIDUAL_GRAVITY_SOFTENING=4", options)
        params = gadget_param_text(cfg, base, base / "test_ic", dry_run=True)
        self.assertIn("SofteningClassOfPartType2 6", params)
        self.assertIn("SofteningComovingClass7", params)

    def test_handoff_rejects_wrong_masses_before_build_or_launch(self):
        from unittest.mock import patch
        from evolution import run_gadget

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wrong.hdf5"
            (Path(tmp)/'manifest.json').write_text('{}')
            with h5py.File(p, "w") as f:
                header=f.create_group("Header").attrs
                header["NumFilesPerSnapshot"] = 1
                header['NumPart_Total']=np.array([0,2,1,0,0,0],dtype=np.uint32)
                header['NumPart_ThisFile']=np.array([0,2,1,0,0,0],dtype=np.uint32)
                for t, masses in [(1, [3.0, 3.0]), (2, [99.0])]:
                    g = f.create_group(f"PartType{t}")
                    g["Coordinates"] = np.zeros((len(masses), 3))
                    g["Velocities"] = np.zeros((len(masses), 3))
                    g["Masses"] = masses
                    g['ParticleIDs']=np.arange(t*10,t*10+len(masses),dtype=np.uint64)
            with patch('evolution.build_gadget') as build, patch('evolution.mpi_ranks',return_value=1):
                with self.assertRaisesRegex(ValueError, "no unique"):
                    run_gadget({"simulation": self.cfg}, Path(tmp), p)
                build.assert_not_called()

# ========================================================================
# test_stellar_blender_assets
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from blender import verify_stellar_candidate_assets


class StellarBlenderAssetsTests(unittest.TestCase):
    def fixture(self,root):
        asset=root/'catalogue.hdf5';asset.write_bytes(b'transport fixture; not scientific HDF5')
        manifest={'schema':'darkmattersim-stellar-candidate-blender-v1','provenance':{'data_contract':{}},
            'assets':[{'path':asset.name,'sha256':file_sha256(asset),'size_bytes':asset.stat().st_size}]}
        path=root/'stellar_candidates_blender.json';path.write_text(json.dumps(manifest))
        path.with_suffix('.json.sha256').write_text(file_sha256(path)+'\n')
        return path,asset

    def test_relocated_bundle_verifies_then_rejects_changed_assets(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            original=Path(temporary)/'original';original.mkdir();path,asset=self.fixture(original)
            relocated=Path(temporary)/'relocated';shutil.copytree(original,relocated);shutil.rmtree(original)
            verify_stellar_candidate_assets(relocated/path.name)
            (relocated/asset.name).write_bytes(b'changed bytes')
            with self.assertRaisesRegex(ValueError,'asset missing or changed'):
                verify_stellar_candidate_assets(relocated/path.name)

    def test_changed_manifest_and_escaping_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path,asset=self.fixture(root)
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'manifest checksum'):verify_stellar_candidate_assets(path)
            manifest=json.loads(path.read_text());manifest['assets'][0]['path']='../outside.hdf5'
            path.write_text(json.dumps(manifest));path.with_suffix('.json.sha256').write_text(file_sha256(path)+'\n')
            with self.assertRaises(ValueError):verify_stellar_candidate_assets(path)

# ========================================================================
# test_xdmf_publication
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from visualization import make_xdmf


class XdmfPublicationTests(unittest.TestCase):
    def fixture(self,root):
        test_stellar_history_snapshot(root,0,.5,[2**63+1,2**64-1],[[1.,2.,3.],[90.,2.,3.]])
        return root/'data/snapshot_000.hdf5'

    def test_public_bundle_relocates_without_source_run_and_preserves_uint64_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'run';root.mkdir()
            original=self.fixture(root);before=digest(original)
            first=make_xdmf(root);second=make_xdmf(root)
            self.assertNotEqual(first[0].parent,second[0].parent)
            self.assertEqual(digest(original),before)
            moved=Path(temporary)/'moved'
            shutil.move(str(first[0].parent),moved)
            shutil.rmtree(root)
            contract=json.loads((moved/'xdmf_contract.json').read_text())
            self.assertEqual(contract['science_acceptance'],'NOT_ACCEPTED')
            self.assertEqual(contract['epochs']['0']['actual_scale_factor'],.5)
            for artifact in contract['artifacts']:
                path=moved/artifact['path']
                self.assertEqual(digest(path),artifact['sha256'])
                document=ET.parse(path)
                for item in document.findall('.//DataItem'):
                    relative,key=item.text.split(':/',1)
                    target=(moved/relative).resolve()
                    self.assertTrue(target.is_relative_to(moved.resolve()))
                    with h5py.File(target,'r') as source:
                        self.assertIn(key,source)
                        if key.endswith('ParticleIDs'):
                            self.assertEqual(item.attrib['NumberType'],'UInt')
                            self.assertEqual(item.attrib['Precision'],'8')
                            np.testing.assert_array_equal(source[key][:],np.array([2**63+1,2**64-1],dtype=np.uint64))

    def test_interrupted_publication_retains_stage_and_prior_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);original=self.fixture(root)
            made=make_xdmf(root);before=digest(made[0]);source_before=digest(original)
            with patch('core.foundation.publish_directory',side_effect=OSError('injected publication failure')):
                with self.assertRaisesRegex(OSError,'injected'):make_xdmf(root)
            self.assertEqual(digest(made[0]),before)
            self.assertEqual(digest(original),source_before)
            stages=list((root/'results/3d').glob('.xdmf-*'))
            self.assertEqual(len(stages),1)
            self.assertTrue((stages[0]/'export_failure.json').is_file())
            self.assertTrue((stages[0]/'xdmf_contract.json').is_file())

    def test_source_mutation_during_copy_cannot_publish_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);original=self.fixture(root)
            actual_copy=shutil.copy2
            def damaged_copy(source,destination,*args,**kwargs):
                result=actual_copy(source,destination,*args,**kwargs)
                with h5py.File(destination,'r+') as h5:h5['PartType4/Coordinates'][0,0]=17.
                return result
            before=digest(original)
            with patch('visualization.shutil.copy2',side_effect=damaged_copy):
                with self.assertRaisesRegex(ValueError,'changed during copy'):make_xdmf(root)
            self.assertEqual(digest(original),before)
            self.assertFalse(list((root/'results/3d').glob('xdmf_*')))

# ========================================================================
# test_suite_shared_contracts
test_suite_shared_contracts_ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(test_suite_shared_contracts_ROOT),str(test_suite_shared_contracts_ROOT/'core')]
from commands import load_json,integer_value
from core.foundation import physical_constants_contract

class SuiteSharedContractTests(unittest.TestCase):
    def test_shared_spectrum_reader_uses_whole_bundle_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);expected={'status':'complete','integrity_status':'SEALED_ARTIFACTS_VERIFIED'}
            with patch('core.linear.verify_spectrum_directory',return_value=expected) as verify:
                self.assertEqual(load_json(root/'spectrum.json'),expected)
            verify.assert_called_once_with(root)
    def test_shared_reader_propagates_corrupt_bundle_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('core.linear.verify_spectrum_directory',side_effect=ValueError('artifact changed')):
                with self.assertRaisesRegex(ValueError,'artifact changed'):load_json(root/'spectrum.json')
    def test_rounded_large_float_integer_rejected_but_exact_uint64_text_retained(self):
        with self.assertRaises(ValueError):integer_value(float(2**53+1),'particle ID',0,2**64-1)
        self.assertEqual(integer_value(str(2**64-1),'particle ID',0,2**64-1),2**64-1)
        self.assertEqual(integer_value(12.,'count',0,20),12)
    def test_constants_are_declared_with_scope_and_definitions(self):
        record=physical_constants_contract();constants=record['constants']
        self.assertEqual(constants['KPC_CM']['value'],KPC_CM)
        self.assertEqual(MPC_CM/1000,KPC_CM)
        self.assertEqual(constants['SECONDS_PER_YEAR']['value'],SECONDS_PER_YEAR)
        self.assertIn('no native engine equivalence',record['scope'])

# ========================================================================
# test_inference_publication
sys.path[:0]=[str(Path(__file__).resolve().parent/'core')]

class InferencePublicationTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'samples.npz';meta=root/'metadata.json'
        np.savez(source,samples=np.arange(10.).reshape(5,2),coordinate=[1.,2.])
        meta.write_text(json.dumps({'labels':['a','b'],'units':['kpc','km/s'],'sample_kind':'population','provenance':{'selection':'all supplied control rows'}}))
        return {'samples_npz':str(source),'metadata_json':str(meta),'output_directory':str(root/'final')}
    def test_partial_prediction_schema_preserves_sources_and_leaves_no_final_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);request=self.fixture(root)
            before={key:digest(request[key]) for key in ('samples_npz','metadata_json')}
            with self.assertRaisesRegex(ValueError,'together'):export_diagnostics(request)
            self.assertFalse((root/'final').exists())
            self.assertTrue(list(root.glob('.final-stage-*/failure.json')))
            self.assertEqual({key:digest(request[key]) for key in before},before)
    def test_killed_publication_keeps_complete_staged_assets_for_inspection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);request=self.fixture(root)
            np.savez(request['samples_npz'],samples=np.arange(10.).reshape(5,2))
            with patch('figures.corner_figure',return_value=[]),patch('core.foundation.publish_directory',side_effect=KeyboardInterrupt('before bundle publication')):
                with self.assertRaises(KeyboardInterrupt):export_diagnostics(request)
            self.assertFalse((root/'final').exists())
            self.assertTrue(list(root.glob('.final-stage-*/diagnostics.json')))
            self.assertTrue(list(root.glob('.final-stage-*/inputs/samples.npz')))

# ========================================================================
# test_halo_growth_publication

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from visualization import make_halo_growth_movie


class CapturingEncoder:
    def __init__(self,path,**kwargs):self.path=Path(path);self.count=0
    def append_data(self,frame):self.count+=1
    def close(self):self.path.write_bytes(b'DEFERRED-ENCODER-TRANSPORT-CONTROL:'+str(self.count).encode())


class HaloGrowthPublicationTests(unittest.TestCase):
    def fixture(self,root):
        test_blender_data_contract_fixture(root);(root/'results').mkdir()
        with h5py.File(root/'results/analysis.h5','w') as source:
            history=source.create_group('main_progenitor_histories/halo_0000')
            history.attrs['final_halo_id']=np.uint64(2**63+1)
            for key,values,units in [('scale_factor',[.5,1.],'dimensionless'),
                ('redshift',[1.,0.],'dimensionless'),('mvir_msun_h',[1.e10,2.e10],'Msun/h')]:
                write_dataset_contract(history.create_dataset(key,data=values,chunks=True,fletcher32=True),units=units)

    def test_public_growth_export_preserves_prior_output_and_declares_display_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            with patch('visualization.imageio.get_writer',side_effect=CapturingEncoder):
                first=make_halo_growth_movie(root)
                second=make_halo_growth_movie(root)
            self.assertNotEqual(first.parent,second.parent)
            manifest=json.loads((first.parent/'halo_growth_contract.json').read_text())
            self.assertEqual(manifest['science_acceptance'],'NOT_ACCEPTED')
            frames=json.loads((first.parent/'halo_growth_frames.json').read_text())
            self.assertEqual(frames['frames'][0]['display_epoch']['scale_factor'],.5)
            self.assertIn('not a simulation output',frames['frames'][0]['state_role'])
            for item in manifest['assets']:self.assertEqual(digest(first.parent/item['path']),item['sha256'])

    def test_failed_growth_publication_retains_complete_staged_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            with patch('visualization.imageio.get_writer',side_effect=CapturingEncoder),patch(
                'core.foundation.publish_directory',side_effect=OSError('injected movie publication failure')):
                with self.assertRaisesRegex(OSError,'injected'):make_halo_growth_movie(root)
            stage=next((root/'results/movies').glob('.halo-growth-*'))
            self.assertTrue((stage/'halo_growth_mergers.mp4').is_file())
            self.assertTrue((stage/'export_failure.json').is_file())
            self.assertFalse(list((root/'results/movies').glob('halo_growth_*')))

# ========================================================================
# test_history_hydro_contracts

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from analysis import (build_star_formation_history,write_star_formation_history_hdf5,
                     hydro_field_maps)
from core.data import (canonical_halo_history_payload,
                      attach_time_series_contract,dataset_to_cgs_factor)


class HistoryHydroContractTests(unittest.TestCase):
    def test_sfr_public_writer_uses_redshift_clock_not_storage_index(self):
        summary={'700':{'actual_redshift':1.,'star_formation':{'sfr_physical_status':'COMPLETE','sfr_total_msun_per_year':2.}},
                 '4':{'actual_redshift':0.,'star_formation':{'sfr_physical_status':'COMPLETE','sfr_total_msun_per_year':3.}},
                 '9':{'actual_redshift':.2,'star_formation':{'sfr_physical_status':'UNAVAILABLE'}}}
        history=build_star_formation_history(summary)
        with tempfile.TemporaryDirectory() as temporary,h5py.File(Path(temporary)/'history.h5','w') as h5:
            write_star_formation_history_hdf5(h5,history)
            group=h5['star_formation_history']
            np.testing.assert_array_equal(group['snapshot'][:],[4,700])
            self.assertEqual(group['snapshot'].attrs['units'],'storage index (not physical time)')
            self.assertEqual(group['total_sfr_msun_per_year'].attrs['units'],'Msun/year')
            epochs=json.loads(group['total_sfr_msun_per_year'].attrs['epoch_json'])['rows']
            self.assertEqual([row['actual_scale_factor'] for row in epochs],[1.,.5])
            self.assertEqual(epochs[0]['cosmic_time_source']['status'],'UNAVAILABLE')
            omitted=json.loads(group['omitted_snapshots/reason'].attrs['epoch_json'])
            self.assertNotIn('rows',omitted)

    def test_history_uint64_extremes_and_missing_original_link_remain_exact(self):
        history={'histories':[{'final_halo_id':2**64-1,'track':[
            {'redshift':1.,'halo_id':2**63+1,'original_halo_id':-1,'mass_msun_h':3.},
            {'redshift':0.,'halo_id':2**64-1,'original_halo_id':2**64-1,'mass_msun_h':4.}]}]}
        with tempfile.TemporaryDirectory() as temporary,h5py.File(Path(temporary)/'history.h5','w') as h5:
            _write_product_hdf5(h5,'data',canonical_halo_history_payload(history))
            attach_time_series_contract(h5['data'],identity_namespaces={'halo':'native:halo','tree_node':'native:tree_node'})
            track=h5['data/histories/item_0000/track']
            first=track['item_0000'];last=track['item_0001']
            self.assertEqual(first['halo_id'].dtype,np.dtype('uint64'))
            self.assertEqual(int(first['halo_id'][()]),2**63+1)
            self.assertFalse(first['original_halo_id_available'][()])
            self.assertEqual(first['original_halo_id_native'][()],-1)
            self.assertEqual(int(last['original_halo_id'][()]),2**64-1)
            self.assertTrue(last['original_halo_id_available'][()])
            self.assertEqual(last['halo_id'].attrs['identity_namespace'],'native:tree_node')
            self.assertEqual(json.loads(last['halo_id'].attrs['epoch_json'])['actual_scale_factor'],1.)
        with self.assertRaises(ValueError):canonical_halo_history_payload({'halo_id':1.1})

    def test_actual_hydro_map_writer_converts_gradients_and_preserves_signed_cooling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);test_stellar_history_snapshot(root,0,.5,[1,2],[[1.,2.,3.],[5.,6.,7.]])
            path=root/'data/snapshot_000.hdf5'
            attrs={'to_cgs':.125,'a_scaling':-2.,'h_scaling':1.}
            with h5py.File(path,'r+') as source:
                source.move('PartType4','PartType0')
                gas=source['PartType0']
                gas['VelocityDivergence']=[-2.,-2.];gas['VelocityDivergence'].attrs.update(attrs)
                gas['VelocityCurl']=[4.,4.];gas['VelocityCurl'].attrs.update(attrs)
                gas['CoolingHeatingEnergy']=[-7.,-7.]
            fields={'Coordinates':np.array([[1.,2.,3.],[5.,6.,7.]]),'Masses':np.ones(2),
                    'VelocityDivergence':np.array([-2.,-2.]),'VelocityCurl':np.array([4.,4.]),
                    'CoolingHeatingEnergy':np.array([-7.,-7.])}
            result=hydro_field_maps(root,{'simulation':{'box_mpc_h':.1},'visualization':{'hydro_map_grid':8}},
                0,[path],fields,np.ones(2))
            factor=dataset_to_cgs_factor(attrs,.5,.5)
            compression=result['fields']['compression']
            np.testing.assert_allclose(compression['pixel_values'][np.isfinite(compression['pixel_values'])],2.*factor)
            cooling=result['fields']['cooling_heating']
            np.testing.assert_allclose(cooling['pixel_values'][np.isfinite(cooling['pixel_values'])],-7.)
            with h5py.File(root/'products.h5','w') as h5:
                _write_product_hdf5(h5,'data',{'diagnostics':{'hydro_field_maps':result}})
                attach_analysis_dataset_units(h5['data'],'hydro')
                self.assertEqual(h5['data/diagnostics/hydro_field_maps/fields/compression/pixel_values'].attrs['units'],'s^-1')
                self.assertEqual(h5['data/diagnostics/hydro_field_maps/fields/vorticity/pixel_values_log10'].attrs['units'],
                    'dimensionless log10(value / (s^-1))')


class MixedStagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def stage(self,name,model='lcdm',bump=0.,redshift=99.):
        source=solver_bundle(self.root/(name+'-source'),model=model,ede_bump=bump,redshift=redshift)
        return stage_zoom_inputs(source,self.root/name,box_mpc_h=10.,parent_level=5,refinement_level=6,
            seeds_by_level={'5':150,'6':160},padding_cells=2,poisson_accuracy=1e-14,lpt_order=2)

    def test_native_mixed_profile_and_background_coefficients_at_multiple_epochs(self):
        values=[]
        for z in (49.,99.,199.):
            name='z'+str(int(z));report=self.stage(name,redshift=z)
            cfg=configparser.ConfigParser();cfg.optionxform=str;cfg.read(self.root/name/'parent.conf')
            self.assertTrue(cfg.getboolean('setup','use_2LPT'))
            self.assertTrue(cfg.getboolean('cosmology','mixed_2lpt'))
            self.assertTrue(cfg.getboolean('setup','do_SPH'))
            self.assertEqual(report['perturbation_order'],2)
            self.assertEqual(verify_staged_zoom_inputs(self.root/name),report)
            values.append(report['second_order_growth']['displacement_coefficient'])
        self.assertGreater(max(values)-min(values),1e-4)

    def test_ede_expansion_changes_native_kernel(self):
        cold=self.stage('cold')
        ede=self.stage('ede',model='ede',bump=.3)
        self.assertGreater(abs(cold['second_order_growth']['displacement_coefficient']-
                               ede['second_order_growth']['displacement_coefficient']),1e-4)
        self.assertEqual(ede['mixed_fluid_formulation']['relative_velocity_order'],1)

    def test_wrong_order_and_implicit_eds_seed_are_rejected(self):
        source=solver_bundle(self.root/'source')
        for kwargs in ({'lpt_order':True},{'lpt_order':3},{'lpt_order':1,'lpt_seed_kind':'eds'},
                       {'lpt_order':2,'lpt_seed_kind':'eds'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                stage_zoom_inputs(source,self.root/'bad',box_mpc_h=10.,parent_level=5,refinement_level=6,
                    seeds_by_level={'5':150,'6':160},padding_cells=2,poisson_accuracy=1e-14,**kwargs)

# ========================================================================
# test_wdm_zoom_staging
from core.cosmology import wdm_parameters


class RelicStagingTests(unittest.TestCase):
    def setUp(self):
        self.f=staging_fixtures.StagingTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.cfg=configuration()
        self.control=self.f.root/'control';self.control.mkdir()
        control_info=copy.deepcopy(self.f.info)
        for name in ('transfer.dat','background.dat','input.ini'):
            shutil.copyfile(self.f.root/name,self.control/name)
        k=np.geomspace(1e-5,1000,128)
        baseline=1/(1+k)**2
        np.savetxt(self.control/'pk.dat',np.column_stack((k,baseline)),header='matter P(k) at redshift z = 99')
        control_info['source_files']['start']['pk']='pk.dat'
        control_info['checksums_sha256']['pk.dat']=file_sha256(self.control/'pk.dat')
        self.control_manifest=self.control/'spectrum.json';self.control_manifest.write_text(json.dumps(control_info))
        self.f.info['model']='wdm';self.f.info['parameters']['model']='wdm'
        self.f.info['class_parameters']['start'].update(wdm_parameters(self.cfg))
        rows=np.column_stack((k,-k*k,-2*k*k,k*k*.1,k*k*.2,np.zeros_like(k),np.zeros_like(k),
                              -k*k*.8,k*k*.15,-k*k))
        np.savetxt(self.f.root/'transfer.dat',rows,header='transfer functions at redshift z = 99\n1:k 2:d_b 3:d_cdm 4:t_b 5:t_cdm 6:h_t_nb_prime 7:k2gamma_nb 8:d_ncdm[0] 9:t_ncdm[0] 10:d_m')
        np.savetxt(self.f.root/'pk.dat',np.column_stack((k,baseline/(1+(k/10)**4))),header='matter P(k) at redshift z = 99')
        self.f.info['source_files']['start']['pk']='pk.dat'
        for name in ('transfer.dat','pk.dat'):
            self.f.info['checksums_sha256'][name]=file_sha256(self.f.root/name)
        self.f.save_solver_input();self.f.save_manifest()
        self.request={'wdm':dict(self.cfg['wdm'],allow_underresolved_benchmark=True),
                      'thermal_seed':self.cfg['simulation']['random_seed'],
                      'lcdm_manifest':str(self.control_manifest)}

    def stage(self,lpt_order=1):
        return stage_zoom_inputs(self.f.manifest,self.f.destination,box_mpc_h=10,parent_level=5,refinement_level=7,
                    seeds_by_level={'5':150,'6':160,'7':170},padding_cells=2,poisson_accuracy=1e-14,
                    wdm_request=self.request,lpt_order=lpt_order)

    def test_relic_transfers_and_prescription_travel_together(self):
        result=self.stage()
        self.assertEqual(result['status'],'STAGED_MODEL_MATCHED_INPUTS')
        self.assertEqual(result['wdm']['thermal_seed'],2**63+71)
        self.assertEqual(set(result['wdm']['resolution_by_level']),{'5','6','7'})
        table=np.loadtxt(self.f.destination/'transfers.txt')
        raw=np.loadtxt(self.f.root/'transfer.dat')
        np.testing.assert_allclose(table,raw[:,[0,7,1,8,3]],rtol=1e-15)
        self.assertEqual(verify_staged_zoom_inputs(self.f.destination),result)
        self.assertTrue((self.f.destination/'wdm-prescription.json').is_file())
        self.assertEqual(result['wdm']['configuration']['cosmology']['omega_cdm'],.25*.7**2)

    def test_missing_relic_recipe_or_bad_normalization_is_rejected(self):
        original=self.request
        self.request=None
        with self.assertRaisesRegex(ValueError,'wdm_request'):
            self.stage()
        self.request=copy.deepcopy(original)
        self.request['wdm']['thermal_relic_mass_keV']=3
        with self.assertRaisesRegex(ValueError,'physical thermal relic'):
            self.stage()
        self.assertFalse(self.f.destination.exists())

    def test_raw_control_spectrum_change_and_disabled_thermal_are_rejected(self):
        (self.control/'pk.dat').write_text('changed')
        with self.assertRaisesRegex(ValueError,'spectrum changed'):
            self.stage()
        self.request['wdm']['include_thermal_velocities']=False
        with self.assertRaisesRegex(ValueError,'velocities enabled'):
            self.stage()

    def test_wrong_control_cosmology_rejected_before_output(self):
        info=json.loads(self.control_manifest.read_text())
        info['parameters']['A_s']=3e-9
        self.control_manifest.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError,'cosmology mismatch'):
            self.stage()
        self.assertFalse(self.f.destination.exists())

# ========================================================================
# test_shape_zoom_publication
sys.path[:0]=[str(Path(__file__).resolve().parent/'core'),str(Path(__file__).resolve().parent)]
from halos import plot_radial_shapes


class ShapeZoomPublicationTests(unittest.TestCase):
    def test_shape_publication_interruption_retains_arrays_and_consumer_rejects_corruption(self):
        control=RadialShapeWorkflowTests();control.setUp();self.addCleanup(control.doCleanups)
        request=dict(control.request);destination=Path(request['output_directory'])
        with patch('core.foundation.publish_directory',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):export_native_halo_shapes(control.root,**request)
        self.assertFalse(destination.exists())
        stage=list(destination.parent.glob('.halo-shape-*'))[0]
        report=json.loads((stage/'report.json').read_text())
        verify_numpy_archive_contract(stage/'parent.npz',report['array_contract'])
        self.assertTrue((stage/'export_failure.json').is_file())
        export_native_halo_shapes(control.root,**request)
        with (destination/'parent.npz').open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum|hash'):
            plot_radial_shapes(destination,destination.parent/'figures')
        self.assertTrue(stage.is_dir())

    def test_zoom_publication_failure_preserves_epoch_provenance_and_retry(self):
        control=ZoomResolutionTests();control.setUp();self.addCleanup(control.doCleanups)
        arguments=control.kwargs();region={'snapshot':0,
            'center_comoving_kpc_h':arguments.pop('center_comoving_kpc_h'),
            'radius_proper_kpc':arguments.pop('radius_proper_kpc')}
        with patch('core.foundation.publish_directory',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                audit_zoom_regions(control.root,regions=[region],**arguments)
        parent=control.root/'results';stage=list(parent.glob('.zoom-audit-*'))[0]
        self.assertFalse((parent/'zoom_contamination').exists())
        report=json.loads((stage/'provenance.json').read_text())
        self.assertIn('epoch_contract',report['records'][0])
        self.assertIn('data_contract',report)
        self.assertTrue((stage/'export_failure.json').is_file())
        result=audit_zoom_regions(control.root,regions=[region],**arguments)
        self.assertTrue(Path(result['directory']).is_dir());self.assertTrue(stage.is_dir())

test_native_checkpoint_reader_ROOT=Path(__file__).resolve().parent
class NativeReceiptReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.directory=Path(cls.temp.name);cls.binary=cls.directory/'reader'
        source=cls.directory/'reader.cc'
        source.write_text('#include "checkpoint_receipt.h"\nint main(int argc,char **argv){native_checkpoint_receipt r;if(argc!=3)return 2;return native_checkpoint_verify(argv[1],argv[2],0,1,r)?0:1;}\n')
        subprocess.run(['c++','-std=c++11','-I',str(test_native_checkpoint_reader_ROOT/'core/software/gadget4/src/io'),str(source),'-lz','-o',str(cls.binary)],capture_output=True,check=True)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def fixture(self,root):
        payload=root/'restart.0';payload.write_bytes(bytes(range(256))*1024)
        record={'schema':'gadget-rank-checkpoint-v1','rank':0,'ranks':1,'generation':17234567890,'a':.2,'size_bytes':payload.stat().st_size,'crc32':f'{zlib.crc32(payload.read_bytes()):08x}'}
        receipt=root/'restart.0.checksum.json';receipt.write_text(json.dumps(record,separators=(',',':'))+'\n');return payload,receipt,record
    def read(self,payload,receipt):return subprocess.run([str(self.binary),str(payload),str(receipt)],capture_output=True,timeout=30).returncode
    def test_saved_writer_format_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload,receipt,_=self.fixture(Path(tmp));self.assertEqual(self.read(payload,receipt),0)
    def test_native_payload_corruption_truncation_and_missing_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload,receipt,_=self.fixture(Path(tmp));data=payload.read_bytes();payload.write_bytes(bytes([data[0]^1])+data[1:])
            self.assertEqual(self.read(payload,receipt),1);payload.write_bytes(data[:-1]);self.assertEqual(self.read(payload,receipt),1)
            payload.write_bytes(data);receipt.unlink();self.assertEqual(self.read(payload,receipt),1)
    def test_malformed_schema_rank_count_time_and_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload,receipt,original=self.fixture(Path(tmp))
            for key,value in [('schema','wrong'),('rank',1),('ranks',2),('generation',0),('generation',-1),('size_bytes',-1),('a',float('nan')),('a',float('inf')),('a',True),('crc32','123'),('crc32','zzzzzzzz')]:
                with self.subTest(key=key,value=value):
                    record=dict(original);record[key]=value;receipt.write_text(json.dumps(record,separators=(',',':'))+'\n');self.assertEqual(self.read(payload,receipt),1)
    def test_extra_or_truncated_receipt_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload,receipt,_=self.fixture(Path(tmp));original=receipt.read_text()
            for value in [original+'{}\n',original.rstrip()+'garbage',original[:-4],'{"schema":"'+('x'*4096)+'"}']:
                receipt.write_text(value);self.assertEqual(self.read(payload,receipt),1)

    def test_isolated_native_writer_zero_time_and_frame(self):
        with tempfile.TemporaryDirectory() as t:
            payload,receipt,old=self.fixture(Path(t));r=dict(schema='gadget-rank-checkpoint-v2',rank=0,ranks=1,generation=old['generation'],native_time=0.,size_bytes=old['size_bytes'],crc32=old['crc32'])
            receipt.write_text(json.dumps(r,separators=(',',':'))+'\n')
            self.assertEqual(self.read(payload,receipt),0)
            r['native_time']=-1.;receipt.write_text(json.dumps(r,separators=(',',':'))+'\n')
            self.assertNotEqual(self.read(payload,receipt),0)

    def test_isolated_native_reader_rejects_ambiguous_clock(self):
        with tempfile.TemporaryDirectory() as t:
            payload,receipt,r=self.fixture(Path(t));r.update(schema='gadget-rank-checkpoint-v2',native_time=.4)
            receipt.write_text(json.dumps(r,separators=(',',':'))+'\n')
            self.assertNotEqual(self.read(payload,receipt),0)

# ========================================================================
# test_ic_resources

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.resources import estimate_run_resources, initial_condition_census, preflight_storage
from evolution import prepare_and_run


class ICResourceTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'simulation':{'particles_per_dimension':32,'pm_grid':32,'physics':'hydro'},
                  'hardware':{'memory_limit_gb':1},'black_holes':{'enabled':False}}

    def test_default_preserves_uniform_calibration_and_explicit_counts_drive_memory(self):
        default=estimate_run_resources(self.cfg,2)
        explicit=estimate_run_resources(self.cfg,2,particle_counts=[32**3,32**3,0,0,0,0])
        self.assertEqual(default['disk_bytes'],explicit['disk_bytes'])
        self.assertEqual(default['memory_bytes'],explicit['memory_bytes'])
        zoom=estimate_run_resources(self.cfg,2,particle_counts=[8,8,100,0,0,0])
        self.assertEqual(zoom['memory_bytes'],256*116+512*8+64*32**3+zoom['periodic_image_memory_allowance_bytes'])
        self.assertLess(zoom['disk_bytes'],default['disk_bytes'])
        self.assertEqual(zoom['particle_count_source'],'external_ic_census')

    def test_invalid_census_and_memory_fail_without_disk_prompt(self):
        for counts in ([0]*6,[True,1,0,0,0,0],[1.,1,0,0,0,0],[1,2],[-1,1,0,0,0,0]):
            with self.subTest(counts=counts),self.assertRaises(ValueError):
                estimate_run_resources(self.cfg,1,particle_counts=counts)
        with patch('builtins.input') as prompt:
            with self.assertRaises(MemoryError):
                preflight_storage(self.cfg,Path('/tmp'),1,particle_counts=[10**8,10**8,10**8,0,0,0])
            prompt.assert_not_called()

    def make_shard(self,path,counts,total,shards):
        with h5py.File(path,'w') as source:
            source.create_group('Header').attrs.update(NumFilesPerSnapshot=shards,
                NumPart_Total=np.array(total,dtype=np.uint32),NumPart_ThisFile=np.array(counts,dtype=np.uint32))
            for t,n in enumerate(counts):
                if n:
                    group=source.create_group(f'PartType{t}')
                    group['Coordinates']=np.zeros((n,3))
                    group['Velocities']=np.zeros((n,3))
                    group['ParticleIDs']=np.arange(n,dtype=np.uint64)

    def test_complete_shards_hashes_and_dataset_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            first,second=[Path(temporary)/name for name in ('ic.0.hdf5','ic.1.hdf5')]
            counts=[2,3,4,0,0,0]
            total=[4,6,8,0,0,0]
            for path in (first,second):
                self.make_shard(path,counts,total,2)
            census=initial_condition_census([first,second])
            self.assertEqual(census['particles_by_type'],total)
            self.assertEqual(len(census['files']),2)
            self.assertTrue(all(len(row['sha256'])==64 for row in census['files']))
            with self.assertRaisesRegex(ValueError,'missing declared shards'):
                initial_condition_census([first])
            with self.assertRaisesRegex(ValueError,'distinct'):
                initial_condition_census([first,first])
            with h5py.File(first,'r+') as source:
                source['Header'].attrs['NumPart_ThisFile']=np.array([1,3,4,0,0,0],dtype=np.uint32)
            with self.assertRaisesRegex(ValueError,'differs from datasets'):
                initial_condition_census([first,second])

    def test_simulation_uses_delivered_counts_and_stops_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            ic=root/'ic.hdf5'
            counts=[8,8,7,0,0,0]
            self.make_shard(ic,counts,counts,1)
            cfg=dict(self.cfg,cosmology={'model':'lcdm'},ede={},wdm={})
            cfg['simulation']=dict(self.cfg['simulation'],pm_grid=1024)
            run=root/'must-not-exist'
            # Isolate the resource handoff, which precedes linear cosmology and
            # the independent physics validators exercised in their own tests.
            with (patch('evolution.require_particle_cosmology'),
                  patch('evolution.validate_supported_options'),
                  patch('evolution.validate_hydro_configuration'),
                  patch('evolution.snapshot_redshifts',return_value=[0.]),
                  patch('evolution.print_run_summary') as summary):
                with self.assertRaisesRegex(MemoryError,'Delivered IC/PM'):
                    prepare_and_run(cfg,run,existing_ic=ic)
                self.assertEqual(summary.call_args.kwargs['ic_census']['particles_by_type'],counts)
            self.assertFalse(run.exists())
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from core.data import convert_pinned_native
from analysis import units_to_mpc_h,mass_to_msun_h,_dataset_to_cgs_factor


class CentralConversionTests(unittest.TestCase):
    def test_h_scaled_conversions_preserve_old_values_and_do_not_guess_epoch(self):
        values=np.array([0.,.0137,37.5,1000.,-9.,np.nan])
        np.testing.assert_array_equal(convert_pinned_native(values,'position_comoving_mpc_h'),values/1000.)
        np.testing.assert_array_equal(convert_pinned_native(values,'mass_msun_h'),values*1e10)
        np.testing.assert_array_equal(units_to_mpc_h(values,7.),values/1000.)
        np.testing.assert_array_equal(mass_to_msun_h(values),values*1e10)
        for quantity in ['position_comoving_mpc_h','mass_msun_h']:
            np.testing.assert_allclose(convert_pinned_native(convert_pinned_native(values,quantity),quantity,inverse=True),values,rtol=2e-16,equal_nan=True)
        self.assertIsNone(units_to_mpc_h(None,3.));self.assertIsNone(mass_to_msun_h(None))
        with self.assertRaises(ValueError):convert_pinned_native(values,'position_proper_kpc')
        with self.assertRaises(ValueError):convert_pinned_native([True,False],'mass_msun_h')

    def test_saved_cgs_factors_match_nontrivial_epoch_and_h_roundtrips(self):
        a,h=.173,.687
        for attrs in [{'to_cgs':3.0856775814913673e21,'a_scaling':1.,'h_scaling':-1.},
                      {'to_cgs':1e5,'a_scaling':.5,'h_scaling':0.},
                      {'to_cgs':1.98847e43,'a_scaling':0.,'h_scaling':-1.}]:
            factor=dataset_to_cgs_factor(attrs,a,h)
            expected=attrs['to_cgs']*a**attrs['a_scaling']*h**attrs['h_scaling']
            self.assertAlmostEqual(factor/expected,1.,places=14)
            self.assertEqual(_dataset_to_cgs_factor(attrs,a,h),factor)
            data=np.array([.7,12.,1e4]);np.testing.assert_allclose(data*factor/factor,data,rtol=2e-16)
        self.assertAlmostEqual(dataset_to_cgs_factor({'to_cgs':1e5,'a_scaling':.5,'h_scaling':0.},a,h)/1e5,native_unit_factor('particle_velocity_peculiar_kms',scale_factor=a,hubble_param=h))

    def test_corrupt_units_and_overflow_cannot_be_plausible_physical_values(self):
        valid={'to_cgs':1.,'a_scaling':0.,'h_scaling':0.}
        for attrs in [{},dict(valid,to_cgs=-1),dict(valid,a_scaling=np.nan),dict(valid,h_scaling=True),dict(valid,to_cgs=1e308,a_scaling=-1e6)]:
            with self.assertRaises(ValueError):dataset_to_cgs_factor(attrs,.1,.7)
        for a,h in [(0.,.7),(.2,0.),(np.nan,.7),(.2,True)]:
            with self.assertRaises(ValueError):dataset_to_cgs_factor(valid,a,h)
        self.assertIsNone(dataset_to_cgs_factor(None,.2,.7))

# ========================================================================
# test_image_product_publication
sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from core.data import export_image_product
from blender import verify_image_publication, recover_image_publication


class FakeImage:
    def save(self, filename): Path(filename).write_bytes(b'fake image transport bytes; not a PNG')


class ImageProductTests(unittest.TestCase):
    def publish(self, source, output):
        arrays = {'angle_rad': np.array([.1, .2]), 'captured': np.array([True, False])}
        specifications = {'angle_rad': {'units': 'rad', 'quantity': 'Observer angle'},
                          'captured': {'units': 'dimensionless', 'quantity': 'Ray capture indicator'}}
        return export_image_product(output, FakeImage(), {'model': 'synthetic transport fixture'},
            arrays, specifications, algorithm=__file__, context={'science_acceptance': 'NOT_ACCEPTED'},
            source_hashes={str(source): durable.file_sha256(source)})

    def test_complete_numeric_contract_immutable_and_corrupt_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); source = root / 'source.fixture'; source.write_bytes(b'source')
            output = root / 'view.png'
            self.publish(source, output)
            record = verify_image_publication(output)
            self.assertEqual(record['science_acceptance'], 'NOT_ACCEPTED')
            verify_model_arrays(root / 'view.png.product' / record['numeric_data']['directory'])
            with self.assertRaises(FileExistsError): self.publish(source, output)
            output.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'aliases changed'): verify_image_publication(output)

    def test_interrupted_public_alias_replays_retained_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); source = root / 'source.fixture'; source.write_bytes(b'source')
            output = root / 'view.png'
            original = durable.atomic_copy
            def interrupted(src, dst):
                if Path(dst) == output: raise KeyboardInterrupt
                return original(src, dst)
            with patch('core.foundation.atomic_copy', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt): self.publish(source, output)
            with self.assertRaisesRegex(ValueError, 'incomplete'): verify_image_publication(output)
            record = recover_image_publication(output, root / 'view.png.product')
            self.assertEqual(record['integrity_status'], 'INTEGRITY_VERIFIED')
            self.assertFalse((root / 'view.png.product/failure.json').exists())
            self.assertTrue((root / 'view.png.product.failure.json').exists())

    def test_changed_numeric_archive_rejects_even_when_public_pixels_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); source = root / 'source.fixture'; source.write_bytes(b'source')
            output = root / 'view.png'; self.publish(source, output)
            record = verify_image_publication(output)
            numeric = root / 'view.png.product' / record['numeric_data']['directory'] / 'arrays.npz'
            numeric.write_bytes(numeric.read_bytes() + b'changed')
            with self.assertRaises(ValueError): verify_image_publication(output)

# ========================================================================
# test_dome_lensing_contracts
test_dome_lensing_contracts_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(test_dome_lensing_contracts_ROOT / 'core'), str(test_dome_lensing_contracts_ROOT / 'tests')]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, test_dome_lensing_contracts_ROOT / 'core' / file)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


dome = module('dome_contract_control', 'visualization.py')
lensing = module('lensing_contract_control', 'blender.py')


class DomeLensingContractTests(unittest.TestCase):
    def test_actual_sample_ids_epoch_inclusion_and_observer_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids = [2**63 + 1, 2**63 + 3]
            test_stellar_history_snapshot(root, 0, .5, ids, [[50, 50, 80], [51, 50, 75]])
            path = root / 'data/snapshot_000.hdf5'
            with h5py.File(path, 'r+') as source:
                source.move('PartType4', 'PartType1')
                counts = np.array([0, 2, 0, 0, 0, 0], dtype=np.uint64)
                source['Header'].attrs['NumPart_Total'] = counts
                source['Header'].attrs['NumPart_ThisFile'] = counts
            coords, weights, box, z, metadata, arrays = dome.load_particles_for_fisheye(
                root, 0, 'dm', 2, return_arrays=True)
            np.testing.assert_array_equal(arrays['particle_ids'], np.array(ids, dtype=np.uint64))
            np.testing.assert_allclose(arrays['represented_mass_msun_h'], arrays['mass_msun_h'])
            np.testing.assert_allclose(arrays['inverse_inclusion_weight'], [1, 1])
            self.assertEqual(metadata['epoch']['actual_scale_factor'], .5)
            self.assertEqual(metadata['data_contract']['coordinates']['position_units'], 'comoving Mpc/h')
            np.testing.assert_allclose(coords, arrays['observer_relative_positions_mpc_h'])

    def test_lensing_reader_and_scene_use_sealed_numeric_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'panorama.fixture'; source.write_bytes(b'source')
            image = root / 'lensing.png'
            export_image_product(image, FakeImage(), {
                'model': 'Schwarzschild vacuum null geodesics; non-rotating isolated black hole',
                'image_size_px': [128, 128]}, {'angles': np.array([.1, .2])},
                {'angles': {'units': 'rad', 'quantity': 'Saved mapping angle'}},
                algorithm=__file__, context={'scope': 'transport fixture'},
                source_hashes={str(source): digest(source)})
            metadata = image.with_suffix('.json')
            record = lensing.verify_lensing_input(image, metadata)
            self.assertEqual(record['integrity_status'], 'INTEGRITY_VERIFIED')
            def fake_builder(img, meta, target, render):
                target.write_bytes(b'fake Blender transport bytes')
                if render: target.with_name(target.stem + '_preview.png').write_bytes(b'fake preview bytes')
            with patch.object(lensing, '_build_lensing_scene_build_to_file', side_effect=fake_builder):
                blend = lensing.build_lensing_scene_build(image, metadata, root / 'lensing.blend', render=True)
            verify_scene_publication(blend)
            image.with_name(image.name + '.publication.json').unlink()
            with self.assertRaises(FileNotFoundError): lensing.verify_lensing_input(image, metadata)

# ========================================================================
# test_deferred_campaign_contract
sys.path[:0]=[str(Path(__file__).resolve().parent/'core')]
from core.foundation import validation_campaign_record
test_deferred_campaign_contract_ROOT=Path(__file__).resolve().parent

class DeferredCampaignContractTests(unittest.TestCase):
    def test_example_spec_binds_real_controls_without_granting_acceptance(self):
        spec=json.loads((test_deferred_campaign_contract_ROOT/'research_spec.example.json').read_text())
        validate_spec(spec)
        record=validation_campaign_record(spec['validation_campaign'])
        self.assertEqual(record['execution_state'],'VALIDATION_PENDING')
        self.assertEqual(record['science_status'],'NOT_ACCEPTED')
        legacy=copy.deepcopy(spec);legacy.pop('validation_campaign');validate_spec(legacy)
    def test_changed_hash_missing_campaign_or_absolute_dependency_rejected(self):
        entry=json.loads((test_deferred_campaign_contract_ROOT/'research_spec.example.json').read_text())['validation_campaign']
        for bad in [dict(entry,sha256='0'*64),dict(entry,path='/absolute/dev/campaign.json'),dict(entry,path='../outside.json'),dict(entry,unexpected=True)]:
            with self.assertRaises(ValueError):validation_campaign_record(bad)

# ========================================================================
# test_comparison_publication
test_comparison_publication_ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(test_comparison_publication_ROOT/'core'))
from analysis import compare_runs,verify_comparison_bundle


class ComparisonPublicationTests(unittest.TestCase):
    def fixture(self,root):
        runs=[root/'A',root/'B']
        for run in runs:
            (run/'results').mkdir(parents=True)
            with h5py.File(run/'results/analysis.h5','w') as source:source['input']=np.arange(2.)
        return runs

    def worker(self,a,b,output):
        with h5py.File(output/'analysis_comparison.h5','w') as result:
            values=result.create_dataset('ratio',data=np.array([1.,2.]),fletcher32=True)
            write_dataset_contract(values,units='dimensionless',quantity='explicit mocked paired transport values')
        atomic_json(output/'comparison_summary.json',{'science_acceptance':'NOT_ACCEPTED'})

    def test_immutable_bundle_and_corrupt_consumer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,b=self.fixture(root)
            with patch('analysis._compare_runs_to_directory',side_effect=self.worker):
                output=Path(compare_runs(a,b))
            self.assertEqual(verify_comparison_bundle(output)['science_acceptance'],'NOT_ACCEPTED')
            with self.assertRaises(FileExistsError):compare_runs(a,b)
            with h5py.File(output/'analysis_comparison.h5','r+') as result:result['ratio'][0]=9.
            with self.assertRaisesRegex(ValueError,'artifact'):verify_comparison_bundle(output)

    def test_failed_stage_retained_with_original_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,b=self.fixture(root);before=digest(a/'results/analysis.h5')
            def interrupted(a,b,output):
                self.worker(a,b,output)
                raise KeyboardInterrupt()
            with patch('analysis._compare_runs_to_directory',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):compare_runs(a,b)
            self.assertFalse((root/'comparison').exists())
            stages=list(root.glob('.comparison-stage-*'));self.assertEqual(len(stages),1)
            self.assertTrue((stages[0]/'analysis_comparison.h5').is_file())
            self.assertTrue((stages[0]/'failure.json').is_file())
            self.assertEqual(digest(a/'results/analysis.h5'),before)

    def test_input_rewrite_blocks_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,b=self.fixture(root)
            def mutate(a,b,output):
                self.worker(a,b,output)
                with h5py.File(a/'results/analysis.h5','r+') as source:source['input'][0]=7.
            with patch('analysis._compare_runs_to_directory',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'analysis input changed'):compare_runs(a,b)
            self.assertFalse((root/'comparison').exists())
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core.release as package_suite


class PortablePackageTests(unittest.TestCase):
    def test_release_preserves_assets_tests_upstream_docs_licenses_and_excludes_project_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'source';root.mkdir()
            files={'README.md':'instructions','tests/test_control.py':'pass','core/software-revisions':'engine revision',
                'core/software/engine/data/table.txt':'1 2 3','core/software/engine/LICENSE.txt':'license',
                'core/software/engine/CMakeLists.txt':'project(engine)','core/software/engine/tests/required.cc':'source',
                'core/software/engine/build/CMakeCache.txt':'/old/mac/path','.venv/bin/python':'environment',
                '.DS_Store':'metadata','__pycache__/module.pyc':'bytecode','AUDIT_FILE_INVENTORY.json':'generated',
                'core/software/engine/tool':b'\x7fELFbinary'}
            for name,value in files.items():
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value if isinstance(value,bytes) else value.encode())
            upstream = root/'core/software/upstream/gadget4'
            shipped = root/'core/software/gadget4'
            for directory, content in [(upstream, 'original'), (shipped, 'modified')]:
                (directory/'src').mkdir(parents=True)
                (directory/'src/control.cc').write_text(content)
                (directory/'LICENSE').write_text('fixture GPL notice')
                (directory/'Makefile').write_text('all:\n\t@true\n')
            provenance = {'schema': 'retained-native-upstream-v1',
                'repository_url': 'https://gitlab.mpcdf.mpg.de/vrs/gadget4.git',
                'commit': '6fb393b5e25907f2ff06211f67d341c5e40e90d1',
                'files_sha256': {name: hashlib.sha256((upstream/name).read_bytes()).hexdigest()
                    for name in ('src/control.cc', 'LICENSE', 'Makefile')}}
            (upstream.parent/'gadget4.provenance.json').write_text(json.dumps(provenance))
            with patch.object(package_suite,'SOURCE_ROOT',root):
                one=package_suite.build_portable_zip(base/'one.zip');two=package_suite.build_portable_zip(base/'two.zip')
            self.assertEqual(one.read_bytes(),two.read_bytes())
            self.assertEqual(len(package_suite.verify_portable_zip(one)['files_sha256']),16)
            with zipfile.ZipFile(one) as archive:
                names=set(archive.namelist());report=json.loads(archive.read('research2/RELEASE.json'))
                for name in ('README.md','tests/test_control.py','core/software/engine/data/table.txt','core/software/engine/LICENSE.txt','core/software/engine/tests/required.cc'):
                    self.assertIn('research2/'+name,names)
                for name in ('AUDIT_FILE_INVENTORY.json','.DS_Store','.venv/bin/python','core/software/engine/tool','core/software/engine/build/CMakeCache.txt'):
                    self.assertNotIn('research2/'+name,names)
                for name,digest in report['files_sha256'].items():self.assertEqual(hashlib.sha256(archive.read('research2/'+name)).hexdigest(),digest)

# ========================================================================
# test_prepared_binary

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from evolution import run_gadget, gadget_binary, completed_gadget_run_record


class PreparedBinaryTests(unittest.TestCase):
    def test_completed_restart_does_not_retain_stale_error_or_initial_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            snapshot,log=root/'snapshot.hdf5',root/'resume.log'
            snapshot.write_bytes(b'verified by the caller'); log.write_text('completed native log')
            previous={'status':'INTERRUPTED','error':'missing final snapshots','log':'initial.log'}
            result=completed_gadget_run_record(previous,{'status':'DEVIATED','rows':[{'files':[str(snapshot)]}]},log)
            self.assertEqual(result['status'],'COMPLETE')
            self.assertNotIn('error',result)
            self.assertEqual(result['log'],str(log.resolve()))
            self.assertEqual(result['previous_failed_attempt']['error'],previous['error'])
            self.assertEqual(previous['status'],'INTERRUPTED')
            self.assertEqual(result['snapshot_files'][0]['size_bytes'],snapshot.stat().st_size)

    def test_local_validated_binary_is_not_copied_onto_itself(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)
            (run/'input').mkdir()
            binary=gadget_binary(run)
            binary.write_bytes(b'fixture executable')
            cfg={'simulation':{}}
            with (patch('evolution.validate_coarse_softening_masses',return_value=None),
                  patch('evolution.initial_condition_files',return_value=[]),
                  patch('evolution.mpi_ranks',return_value=1),
                  patch('core.native.preflight_native_ic_capacity',return_value={}),
                  patch('core.data.validate_gadget_native_particle_ids'),
                  patch('evolution.update_manifest'),
                  patch('evolution.gadget_config_text',return_value='fixture options'),
                  patch('evolution.snapshot_redshifts',return_value=[0]),
                  patch('evolution.write_output_list',side_effect=RuntimeError('stop before launching')),
                  patch('evolution.atomic_copy') as copying):
                with self.assertRaisesRegex(RuntimeError,'stop before launching'):
                    run_gadget(cfg,run,run/'ic.hdf5',validated_binary=binary)
                copying.assert_not_called()
            self.assertEqual(binary.read_bytes(),b'fixture executable')

# ========================================================================
# test_halo_native_ownership
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
import halos


class HaloNativeOwnershipTests(unittest.TestCase):
    def test_public_routes_reject_another_thread_owner_before_any_native_work(self):
        routes=[(halos.run_gadget_groups,{}),(halos.rockstar_config,{}),(halos.run_rockstar,{}),
                (halos.run_consistent_trees,None),(halos.prepare_rockstar_host_catalogues,{}),(halos.run_all,{})]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with native_run_lock(root),concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                for function,cfg in routes:
                    args=(root,) if cfg is None else (root,cfg)
                    future=pool.submit(function,*args)
                    with self.assertRaisesRegex(RuntimeError,'owned'):future.result(timeout=10)
            self.assertFalse((root/'data').exists())

    def test_skipped_rockstar_status_uses_complete_durable_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch('halos.rockstar_worker_count',return_value=4),patch('halos.cpu_thread_budget',return_value=4):
                status=halos.run_rockstar(root,{})
            path=root/'data/rockstar/rockstar_status.json'
            self.assertEqual(json.loads(path.read_text()),status)
            self.assertEqual(status['status'],'SKIPPED')

# ========================================================================
# test_native_manifest

test_native_manifest_CORE = Path(__file__).resolve().parent / 'core'
sys.path.insert(0, str(test_native_manifest_CORE))
from evolution import _write_json_atomic, record_manifest_fields, update_manifest


class NativeManifestTests(unittest.TestCase):
    def test_disk_full_and_nonfinite_keep_previous_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'manifest.json'
            _write_json_atomic(path, {'status': 'RUNNING'})
            original = path.read_bytes()
            for value, failure in [({'status': 'COMPLETE'}, OSError(errno.ENOSPC, 'disk full')),
                                   ({'measurement': float('nan')}, None)]:
                with self.subTest(value=value):
                    if failure:
                        with patch('core.foundation.os.fsync', side_effect=failure):
                            with self.assertRaises(OSError):
                                _write_json_atomic(path, value)
                    else:
                        with self.assertRaises(ValueError):
                            _write_json_atomic(path, value)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(list(root.iterdir()), [path])

    def test_stage_fields_are_one_commit_and_preserve_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'manifest.json'
            _write_json_atomic(path, {'seed': 718, 'input_artifacts': {'sha256': 'immutable'},
                                      'status': 'RUNNING'})
            fields = {'status': 'SIMULATION_INCOMPLETE',
                      'gadget_run': {'status': 'INCOMPLETE_OUTPUTS', 'error': 'missing snapshot'}}
            with patch('core.foundation.os.replace', wraps=os.replace) as replacing:
                record_manifest_fields(root, fields)
                self.assertEqual(replacing.call_count, 1)
            saved = json.loads(path.read_text())
            self.assertEqual({k: saved[k] for k in fields}, fields)
            self.assertEqual(saved['seed'], 718)
            update_manifest(root, 'schedule', {'status': 'FAIL'})
            self.assertEqual(json.loads(path.read_text())['input_artifacts'], {'sha256': 'immutable'})

    def test_malformed_original_fails_without_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'manifest.json'
            path.write_bytes(b'{"status":')
            with self.assertRaises(json.JSONDecodeError):
                record_manifest_fields(root, {'status': 'COMPLETE'})
            self.assertEqual(path.read_bytes(), b'{"status":')
            self.assertEqual(list(root.iterdir()), [path])

    def test_file_fsync_precedes_rename_and_directory_fsync(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            events = []
            original_replace = os.replace
            original_fsync = os.fsync
            def syncing(fd):
                events.append('fsync')
                return original_fsync(fd)
            def replacing(source, target):
                events.append('rename')
                return original_replace(source, target)
            with patch('core.foundation.os.fsync', side_effect=syncing), \
                 patch('core.foundation.os.replace', side_effect=replacing):
                _write_json_atomic(path, {'status': 'COMPLETE'})
            self.assertEqual(events, ['fsync', 'rename', 'fsync'])

    def test_directory_sync_failure_is_reported_after_complete_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            with patch('core.foundation.fsync_directory', side_effect=OSError(errno.EIO, 'directory failure')):
                with self.assertRaises(OSError):
                    _write_json_atomic(path, {'status': 'COMPLETE'})
            # Rename already happened; callers must not claim durable success.
            self.assertEqual(json.loads(path.read_text()), {'status': 'COMPLETE'})
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_sigkill_before_rename_keeps_old_stage_then_replay_commits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'manifest.json'
            _write_json_atomic(path, {'status': 'RUNNING', 'seed': 81})
            original = path.read_bytes()
            script = """
import os, signal, sys
sys.path.insert(0, sys.argv[1])
import core.foundation as research_acceptance
from evolution import record_manifest_fields
real_fsync = os.fsync
def kill_after_flush(fd):
    real_fsync(fd)
    os.kill(os.getpid(), signal.SIGKILL)
research_acceptance.os.fsync = kill_after_flush
record_manifest_fields(sys.argv[2], {'status':'COMPLETE','gadget_run':{'status':'COMPLETE'}})
"""
            result = subprocess.run([sys.executable, '-c', script, str(test_native_manifest_CORE), str(root)],
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, -9, result.stderr.decode())
            self.assertEqual(path.read_bytes(), original)
            orphaned = set(root.iterdir()) - {path}
            self.assertEqual(len(orphaned), 1)
            self.assertEqual(json.loads(next(iter(orphaned)).read_text())['status'], 'COMPLETE')
            record_manifest_fields(root, {'status': 'COMPLETE', 'gadget_run': {'status': 'COMPLETE'}})
            saved = json.loads(path.read_text())
            self.assertEqual(saved['seed'], 81)
            self.assertEqual(saved['status'], saved['gadget_run']['status'])
            self.assertTrue(all(p.exists() for p in orphaned))

# ========================================================================
# test_zoom_host_tracking

sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
import verification as test_zoom_host_tracking_fixtures
from zoom import recover_host_regions


class HostTrackingTests(unittest.TestCase):
    def setUp(self):
        fixture=test_zoom_host_tracking_fixtures.CatalogueTargetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture=fixture
        self.root=fixture.root
        with h5py.File(fixture.snap,'r+') as source:
            source['Header'].attrs.update(Time=.5,Redshift=1.)
        with h5py.File(fixture.catalogue,'r+') as source:
            source['Header'].attrs['Time']=.5
        self.next_snap=self.root/'data/snapshot_001.hdf5'
        self.next_cat=self.root/'data/fof_subhalo_tab_001.hdf5'
        shutil.copy2(fixture.snap,self.next_snap)
        shutil.copy2(fixture.catalogue,self.next_cat)
        self.ids=fixture.fixture.ids
        with h5py.File(self.next_snap,'r+') as source:
            source['Header'].attrs.update(Time=1.,Redshift=0.)
            source['PartType1/ParticleIDs'][:]=self.ids[[2,0,1]]
            source['PartType1/Coordinates'][:]=[[50.,0,0],[39.,0,0],[41.,0,0]]
        with h5py.File(self.next_cat,'r+') as source:
            source['Header'].attrs['Time']=1.
            source['Group/GroupOffsetType'][:,1]=[0,1]
            source['Group/GroupLenType'][:,1]=[1,2]
            source['Group/GroupPos'][:]=[[50.,0,0],[40.,0,0]]
            source['Group/Group_R_Crit200'][:]=[1.,.7]

    def request(self):
        return dict(anchor_snapshot=0,anchor_host_row=0,snapshots=[0,1],aperture_r200c_factor=4.,
                    minimum_anchor_fraction=.9,minimum_shared_particles=2,maximum_runner_up_fraction=.1,memory_gib=1.)

    def test_exact_uint64_ids_follow_changed_row_and_actual_aperture(self):
        report,ids=recover_host_regions(self.root,**self.request())
        self.assertEqual(report['status'],'PASS')
        np.testing.assert_array_equal(ids,self.ids[:2])
        self.assertEqual([row['host_fof_row'] for row in report['records']],[0,1])
        self.assertEqual([row['shared_anchor_particles'] for row in report['records']],[2,2])
        self.assertEqual(report['regions'][1]['center_comoving_kpc_h'],[40.,0.,0.])
        self.assertAlmostEqual(report['regions'][0]['radius_proper_kpc'],2.)
        self.assertAlmostEqual(report['regions'][1]['radius_proper_kpc'],4.)
        self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')

    def test_equal_split_is_unresolved_instead_of_reusing_old_row(self):
        with h5py.File(self.next_snap,'r+') as source:
            source['PartType1/ParticleIDs'][:]=self.ids[[0,2,1]]
        report,_=recover_host_regions(self.root,**self.request())
        self.assertEqual(report['status'],'FAIL')
        self.assertEqual(report['records'][1]['status'],'UNRESOLVED_IDENTITY')
        self.assertNotIn('region',report['records'][1])
        self.assertEqual(len(report['regions']),1)

    def test_duplicate_anchor_id_and_reversed_epoch_are_rejected(self):
        request=self.request();request['snapshots']=[1,0]
        with self.assertRaisesRegex(ValueError,'increase in saved time'):
            recover_host_regions(self.root,**request)
        with h5py.File(self.next_snap,'r+') as source:
            source['PartType1/ParticleIDs'][:]=self.ids[[0,0,1]]
        with self.assertRaisesRegex(ValueError,'more than once'):
            recover_host_regions(self.root,**self.request())

    def test_memory_gate_and_invalid_thresholds_before_id_payload(self):
        request=self.request();request['memory_gib']=.001
        with self.assertRaises(MemoryError):
            recover_host_regions(self.root,**request)
        request=self.request();request['maximum_runner_up_fraction']=.9
        with self.assertRaises(ValueError):
            recover_host_regions(self.root,**request)

    def test_other_run_with_coincident_ids_cannot_share_native_namespace(self):
        with h5py.File(self.next_snap,'r+') as source:
            source['Parameters'].attrs['InitCondFile']='another-simulation/ics'
        with self.assertRaisesRegex(ValueError,'recorded IC namespace'):
            recover_host_regions(self.root,**self.request())

    def test_grouped_recovery_retains_exact_native_source_lineage(self):
        (self.root/'input').mkdir()
        inputs=[]
        for snap,path in enumerate((self.fixture.snap,self.next_snap)):
            original=self.root/f'original_{snap}.hdf5'
            shutil.copy2(path,original)
            with h5py.File(path,'r+') as source:
                source['Parameters'].attrs['InitCondFile']=str(original.with_suffix(''))
            inputs.append({'inputs_sha256':{str(original):digest(original)},
                           'outputs':[{'path':str(path),'sha256':digest(path)}]})
        receipt=self.root/'input/halo_finding.json'
        atomic_json(receipt,{'status':'COMPLETE','snapshots':inputs})
        report,_=recover_host_regions(self.root,**self.request())
        self.assertEqual(report['status'],'PASS')
        for row in report['records']:
            paths={item['path'] for item in row['source_files']}
            self.assertIn(str(receipt.resolve()),paths)
            self.assertIn(str((self.root/f"original_{row['snapshot']}.hdf5").resolve()),paths)
        with h5py.File(self.root/'original_1.hdf5','r+') as source:
            source['Parameters'].attrs['InitCondFile']='another-simulation/ics'
        with self.assertRaisesRegex(ValueError,'native source changed'):
            recover_host_regions(self.root,**self.request())

    def test_declared_resolution_must_match_actual_initial_ic_mass(self):
        with self.assertRaisesRegex(ValueError,'actual recorded IC inventory'):
            audit_zoom_regions(self.root,host_tracking=self.request(),fine_dm_mass_msun=1e9,
                maximum_coarse_mass_fraction=1.,minimum_fine_dm_count=0,minimum_stellar_count=0)
        request=self.request();request['minimum_anchor_fraction']=True
        with self.assertRaises(ValueError):
            recover_host_regions(self.root,**request)

    def audit(self,stars=0):
        return audit_zoom_regions(self.root,host_tracking=self.request(),
            fine_dm_mass_msun=1e10/.7,maximum_coarse_mass_fraction=0.,minimum_fine_dm_count=2,
            minimum_stellar_count=stars,output_directory=self.root/'tracked-audit')

    def test_tracked_apertures_sampling_provenance_and_exact_ids_are_retained(self):
        result=self.audit()
        self.assertEqual(result['status'],'PASS')
        destination=Path(result['directory'])
        report=json.loads((destination/'provenance.json').read_text())
        self.assertEqual(report['host_tracking']['records'][1]['host_fof_row'],1)
        self.assertEqual([row['fine_dm_count_inside'] for row in report['records']],[2,2])
        with h5py.File(destination/'anchor_particles.hdf5') as source:
            np.testing.assert_array_equal(source['ParticleIDs'][:],self.ids[:2])
        self.assertTrue((destination/'zoom_contamination.pdf').is_file())

    def test_clean_aperture_cannot_pass_missing_required_stellar_sampling(self):
        result=self.audit(stars=1)
        self.assertEqual(result['status'],'FAIL')
        report=json.loads((Path(result['directory'])/'provenance.json').read_text())
        self.assertEqual(report['records'][1]['contamination_status'],'PASS')
        self.assertEqual(report['records'][1]['status'],'FAIL_SAMPLING')

    def test_unresolved_identity_retains_failed_report_and_does_not_measure_wrong_host(self):
        with h5py.File(self.next_snap,'r+') as source:
            source['PartType1/ParticleIDs'][:]=self.ids[[0,2,1]]
        result=self.audit()
        report=json.loads((Path(result['directory'])/'provenance.json').read_text())
        self.assertEqual(result['status'],'FAIL')
        self.assertEqual(report['records'],[])
        self.assertEqual(report['host_tracking']['status'],'FAIL')

# ========================================================================
# test_ic_lpt
from initial_conditions import second_order_growth, mixed_fluid_2lpt, SecondOrderGrowth,radiation_growing_history


class GrowthTests(unittest.TestCase):
    def test_regular_growing_history_matches_exact_meszaros_solution(self):
        a=np.geomspace(1e-10,.1,1200);om,orr=.3,9e-5
        h=np.sqrt(om/a**3+orr/a**4)
        d,f=radiation_growing_history(a,h,omega_m0=om,omega_r0=orr,hubble0=1.)
        exact=1+1.5*om*a/orr
        np.testing.assert_allclose(d,exact/exact[-1],rtol=2e-7)
        np.testing.assert_allclose(f,1.5*om*a/orr/exact,rtol=2e-7)
        with self.assertRaisesRegex(ValueError,'too late'):
            radiation_growing_history(a[500:],h[500:],omega_m0=om,omega_r0=orr,hubble0=1.)
        with self.assertRaisesRegex(ValueError,'not radiation/matter'):
            radiation_growing_history(a,h*2,omega_m0=om,omega_r0=orr,hubble0=1.)

    def history(self):
        a = np.geomspace(1e-6,1,500)
        return a,a**-1.5,a,np.ones_like(a)

    def test_eds_at_multiple_user_selected_epochs(self):
        for z in (9,49,99,199):
            with self.subTest(z=z):
                result = second_order_growth(*self.history(),omega_m0=1,hubble0=1,
                    target_a=1/(1+z),seed_kind='eds')
                self.assertAlmostEqual(result.displacement_coefficient,3/7,places=8)
                self.assertAlmostEqual(result.velocity_coefficient,6/7,places=8)
                self.assertAlmostEqual(result.linear_growth_rate,1.,places=12)

    def test_normalization_cancels(self):
        a,h,d,f = self.history()
        values = [second_order_growth(a,h,d*scale,f,omega_m0=1,hubble0=1,
                  target_a=.01,seed_kind='eds') for scale in (1e-5,1,1e5)]
        np.testing.assert_allclose([v.displacement_coefficient for v in values],3/7,rtol=1e-8)

    def test_supplied_background_changes_kernel(self):
        a,h,d,f = self.history()
        base = second_order_growth(a,h,d,f,omega_m0=1,hubble0=1,target_a=.1,seed_kind='eds')
        # A synthetic expansion perturbation tests use of the supplied H;
        # it is not claimed to be a solved EDE cosmology.
        modified = h*np.sqrt(1+.4*np.exp(-((np.log(a)-np.log(.005))/.5)**2))
        other = second_order_growth(a,modified,d,f,omega_m0=1,hubble0=1,target_a=.1,seed_kind='eds')
        self.assertGreater(abs(base.displacement_coefficient-other.displacement_coefficient),.001)

    def test_radiation_regular_seed_and_explicit_seed_agree(self):
        a = np.geomspace(1e-10,.1,700)
        om,orr = .3,9e-5
        h = np.sqrt(om/a**3+orr/a**4)
        y = om*a/orr
        d = 1+1.5*y
        f = 1.5*y/d
        y0 = y[0]
        initial = [-1.5*y0-1.125*y0*y0+.0625*y0**3,
                   -1.5*y0-2.25*y0*y0+.1875*y0**3]
        radiation = second_order_growth(a,h,d,f,omega_m0=om,omega_r0=orr,hubble0=1,
                    target_a=.01,seed_kind='radiation_regular')
        explicit = second_order_growth(a,h,d,f,omega_m0=om,omega_r0=orr,hubble0=1,
                    target_a=.01,seed_kind='explicit',initial_second_order=initial)
        self.assertAlmostEqual(radiation.displacement_coefficient,explicit.displacement_coefficient,places=12)
        self.assertTrue(.3 < radiation.displacement_coefficient < .6)

    def test_seed_and_coverage_rejections(self):
        for kwargs in ({'target_a':2}, {'seed_kind':'LCDM'}, {'omega_r0':1e-4},
                       {'seed_kind':'explicit'}, {'omega_m0':.3}):
            settings = dict(omega_m0=1,hubble0=1,target_a=.01,seed_kind='eds')
            settings.update(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                second_order_growth(*self.history(),**settings)


class MixedFluidTests(unittest.TestCase):
    def setUp(self):
        self.n = 16
        self.q = np.indices((self.n,)*3)/self.n
        self.x,self.y,self.z = self.q
        self.growth = SecondOrderGrowth(.01,3/7,6/7,1.,1e-6,'eds',1e-10)
        self.kw = dict(baryon_fraction=.2,box_mpc_h=1.,growth=self.growth,
                       aH_km_s_Mpc=100.,hubble_parameter=.7)

    def test_one_plane_wave_has_exact_first_order_and_no_second_order(self):
        delta = .01*np.cos(2*np.pi*self.x+.3)
        result = mixed_fluid_2lpt(delta,delta,**self.kw)
        np.testing.assert_allclose(result['displacement'][...,0],-.01*np.sin(2*np.pi*self.x+.3)/(2*np.pi),atol=1e-16)
        np.testing.assert_allclose(result['second_order_displacement'],0.,atol=1e-16)
        np.testing.assert_allclose(result['velocity_km_s'],result['displacement']*100/.7,atol=1e-14)

    def test_crossed_plane_waves_have_analytic_quadratic_source(self):
        delta = .01*np.cos(2*np.pi*self.x)+.02*np.cos(2*np.pi*self.y)
        result = mixed_fluid_2lpt(delta,delta,**self.kw)
        np.testing.assert_allclose(result['second_order_source'],.0002*np.cos(2*np.pi*self.x)*np.cos(2*np.pi*self.y),atol=1e-17)
        np.testing.assert_allclose(result['second_order_velocity_km_s'],result['second_order_displacement']*2*100/.7,atol=1e-14)

    def test_compensated_relative_density_changes_masses_not_common_flow(self):
        relative = .03*np.sin(2*np.pi*self.x)
        total = .01*np.cos(2*np.pi*self.y)
        dark,baryon = total-.2*relative,total+.8*relative
        result = mixed_fluid_2lpt(dark,baryon,**self.kw)
        reference = mixed_fluid_2lpt(total,total,**self.kw)
        np.testing.assert_allclose(result['displacement'],reference['displacement'],atol=1e-17)
        np.testing.assert_allclose(.8*result['mass_weights']['dark']+.2*result['mass_weights']['baryon'],1.,atol=1e-15)
        np.testing.assert_allclose(result['compensated_density']['dark'],-.2*relative,atol=1e-17)

    def test_second_order_scales_quadratically_and_preserves_phase(self):
        delta = .01*(np.cos(2*np.pi*self.x+.7)+np.cos(2*np.pi*self.y-.2))
        a = mixed_fluid_2lpt(delta,delta,**self.kw)
        b = mixed_fluid_2lpt(2*delta,2*delta,**self.kw)
        np.testing.assert_allclose(b['second_order_displacement'],4*a['second_order_displacement'],atol=1e-17)
        np.testing.assert_allclose(b['first_order_displacement'],2*a['first_order_displacement'],atol=1e-17)

    def test_zero_fields_produce_exact_zero_flow_and_unit_masses(self):
        zero = np.zeros((16,)*3)
        result = mixed_fluid_2lpt(zero,zero,**self.kw)
        self.assertTrue(np.array_equal(result['displacement'],np.zeros((16,16,16,3))))
        self.assertTrue(np.array_equal(result['mass_weights']['dark'],np.ones((16,)*3)))

    def test_coarse_fine_common_nodes_preserve_mixed_quadratic_flow(self):
        from initial_conditions import validate_refinement_continuity
        results=[]
        for n in (16,32):
            x,y,z=np.indices((n,)*3)/n
            common=.01*np.cos(2*np.pi*x+.3)+.02*np.cos(2*np.pi*y-.2)
            relative=.03*np.sin(2*np.pi*z)
            result=mixed_fluid_2lpt(common-.2*relative,common+.8*relative,**self.kw)
            results.append(result)
        q=np.moveaxis(np.indices((16,)*3)/16,0,-1).reshape(-1,3)
        for field in ('displacement','velocity_km_s','second_order_displacement'):
            coarse=results[0][field].reshape(-1,3)
            fine=results[1][field][::2,::2,::2].reshape(-1,3)
            report=validate_refinement_continuity(q,coarse,q,fine,box=1,tolerance=1e-13)
            self.assertEqual(report['status'],'PASS')

    def test_relative_velocity_nonpositive_mass_nyquist_and_budget_rejected(self):
        delta = .01*np.cos(2*np.pi*self.x)
        with self.assertRaisesRegex(ValueError,'relative velocity'):
            mixed_fluid_2lpt(delta,delta,relative_theta=delta,**self.kw)
        with self.assertRaisesRegex(ValueError,'homogeneous'):
            mixed_fluid_2lpt(delta+1,delta,**self.kw)
        with self.assertRaisesRegex(ValueError,'nonpositive'):
            mixed_fluid_2lpt(-4*np.cos(2*np.pi*self.x),4*np.cos(2*np.pi*self.x),**self.kw)
        with self.assertRaisesRegex(ValueError,'Nyquist'):
            nyquist = np.cos(np.pi*np.indices((16,)*3)[0])
            mixed_fluid_2lpt(nyquist,nyquist,**self.kw)
        with self.assertRaises(MemoryError):
            mixed_fluid_2lpt(delta,delta,memory_gib=.001,**self.kw)

    def test_retained_long_modes_agree_on_two_refinement_meshes(self):
        delta = .01*(np.cos(2*np.pi*self.x)+np.cos(2*np.pi*self.y))
        coarse = mixed_fluid_2lpt(delta,delta,**self.kw)
        q = np.indices((32,)*3)/32
        fine_delta = .01*(np.cos(2*np.pi*q[0])+np.cos(2*np.pi*q[1]))
        fine = mixed_fluid_2lpt(fine_delta,fine_delta,**self.kw)
        np.testing.assert_allclose(coarse['displacement'],fine['displacement'][::2,::2,::2],atol=1e-17)
        np.testing.assert_allclose(coarse['velocity_km_s'],fine['velocity_km_s'][::2,::2,::2],atol=1e-14)

class RelativeVelocityReferenceTests(unittest.TestCase):
    def test_declared_linear_relative_velocity_has_correct_species_momentum(self):
        n=16;q=np.indices((n,)*3)/n
        zero=np.zeros((n,)*3);theta=.01*np.cos(2*np.pi*q[0])
        growth=SecondOrderGrowth(.01,3/7,6/7,1.,1e-6,'eds',1e-10)
        result=mixed_fluid_2lpt(zero,zero,baryon_fraction=.2,box_mpc_h=1.,growth=growth,
            aH_km_s_Mpc=100,hubble_parameter=.7,relative_theta=theta,relative_velocity_order=1)
        dark,baryon=result['species_velocities_km_s'].values()
        np.testing.assert_allclose(.8*dark+.2*baryon,0.,atol=1e-17)
        np.testing.assert_allclose((baryon-dark)[...,0],.01*np.sin(2*np.pi*q[0])/(2*np.pi),atol=1e-17)
        self.assertEqual(result['relative_velocity_order'],1)

# ========================================================================
# test_ic_transform_publication

test_ic_transform_publication_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(test_ic_transform_publication_ROOT),str(test_ic_transform_publication_ROOT/'core')]
from core.data import (transactional_ic_transform, replay_ic_transform,
                       assert_ic_publication_complete)
from evolution import rescale_monofonic_ic, initial_condition_files


class ICTransformPublicationTests(unittest.TestCase):
    def fixture(self, root, name='ics.hdf5'):
        path=root/name
        with h5py.File(path,'w') as saved:
            header=saved.create_group('Header')
            header.attrs['BoxSize']=np.array([2.])
            header.attrs['Time']=np.array([.25])
            header.attrs['NumFilesPerSnapshot']=1
            group=saved.create_group('PartType1')
            group.create_dataset('Coordinates',data=[[2.1,-.1,1.]],fletcher32=True)
            group.create_dataset('ParticleIDs',data=np.array([2**64-1],dtype=np.uint64),fletcher32=True)
        return path

    def transform(self, paths):
        for path in paths:
            with h5py.File(path,'r+') as saved:
                saved['PartType1/Coordinates'][:]*=2
        return {'model':'declared arithmetic control','science_status':'NOT_ACCEPTED'}

    def test_actual_monofonic_transform_preserves_original_and_exact_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();path=self.fixture(root);original=path.read_bytes()
            rescale_monofonic_ic(path,2.)
            with h5py.File(path) as saved:
                np.testing.assert_allclose(saved['PartType1/Coordinates'][:],[[100,1900,1000]],atol=1e-12)
                self.assertEqual(saved['PartType1/ParticleIDs'][0],np.uint64(2**64-1))
                self.assertEqual(saved['Header'].attrs['BoxSize'],2000.)
            journal=next(root.glob('.ic-transform-*.json'))
            record=json.loads(journal.read_text())
            self.assertEqual((root/record['archive']/'original'/path.name).read_bytes(),original)
            self.assertEqual(replay_ic_transform(journal)['status'],'COMPLETE')

    def test_failed_final_model_gate_keeps_every_original_byte(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();paths=[self.fixture(root,f'ics.{i}.hdf5') for i in range(2)]
            before=[p.read_bytes() for p in paths]
            def failing(staged):
                self.transform(staged)
                raise ValueError('physical validation failed after staged write')
            with self.assertRaisesRegex(ValueError,'physical validation'):
                transactional_ic_transform(paths,failing,operation='control',configuration={})
            self.assertEqual([p.read_bytes() for p in paths],before)
            self.assertTrue(list(root.glob('.ic-transform-*-stage-*/failure.json')))
            self.assertEqual(list(root.glob('.ic-transform-*.json')),[])

    def test_interrupted_multishard_commit_blocks_discovery_and_replays(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();paths=[self.fixture(root,f'ics.{i}.hdf5') for i in range(2)]
            import core.foundation as research_acceptance
            original=research_acceptance.atomic_copy
            def copying(source,destination):
                result=original(source,destination)
                if Path(destination)==paths[0]:raise KeyboardInterrupt('publication gap')
                return result
            with patch('core.foundation.atomic_copy',side_effect=copying):
                with self.assertRaises(KeyboardInterrupt):
                    transactional_ic_transform(paths,self.transform,operation='control',configuration={})
            with self.assertRaisesRegex(ValueError,'Incomplete IC'):
                initial_condition_files(paths[0])
            journal=next(root.glob('.ic-transform-*.json'))
            self.assertEqual(replay_ic_transform(journal)['status'],'COMPLETE')
            assert_ic_publication_complete(paths)
            for path in paths:
                with h5py.File(path) as saved:
                    np.testing.assert_array_equal(saved['PartType1/Coordinates'][:],[[4.2,-.2,2.]])

    def test_corrupt_archive_or_changed_destination_prevents_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();path=self.fixture(root)
            transactional_ic_transform([path],self.transform,operation='control',configuration={})
            journal=next(root.glob('.ic-transform-*.json'));record=json.loads(journal.read_text())
            (root/record['archive']/'transformed'/path.name).write_bytes(b'corrupt')
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'archive is corrupt'):replay_ic_transform(journal)
            self.assertEqual(path.read_bytes(),before)

    def test_disk_full_and_wrong_box_do_not_edit_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();path=self.fixture(root);before=digest(path)
            with patch('core.foundation.os.fsync',side_effect=OSError(errno.ENOSPC,'full')):
                with self.assertRaises(OSError):rescale_monofonic_ic(path,2.)
            self.assertEqual(digest(path),before)
            with self.assertRaises(ValueError):rescale_monofonic_ic(path,3.)
            self.assertEqual(digest(path),before)

    def test_real_sigkill_commit_replays_without_applying_transform_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();path=self.fixture(root)
            script='''
import os, signal, sys
from pathlib import Path
sys.path[:0]=[sys.argv[1]]
import core.foundation as research_acceptance
from evolution import rescale_monofonic_ic
original=research_acceptance.atomic_copy
destination=Path(sys.argv[2]).resolve()
def copying(source,target):
    value=original(source,target)
    if Path(target)==destination:os.kill(os.getpid(),signal.SIGKILL)
    return value
research_acceptance.atomic_copy=copying
rescale_monofonic_ic(destination,2.)
'''
            result=subprocess.run([sys.executable,'-c',script,str(test_ic_transform_publication_ROOT/'core'),str(path)],capture_output=True,timeout=60)
            self.assertEqual(result.returncode,-9,result.stderr.decode())
            replay_ic_transform(next(root.glob('.ic-transform-*.json')))
            with h5py.File(path) as saved:
                np.testing.assert_allclose(saved['PartType1/Coordinates'][:],[[100,1900,1000]],atol=1e-12)

# ========================================================================
# test_snapshot_schedule

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from evolution import collect_gadget_snapshot_schedule


class SnapshotScheduleTests(unittest.TestCase):
    def test_saved_epochs_must_be_finite_consistent_and_increasing(self):
        cfg = {'cosmology': {'model': 'lcdm'}, 'simulation': {
            'snapshot_redshift_list': [1., 0.], 'start_redshift': 99.,
            'end_redshift': 0., 'num_files_per_snapshot': 1}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'data').mkdir()
            paths = [root / 'data' / f'snapshot_{i:03d}.hdf5' for i in range(2)]
            for path, a, z in zip(paths, [.5, 1.], [1., 0.]):
                with h5py.File(path, 'w') as source:
                    header = source.create_group('Header')
                    header.attrs.update(Time=a, Redshift=z, NumFilesPerSnapshot=1)
            self.assertEqual(collect_gadget_snapshot_schedule(root, cfg)['status'], 'MATCHED')
            for a, z, message in [(1., float('nan'), 'invalid'), (.6, 0., 'inconsistent'),
                                   (.5, 1., 'increase strictly')]:
                with h5py.File(paths[1], 'r+') as source:
                    source['Header'].attrs.update(Time=a, Redshift=z)
                with self.subTest(a=a, z=z):
                    with self.assertRaisesRegex(ValueError, message):
                        collect_gadget_snapshot_schedule(root, cfg)

TEST_GROUPS = {'test_visualization_numpy_contracts': ['VisualizationNumpyTests'], 'test_product_contract_cli': ['ProductContractCliTests'], 'test_zoom_target': ['TargetTests'], 'test_zoom_target_publication': ['TargetPublicationTests'], 'test_native_source_state': ['NativeSourceStateTests'], 'test_gas_initial_temperature': ['InitialTemperatureTests'], 'test_analysis_ownership': ['AnalysisOwnershipTests'], 'test_zoom_initial_conditions': ['ZoomAuditTests'], 'test_ic_wdm': ['ThermalTests'], 'test_modular_cosmology_publication': ['ModularCosmologyTests'], 'test_comparison_contracts': ['ComparisonContractTests'], 'test_zoom_staging': ['StagingTests'], 'test_observation_model': ['ObservationModelTests'], 'test_observer_publication': ['ObserverPublicationTests'], 'test_zoom_catalogue_target': ['CatalogueTargetTests'], 'test_halo_shape_profiles': ['HaloShapeProfilesTests'], 'test_stellar_history': ['StellarHistoryTests'], 'test_research_outputs': ['ResearchOutputTests'], 'test_scientific_table_publication': ['ScientificTableTests'], 'test_analysis_contract': ['AnalysisContractTests'], 'test_analysis_dataset_semantics': ['AnalysisDatasetSemanticsTests'], 'test_figure_review_publication': ['FigureReviewTests'], 'test_native_text_writes': ['NativeTextWriteTests'], 'test_zoom_window': ['ZoomWindowTests'], 'test_galaxy_publication': ['GalaxyPublicationTests'], 'test_snapshot_density_modes': ['SnapshotDensityModesTests'], 'test_research_acceptance': ['AcceptanceTests'], 'test_saved_dataset_units': ['SavedUnitTests'], 'test_build_cache_receipt': ['BuildCacheReceiptTests'], 'test_resolution_budget': ['ResolutionBudgetTests'], 'test_observer_coordinates': ['ObserverCoordinatesTests'], 'test_radial_shape_workflow': ['RadialShapeWorkflowTests'], 'test_radial_shape_batch': ['RadialShapeBatchTests'], 'test_start_epoch_campaign': ['CampaignTests'], 'test_data_contract': ['DataContractTests'], 'test_blender_data_contract': ['BlenderContractTests'], 'test_blender_single_publication': ['BlenderSinglePublicationTests'], 'test_blender_scene_routes': ['BlenderSceneRouteTests'], 'test_finder_comparison': ['FinderComparisonTests'], 'test_exact_output_timeline': ['ExactOutputTimelineTests'], 'test_native_catalogue_bundles': ['NativeCatalogueBundleTests'], 'test_inference_figures': ['FigureTests'], 'test_zoom_resolution': ['ZoomResolutionTests'], 'test_numpy_contract_integrity': ['NumpyContractIntegrityTests'], 'test_native_identity_domains': ['NativeIdentityDomainTests'], 'test_native_execution_receipts': ['NativeExecutionTests'], 'test_native_source_manifests': ['NativeSourceManifestTests'], 'test_sealed_blender_invocation': ['SealedBlenderInvocationTests'], 'test_filament_publication': ['FilamentPublicationTests'], 'test_block07_08_integration': ['BlockIntegrationTests'], 'test_zoom_phases': ['PhaseTests'], 'test_ic_validation': ['BundleTests', 'RealizationTests', 'GeneratorTests', 'PublicAuditTests', 'ZoomAuditTestsForExport'], 'test_snapshot_intake_publication': ['SnapshotIntakeTests'], 'test_paired_report_publication': ['PdfFixture', 'PairedReportPublicationTests'], 'test_analysis_finder_contracts': ['AnalysisFinderContractTests'], 'test_native_checkpoints': ['CheckpointTests'], 'test_identity_graph': ['IdentityGraphTests'], 'test_event_cadence': ['EventCadenceTests'], 'test_external_hdf5_contract': ['ExternalHdf5ContractTests'], 'test_cosmology_product_publication': ['CosmologyProductTests'], 'test_hdf5_contract_integrity': ['Hdf5ContractIntegrityTests'], 'test_zoom_generation': ['ZoomGenerationTests', 'GeneratedZoomReaderTests'], 'test_atomic_native_copy': ['AtomicNativeCopyTests'], 'test_solver_route_contracts': ['SolverRouteTests'], 'test_native_numerical_controls': ['AnalyticControlTests', 'NativeHarnessContractTests', 'IsolatedCheckpointClockTests', 'ExecutedNativeCampaignTests'], 'test_science_types': ['ScienceTypesTests'], 'test_checkpoint_clock': ['ClockEnvelopeTests', 'NativePeriodicEpochTests'], 'test_ic_mixed_contract': ['MixedContractTests'], 'test_native_hdf5_sets': ['NativeSetTests'], 'test_stellar_finder_publication': ['StellarFinderPublicationTests'], 'test_black_hole_model': ['BlackHoleReferenceTests'], 'test_ic_species_reference': ['SpeciesReferenceTests'], 'test_model_exports': ['ModelExportTests', 'NumericAuditTests'], 'test_identity_history': ['IdentityHistoryTests'], 'test_native_run_lock': ['NativeRunLockTests'], 'test_finder_adapters': ['FinderAdapterTests'], 'test_scene_product_publication': ['ScenePublicationTests'], 'test_zoom_target_coverage': ['CoverageTests'], 'test_subhalo_publication': ['SubhaloPublicationTests'], 'test_checkpoint_recovery': ['RecoveryTests'], 'test_zoom_resources': ['ZoomResourceTests'], 'test_black_hole_analysis': ['BlackHoleAnalysisTests'], 'test_history_publication': ['HistoryPublicationTests'], 'test_zoom_handoff': ['HandoffTests'], 'test_map_publication': ['MapPublicationTests'], 'test_comparison_map_publication': ['ComparisonMapPublicationTests'], 'test_resource_limits': ['ResourceLimitTests'], 'test_native_portability_patches': ['NativePortabilityPatchTests'], 'test_catalogue_kinematics': ['CatalogueKinematicsTests'], 'test_isolated_merger_publication': ['IsolatedMergerTests'], 'test_catalogue_publication': ['CataloguePublicationTests'], 'test_scientific_product': ['ScientificProductTests'], 'test_zoom_softening': ['ZoomSofteningTests'], 'test_stellar_blender_assets': ['StellarBlenderAssetsTests'], 'test_xdmf_publication': ['XdmfPublicationTests'], 'test_suite_shared_contracts': ['SuiteSharedContractTests'], 'test_inference_publication': ['InferencePublicationTests'], 'test_halo_growth_publication': ['CapturingEncoder', 'HaloGrowthPublicationTests'], 'test_history_hydro_contracts': ['HistoryHydroContractTests'], 'test_mixed_zoom_staging': ['MixedStagingTests'], 'test_wdm_zoom_staging': ['RelicStagingTests'], 'test_shape_zoom_publication': ['ShapeZoomPublicationTests'], 'test_native_checkpoint_reader': ['NativeReceiptReaderTests'], 'test_ic_resources': ['ICResourceTests'], 'test_central_conversions': ['CentralConversionTests'], 'test_image_product_publication': ['FakeImage', 'ImageProductTests'], 'test_dome_lensing_contracts': ['DomeLensingContractTests'], 'test_deferred_campaign_contract': ['DeferredCampaignContractTests'], 'test_comparison_publication': ['ComparisonPublicationTests'], 'test_portable_package': ['PortablePackageTests'], 'test_prepared_binary': ['PreparedBinaryTests'], 'test_halo_native_ownership': ['HaloNativeOwnershipTests'], 'test_native_manifest': ['NativeManifestTests'], 'test_zoom_host_tracking': ['HostTrackingTests'], 'test_ic_lpt': ['GrowthTests', 'MixedFluidTests', 'RelativeVelocityReferenceTests'], 'test_ic_transform_publication': ['ICTransformPublicationTests'], 'test_snapshot_schedule': ['SnapshotScheduleTests']}

class StreamHeatingCalibrationTests(unittest.TestCase):
    def fixture(self, root):
        from core.native import run_native_command
        ids = [2**63+11, 2**63+12, 2**63+13]
        potential = root/'potential.json'; atomic_json(potential, {'kind': 'control_fixture'})
        request = {'schema': 'stream-heating-calibration-v1', 'observable': 'stream',
            'selection': 'frozen tracer control', 'identity_namespace': 'control:stars', 'particle_ids': ids,
            'species': 4, 'bin_coordinate': 1., 'physics_fingerprint': digest(potential),
            'smooth_potential': {'path': str(potential), 'sha256': digest(potential)},
            'maximum_initial_velocity_difference_km_s': 1e-12,
            'maximum_initial_position_difference_proper_kpc': 1e-12, 'memory_gib': .1}
        for kind in ('granular', 'smooth'):
            run = root/kind; run.mkdir()
            background = run/'background.dat'
            background.write_text('# 1:z 2:proper time [Gyr] 3:Omega_r(z)\n1 1 0\n0 2 0\n')
            atomic_json(run/'manifest.json', {'fixture': 'ANALYTIC_CONTROL', 'input_artifacts': {
                'boltzmann_background': {'path': str(background), 'sha256': digest(background)}}})
            sets = []
            for number, a in enumerate((.5, 1.)):
                order = [2, 0, 1] if number else [0, 1, 2]
                test_stellar_history_snapshot(run, number, a, [ids[i] for i in order], np.zeros((3, 3)))
                path = run/'data'/f'snapshot_{number:03d}.hdf5'
                physical = np.zeros((3, 3)); physical[:, 0] = 10.
                if kind == 'granular' and number: physical[:, 0] += [-1., 0., 1.]
                with h5py.File(path, 'r+') as f:
                    f['PartType4/Velocities'][:] = physical[order]/np.sqrt(a)
                    f['PartType4/ParticleIDs'].attrs['identity_namespace'] = request['identity_namespace']
                    f['Header'].attrs.update(Research2Generation='control-'+str(number), Research2ShardCount=1)
                atomic_json(path.with_suffix('.publication.json'), {'schema': 'native-hdf5-set-publication-v1',
                    'state': 'COMPLETE', 'generation': 'control-'+str(number), 'shard_count': 1})
                sets.append([str(path)])
            receipt = run/'execution.json'
            run_native_command([sys.executable, '-c', 'pass'], cwd=run, receipt_path=receipt,
                               inputs=[potential, background], stdout=subprocess.DEVNULL)
            request[kind] = {'run_directory': str(run), 'snapshot_sets': sets, 'execution_receipt': str(receipt)}
        return request

    def test_mass_weighted_statistics_preserve_bulk_error_and_signed_growth(self):
        from zoom import stream_heating_statistics
        smooth = np.zeros((2, 3, 3)); granular = smooth.copy()
        granular[1, :, 0] = [1., 3., 0.]
        result = stream_heating_statistics(granular, smooth, [1., 1., 2.])
        self.assertAlmostEqual(result['heating_variance_km2_s2'][1], 2.5)
        self.assertAlmostEqual(result['incoherent_variance_km2_s2'][1], 1.5)
        self.assertEqual(result['mean_kick_km_s'][1, 0], 1.)
        granular[1, :, 0] = 7.
        result = stream_heating_statistics(granular, smooth, [1., 1., 2.])
        self.assertEqual(result['incoherent_variance_km2_s2'][1], 0.)
        self.assertEqual(result['heating_variance_km2_s2'][1], 49.)

    def test_calibration_retains_initial_anisotropy_and_bulk_velocity_separately(self):
        from zoom import stream_heating_statistics
        # Two correlated directions and unequal masses have an independent
        # population-covariance oracle, including the off-diagonal sign.
        initial=np.array([[1.,2.,0.],[3.,-2.,0.],[0.,0.,0.]])
        smooth=np.stack([initial+100.,initial+200.]);granular=smooth.copy()
        granular[1]+=np.array([7.,0.,0.])
        result=stream_heating_statistics(granular,smooth,[1.,1.,2.])
        expected=np.array([[1.5,-1.,0.],[-1.,2.,0.],[0.,0.,0.]])
        np.testing.assert_allclose(result['smooth_velocity_covariance_km2_s2'][0],expected)
        np.testing.assert_allclose(result['granular_velocity_covariance_km2_s2'],np.broadcast_to(expected,(2,3,3)))
        np.testing.assert_allclose(result['smooth_mean_velocity_km_s'][0],[101.,100.,100.])
        self.assertEqual(result['signed_dispersion_excess_km2_s2'][1],0.)
        self.assertEqual(result['heating_variance_km2_s2'][1],49.)

    def test_real_publication_reorder_clock_and_source_corruption(self):
        from zoom import publish_stream_heating_calibration, verify_stream_heating_calibration
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); request = self.fixture(root); path = root/'heating.json'
            result = publish_stream_heating_calibration(request, path)
            self.assertEqual(result['science_acceptance'], 'NOT_ACCEPTED')
            self.assertEqual(result['resolution_measurement']['origin'], 'synthetic_control')
            self.assertEqual(result['resolution_measurement']['duration_myr'], 1000.)
            self.assertEqual(verify_stream_heating_calibration(path), result)
            with np.load(str(path)+'.data/arrays.npz', allow_pickle=False) as arrays:
                self.assertAlmostEqual(arrays['null_stream_heating'][0], 2/3)
                self.assertEqual(arrays['particle_ids'].tolist(), request['particle_ids'])
            source = Path(request['smooth']['snapshot_sets'][0][0])
            with h5py.File(source, 'r+') as f: f['PartType4/Velocities'][0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'source changed'): verify_stream_heating_calibration(path)

    def test_mismatched_initial_state_missing_ids_and_unsealed_sets_rejected(self):
        from zoom import publish_stream_heating_calibration
        for failure in ('initial', 'position', 'missing', 'unsealed', 'epoch', 'units'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve(); request = self.fixture(root)
                number = 0 if failure in ('initial', 'position') else 1
                path = Path(request['smooth']['snapshot_sets'][number][0])
                with h5py.File(path, 'r+') as f:
                    if failure == 'initial': f['PartType4/Velocities'][0, 0] += 1
                    elif failure == 'position': f['PartType4/Coordinates'][0, 0] += 1
                    elif failure == 'missing': f['PartType4/ParticleIDs'][0] = 3
                    elif failure == 'epoch': f['Header'].attrs['Time'] = .8
                    elif failure == 'units': del f['PartType4/Velocities'].attrs['a_scaling']
                if failure == 'unsealed': path.with_suffix('.publication.json').unlink()
                with self.assertRaises(ValueError): publish_stream_heating_calibration(request, root/'failed.json')
                self.assertFalse((root/'failed.json').exists())

    def test_statistics_reject_empty_failed_or_unmatched_populations(self):
        from zoom import stream_heating_statistics
        for velocities, masses in [(np.zeros((2, 2, 3)), [1, 1]), (np.full((2, 3, 3), np.nan), [1, 1, 1]),
                                   (np.zeros((2, 3, 3)), [1, 0, 1])]:
            with self.assertRaises(ValueError): stream_heating_statistics(velocities, velocities, masses)


class StartEpochAnalysisBridgeTests(unittest.TestCase):
    def fixture(self, root):
        from analysis import _write_analysis_provenance
        spec = campaign_fixture(root/'source')
        spec['start_epoch_campaign']['observables']['shape']['bin_labels'] = ['1', '2']
        plan = build_start_epoch_campaign(spec, spec['start_epoch_campaign']['redshifts'], root/'campaign')
        campaign = root/'campaign/campaign.json'
        inputs = synthetic_ic_phase_inputs(root/'ics', plan)
        phase = root/'phases.json'
        audit_start_epoch_phases(campaign, inputs, phase, modes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        branch = plan['branches'][0]; run = root/'run'; run.mkdir()
        (run/'config.yaml').write_bytes(Path(branch['configuration']['path']).read_bytes())
        selected = []
        for number, a in enumerate((.1, .5, 1.)):
            test_stellar_history_snapshot(run, number, a, [11, 12, 13], np.zeros((3, 3)))
            snapshot = run/'data'/f'snapshot_{number:03d}.hdf5'
            with h5py.File(snapshot, 'r+') as f:
                f['Parameters'].attrs.update(TimeBegin=branch['start_scale_factor'], HubbleParam=.7)
            selected.append([str(snapshot)])
        directory = root/'analysis'; directory.mkdir(); file = directory/'analysis_products.h5'
        with h5py.File(file, 'w') as handle:
            for number, a in enumerate((.1, .5, 1.)):
                group = handle.create_group(f'snap_{number:03d}/shape'); group.attrs['status'] = 'COMPLETE'
                for name, values in (('value', [.7, .5]), ('bin', [1., 2.])):
                    dataset = group.create_dataset(name, data=values)
                    write_dataset_contract(dataset, units='dimensionless', quantity='fixture '+name,
                                           epoch=epoch_contract(a), provenance={'fixture': 'analysis bridge control'})
        _write_analysis_provenance(directory, run, ['shape'], 'control')
        products = {'shape': [{'directory': str(directory), 'file': file.name,
            'dataset': f'snap_{n:03d}/shape/value', 'coordinate_dataset': f'snap_{n:03d}/shape/bin',
            'snapshot_set': files} for n, files in enumerate(selected)]}
        return campaign, branch, products, phase, file

    def test_sealed_analysis_producer_connected_and_source_mutation_rejected(self):
        from evolution import write_start_epoch_analysis_measurements
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); campaign, branch, products, phase, file = self.fixture(root)
            path = root/'measurement.json'
            result = write_start_epoch_analysis_measurements(campaign, branch['id'], products, path, ic_phase_audit=phase)
            self.assertEqual(result['actual_scale_factors'], [.1, .5, 1.])
            self.assertEqual(result['science_acceptance'], 'NOT_ACCEPTED')
            self.assertIn(str(file), result['sources'])
            with np.load(str(path)+'.data/arrays.npz') as arrays:
                np.testing.assert_array_equal(arrays['observable_shape'], [[.7, .5]]*3)
            with h5py.File(file, 'r+') as f: f['snap_001/shape/value'][0] += 1
            with self.assertRaisesRegex(ValueError, 'artifact missing or changed'):
                write_start_epoch_analysis_measurements(campaign, branch['id'], products, root/'corrupt.json', ic_phase_audit=phase)
            self.assertFalse((root/'corrupt.json').exists())

    def test_wrong_bins_or_missing_observable_are_rejected_before_publication(self):
        from evolution import write_start_epoch_analysis_measurements
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); campaign, branch, products, phase, file = self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'exactly'):
                write_start_epoch_analysis_measurements(campaign, branch['id'], {}, root/'empty.json', ic_phase_audit=phase)
            products['shape'][0]['coordinate_dataset'] = 'snap_000/shape/value'
            with self.assertRaisesRegex(ValueError, 'bin coordinates'):
                write_start_epoch_analysis_measurements(campaign, branch['id'], products, root/'wrong.json', ic_phase_audit=phase)
            self.assertFalse((root/'wrong.json').exists())


class IdentityBatchAtomicityTests(unittest.TestCase):
    def graph(self, maximum_nodes=20):
        from halos import IdentityGraph
        graph = IdentityGraph(maximum_nodes=maximum_nodes, maximum_edges=20)
        graph.add_epoch(0, epoch_contract(.5)); graph.add_epoch(1, epoch_contract(1.))
        graph.add_node('halo', 2**64-1, 0, kind='halo')
        return graph

    def test_membership_failure_rolls_back_new_nodes_edges_and_duplicate_index(self):
        graph = self.graph(maximum_nodes=3)
        owner = ('halo', 2**64-1, 0)
        with self.assertRaises(MemoryError):
            graph.add_membership(owner, 'dm', [1, 2, 3], bound=True, evidence={'origin': 'native_catalogue'})
        self.assertEqual(len(graph.nodes), 1); self.assertEqual(graph.edges, [])
        self.assertEqual(graph._edge_keys, set())
        graph.add_membership(owner, 'dm', [1, 2], bound=True, evidence={'origin': 'native_catalogue'})
        self.assertEqual(len(graph.nodes), 3); self.assertEqual(len(graph.edges), 2)

    def test_node_and_membership_batch_rejection_preserves_original_state(self):
        graph = self.graph()
        with self.assertRaises(ValueError):
            graph.add_epoch_nodes(0, [{'namespace': 'stars', 'id': 7, 'kind': 'ssp'},
                                      {'namespace': 'stars', 'id': 7, 'kind': 'ssp'}])
        self.assertEqual(len(graph.nodes), 1)
        good = {'owner': ('halo', 2**64-1, 0), 'particle_namespace': 'dm', 'exact_particle_ids': [9],
                'bound': True, 'evidence': {'origin': 'native_catalogue'}}
        with self.assertRaises(ValueError): graph.add_membership_edges([good, {'owner': None}])
        self.assertEqual(len(graph.nodes), 1); self.assertEqual(graph.edges, [])

    def test_cycle_and_unverified_parent_zoom_batch_roll_back(self):
        graph = self.graph()
        graph.add_epoch_nodes(0, [{'namespace': 'bh', 'id': 1, 'kind': 'bh'}, {'namespace': 'bh', 'id': 2, 'kind': 'bh'}])
        first = ('bh', 1, 0); second = ('bh', 2, 0)
        links = [{'source': first, 'target': second, 'kind': 'bh_merger', 'evidence': {'origin': 'native_ancestry'}},
                 {'source': second, 'target': first, 'kind': 'bh_merger', 'evidence': {'origin': 'native_ancestry'}}]
        with self.assertRaisesRegex(ValueError, 'cycle'): graph.add_ancestry_edges(links)
        self.assertEqual(graph.edges, []); self.assertEqual(graph._edge_keys, set())
        with self.assertRaises(ValueError): graph.add_parent_zoom_edges([
            {'source': first, 'target': second, 'evidence': {'origin': 'lagrangian_identity'}},
            {'source': second, 'target': first, 'evidence': {'origin': 'row_equality'}}])
        self.assertEqual(graph.edges, [])




class ConsolidatedWorkflowTests(unittest.TestCase):




    def test_actual_gallery_pdf_is_available_and_integrity_verified(self):
        from figures import review_figure_folder, verify_figure_review
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); figure, axis = plt.subplots(); axis.plot([0., 1.], [1., 2.])
            export_figure(figure, root/'control', formats=['png'])
            report = review_figure_folder(root, create_gallery=True)
            self.assertEqual(report['status'], 'COMPLETE')
            self.assertIsNotNone(report['gallery_pdf'])
            self.assertTrue((root/report['gallery_pdf']).read_bytes().startswith(b'%PDF'))
            self.assertEqual(verify_figure_review(root/report['review_bundle']), report)



TEST_GROUPS['test_stream_heating_calibration'] = ['StreamHeatingCalibrationTests']
TEST_GROUPS['test_consolidated_workflows'] = ['ConsolidatedWorkflowTests']
TEST_GROUPS['test_start_epoch_analysis_bridge'] = ['StartEpochAnalysisBridgeTests']
TEST_GROUPS['test_identity_batches'] = ['IdentityBatchAtomicityTests']


class AnalyticSphericalPotentialTests(unittest.TestCase):
    def test_harmonic_cartesian_derivatives_include_origin(self):
        from core.action_angle import AnalyticSphericalPotential

        omega = 2.5
        positions = np.array([[0., 0., 0.], [1., -2., 3.]])
        potential, acceleration, hessian = AnalyticSphericalPotential(
            'harmonic', omega_kms_kpc=omega).evaluate(positions)
        np.testing.assert_allclose(potential, .5 * omega**2 * np.sum(positions**2, axis=1))
        np.testing.assert_allclose(acceleration, -omega**2 * positions)
        np.testing.assert_allclose(hessian, np.broadcast_to(omega**2 * np.eye(3), (2, 3, 3)))


class HaloStructureTests(unittest.TestCase):
    def sample(self, x=None, v=None, mass=None, first_id=0):
        x = np.array([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0]]) if x is None else np.asarray(x,dtype=float)
        v = np.array([[1.,2,0],[-1.,-2,0],[2.,1,0],[-2.,-1,0]]) if v is None else np.asarray(v,dtype=float)
        mass = np.ones(len(x)) if mass is None else np.asarray(mass,dtype=float)
        return {'offsets_proper_kpc':x,'velocities_peculiar_km_s':v,'masses_msun':mass,
                'particle_ids':np.arange(first_id,first_id+len(x),dtype=np.uint64)}

    def settings(self):
        return {'lmax':4,'mmax':4,'grid_size_r':20,'grid_size_z':20,
                'rmin_kpc':.05,'rmax_kpc':10.,'zmin_kpc':.05,'zmax_kpc':10.}

    def test_complete_component_mass_covariance_spin_and_degenerate_bins(self):
        from halos import measure_mass_density_kinematics_spin
        sample = self.sample()
        p = measure_mass_density_kinematics_spin({'dm':sample},[0,.5,1.5,2.],[0,0,0],gravity_constant=2)
        dm = p['dm']
        np.testing.assert_array_equal(dm['shell_count'],[0,4,0])
        np.testing.assert_array_equal(dm['enclosed_mass_msun'],[0,4,4])
        np.testing.assert_allclose(dm['velocity_covariance_km2_s2'][1],[[2.5,2,0],[2,2.5,0],[0,0,0]])
        np.testing.assert_allclose(dm['circular_velocity_spherical_km_s'],[0,np.sqrt(8/1.5),2])
        self.assertTrue(np.isnan(dm['anisotropy_beta'][0]))
        self.assertTrue(dm['kinematics_supported'][1])
        self.assertAlmostEqual(dm['density_shot_noise_msun_kpc3'][1],dm['density_msun_kpc3'][1]/2)
        np.testing.assert_allclose(dm['shell_mass_msun'],p['total']['shell_mass_msun'])
        changed = self.sample(v=sample['velocities_peculiar_km_s']+[100,20,-5])
        shifted = measure_mass_density_kinematics_spin({'dm':changed},[0,.5,1.5,2.],[100,20,-5],gravity_constant=2)
        np.testing.assert_allclose(shifted['dm']['velocity_covariance_km2_s2'],dm['velocity_covariance_km2_s2'],equal_nan=True)

    def test_duplicate_namespace_overlap_nonfinite_and_unphysical_inputs_rejected(self):
        from halos import measure_mass_density_kinematics_spin
        sample = self.sample()
        with self.assertRaisesRegex(ValueError,'overlap'):
            measure_mass_density_kinematics_spin({'dm':sample,'gas':sample},[0,1,2],[0,0,0])
        for key,value in [('particle_ids',np.array([0.,1.,2.,3.])),('masses_msun',[1,1,-1,1]),
                          ('offsets_proper_kpc',np.ones((3,3))),('velocities_peculiar_km_s',np.full((4,3),np.nan))]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                measure_mass_density_kinematics_spin({'dm':sample|{key:value}},[0,1,2],[0,0,0])

    def test_direct_field_kepler_force_hessian_and_translation(self):
        from halos import direct_host_field
        query = np.array([[2.,0,0],[0,3.,0],[0,0,4.]])
        p,a,h = direct_host_field([[0,0,0]],[5],query,gravity_constant=2,chunk_rows=1)
        r = np.linalg.norm(query,axis=1)
        np.testing.assert_allclose(p,-10/r)
        np.testing.assert_allclose(a,-10*query/r[:,None]**3)
        np.testing.assert_allclose(np.trace(h,axis1=1,axis2=2),0,atol=1e-14)
        for k,point in enumerate(query):
            np.testing.assert_allclose(h[k],10*(np.eye(3)/r[k]**3-3*np.outer(point,point)/r[k]**5))
        moved = direct_host_field([[10,4,-2]],[5],query+[10,4,-2],gravity_constant=2)
        for actual,expected in zip(moved,(p,a,h)):np.testing.assert_allclose(actual,expected)
        with self.assertRaisesRegex(ValueError,'coincides'):
            direct_host_field([[0,0,0]],[1],[[0,0,0]],gravity_constant=1)

    def test_agama_analytic_plummer_derivatives_rotation_and_reload(self):
        from halos import pinned_agama_backend,ReconstructedHostPotential
        agama,binding = pinned_agama_backend()
        g = binding['G_kpc_km2_s2_msun']
        mass,scale = 4e9,.7
        angle = .63
        rotation = np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
        model = ReconstructedHostPotential([('analytic',agama.Potential(type='Plummer',mass=mass,scaleRadius=scale),rotation)])
        q = np.array([[1.,2,3],[-1.,.2,.4],[0,0,0]])
        phi,acc,h = model.evaluate(q)
        r2 = np.sum(q*q,axis=1)+scale**2
        np.testing.assert_allclose(phi,-g*mass/np.sqrt(r2),rtol=1e-12)
        np.testing.assert_allclose(acc,-g*mass*q/r2[:,None]**1.5,rtol=1e-12,atol=1e-11)
        expected = g*mass*(np.eye(3)[None,:,:]/r2[:,None,None]**1.5-3*q[:,:,None]*q[:,None,:]/r2[:,None,None]**2.5)
        np.testing.assert_allclose(h,expected,rtol=1e-12,atol=1e-11)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'lossy'):
                model.export_coefficients(Path(d)/'analytic')
            from halos import fit_component_multipole_expansion
            x = np.random.default_rng(348).normal(size=(200,3))
            sample = self.sample(x=x,v=np.zeros_like(x),mass=np.full(200,mass/200))
            model,binding = fit_component_multipole_expansion({'dm':sample},
                {'dm':{'kind':'spheroidal','disk_rotation':rotation.tolist()}},self.settings())
            q = np.array([[1,2,3],[2,1,3],[0,0,5.]])
            phi,acc,h = model.evaluate(q)
            records = model.export_coefficients(Path(d)/'field')
            fresh = ReconstructedHostPotential.from_coefficients(Path(d)/'field',records)
            for order,(actual,expected) in enumerate(zip(fresh.evaluate(q),(phi,acc,h))):
                step = 1e-3*np.maximum(1.,np.linalg.norm(q,axis=1))
                envelope = 256*np.finfo(float).eps*np.maximum(1.,abs(phi))/step**order
                envelope = envelope.reshape((-1,)+(1,)*(actual.ndim-1))
                self.assertTrue(np.all(abs(actual-expected)<=envelope+5e-12*abs(expected)))
            (Path(d)/'field'/records[0]['file']).write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'binding'):
                ReconstructedHostPotential.from_coefficients(Path(d)/'field',records)

    def test_spherical_isopotential_has_no_spurious_orientation(self):
        from halos import pinned_agama_backend,ReconstructedHostPotential,measure_isopotential_geometry
        agama,binding = pinned_agama_backend()
        model = ReconstructedHostPotential([('sphere',agama.Potential(type='Plummer',mass=1e9,scaleRadius=1),np.eye(3))])
        geometry = measure_isopotential_geometry(model,[1,2],radial_bracket_kpc=[.1,4],directions=32,
            maximum_shape_residual=.01,minimum_eigenvalue_gap=.01)
        for row,radius in zip(geometry['levels'],[1,2]):
            self.assertEqual(row['status'],'MEASURED')
            np.testing.assert_allclose(row['axis_lengths_proper_kpc'],radius,rtol=1e-10)
            self.assertEqual(row['orientation_resolved'],[False]*3)
            self.assertIsNone(row['triaxiality'])

    def test_actual_spheroidal_and_disk_backend_and_frozen_direct_budgets(self):
        from halos import fit_component_multipole_expansion,compare_with_independent_direct_force_samples
        rng = np.random.default_rng(7382)
        x = rng.normal(size=(200,3))
        sample = self.sample(x=x,v=np.zeros_like(x),mass=np.full(200,1e7))
        for kind in ['spheroidal','disk']:
            with self.subTest(kind=kind):
                model,binding = fit_component_multipole_expansion({'dm':sample},
                    {'dm':{'kind':kind,'disk_rotation':np.eye(3).tolist()}},self.settings())
                budget = {'force_relative_tolerance':.03,'force_floor_km2_s2_kpc':.01,
                          'tidal_relative_tolerance':.05,'tidal_floor_km2_s2_kpc2':.01}
                result,data = compare_with_independent_direct_force_samples(model,{'dm':sample},
                    [[30,0,0],[0,30,0],[0,0,30]],budget,gravity_constant=binding['G_kpc_km2_s2_msun'])
                self.assertEqual(result['status'],'PASS',result)
                failed,_ = compare_with_independent_direct_force_samples(model,{'dm':sample},
                    [[1.1,.2,.3],[.2,1.1,.3],[.1,.2,1.3]],
                    budget|{'force_relative_tolerance':1e-10,'tidal_relative_tolerance':1e-10},
                    gravity_constant=binding['G_kpc_km2_s2_msun'])
                self.assertEqual(failed['status'],'FAIL')

    def test_declared_nfw_concentration_and_unbracketed_density(self):
        from halos import measure_declared_concentration
        radius = np.geomspace(.2,30,20)
        rs = 2.
        rho = 1e7/(radius/rs)/(1+radius/rs)**2
        enclosed = 4*np.pi*1e7*rs**3*(np.log1p(radius/rs)-radius/(radius+rs))
        profile = {'radius_proper_kpc':radius,'radial_outer_edge_proper_kpc':radius,
                   'density_msun_kpc3':rho,'density_shot_noise_msun_kpc3':rho*.02,
                   'effective_count':np.full(len(radius),2500.),'enclosed_mass_msun':enclosed}
        request = {'reference_density_msun_kpc3':1000.,'overdensity_delta':200.,
                   'definition_label':'supplied critical density at actual epoch','fit_min_kpc':.3,
                   'fit_max_kpc':25.,'rs_min_kpc':.1,'rs_max_kpc':10.,'minimum_effective_count':10}
        result = measure_declared_concentration({'total':profile},request)
        self.assertEqual(result['status'],'MEASURED')
        self.assertAlmostEqual(result['components']['total']['rs_proper_kpc'],rs,places=8)
        unresolved = measure_declared_concentration({'total':profile},request|{'reference_density_msun_kpc3':1e20})
        self.assertEqual(unresolved['status'],'UNBRACKETED_OVERDENSITY')

    def native_fixture(self):
        control = CatalogueTargetTests();control.setUp();self.addCleanup(control.doCleanups)
        rng = np.random.default_rng(382)
        n = 256
        x = rng.normal(size=(n,3))*.5
        with h5py.File(control.snap,'r+') as h:
            group = h['PartType1']
            original_units = {key:dict(group[key].attrs) for key in ['Coordinates','Velocities','Masses'] if key in group}
            for key,value in {'Coordinates':x%100,'Velocities':rng.normal(size=(n,3)),
                              'ParticleIDs':np.arange(2**63+1,2**63+n+1,dtype=np.uint64)}.items():
                if key in group:del group[key]
                group[key] = value
                if key in original_units:group[key].attrs.update(original_units[key])
            if not len(group['Velocities'].attrs):
                group['Velocities'].attrs.update(to_cgs=1e5,a_scaling=.5,h_scaling=0.)
            if 'Masses' in group:del group['Masses']
            group['Masses'] = np.ones(n)*1e-3
            mass_table = np.asarray(h['Header'].attrs['MassTable']).copy();mass_table[1]=0
            h['Header'].attrs['MassTable']=mass_table
            group['Masses'].attrs.update(original_units.get('Masses',{'to_cgs':1.98847e43,'a_scaling':0.,'h_scaling':-1.}))
            count = np.zeros(6,dtype=np.uint32);count[1] = n
            for key in ['NumPart_ThisFile','NumPart_Total']:h['Header'].attrs[key]=count
            h['Header'].attrs['NumPart_Total_HighWord']=np.zeros(6,dtype=np.uint32)
            h['Header'].attrs['Research2Generation']='block11-test'
            h['Header'].attrs['Research2ShardCount']=1
        with h5py.File(control.catalogue,'r+') as h:
            h['Group/GroupOffsetType'][1,1]=n-1
            h['Group/GroupLenType'][:,1]=[n-1,1]
        atomic_json(control.snap.with_suffix('.publication.json'),
                    {'schema':'native-hdf5-set-publication-v1','state':'COMPLETE','generation':'block11-test','shard_count':1})
        atomic_json(control.root/'manifest.json',{'fixture':True})
        shape = dict(window='spherical_shell',estimator='mass_quadrupole',minimum_particles=8,tolerance=1e-5,
                     maximum_iterations=50,minimum_axis_ratio=.1,bootstrap_realizations=32,seed=883,
                     minimum_bootstrap_success_fraction=.5,minimum_eigenvalue_gap=.03,maximum_orientation_error_deg=20.)
        request = {'schema':'native-halo-structure-v1','run_directory':str(control.root),'snapshot':0,'host_row':0,
                   'components':{'dm':{'species':[1],'kind':'spheroidal','disk_rotation':np.eye(3).tolist()}},
                   'support_radius_proper_kpc':4.,'radial_edges_proper_kpc':[0,.5,1.5,4.],
                   'bulk_velocity_peculiar_km_s':[0,0,0],'shape_controls':shape,'expansion_sequence':[self.settings()],
                   'calibration_positions_proper_kpc':[[30,0,0],[0,30,0],[0,0,30]],
                   'validation_positions_proper_kpc':[[25,2,0],[0,25,2],[2,0,25]],
                   'field_error_budget':{'force_relative_tolerance':.04,'force_floor_km2_s2_kpc':.001,
                                         'tidal_relative_tolerance':.05,'tidal_floor_km2_s2_kpc2':.001},
                   'equipotential_controls':{'reference_radii_kpc':[2.],'radial_bracket_kpc':[.1,5.],
                                             'directions':24,'maximum_shape_residual':.15,'minimum_eigenvalue_gap':.03},
                   'concentration':None,'memory_gib':1.}
        return control,request

    def test_actual_native_publication_exact_ids_source_and_coefficients(self):
        from halos import publish_halo_structure,verify_halo_structure
        control,request = self.native_fixture()
        path = control.root/'structure.json'
        report = publish_halo_structure(request,path)
        self.assertEqual(report['source_origin'],'synthetic_control')
        self.assertEqual(report['field_validation_status'],'PASS',report['heldout_field_validation'])
        self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')
        self.assertIn('total',report['density_shapes'])
        self.assertEqual(verify_halo_structure(path)['status'],'MEASURED')
        with np.load(path.with_name(path.name+'.data')/'arrays.npz') as saved:
            self.assertEqual(saved['parent_dm_particle_ids'].dtype,np.dtype('uint64'))
            self.assertTrue(np.min(saved['parent_dm_particle_ids'])>2**63)
            self.assertEqual(saved['profile_total_shell_count'].sum(),256)
        coefficient = path.parent/report['potential_coefficients'][0]['file']
        old=coefficient.read_bytes();coefficient.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'binding'):verify_halo_structure(path)
        coefficient.write_bytes(old)
        with h5py.File(control.snap,'r+') as h:h['PartType1/Coordinates'][0,0]+=1.
        with self.assertRaisesRegex(ValueError,'source changed'):verify_halo_structure(path)

    def test_native_stage_rejects_unpublished_wrong_units_and_holdout_overlap(self):
        from halos import publish_halo_structure
        control,request = self.native_fixture()
        with self.assertRaisesRegex(ValueError,'overlap'):
            publish_halo_structure(request|{'validation_positions_proper_kpc':request['calibration_positions_proper_kpc']},control.root/'overlap.json')
        marker = control.snap.with_suffix('.publication.json')
        marker.unlink()
        with self.assertRaisesRegex(ValueError,'marker'):
            publish_halo_structure(request,control.root/'unpublished.json')
        atomic_json(marker,{'schema':'native-hdf5-set-publication-v1','state':'COMPLETE','generation':'block11-test','shard_count':1})
        with h5py.File(control.snap,'r+') as h:
            h['PartType1/Coordinates'].attrs.update(to_cgs=1.,a_scaling=0.,h_scaling=0.)
        with self.assertRaisesRegex(ValueError,'units'):
            publish_halo_structure(request,control.root/'wrong_units.json')

    def test_public_cli_figure_and_verification_reloads_the_actual_product(self):
        from figures import verify_figure_export
        control,request = self.native_fixture()
        source,path,figure = control.root/'request.json',control.root/'cli.json',control.root/'halo_view'
        atomic_json(source,request)
        completed = subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'commands.py'),
            'halo-structure',str(source),'--output',str(path),'--figure-prefix',str(figure)],capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        report = json.loads(path.read_text())
        self.assertEqual(report['field_validation_status'],'PASS')
        sidecar = verify_figure_export(figure.with_suffix('.figure.json'))
        self.assertTrue(figure.with_suffix('.png').is_file())
        self.assertTrue(figure.with_suffix('.pdf').is_file())
        completed = subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'commands.py'),
            'halo-structure',str(source),'--output',str(path),'--verify'],capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertIn('NOT_ACCEPTED',completed.stdout)

    def test_withheld_failure_is_retained_and_not_used_for_refinement(self):
        from halos import publish_halo_structure
        control,request = self.native_fixture()
        request['validation_positions_proper_kpc'] = [[.1,.2,.3],[.2,.4,.1],[.7,.1,.2]]
        request['expansion_sequence'].append(request['expansion_sequence'][0]|{'lmax':6,'mmax':6,'grid_size_r':32})
        result = publish_halo_structure(request,control.root/'holdout_failure.json')
        self.assertEqual(len(result['refinement_history']),1)
        self.assertEqual(result['heldout_field_validation']['status'],'FAIL')
        self.assertEqual(result['field_validation_status'],'UNRESOLVED')
        self.assertEqual(result['science_acceptance'],'NOT_ACCEPTED')

    def test_twists_preserve_spherical_degeneracy_and_unsigned_axes(self):
        from halos import radial_shape_twists
        a = {'status':'MEASURED','axes_columns':np.eye(3).tolist(),'orientation_resolved':[True,True,True]}
        b = a|{'axes_columns':(-np.eye(3)).tolist()}
        self.assertEqual(radial_shape_twists([a,b],orientation_key='orientation_resolved')[0]['unsigned_axis_twist_deg'],[0,0,0])
        b['orientation_resolved']=[False]*3
        self.assertEqual(radial_shape_twists([a,b],orientation_key='orientation_resolved')[0]['unsigned_axis_twist_deg'],[None]*3)


TEST_GROUPS['test_halo_structure'] = ['HaloStructureTests']


class EnvironmentPhysicsTests(unittest.TestCase):
    def test_periodic_filament_regions_proximity_bounds_and_empty_support(self):
        from analysis import measure_periodic_filament_regions,measure_filament_proximity
        classes=np.zeros((5,5,5),dtype=np.uint8)
        classes[0,0,0]=classes[-1,0,0]=classes[2,2,2]=2
        regions=measure_periodic_filament_regions(classes,box_comoving_kpc=5.,density_supported=np.ones_like(classes,dtype=bool))
        self.assertEqual(regions['metadata']['component_count'],2)
        self.assertEqual(regions['component_ids'][0,0,0],regions['component_ids'][-1,0,0])
        np.testing.assert_array_equal(regions['voxel_count'],[2,1])
        distances=measure_filament_proximity(regions,[[4.9,0,0],[9.9,0,0]],box_comoving_kpc=5.)
        np.testing.assert_allclose(distances['nearest_node_distance_comoving_kpc'],[.1,.1],atol=1e-14)
        np.testing.assert_array_equal(distances['distance_lower_comoving_kpc'],[0,0])
        empty=measure_periodic_filament_regions(classes,box_comoving_kpc=5.,density_supported=np.zeros_like(classes,dtype=bool))
        proximity=measure_filament_proximity(empty,[[0,0,0]],box_comoving_kpc=5.)
        self.assertEqual(empty['metadata']['component_count'],0)
        self.assertFalse(proximity['supported'][0]);self.assertTrue(np.isnan(proximity['distance_upper_comoving_kpc'][0]))

    def test_spectral_filter_ringing_is_measured_without_clipping(self):
        from analysis import solve_periodic_peculiar_potential
        contrast=np.full((9,9,9),-1.);contrast[0,0,0]=9**3-1
        field=solve_periodic_peculiar_potential(contrast,scale_factor=1.,mean_density_proper_msun_kpc3=1.,
            box_comoving_kpc=9.,smoothing_comoving_kpc=.3,memory_gib=.1)
        smoothed=field['density_contrast_smoothed']
        self.assertLess(smoothed.min(),-1.)
        self.assertEqual(field['metadata']['negative_smoothed_density_cells'],int((smoothed < -1-1e-12).sum()))
        self.assertEqual(field['metadata']['density_filter_positivity'],'UNRESOLVED_SPECTRAL_RINGING')
        self.assertTrue(np.all(~field['filtered_density_supported'][smoothed < -1-1e-12]))
        self.assertAlmostEqual(smoothed.mean(),0,places=12)

    def test_actual_shell_tensor_sampling_mass_weighting_empty_bins_and_periodicity(self):
        from analysis import measure_radial_environment,sample_periodic_tensor_grid
        n=9;grid=np.zeros((n,n,n,3,3))
        grid[...,0,0]=-.4+.1*np.cos(2*np.pi*np.arange(n)/n)[:,None,None]
        grid[...,1,1]=.2;grid[...,2,2]=.5
        points=np.array([[.3,0,0],[.6,0,0],[1.4,0,0]])
        weights=np.array([1.,3.,2.]);components={'dm':{'offsets_proper_kpc':points,'masses_msun':weights}}
        options=dict(center_comoving_kpc=[8.8,0,0],scale_factor=.5,box_comoving_kpc=9.,threshold=.1,eigenvalue_gap_floor=.001)
        result=measure_radial_environment(grid,components,[0,1,2,3],chunk_rows=1,**options)
        actual=sample_periodic_tensor_grid(grid,np.array([8.8,0,0])+points/.5,9.)
        mean=np.average(actual[:2],axis=0,weights=weights[:2])
        np.testing.assert_allclose(result['mean_dimensionless_tidal_hessian'][0],mean,atol=1e-15)
        spread=np.sqrt(np.average(np.sum((actual[:2]-mean)**2,axis=(1,2)),weights=weights[:2]))
        self.assertAlmostEqual(result['tensor_frobenius_scatter'][0],spread)
        np.testing.assert_array_equal(result['particle_count'],[2,1,0])
        np.testing.assert_array_equal(result['web_class'],[2,2,255])
        self.assertTrue(np.isnan(result['mean_dimensionless_tidal_hessian'][2]).all())
        np.testing.assert_allclose(np.trace(result['mean_dimensionless_trace_free_hessian'][:2],axis1=1,axis2=2),0,atol=1e-15)
        shifted=measure_radial_environment(grid,components,[0,1,2,3],chunk_rows=2,
            **(options|{'center_comoving_kpc':[17.8,-9,9]}))
        np.testing.assert_allclose(shifted['mean_dimensionless_tidal_hessian'],result['mean_dimensionless_tidal_hessian'],atol=1e-15)
        ringing=measure_radial_environment(grid,components,[0,1,2,3],chunk_rows=2,
            filtered_density_contrast=np.full((9,9,9),-2.),**options)
        np.testing.assert_array_equal(ringing['unsupported_density_count'],[2,1,0])
        self.assertFalse(ringing['filament_axis_supported'].any())

    def test_signed_spin_unsigned_axes_missingness_and_actual_cadence(self):
        from analysis import measure_reorientation_series
        vectors=np.array([[[1.,0,0],[0,0,0]],[[-1.,0,0],[0,0,0]],[[0,1.,0],[0,0,0]]])
        flags=np.array([[True,False]]*3)
        spin=measure_reorientation_series(vectors,flags,[10,20,100],unsigned=False,maximum_interval_myr=20)
        axis=measure_reorientation_series(vectors,flags,[10,20,100],unsigned=True,maximum_interval_myr=20)
        np.testing.assert_allclose(spin['change_angle_deg'][:,0],[180,90])
        np.testing.assert_allclose(axis['change_angle_deg'][:,0],[0,90])
        self.assertTrue(np.isnan(spin['change_angle_deg'][:,1]).all())
        np.testing.assert_array_equal(spin['accepted_interval'],[[True,False],[False,False]])
        with self.assertRaises(ValueError):measure_reorientation_series(vectors,flags,[10,10,100],unsigned=False,maximum_interval_myr=20)
        from analysis import _proper_clock_error_bounds_myr
        bounds=_proper_clock_error_bounds_myr([{'cosmic_time_numerical_error_gyr':{'status':'MODEL_CONDITIONAL_BOUND','absolute_bound_gyr':.00001}},
                                              {'cosmic_time_numerical_error_gyr':{'status':'UNBOUNDED'}}])
        self.assertAlmostEqual(bounds[0],.01);self.assertTrue(np.isnan(bounds[1]))

    def field(self, contrast, a=.5, rho=2e4, smoothing=0):
        from analysis import solve_periodic_peculiar_potential
        return solve_periodic_peculiar_potential(contrast,scale_factor=a,
            mean_density_proper_msun_kpc3=rho,box_comoving_kpc=10.,
            smoothing_comoving_kpc=smoothing,memory_gib=1.)

    def test_known_fourier_force_hessian_poisson_trace_and_scale_factors(self):
        from core.foundation import G_MPC_KMS2_MSUN
        n = 11;grid = np.arange(n)*10/n
        k = 2*np.pi/10
        phase = k*(grid[:,None,None]+grid[None,:,None])
        delta = np.broadcast_to(.2*np.cos(phase),(n,n,n)).copy()
        a,rho = .3,2e4
        coefficient = 4*np.pi*G_MPC_KMS2_MSUN*1000*rho
        field = self.field(delta,a=a,rho=rho)
        np.testing.assert_allclose(field['potential_km2_s2'],-coefficient*a*a*delta/(2*k*k),rtol=1e-12,atol=1e-11)
        sine = np.broadcast_to(.2*np.sin(phase),(n,n,n))
        expected_acc = -coefficient*a*sine/(2*k)
        np.testing.assert_allclose(field['acceleration_km2_s2_proper_kpc'][...,0],expected_acc,rtol=1e-12,atol=1e-11)
        np.testing.assert_allclose(field['acceleration_km2_s2_proper_kpc'][...,1],expected_acc,rtol=1e-12,atol=1e-11)
        for i,j in [(0,0),(1,1),(0,1)]:
            np.testing.assert_allclose(field['hessian_km2_s2_proper_kpc2'][...,i,j],coefficient*delta/2,rtol=1e-12,atol=1e-11)
        np.testing.assert_allclose(np.trace(field['trace_free_hessian_km2_s2_proper_kpc2'],axis1=-2,axis2=-1),0,atol=1e-12)
        later = self.field(delta,a=a*2,rho=rho)
        np.testing.assert_allclose(later['potential_km2_s2'],4*field['potential_km2_s2'],rtol=1e-12,atol=1e-11)
        np.testing.assert_allclose(later['hessian_km2_s2_proper_kpc2'],field['hessian_km2_s2_proper_kpc2'],rtol=1e-12,atol=1e-11)

    def test_gaussian_source_smoothing_and_uniform_box(self):
        n=9;k=2*np.pi/10
        delta=np.broadcast_to(.4*np.cos(k*np.arange(n)[:,None,None]*10/n),(n,n,n)).copy()
        raw=self.field(delta);filtered=self.field(delta,smoothing=.7)
        attenuation=np.exp(-.5*k*k*.7**2)
        np.testing.assert_allclose(filtered['potential_km2_s2'],raw['potential_km2_s2']*attenuation,rtol=1e-12,atol=1e-11)
        uniform=self.field(np.zeros((n,n,n)))
        for key in ['potential_km2_s2','acceleration_km2_s2_proper_kpc','hessian_km2_s2_proper_kpc2']:
            np.testing.assert_array_equal(uniform[key],0)

    def test_even_mesh_nonzero_mean_unphysical_density_and_workspace_rejected(self):
        from analysis import solve_periodic_peculiar_potential
        with self.assertRaisesRegex(ValueError,'odd'):
            self.field(np.zeros((8,8,8)))
        with self.assertRaisesRegex(ValueError,'zero mean'):
            self.field(np.full((9,9,9),.1))
        delta=np.zeros((9,9,9));delta[0,0,0]=-2;delta[0,0,1]=2
        with self.assertRaisesRegex(ValueError,'delta'):
            self.field(delta)
        with self.assertRaises(MemoryError):
            solve_periodic_peculiar_potential(np.zeros((9,9,9)),scale_factor=.5,mean_density_proper_msun_kpc3=1.,
                box_comoving_kpc=10.,smoothing_comoving_kpc=0,memory_gib=.001)

    def test_web_signed_full_tensor_and_degenerate_filament_axes(self):
        from analysis import classify_eigenstructure
        tensor=np.array([np.diag(v) for v in [[-1,-1,-1],[-1,-1,1],[-1,1,2],[1,2,3],[0,0,2]]])
        web=classify_eigenstructure(tensor,threshold=0,eigenvalue_gap_floor=.01)
        np.testing.assert_array_equal(web['web_class'],[0,1,2,3,1])
        np.testing.assert_array_equal(web['filament_axis_supported'],[False,False,True,False,False])
        np.testing.assert_array_equal(web['eigenvectors_columns'][2,:,0],[1,0,0])
        nearly=np.array([np.diag([1.,1.001,3.])])
        web=classify_eigenstructure(nearly,threshold=2,eigenvalue_gap_floor=.01)
        self.assertFalse(web['axis_supported'][0,0])

    def test_periodic_tensor_interpolation_before_eigenvectors(self):
        from analysis import sample_periodic_tensor_grid
        grid=np.arange(27,dtype=float).reshape(3,3,3)
        np.testing.assert_allclose(sample_periodic_tensor_grid(grid,[[.5,.5,.5],[3.5,.5,.5]],3),[6.5,6.5])
        matrices=grid[...,None,None]*np.eye(3)
        np.testing.assert_allclose(sample_periodic_tensor_grid(matrices,[[0,0,0]],3),np.zeros((1,3,3)))

    def test_isotropic_null_and_zero_spin_selection(self):
        from analysis import classify_eigenstructure,measure_spin_shape_filament_angles
        n=256
        web=classify_eigenstructure(np.repeat(np.diag([-1,1,2])[None,:,:],n,axis=0),threshold=0,eigenvalue_gap_floor=.01)
        vectors=np.repeat([[1.,0,0]],n,axis=0);vectors[0]=0
        result=measure_spin_shape_filament_angles(vectors,web,randomized_realizations=64,seed=313)
        self.assertTrue(np.isnan(result['cosine_unsigned'][0]))
        np.testing.assert_array_equal(result['cosine_unsigned'][1:],1)
        self.assertLess(abs(result['isotropic_null_mean_cosine'].mean()-.5),.01)
        again=measure_spin_shape_filament_angles(vectors,web,randomized_realizations=64,seed=313)
        np.testing.assert_array_equal(again['isotropic_null_mean_cosine'],result['isotropic_null_mean_cosine'])

    def slope(self, radius, density, covariance=None, count=None, window=5):
        from analysis import uncertainty_aware_log_density_derivative
        n=len(radius)
        return uncertainty_aware_log_density_derivative(radius,density,
            np.eye(n)*1e-8 if covariance is None else covariance,window_bins=window,polynomial_degree=3,
            minimum_effective_count=10,effective_count=np.full(n,1000.) if count is None else count)

    def feature(self, radius, slope):
        from analysis import locate_supported_slope_feature
        return locate_supported_slope_feature(radius,slope,search_interval_proper_kpc=[radius[2],radius[-3]],
            minimum_depth=.5,bootstrap_realizations=64,seed=771,
            maximum_fractional_radius_error=.2,minimum_success_fraction=.8)

    def test_power_law_derivative_has_no_spurious_splashback(self):
        r=np.geomspace(.1,10,31)
        product=self.slope(r,1e8*r**-2.7)
        np.testing.assert_allclose(product['logarithmic_slope'],-2.7,rtol=1e-11,atol=1e-11)
        self.assertEqual(self.feature(r,product)['status'],'NO_SUPPORTED_FEATURE')

    def test_known_valley_uncertainty_and_multiple_minimum_ambiguity(self):
        from scipy.special import erf
        x=np.linspace(-1,1,61);r=np.exp(x);width=.2
        log_rho=-2*x-3*width*np.sqrt(np.pi/2)*erf(x/(np.sqrt(2)*width))
        product=self.slope(r,np.exp(log_rho))
        result=self.feature(r,product)
        self.assertEqual(result['status'],'MEASURED',result)
        self.assertAlmostEqual(result['candidates'][0]['radius_proper_kpc'],1,places=3)
        self.assertAlmostEqual(result['candidates'][0]['depth'],3,delta=.03)
        self.assertAlmostEqual(result['candidates'][0]['half_depth_width_ln_radius'],2*np.sqrt(2*np.log(2))*width,delta=.01)
        ambiguous=product|{'logarithmic_slope':-2-3*np.exp(-((x-.4)/.1)**2)-3*np.exp(-((x+.4)/.1)**2)}
        self.assertEqual(self.feature(r,ambiguous)['status'],'AMBIGUOUS_MINIMA')

    def test_missing_bins_break_slope_support_and_covariance_rejection(self):
        r=np.geomspace(.1,10,21);density=r**-2;density[10]=0
        product=self.slope(r,density)
        self.assertFalse(product['supported'][10])
        self.assertTrue(np.isnan(product['logarithmic_slope'][10]))
        self.assertTrue(np.all(product['derivative_operator'][:10,11:]==0))
        self.assertTrue(np.all(product['derivative_operator'][11:,:10]==0))
        covariance=np.eye(len(r));covariance[0,0]=-1
        with self.assertRaisesRegex(ValueError,'semidefinite'):
            self.slope(r,density,covariance=covariance)

    def test_correlated_covariance_matches_independent_realizations(self):
        r=np.geomspace(.1,10,21)
        covariance=.001*np.eye(21)+.002*np.ones((21,21))
        product=self.slope(r,r**-2,covariance=covariance)
        rng=np.random.default_rng(981)
        draws=rng.multivariate_normal(np.zeros(21),covariance,size=12000)
        slopes=draws @ product['derivative_operator'].T
        empirical=np.var(slopes,axis=0)
        np.testing.assert_allclose(empirical,np.diag(product['slope_covariance']),rtol=.045,atol=1e-8)

    def test_first_infall_pericenter_apocenter_actual_clock_and_missing_history(self):
        from analysis import compare_first_apocenter_particles
        ids=np.array([2**63+3,2**64-1],dtype=np.uint64)
        time=np.arange(8)*10.
        radii=np.column_stack(([3,2.4,1.8,.7,.9,1.3,1,.7],[3,2.4,1.8,.7,.9,1.3,np.nan,.7]))
        result=compare_first_apocenter_particles(ids,time,radii,np.full(8,2.),maximum_bracket_myr=25,
            profile_feature_radius_kpc=1.3)
        self.assertEqual(result['supported_count'],1)
        self.assertEqual(result['particles'][0]['first_apocenter_bracket_myr'],[40,60])
        self.assertEqual(result['particles'][0]['particle_id'],int(ids[0]))
        self.assertEqual(result['particles'][1]['status'],'MISSING_ORBITAL_SUPPORT')
        self.assertEqual(result['median_radius_over_profile_feature'],1)
        coarse=compare_first_apocenter_particles(ids,time,radii,np.full(8,2.),maximum_bracket_myr=15,
            profile_feature_radius_kpc=1.3)
        self.assertEqual(coarse['supported_count'],0)
        self.assertEqual(coarse['particles'][0]['status'],'CADENCE_UNRESOLVED')

    def native_request(self, root, products):
        return {'schema':'native-environment-v1','run_directory':str(root),'snapshot':0,'mesh_size':9,
            'smoothing_comoving_kpc':[0.,5.],'web_threshold':.1,'eigenvalue_gap_floor':.001,
            'halo_structure_products':products,'splashback_component':'dm',
            'orbital_history':None,
            'slope_window_study':[5,7,9],
            'slope_controls':{'window_bins':5,'polynomial_degree':3,'minimum_effective_count':10},
            'splashback_controls':{'search_interval_proper_kpc':[.5,3.],'minimum_depth':.5,
                'bootstrap_realizations':64,'seed':571,'maximum_fractional_radius_error':.2,'minimum_success_fraction':.8},
            'alignment_controls':{'randomized_realizations':32,'seed':713},'memory_gib':1.}

    def test_native_mass_mesh_publication_profiles_and_source_corruption(self):
        from halos import publish_halo_structure
        from analysis import publish_native_environment,verify_native_environment
        c=HaloStructureTests();fixture,request=c.native_fixture();self.addCleanup(c.doCleanups)
        profile=fixture.root/'structure.json';publish_halo_structure(request,profile)
        environmental=self.native_request(fixture.root,[str(profile)])
        path=fixture.root/'environment.json'
        report=publish_native_environment(environmental,path)
        self.assertEqual(report['source_origin'],'synthetic_control')
        self.assertEqual(report['massive_particle_rows'],256)
        self.assertEqual(len(report['smoothing_fields']),2)
        self.assertEqual(report['halo_measurements'][0]['splashback']['status'],'NO_SUPPORTED_FEATURE')
        verify_native_environment(path)
        from analysis import plot_native_environment
        plotted=plot_native_environment(path,fixture.root/'environment_profile',smoothing_index=0,halo_index=0)
        self.assertTrue(Path(plotted['figure']).is_file())
        self.assertTrue((fixture.root/'environment_profile.png').is_file())
        self.assertTrue((fixture.root/'environment_profile.pdf').is_file())
        self.assertEqual(report['halo_measurements'][0]['slope_window_sensitivity_status'],'UNRESOLVED_WINDOWS_RETAINED')
        with np.load(path.with_name(path.name+'.data')/'arrays.npz') as saved:
            self.assertEqual(saved['smoothing_00_web_web_class'].shape,(9,9,9))
            self.assertLess(abs(saved['density_contrast_unsmoothed'].mean()),1e-12)
        with h5py.File(fixture.snap,'r+') as h:h['PartType1/Masses'][0]*=2
        with self.assertRaisesRegex(ValueError,'source changed'):verify_native_environment(path)

    def test_native_environment_cli_and_even_grid_rejection(self):
        c=HaloStructureTests();fixture,request=c.native_fixture();self.addCleanup(c.doCleanups)
        source,path=fixture.root/'environment_request.json',fixture.root/'environment.json'
        environmental=self.native_request(fixture.root,[])
        atomic_json(source,environmental)
        for extra in [[],['--verify','--figure-prefix',str(fixture.root/'field_plot'),'--figure-smoothing-index','1']]:
            completed=subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'commands.py'),
                'environment',str(source),'--output',str(path),*extra],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
        self.assertTrue((fixture.root/'field_plot.png').is_file())
        from analysis import publish_native_environment
        with self.assertRaisesRegex(ValueError,'odd'):
            publish_native_environment(environmental|{'mesh_size':8},fixture.root/'even.json')

    def test_native_first_apocenter_bridge_exact_reordering_clock_and_graph_binding(self):
        from halos import publish_halo_structure,IdentityGraph
        from analysis import compare_native_first_apocenters
        c=HaloStructureTests();fixture,base=c.native_fixture();self.addCleanup(c.doCleanups)
        root=fixture.root
        initial=root/'initial_source.hdf5';shutil.copy2(fixture.snap,initial)
        scale=np.linspace(.5,1,8)
        clock=root/'background.dat'
        clock.write_text('# 1:z 2:proper time [Gyr] 3:Omega_r(z)\n'+''.join(f'{1/a-1:.17g} {1+i*.01:.17g} 0\n' for i,a in enumerate(scale)))
        atomic_json(root/'manifest.json',{'fixture':True,'gadget_handoff':{'initial_conditions':[{'path':str(initial),'sha256':digest(initial)}]},
            'input_artifacts':{'boltzmann_background':{'path':str(clock),'sha256':digest(clock)}}})
        graph=IdentityGraph(maximum_nodes=32,maximum_edges=32)
        sources={};observations=[];keys=[]
        initial_snapshot=fixture.snap.read_bytes();initial_catalogue=fixture.catalogue.read_bytes()
        tracer=np.uint64(2**63+1)
        radial=[3,2.4,1.8,.7,.9,1.3,1,.7]
        for i,a in enumerate(scale):
            snapshot=root/'data'/f'snapshot_{i:03d}.hdf5';catalogue=root/'data'/f'fof_subhalo_tab_{i:03d}.hdf5'
            snapshot.write_bytes(initial_snapshot);catalogue.write_bytes(initial_catalogue)
            with h5py.File(snapshot,'r+') as h:
                h['Header'].attrs['Time']=a
                if 'Redshift' in h['Header'].attrs:h['Header'].attrs['Redshift']=1/a-1
                h['Header'].attrs['Research2Generation']='orbit-'+str(i)
                group=h['PartType1'];order=np.arange(256)[::-1] if i%2 else np.arange(256)
                coordinates=group['Coordinates'][:]
                coordinates[0]=[radial[i]*.7/a,0,0]
                for key in ['Coordinates','Velocities','Masses','ParticleIDs']:
                    value=coordinates if key=='Coordinates' else group[key][:]
                    group[key][:]=value[order]
            atomic_json(snapshot.with_suffix('.publication.json'),{'schema':'native-hdf5-set-publication-v1',
                'state':'COMPLETE','generation':'orbit-'+str(i),'shard_count':1})
            with h5py.File(catalogue,'r+') as h:
                h['Header'].attrs['Time']=a
                h['Group/Group_R_Crit200'][0]=2*.7/a
            request=base|{'snapshot':i}
            path=root/f'orbit_structure_{i}.json'
            result=publish_halo_structure(request,path)
            key=('control-host',i,i);keys.append(key)
            graph.add_epoch(i,result['source_context']['epoch'])
            graph.add_node(*key,kind='halo',attributes={'id':0,'position_proper_kpc':[0,0,0]})
            sources.update(result['source_context']['source_hashes'])
            if i:graph.add_edge(keys[i-1],key,'descendant',evidence={'origin':'native_ancestry','fixture':True,'source_hashes':sources.copy()})
            observations.append({'structure_product':str(path),'graph_node':list(key)})
        graph_path=root/'orbital_graph';graph.publish_indexed_queries(graph_path,source_hashes=sources)
        history={'structure_observations':observations,'identity_graph_directory':str(graph_path),
            'component':'dm','particle_ids':[int(tracer)],'maximum_bracket_myr':25.}
        result,arrays,bindings=compare_native_first_apocenters(history,profile_feature_radius_kpc=1.3,memory_gib=1.)
        self.assertEqual(result['supported_count'],1,result)
        self.assertEqual(result['clock_bounded_supported_count'],0)
        np.testing.assert_allclose(arrays['orbital_particle_radius_proper_kpc'][:,0],radial,rtol=1e-12)
        self.assertAlmostEqual(result['median_radius_over_profile_feature'],1,places=12)
        self.assertIn(str(clock.resolve()),bindings)
        from analysis import publish_native_environment,publish_environment_history,verify_environment_history
        environmental_observations=[]
        for i,observation in enumerate(observations):
            path=root/f'orbit_environment_{i}.json'
            native_request=self.native_request(root,[observation['structure_product']])|{'snapshot':i}
            if i==len(observations)-1:native_request['orbital_history']=history
            product=publish_native_environment(native_request,path)
            if i==len(observations)-1:self.assertEqual(product['first_apocenter_check']['supported_count'],1)
            environmental_observations.append({'environment_product':str(path),
                'halo_structure_product':observation['structure_product'],'graph_node':observation['graph_node']})
        history_request={'schema':'native-environment-history-v1','identity_graph_directory':str(graph_path),
            'observations':environmental_observations,'maximum_interval_myr':15.,'memory_gib':1.}
        history_path=root/'environment_history.json'
        measured=publish_environment_history(history_request,history_path)
        self.assertEqual(measured['source_origin'],'synthetic_control')
        self.assertEqual(len(measured['epoch_contracts']),8)
        self.assertEqual(verify_environment_history(history_path)['graph_nodes'],[list(key) for key in keys])
        with np.load(history_path.with_name(history_path.name+'.data')/'arrays.npz') as saved:
            np.testing.assert_allclose(saved['proper_time_myr'],1000+10*np.arange(8))
            self.assertEqual(saved['spin_change_angle_deg'].shape,(7,3))
            np.testing.assert_array_equal(saved['native_progenitor_count'],[0,1,1,1,1,1,1,1])
            self.assertFalse(saved['multiple_progenitor_observation'].any())
            self.assertTrue(np.isnan(saved['proper_clock_error_myr']).all())
            self.assertFalse(saved['spin_clock_bounded_accepted_interval'].any())
            mass=saved['enclosed_aperture_mass_msun']
            np.testing.assert_allclose(saved['aperture_mass_change_msun_per_myr'],np.diff(mass,axis=0)/10)
        # A verified later product with another scientific membership
        # definition cannot become apparent same-estimator mass evolution.
        from unittest.mock import patch
        actual_verify=__import__('halos').verify_halo_structure
        changed_path=Path(observations[1]['structure_product']).resolve()
        def changed_definition(path):
            product=actual_verify(path)
            if Path(path).resolve()==changed_path:
                product=copy.deepcopy(product)
                product['request']['components']['dm']['membership']='subfind_central_bound'
            return product
        with patch('halos.verify_halo_structure',side_effect=changed_definition):
            with self.assertRaisesRegex(ValueError,'matching field definitions'):
                publish_environment_history(history_request,root/'mixed_membership_history.json')
        self.assertFalse((root/'mixed_membership_history.json').exists())
        public_request=root/'environment_history_request.json';atomic_json(public_request,history_request)
        public_path=root/'environment_history_cli.json'
        for extra in ([],['--verify']):
            completed=subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'commands.py'),'environment',
                str(public_request),'--history','--output',str(public_path),*extra],capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
        wrong_history=copy.deepcopy(history_request)
        wrong_history['observations'][1]['graph_node']=list(keys[-1])
        with self.assertRaisesRegex(ValueError,'actual epochs|identity'):
            publish_environment_history(wrong_history,root/'wrong_history.json')
        wrong=copy.deepcopy(history);wrong['structure_observations'][1]['graph_node']=list(keys[-1])
        with self.assertRaisesRegex(ValueError,'epoch|identity'):
            compare_native_first_apocenters(wrong,profile_feature_radius_kpc=1.3,memory_gib=1.)
        lossy=copy.deepcopy(history);lossy['structure_observations'][0]['graph_node'][1]=float(keys[0][1])
        with self.assertRaisesRegex(ValueError,'uint64|exact'):
            compare_native_first_apocenters(lossy,profile_feature_radius_kpc=1.3,memory_gib=1.)
        clock.write_text('changed source clock')
        with self.assertRaisesRegex(ValueError,'source changed'):
            compare_native_first_apocenters(history,profile_feature_radius_kpc=1.3,memory_gib=1.)
        with self.assertRaisesRegex(ValueError,'source changed'):verify_environment_history(history_path)


TEST_GROUPS['test_environment_physics'] = ['EnvironmentPhysicsTests']


class NativeSSPBirthTests(unittest.TestCase):
    def test_legacy_current_mass_is_measured_and_cannot_hide_nonfinite_or_negative_values(self):
        from stars import validate_ssp_birth_arrays
        for value in (np.nan,np.inf,-1.):
            arrays=self.arrays();arrays['SSPCurrentMass'][2]=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'current mass'):
                validate_ssp_birth_arrays(arrays,factor_sn=.1,actual_native_time=.5,comoving=True)

    def test_sealed_native_birth_bridge_public_graph_query_and_missing_parent(self):
        from stars import audit_native_ssp_birth_state
        from halos import IdentityGraph,IdentityQueries,add_native_ssp_birth_audit
        from core.data import epoch_contract
        helper=HaloStructureTests();self.addCleanup(helper.doCleanups)
        control,_=helper.native_fixture();arrays=self.arrays()
        with h5py.File(control.snap,'r+') as f:
            f['Parameters'].attrs.update(ComovingIntegrationOn=1,FactorSN=.1,UnitMass_in_g=1.98847e43)
            count=np.asarray(f['Header'].attrs['NumPart_ThisFile']).copy();count[4]=3
            f['Header'].attrs.update(NumPart_ThisFile=count,NumPart_Total=count)
            group=f.create_group('PartType4');group['Coordinates']=np.zeros((3,3));group['Masses']=arrays['SSPCurrentMass']
            for name,values in arrays.items():
                group[name]=values
                if name!='ParticleIDs':group[name].attrs.update(to_cgs=1.98847e43 if name.endswith('Mass') else 1.,
                    a_scaling=0.,h_scaling=-1. if name.endswith('Mass') else 0.)
        path=control.root/'birth.json'
        report=audit_native_ssp_birth_state({'run_directory':str(control.root),'snapshot':0,'memory_gib':.1},path)
        namespace=report['provenance']['identity_namespaces']['native_particle']
        def make_graph():
            graph=IdentityGraph(maximum_nodes=10,maximum_edges=10)
            graph.add_epoch(0,report['epoch']);graph.add_epoch(9,epoch_contract(.05));graph.add_epoch(2,epoch_contract(.2))
            for pid in (12,9,7):graph.add_node(namespace,pid,0,kind='ssp')
            graph.add_node(namespace,7,9,kind='gas')
            graph.add_node(namespace,8,2,kind='gas') # Later gas cannot prove the birth observation.
            return graph
        graph=make_graph()
        with self.assertRaises(MemoryError):add_native_ssp_birth_audit(graph,path,particle_namespace=namespace,maximum_bytes=1)
        with self.assertRaisesRegex(ValueError,'namespace'):add_native_ssp_birth_audit(graph,path,particle_namespace='wrong',maximum_bytes=32*1024**2)
        sources=add_native_ssp_birth_audit(graph,path,particle_namespace=namespace,maximum_bytes=32*1024**2)
        self.assertEqual(len(graph.edges),1)
        self.assertEqual(graph.edges[0]['source'],(namespace,7,9))
        self.assertEqual(graph.nodes[(namespace,9,0)]['attributes']['birth_parent_status'],'PARENT_ID_MEASURED_OBSERVATION_UNAVAILABLE')
        self.assertEqual(graph.nodes[(namespace,12,0)]['attributes']['birth_parent_status'],'LEGACY_BIRTH_UNAVAILABLE')
        add_native_ssp_birth_audit(graph,path,particle_namespace=namespace,maximum_bytes=32*1024**2)
        self.assertEqual(len(graph.edges),1)
        destination=control.root/'lineage';graph.publish(destination,source_hashes=sources)
        query=IdentityQueries(destination)
        self.assertEqual(query.birth_parent((namespace,7,0))['parents'],[(namespace,7,9)])
        self.assertEqual(query.birth_parent((namespace,9,0))['status'],'UNAVAILABLE')
        rejected=make_graph();rejected.add_node(namespace,8,9,kind='gas');rejected.maximum_edges=1
        with self.assertRaisesRegex(MemoryError,'edge budget'):
            add_native_ssp_birth_audit(rejected,path,particle_namespace=namespace,maximum_bytes=32*1024**2)
        self.assertEqual(rejected.edges,[])
        self.assertTrue(all('ssp_birth_audit_sha256' not in node['attributes'] for node in rejected.nodes.values()))
        fresh=make_graph()
        request={'limits':{'maximum_nodes':10,'maximum_edges':10},
            'epochs':[{'storage_index':e,'epoch_contract':c} for e,c in fresh.epochs.items()],
            'nodes':[{'namespace':k[0],'id':k[1],'epoch':k[2],**v} for k,v in fresh.nodes.items()],
            'edges':[],'source_hashes':{},'ssp_birth_audits':[{'path':str(path),'particle_namespace':namespace,'maximum_bytes':32*1024**2}]}
        request_path=control.root/'graph-request.json';atomic_json(request_path,request)
        result=subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'commands.py'),
            'build-identity-graph',str(request_path),str(control.root/'cli-lineage')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(IdentityQueries(control.root/'cli-lineage').birth_parent((namespace,7,0))['parents'],[(namespace,7,9)])
        with h5py.File(control.snap,'r+') as f:f['PartType4/SSPParentGasID'][0]=8
        with self.assertRaisesRegex(ValueError,'source changed'):query.birth_parent((namespace,7,0))

    def test_native_birth_sources_and_pinned_reference_ship_in_portable_release(self):
        from core.release import source_files
        project=Path(__file__).resolve().parent
        paths={p.relative_to(project).as_posix() for p in source_files()}
        reference=json.loads((project/'configs/hydro_audit.algorithms.json').read_text())['records'][0]['upstream_revision_or_asset_hash']['independent_philox_reference']
        for path in (reference['asset'],reference['license_asset'],
                     'core/software/gadget4/src/data/research_stellar_birth.h',
                     'core/patches/gadget4-ssp-birth-state.patch'):
            self.assertIn(path,paths)
        self.assertEqual(digest(project/reference['license_asset']),reference['license_sha256'])

    def test_native_patch_replays_exact_source_including_new_header(self):
        # The 13-only reverse patch is historical after native 14/15 integration.
        # The current full-file overlay must reproduce every declared native
        # file and reject unbound local edits, without a compiler or simulation.
        import zipfile
        from evolution import native_source_overlay_manifest,install_native_source_overlay
        project=Path(__file__).resolve().parent
        manifest=project/'core/patches/gadget4-source/native-source-current.json'
        record,archive=native_source_overlay_manifest(manifest)
        with zipfile.ZipFile(archive) as z:
            for entry in record['files']:
                self.assertEqual(z.read(entry['path']),(project/'core/software/gadget4'/entry['path']).read_bytes(),entry['path'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for entry in record['files']:(root/entry['path']).parent.mkdir(parents=True,exist_ok=True)
            expected={e['path']:None for e in record['files']}
            result=install_native_source_overlay(manifest,root,expected_existing_hashes=expected)
            self.assertEqual(result['status'],'COMPLETE')
            for entry in record['files']:
                self.assertEqual((root/entry['path']).read_bytes(),(project/'core/software/gadget4'/entry['path']).read_bytes())
            conflict=root/'src/data/research_source_state.h';conflict.write_bytes(conflict.read_bytes()+b'\n// unbound edit\n')
            with self.assertRaises(ValueError):install_native_source_overlay(manifest,root,expected_existing_hashes=expected)

    def arrays(self):
        return {'ParticleIDs':np.array([7,9,12],dtype=np.uint64),
            'SSPParentGasID':np.array([7,8,0],dtype=np.uint64),
            'SSPBirthStateVersion':np.array([1,1,0],dtype=np.uint64),
            'SSPBirthMode':np.array([1,2,0],dtype=np.uint64),
            'StellarRandomCounter':np.array([11,17,0],dtype=np.uint64),
            'SSPInitialMass':np.array([10.,5.,0.]),'SSPInitialLockedMass':np.array([9.,4.5,0.]),
            'SSPPromptReturnedMass':np.array([1.,.5,0.]),'SSPCurrentMass':np.array([9.,4.5,2.]),
            'SSPBirthScaleFactor':np.array([.1,.1,0.]),'SSPBirthNativeTime':np.array([.1,.1,0.]),
            'SSPBirthMetallicity':np.array([.02,0.,0.])}

    def test_audit_rejects_double_recycling_bad_ids_and_false_legacy_births(self):
        from stars import validate_ssp_birth_arrays
        arrays=self.arrays()
        result=validate_ssp_birth_arrays(arrays,factor_sn=.1,actual_native_time=.5,comoving=True)
        self.assertEqual(result['supported_count'],2)
        self.assertEqual(result['legacy_birth_unavailable_count'],1)
        for name,value in [('SSPCurrentMass',9.1),('SSPInitialMass',9.),('SSPBirthNativeTime',.6),
                           ('SSPBirthMetallicity',1.1),('SSPParentGasID',8),('SSPBirthStateVersion',2)]:
            bad={k:v.copy() for k,v in arrays.items()};bad[name][0]=value
            with self.subTest(field=name),self.assertRaises(ValueError):
                validate_ssp_birth_arrays(bad,factor_sn=.1,actual_native_time=.5,comoving=True)
        arrays['SSPParentGasID']=arrays['SSPParentGasID'].astype(float)
        with self.assertRaisesRegex(ValueError,'uint64'):
            validate_ssp_birth_arrays(arrays,factor_sn=.1,actual_native_time=.5,comoving=True)

    def test_native_snapshot_audit_public_command_and_source_corruption(self):
        from stars import audit_native_ssp_birth_state,verify_native_ssp_birth_audit
        helper=HaloStructureTests();self.addCleanup(helper.doCleanups)
        control,_=helper.native_fixture();arrays=self.arrays()
        with h5py.File(control.snap,'r+') as f:
            time=float(f['Header'].attrs['Time'])
            f['Parameters'].attrs.update(ComovingIntegrationOn=1,FactorSN=.1,UnitMass_in_g=1.98847e43)
            count=np.asarray(f['Header'].attrs['NumPart_ThisFile']).copy();count[4]=3
            f['Header'].attrs.update(NumPart_ThisFile=count,NumPart_Total=count)
            group=f.create_group('PartType4');group['Coordinates']=np.zeros((3,3))
            group['Masses']=arrays['SSPCurrentMass']
            for name,values in arrays.items():
                group[name]=values
                if name!='ParticleIDs':group[name].attrs.update(to_cgs=1.98847e43 if name.endswith('Mass') else 1.,
                    a_scaling=0.,h_scaling=-1. if name.endswith('Mass') else 0.)
        request={'run_directory':str(control.root),'snapshot':0,'memory_gib':.1}
        source=control.root/'ssp-request.json';atomic_json(source,request)
        path=control.root/'birth.json';report=audit_native_ssp_birth_state(request,path)
        self.assertEqual(report['supported_count'],2);self.assertEqual(report['source_origin'],'synthetic_control')
        self.assertEqual(verify_native_ssp_birth_audit(path),report)
        with np.load(path.with_name(path.name+'.data')/'arrays.npz') as data:
            self.assertEqual(data['SSPParentGasID'].dtype,np.dtype('uint64'))
            np.testing.assert_array_equal(data['birth_supported'],[True,True,False])
        argv=[sys.executable,'-B',str(Path(__file__).parent/'commands.py'),'audit-hydro-ssp',str(source),'--output',str(control.root/'cli-birth.json')]
        run=subprocess.run(argv,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        run=subprocess.run(argv+['--verify'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        with h5py.File(control.snap,'r+') as f:f['PartType4/SSPInitialMass'].attrs['h_scaling']=0.
        with self.assertRaisesRegex(ValueError,'declared native SSP units'):
            audit_native_ssp_birth_state(request,control.root/'bad-units.json')
        with h5py.File(control.snap,'r+') as f:f['PartType4/SSPInitialMass'].attrs['h_scaling']=-1.
        with h5py.File(control.snap,'r+') as f:f['PartType4/SSPCurrentMass'][0]+=1.
        with self.assertRaisesRegex(ValueError,'source changed'):verify_native_ssp_birth_audit(path)
        with self.assertRaisesRegex(ValueError,'gravity mass'):
            audit_native_ssp_birth_state(request,control.root/'bad.json')

    def test_native_known_answers_mass_accounting_and_restart_draws(self):
        code = r'''
#include "data/particle_data.h"
#include "data/research_stellar_birth.h"
#include <cstring>
#include <cstdio>
int main() {
  uint32_t c[4]={0,0,0,0}, k[2]={0,0};
  research_stellar_philox(c,k);
  const uint32_t expected[4]={0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8};
  for(int j=0;j<4;j++) if(c[j]!=expected[j]) return 1;
  uint64_t seed;
  if(!research_stellar_seed("18446744073709551615",seed)||seed!=UINT64_MAX) return 2;
  for(const char *bad:{"18446744073709551616","-1","1.0","","+1"," 1"})
    if(research_stellar_seed(bad,seed)) return 3;
  particle_data star;
  if(!research_stellar_record_birth(star.ResearchSourceState,ID_MAX,9.,.25,.25,.02,.1,1)) return 4;
  const research_source_state &s=star.ResearchSourceState;
  if(fabs(s.stellar_initial_mass-10.)>1e-14 || fabs(s.stellar_prompt_returned_mass-1.)>1e-14 ||
     s.stellar_initial_locked_mass!=9. || s.stellar_parent_gas_id!=ID_MAX || s.stellar_birth_mode!=1) return 5;
  if(research_stellar_record_birth(star.ResearchSourceState,ID_MAX,9.,.25,.25,.02,.1,1)) return 6;
  particle_data child;
  if(!research_stellar_record_birth(child.ResearchSourceState,ID_MAX,4.5,2.,0.,0.,.1,2)) return 7;
  if(child.ResearchSourceState.stellar_birth_scale_factor!=0 || child.ResearchSourceState.stellar_initial_mass!=5.) return 8;
  particle_data invalid;
  if(research_stellar_record_birth(invalid.ResearchSourceState,1,9.,.25,.25,.02,1.,1) ||
     invalid.ResearchSourceState.stellar_birth_state_version) return 9;
  star.ResearchSourceState.stellar_random_counter=UINT64_MAX-1;
  research_source_state restored;
  memcpy(&restored,&s,sizeof(restored));
  for(unsigned lane=0;lane<3;lane++) {
    double a=research_stellar_uniform(UINT64_MAX,ID_MAX,s.stellar_random_counter,lane);
    double b=research_stellar_uniform(UINT64_MAX,ID_MAX,restored.stellar_random_counter,lane);
    if(a!=b || !(a>0 && a<1)) return 10;
  }
  int accepted=0; double mean=0;
  for(uint64_t id=1;id<=100000;id++) {
    double u=research_stellar_uniform(17,id,0,2); accepted+=(u<.2); mean+=u;
  }
  if(accepted<19300 || accepted>20700 || fabs(mean/100000-.5)>.005) return 11;
  puts("PASS published Philox KAT; SH03 gross/locked/prompt accounting; exact parents; restart draws; independent-ID sampling");
}
'''
        project=Path(__file__).resolve().parent
        reference=json.loads((project/'configs/hydro_audit.algorithms.json').read_text())['records'][0]['upstream_revision_or_asset_hash']['independent_philox_reference']
        asset=project/reference['asset'];self.assertEqual(digest(asset),reference['sha256'])
        vectors=[line.split()[2:] for line in asset.read_text().splitlines() if line.startswith('philox4x32 10 ')]
        self.assertEqual(len(vectors),3)
        probes=[]
        for row in vectors:
            self.assertEqual(len(row),10)
            literal=lambda values:','.join('0x'+v+'u' for v in values)
            probes.append('{ uint32_t c[4]={'+literal(row[:4])+'},k[2]={'+literal(row[4:6])+'},e[4]={'+literal(row[6:])+'};research_stellar_philox(c,k);for(int j=0;j<4;j++)if(c[j]!=e[j])return 12; }')
        code=code.replace('  uint64_t seed;','\n'.join(probes)+'\n  uint64_t seed;')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'gadgetconfig.h').write_text('#define IDS_64BIT\n#define POSITIONS_IN_64BIT\n#define DOUBLEPRECISION 1\n#define MULTIPOLE_ORDER 2\n#define STARFORMATION\n#define COOLING\n')
            (root/'research_restart_fingerprint.h').write_text('static const char research_restart_fingerprint[]="'+('a'*64)+'";\n')
            (root/'birth.cc').write_text(code)
            includes=shlex.split(subprocess.check_output(['pkg-config','--cflags','gsl','hdf5'],text=True))
            src=Path(__file__).resolve().parent/'core/software/gadget4/src'
            cmd=['mpicxx','-std=c++11','-I',str(root),'-I',str(src),*includes]
            built=subprocess.run(cmd+[str(root/'birth.cc'),'-o',str(root/'birth')],capture_output=True,text=True,timeout=60)
            self.assertEqual(built.returncode,0,built.stderr)
            run=subprocess.run([str(root/'birth')],capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('PASS published Philox KAT',run.stdout)
            # Compile the actual production conversion/spawn integration too.
            for source in ('cooling_sfr/starformation.cc','io/snap_io.cc','data/allvars.cc'):
                syntax=subprocess.run(cmd+['-fsyntax-only',str(src/source)],capture_output=True,text=True,timeout=60)
                self.assertEqual(syntax.returncode,0,source+'\n'+syntax.stderr)

    def test_exact_seed_configuration_and_public_native_parameters(self):
        from evolution import hydro_star_formation_parameters
        for seed in (0,2**63+3,2**64-1):
            self.assertEqual(hydro_star_formation_parameters({'stellar_random_seed':seed})['StellarRandomSeed'],str(seed))
        for seed in (-1,2**64,True,1.,'17'):
            with self.assertRaises(ValueError):hydro_star_formation_parameters({'stellar_random_seed':seed})


class NativeSSPJournalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.workspace.cleanup)
        cls.directory=Path(cls.workspace.name);cls.binary=cls.directory/'journal-control'
        (cls.directory/'gadgetconfig.h').write_text('#define IDS_64BIT\n')
        src=Path(__file__).resolve().parent/'core/software/gadget4/src'
        code=r'''
#include "data/research_stellar_birth.h"
#include <cstring>
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);int rank,ranks;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  uint64_t sequence=argc>3?strtoull(argv[3],NULL,10):0,count=argc>4?strtoull(argv[4],NULL,10):0;
  const uint64_t old_sequence=sequence,old_count=count;
  const int comoving=strcmp(argv[2],"isolated")!=0;
  std::vector<research_stellar_birth_event> events;
  if(strcmp(argv[2],"empty")) for(int i=0;i<2;i++) if(i%ranks==rank) {
    research_stellar_birth_event e={};e.ssp_id=i?9:7;e.parent_id=i?8:7;e.mode=i?2:1;e.opportunity_counter=UINT64_MAX;
    e.birth_time=.6;e.scale_factor=comoving?.6:0.;e.metallicity=.02;e.gross_mass=i?5:10;e.locked_mass=i?4.5:9;e.prompt_mass=i?.5:1;
    e.gas_mass_before=i?6:9;e.gas_mass_after=i?1.5:0;
    e.gas_velocity_before[0]=3;e.gas_velocity_after[0]=i?3:0;e.ssp_velocity[0]=3;
    if(!strcmp(argv[2],"conflict"))e.metallicity=.03;
    events.push_back(e);
  }
  const char *fp="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  bool ok=research_stellar_publish_journal(argv[1],events,sequence,count,17,.6,comoving,1.98847e43,3.0856775814913673e16,
    1e5,3.0856775814913673e21,.7,.1,UINT64_MAX,fp,MPI_COMM_WORLD);
  int bad=0;
  if(!strcmp(argv[2],"conflict"))bad=ok||sequence!=old_sequence||count!=old_count;
  else if(!strcmp(argv[2],"empty"))bad=!ok||sequence!=old_sequence||count!=old_count;
  else bad=!ok||sequence!=old_sequence+1||count!=old_count+2;
  int global;MPI_Allreduce(&bad,&global,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  if(rank==0)puts(global?"FAIL":"PASS native MPI journal publication and counter continuity");
  MPI_Finalize();return global;
}
'''
        source=cls.directory/'control.cc';source.write_text(code)
        run=subprocess.run(['mpicxx','-std=c++11','-I',str(cls.directory),'-I',str(src),str(source),'-lz','-o',str(cls.binary)],capture_output=True,text=True,timeout=60)
        if run.returncode:raise RuntimeError(run.stderr)

    def parameters(self):
        return {'unit_mass_g':1.98847e43,'unit_time_s':3.0856775814913673e16,
            'unit_velocity_cm_s':1e5,'unit_length_cm':3.0856775814913673e21,
            'hubble_param':.7,'factor_sn':.1,'comoving':1,'seed':str(2**64-1)}

    def run_journal(self,path,ranks=1,mode='write',prefix=(0,0)):
        run=subprocess.run(['mpirun','--oversubscribe','-np',str(ranks),str(self.binary),str(path),mode,*map(str,prefix)],
            capture_output=True,text=True,timeout=40)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.assertIn('PASS native MPI',run.stdout)

    def read(self,path,prefix=(0,0),stop=(1,2),memory=32*1024**2):
        from stars import read_native_ssp_birth_journal
        return read_native_ssp_birth_journal(path,start_commit=prefix[0],stop_commit=stop[0],
            start_count=prefix[1],stop_count=stop[1],native_time_interval=[.5,.75],
            expected_parameters=self.parameters(),memory_bytes=memory)

    def test_actual_mpi_empty_ranks_exact_ids_replay_conflict_and_inherited_branch(self):
        for ranks in (1,2,3):
            with self.subTest(ranks=ranks),tempfile.TemporaryDirectory() as temp:
                path=Path(temp);self.run_journal(path,ranks)
                arrays,commits,sources=self.read(path)
                self.assertEqual(len(commits[0]['ranks']),ranks)
                np.testing.assert_array_equal(np.sort(arrays['ssp_id']),[7,9])
                self.assertEqual(arrays['opportunity_counter'].dtype,np.dtype('uint64'))
                self.assertTrue(np.all(arrays['opportunity_counter']==np.uint64(2**64-1)))
                before={str(p):digest(p) for p in path.iterdir()}
                self.run_journal(path,ranks) # same checkpoint prefix: exact immutable replay
                self.run_journal(path,ranks,'conflict')
                self.assertEqual(before,{str(p):digest(p) for p in path.iterdir()})
                self.assertFalse(list(path.glob('*.stage.*')))
                with self.assertRaises(MemoryError):self.read(path,memory=1)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);self.run_journal(path,2,'empty',(4,13));self.assertEqual(list(path.iterdir()),[])
            self.run_journal(path,2,prefix=(4,13))
            arrays,commits,_=self.read(path,prefix=(4,13),stop=(5,15))
            self.assertEqual(commits[0]['previous_event_count'],13);self.assertEqual(len(arrays['ssp_id']),2)

    def test_missing_chunks_corruption_duplicate_keys_and_false_counts_rejected(self):
        for kind in ('chunk','crc','commit','counter','duplicate','float_id','symlink','extra_row'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as temp:
                path=Path(temp);self.run_journal(path,2)
                commit=next(path.glob('*.commit.json'));record=json.loads(commit.read_text())
                chunk=path/record['ranks'][0]['name']
                if kind=='chunk':chunk.unlink()
                elif kind=='crc':chunk.write_text(chunk.read_text().replace('0.02','0.03'))
                elif kind=='commit':commit.unlink()
                elif kind=='counter':record['previous_event_count']=1;atomic_json(commit,record)
                elif kind=='duplicate':commit.write_text(commit.read_text().replace('"sequence":0','"sequence":0,"sequence":0'))
                elif kind=='symlink':
                    renamed=chunk.with_suffix('.original');chunk.rename(renamed);chunk.symlink_to(renamed)
                else:
                    import zlib
                    text=chunk.read_text()
                    if kind=='float_id':text=text.replace('\n7\t','\n7.0\t')
                    else:text+=text.splitlines(keepends=True)[1]
                    chunk.write_text(text);record['ranks'][0].update(bytes=chunk.stat().st_size,crc32=zlib.crc32(chunk.read_bytes()))
                    atomic_json(commit,record)
                with self.assertRaises(ValueError):self.read(path)

    def test_public_native_snapshot_interval_cli_residuals_and_source_corruption(self):
        self._public_snapshot_interval()

    def test_isolated_event_units_and_epoch_are_not_cosmological(self):
        self._public_snapshot_interval(comoving=0)

    def _public_snapshot_interval(self,comoving=1):
        from stars import audit_native_ssp_birth_events,verify_native_ssp_birth_events
        helper=HaloStructureTests();self.addCleanup(helper.doCleanups)
        control,_=helper.native_fixture();second=control.snap.with_name('snapshot_001.hdf5')
        shutil.copy2(control.snap,second)
        for index,path in enumerate((control.snap,second)):
            with h5py.File(path,'r+') as f:
                f['Header'].attrs.update(Time=.5 if index==0 else .75,Redshift=(1. if index==0 else 1/.75-1) if comoving else 0.,StellarBirthEventCount=np.uint64(2*index),
                    StellarBirthCommitCount=np.uint64(index),StellarBirthJournalSchema=np.uint64(1))
                parameters=self.parameters()
                parameters['comoving']=comoving
                for key,native in {'unit_mass_g':'UnitMass_in_g','unit_time_s':'UnitTime_in_s',
                    'unit_velocity_cm_s':'UnitVelocity_in_cm_per_s','unit_length_cm':'UnitLength_in_cm',
                    'hubble_param':'HubbleParam','factor_sn':'FactorSN','comoving':'ComovingIntegrationOn','seed':'StellarRandomSeed'}.items():
                    f['Parameters'].attrs[native]=parameters[key]
            atomic_json(path.with_suffix('.publication.json'),{'schema':'native-hdf5-set-publication-v1',
                'state':'COMPLETE','generation':'block11-test','shard_count':1})
        self.run_journal(control.root/'data',2,mode='write' if comoving else 'isolated')
        request={'run_directory':str(control.root),'snapshots':[0,1],'memory_gib':.1}
        output=control.root/'events.json';report=audit_native_ssp_birth_events(request,output)
        self.assertEqual(report['event_count'],2);self.assertEqual(report['source_origin'],'synthetic_control')
        if not comoving:
            self.assertEqual(report['epoch']['canonical_time'],'isolated_native_elapsed_time')
            self.assertIsNone(report['epoch']['actual_scale_factor'])
            self.assertIsNone(report['epoch']['cosmic_time_gyr'])
        self.assertEqual(verify_native_ssp_birth_events(output),report)
        with np.load(str(output)+'.data/arrays.npz') as arrays:
            self.assertEqual(arrays['ssp_id'].dtype,np.dtype('uint64'))
            np.testing.assert_allclose(arrays['source_mass_residual_msun'],0)
            for k in 'xyz':np.testing.assert_allclose(arrays['source_momentum_residual_'+k],0)
            np.testing.assert_allclose(arrays['ssp_v_x'],5. if comoving else 3.)
        source=control.root/'request.json';atomic_json(source,request)
        argv=[sys.executable,'-B',str(Path(__file__).parent/'commands.py'),'audit-hydro-events',str(source),
            '--output',str(control.root/'cli-events.json')]
        run=subprocess.run(argv,capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
        run=subprocess.run(argv+['--verify'],capture_output=True,text=True);self.assertEqual(run.returncode,0,run.stderr)
        with h5py.File(second,'r+') as f:f['Header'].attrs['StellarBirthEventCount']=np.uint64(3)
        with self.assertRaisesRegex(ValueError,'source changed'):verify_native_ssp_birth_events(output)
        with self.assertRaises(ValueError):audit_native_ssp_birth_events(request,control.root/'bad.json')


TEST_GROUPS['test_native_ssp_birth'] = ['NativeSSPBirthTests','NativeSSPJournalTests']


class HierarchyReferenceTests(unittest.TestCase):
    def test_conservative_prolongation_cell_moments_and_mass(self):
        from core.zoom import conservative_cell_prolongation
        random=np.random.default_rng(71).normal(size=(8,8,8));saved=random.copy()
        for factor in (2,4):
            for degree in (0,3,5):
                fine=conservative_cell_prolongation(random,factor,degree=degree)
                np.testing.assert_allclose(fine.reshape(8,factor,8,factor,8,factor).mean(axis=(1,3,5)),random,atol=2e-14,rtol=2e-14)
        np.testing.assert_array_equal(random,saved)
        for degree in (3,5):
            fine=conservative_cell_prolongation(random,2,degree=degree)
            for axis in range(3):
                reflected=conservative_cell_prolongation(np.flip(random,axis),2,degree=degree)
                np.testing.assert_allclose(reflected,np.flip(fine,axis),rtol=3e-13,atol=3e-13)
            np.testing.assert_allclose(conservative_cell_prolongation(random.transpose(2,0,1),2,degree=degree),
                fine.transpose(2,0,1),rtol=3e-13,atol=3e-13)
        factor=4;x=np.arange(8,dtype=float)
        for power in range(6):
            coarse=((x+.5)**(power+1)-(x-.5)**(power+1))/(power+1)
            values=np.broadcast_to(coarse[:,None,None],(8,8,8))
            fine=conservative_cell_prolongation(values,factor,degree=5)
            left=np.arange(8*factor)/factor-.5;right=left+1/factor
            exact=(right**(power+1)-left**(power+1))/(power+1)*factor
            np.testing.assert_allclose(fine[12:20,12:20,12:20],np.broadcast_to(exact[12:20,None,None],(8,8,8)),rtol=3e-14,atol=5e-9)

    def test_hierarchy_leaf_averages_and_periodic_translation(self):
        from core.zoom import compose_hierarchy_density_reference
        rng=np.random.default_rng(13)
        hierarchy=[{'level':3,'origin':[0,0,0],'values':rng.normal(size=(8,8,8))},
            {'level':4,'origin':[12,2,4],'values':rng.normal(size=(8,8,8))},
            {'level':5,'origin':[28,8,12],'values':rng.normal(size=(8,8,8))}]
        for degree in (0,3,5):
            field,report=compose_hierarchy_density_reference(hierarchy,degree=degree)
            self.assertFalse(report['launch_authorized']);self.assertEqual(report['science_acceptance'],'NOT_ACCEPTED')
            for i,r in enumerate(hierarchy):
                factor=2**(5-r['level']);averaged=field.reshape(2**r['level'],factor,2**r['level'],factor,2**r['level'],factor).mean(axis=(1,3,5))
                indices=[(o+np.arange(n))%(2**r['level']) for o,n in zip(r['origin'],r['values'].shape)];actual=averaged[np.ix_(*indices)];mask=np.ones(r['values'].shape,bool)
                if i+1<len(hierarchy):
                    child=hierarchy[i+1];start=(np.array(child['origin'])//2-np.array(r['origin']))%(2**r['level']);shape=np.array(child['values'].shape)//2
                    mask[np.ix_(*[(a+np.arange(n))%size for a,n,size in zip(start,shape,r['values'].shape)])]=False
                np.testing.assert_allclose(actual[mask],r['values'][mask],rtol=3e-13,atol=3e-13)
            shift=np.array([1,2,-1]);translated=[]
            for i,r in enumerate(hierarchy):
                translated.append({'level':r['level'],'origin':[0,0,0] if i==0 else ((np.array(r['origin'])+shift*2**(r['level']-3))%(2**r['level'])).tolist(),
                    'values':np.roll(r['values'],tuple(shift),axis=(0,1,2)) if i==0 else r['values']})
            other,_=compose_hierarchy_density_reference(translated,degree=degree)
            np.testing.assert_allclose(other,np.roll(field,tuple(shift*4),axis=(0,1,2)),rtol=3e-13,atol=3e-13)

    def test_independent_periodic_wave_response_units_dc_and_translation(self):
        from core.zoom import periodic_hierarchy_displacement_reference
        side=16;box=25.;modes=np.array([1,2,0]);grid=np.indices((side,)*3);phase=2*np.pi*np.sum(modes[:,None,None,None]*(grid+.5),axis=0)/side+.37
        density=.001*np.cos(phase);expected=-.001*box/(2*np.pi)*np.sin(phase)[...,None]*modes/np.sum(modes*modes)
        actual,report=periodic_hierarchy_displacement_reference(density,box)
        np.testing.assert_allclose(actual,expected,atol=1e-16,rtol=2e-13)
        self.assertFalse(report['launch_authorized'])
        for degree in (0,3,5):
            response,_=periodic_hierarchy_displacement_reference(density,box,oversample=2,degree=degree)
            translated,_=periodic_hierarchy_displacement_reference(np.roll(density,(2,-1,3),axis=(0,1,2))+7.,box,oversample=2,degree=degree)
            np.testing.assert_allclose(translated,np.roll(response,(2,-1,3),axis=(0,1,2)),atol=2e-14,rtol=2e-12)

    def test_reference_rejects_bad_geometry_inputs_and_resource_budgets(self):
        from core.zoom import conservative_cell_prolongation,compose_hierarchy_density_reference,periodic_hierarchy_displacement_reference
        base={'level':2,'origin':[0,0,0],'values':np.zeros((4,4,4))}
        for arguments in ({'factor':True},{'factor':2,'degree':False},{'factor':2,'memory_gib':None},{'factor':2,'memory_gib':1e-9}):
            with self.assertRaises(ValueError):conservative_cell_prolongation(base['values'],**arguments)
        for child in ({'level':4,'origin':[0,0,0],'values':np.ones((4,4,4))},
                      {'level':3,'origin':[1,0,0],'values':np.ones((4,4,4))}):
            with self.assertRaises(ValueError):compose_hierarchy_density_reference([base,child])
        bad=base['values'].copy();bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):compose_hierarchy_density_reference([{**base,'values':bad}])
        with self.assertRaises(ValueError):compose_hierarchy_density_reference([base],memory_gib=1e-9)
        with self.assertRaises(ValueError):periodic_hierarchy_displacement_reference(base['values'],25.,oversample=4,memory_gib=1e-9)
        with self.assertRaises(ValueError):periodic_hierarchy_displacement_reference(base['values'],True)
        with self.assertRaises(ValueError):periodic_hierarchy_displacement_reference(base['values'],25.,oversample=3)
        with self.assertRaises(ValueError):compose_hierarchy_density_reference([
            {'level':3,'origin':[0,0,0],'values':np.zeros((8,)*3)},
            {'level':4,'origin':[4,4,4],'values':np.zeros((4,)*3)},
            {'level':5,'origin':[0,0,0],'values':np.zeros((4,)*3)}])

    def test_bounded_reference_matches_full_at_periodic_crossings(self):
        from core.zoom import compose_hierarchy_density_reference,sample_hierarchy_density_reference
        rng=np.random.default_rng(711)
        hierarchy=[{'level':3,'origin':[0,0,0],'values':rng.normal(size=(8,)*3)},
            {'level':4,'origin':[12,2,4],'values':rng.normal(size=(8,)*3)},
            {'level':5,'origin':[28,8,12],'values':rng.normal(size=(8,)*3)}]
        originals=[r['values'].copy() for r in hierarchy]
        for degree in (0,3,5):
            full,_=compose_hierarchy_density_reference(hierarchy,degree=degree)
            for origin,shape in (([-3,7,29],[15,17,11]),([29,30,31],[40,8,8]),([28,8,12],[8,8,8])):
                bounded,receipt=sample_hierarchy_density_reference(hierarchy,origin,shape,degree=degree)
                indices=[(o+np.arange(n))%32 for o,n in zip(origin,shape)]
                np.testing.assert_array_equal(bounded,full[np.ix_(*indices)])
                self.assertFalse(receipt['global_finest_grid_allocated'])
                self.assertFalse(receipt['launch_authorized'])
        for before,record in zip(originals,hierarchy):np.testing.assert_array_equal(before,record['values'])

    def test_bounded_reference_deep_hierarchy_avoids_global_fine_grid(self):
        from core.zoom import sample_hierarchy_density_reference
        hierarchy=[{'level':level,'origin':[0,0,0],'values':np.ones((8,)*3)} for level in range(3,19)]
        values,receipt=sample_hierarchy_density_reference(hierarchy,[-4,2,7],[24,24,24],degree=5,memory_gib=.03)
        np.testing.assert_allclose(values,1.,rtol=5e-13,atol=5e-13)
        self.assertFalse(receipt['global_finest_grid_allocated'])
        self.assertEqual(receipt['levels'][-1],18)
        self.assertLess(sum(r['cells'] for r in receipt['evaluated_rectangles']),24**3*16)
        self.assertLess(receipt['estimated_workspace_bytes'],.03*1024**3)

    def test_bounded_reference_preflight_rejects_bad_requests(self):
        from core.zoom import sample_hierarchy_density_reference
        h=[{'level':3,'origin':[0,0,0],'values':np.ones((8,)*3)}]
        for origin,shape,kwargs in (([0,0,0],[0,8,8],{}),([.5,0,0],[8,8,8],{}),
                ([0,0,0],[8,8,8],{'degree':4}),([0,0,0],[100000,100000,100000],{'memory_gib':.01}),
                ([0,0,0],[8,8,8],{'memory_gib':None})):
            with self.assertRaises(ValueError):sample_hierarchy_density_reference(h,origin,shape,**kwargs)

TEST_GROUPS['test_hierarchy_reference']=['HierarchyReferenceTests']


class HMFFittingFunctionTests(unittest.TestCase):
    """Numerical controls and a saved-input-to-figure integration check."""
    def fixture(self, root, model='lcdm'):
        root=Path(root);(root/'input').mkdir(parents=True,exist_ok=True)
        k=np.geomspace(1e-5,1e3,4096)
        pk=root/'input/pk.dat';np.savetxt(pk,np.column_stack((k,20*k**-2)),header='redshift z=0')
        bg=root/'input/background.dat'
        bg.write_text('# 1:z 2:H [1/Mpc] 3:Omega_r(z) 4:gr.fac. D 5:gr.fac. f\n99 1 0.01 .01 1\n9 1 .001 .1 1\n1 1 .0001 .5 1\n0 1 .00001 1 1\n')
        from core.foundation import file_sha256
        artifacts={name:{'path':str(path),'sha256':file_sha256(path)} for name,path in
                   [('linear_power_spectrum',pk),('boltzmann_background',bg)]}
        artifacts['linear_power_spectrum']['redshift']=0.
        (root/'manifest.json').write_text(json.dumps({'model':model,'input_artifacts':artifacts}))
        meta={'model':model,'engine':'axiclass' if model=='ede' else 'class','pk_file':str(pk),
              'background_file':str(bg),'pk_sha256':file_sha256(pk),'output_redshift':0.}
        (root/'input'/f'{model}_cosmology.json').write_text(json.dumps(meta))
        cfg={'cosmology':{'H0':70.,'omega_b':.0245,'omega_cdm':.1225},
             'analysis':{'hmf_mass_definition':'fof'},'simulation':{'box_mpc_h':100.,'physics':'dm'},
             'halos':{'fof_linking_length':.2}}
        edges=np.geomspace(1e10,1e15,9);mass=np.sqrt(edges[:-1]*edges[1:])
        counts=np.array([3000,1200,400,90,20,4,1,0])
        measured=(mass,counts/1e6/np.diff(np.log10(edges)),counts/1e6/np.diff(edges),counts,edges)
        return cfg,measured,k,20*k**-2

    def test_press_schechter_normalization_and_st_normalization(self):
        from scipy.integrate import simpson
        from core.hmf_fitting_functions import multiplicity_functions
        lognu=np.linspace(-24,4,20001);sigma=1.686/np.exp(lognu)
        fits=multiplicity_functions(sigma,np.full_like(sigma,1/6))
        self.assertAlmostEqual(simpson(fits['press_schechter'],x=lognu),1.,places=8)
        self.assertAlmostEqual(simpson(fits['sheth_tormen'],x=lognu),1.,delta=1e-4)

    def test_reed_spectral_slope_and_watson_coefficients(self):
        from core.hmf_fitting_functions import multiplicity_functions
        first=multiplicity_functions(np.array([1.]),np.array([1/6]))
        second=multiplicity_functions(np.array([1.]),np.array([1/12]))
        # Only Reed07 depends on n_eff. Its suppression changes by exp(-3*C).
        expected=np.exp(-3*.03*1.686**.6)
        self.assertAlmostEqual(float(second['reed_2007'][0]/first['reed_2007'][0]),expected,places=13)
        self.assertAlmostEqual(float(first['watson_fof'][0]),.282*(2.163**1.406+1)*np.exp(-1.21),places=14)
        for name in first:
            if name!='reed_2007':np.testing.assert_array_equal(first[name],second[name])
        with self.assertRaises(ValueError):multiplicity_functions([1.],[0.])

    def test_power_law_variance_and_bin_integrated_predictions(self):
        from core.hmf_fitting_functions import binned_predictions
        with tempfile.TemporaryDirectory() as tmp:
            cfg,measured,k,power=self.fixture(tmp)
            fits,sigma,slope,covered=binned_predictions(k,power,.3,measured[4],0.)
            # P~k^-2 gives sigma~M^-1/6 and n_eff=-2, independently of amplitude.
            np.testing.assert_allclose(slope,1/6,rtol=.002)
            np.testing.assert_allclose(sigma/sigma[0],(measured[0]/measured[0][0])**(-1/6),rtol=.002)
            self.assertTrue(covered.all())
            # Coarse-bin means must reproduce sums of half-width-bin integrals.
            refined_edges=np.geomspace(measured[4][0],measured[4][-1],17)
            refined,_,_,_=binned_predictions(k,power,.3,refined_edges,0.)
            for name in fits:
                np.testing.assert_allclose(fits[name]['dn_dlog10M'],
                    refined[name]['dn_dlog10M'].reshape(-1,2).mean(axis=1),rtol=2e-5)
            restricted=binned_predictions(k[k<30],power[k<30],.3,measured[4],0.)
            self.assertFalse(restricted[3][0])
            self.assertTrue(np.isnan(restricted[0]['watson_fof']['dn_dlog10M'][0]))

    def test_real_saved_input_plots_tables_and_original_outputs_preserved(self):
        from core.hmf_fitting_functions import write_hmf_fit_comparisons
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,measured,_,_=self.fixture(root)
            old=root/'results/HMF_000.png';old.parent.mkdir();old.write_bytes(b'old plot sentinel')
            result=write_hmf_fit_comparisons(root,cfg,0,'lcdm',0.,measured)
            self.assertEqual(result['status'],'DIAGNOSTIC_ONLY')
            self.assertEqual(len(result['artifacts']),12)
            self.assertEqual(old.read_bytes(),b'old plot sentinel')
            output=root/'results/hmf_fitting_functions'
            for artifact in result['artifacts']:self.assertTrue((output/artifact).is_file())
            with (output/'comparison_000.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),8)
            self.assertEqual(float(rows[-1]['count']),0)
            self.assertTrue(np.isfinite(float(rows[-1]['poisson_68_upper'])))

    def test_metadata_integrity_mass_definition_and_ede_scope(self):
        from core.hmf_fitting_functions import write_hmf_fit_comparisons,_linear_inputs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,measured,_,_=self.fixture(root,'ede')
            k,power,omega,provenance=_linear_inputs(root,cfg,'ede',1.)
            np.testing.assert_allclose(power,5*k**-2)
            self.assertAlmostEqual(omega,.3)
            cfg['analysis']['hmf_mass_definition']='m200c'
            report=write_hmf_fit_comparisons(root,cfg,0,'ede',0.,measured)
            self.assertEqual(report['status'],'SKIPPED')
            self.assertIn('SO masses',report['reason'])
            cfg['analysis']['hmf_mass_definition']='fof'
            with patch('core.hmf_fitting_functions._write_plots_and_table',return_value=[]):
                report=write_hmf_fit_comparisons(root,cfg,0,'ede',0.,measured)
            self.assertEqual(report['status'],'DIAGNOSTIC_ONLY')
            self.assertIn('EXTRAPOLATION',report['cosmology_scope'])
            with (root/'input/pk.dat').open('a') as stream:stream.write('\n')
            report=write_hmf_fit_comparisons(root,cfg,0,'ede',0.,measured)
            self.assertEqual(report['status'],'SKIPPED')
            self.assertIn('checksum',report['reason'])

TEST_GROUPS['test_hmf_fitting_functions']=['HMFFittingFunctionTests']


class WorkstationRegressionTests(unittest.TestCase):
    def test_requested_outputs_are_saved_and_tree_prerequisite_is_enabled(self):
        from commands import configure_analysis_outputs
        for selection in (['all'],['available'],['hmf','histories','maps']):
            cfg={'analysis':{'all':False,'products':['hmf']},'halos':{'rockstar':True,'consistent_trees':False}}
            configure_analysis_outputs(cfg,selection)
            self.assertEqual(cfg['analysis']['requested_outputs'],selection)
            self.assertTrue(cfg['halos']['consistent_trees'])
            if selection in (['all'],['available']):
                self.assertTrue(cfg['analysis']['all']);self.assertEqual(cfg['analysis']['products'],[])
            else:
                self.assertEqual(cfg['analysis']['products'],['hmf','histories'])

    def test_selected_hmf_publishes_evolution_figures_and_retains_source(self):
        from core.analysis import analyze_selected_products
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,cfg=catalogue_fixture(root)
            cfg['analysis']['rare_halo_thresholds_msun_h']=[1.e10]
            (root/'config.yaml').write_text(yaml.safe_dump(cfg))
            before=file_sha256(root/'data/snapshot_000.hdf5')
            # Native halo measurement is represented by a controlled producer;
            # actual summary propagation, plotting and publication run normally.
            payload={'status':'COMPLETE','rare_halos':{'maximum_mass_msun_h':2.e12,'10000000000.0':1}}
            with patch('core.analysis._analyze_one_product',return_value=payload), \
                    patch('figures.generate_single_run_story_figures',return_value={'status':'UNAVAILABLE'}):
                result=analyze_selected_products(root,['hmf'],include_pdf_report=False)
            output=Path(result['output_directory'])
            self.assertEqual(result['skipped']['cosmic_web_story'],'UNAVAILABLE')
            self.assertEqual(result['snapshots']['0']['rare_halos'],payload['rare_halos'])
            self.assertTrue((output/'maximum_halo_mass_vs_z.png').is_file())
            self.assertTrue((output/'rare_halo_abundance_vs_z.png').is_file())
            from commands import verified_analysis_figure_gallery
            gallery=verified_analysis_figure_gallery(root,output,root/'exported_scene')
            self.assertTrue(any(row['title']=='maximum_halo_mass_vs_z' for row in gallery))
            for row in gallery:self.assertEqual(file_sha256(root/'exported_scene'/row['path']),row['sha256'])
            self.assertEqual(file_sha256(root/'data/snapshot_000.hdf5'),before)

    def test_histograms_preserve_samples_for_constant_and_one_ulp_ranges(self):
        from core.analysis import histogram_pdf
        for center in (0., -4., 4., 1.e100, -1.e100):
            values=np.array([center,center,np.nextafter(center,np.inf),np.nan,np.inf])
            with np.errstate(all='raise'):
                density,edges=histogram_pdf(values,100)
            self.assertTrue(np.isfinite(density).all())
            self.assertTrue((np.diff(edges)>0).all())
            self.assertAlmostEqual(float(np.sum(density*np.diff(edges))),1.,places=14)
            counts,_=np.histogram(values[np.isfinite(values)],bins=edges)
            self.assertEqual(int(counts.sum()),3)
        for values in ([],[np.nan,np.inf]):
            with self.assertRaisesRegex(ValueError,'no finite'):histogram_pdf(values,100)

    def test_histogram_rejects_invalid_bin_counts(self):
        from core.analysis import histogram_pdf
        for bins in (True,0,-1,2.5):
            with self.assertRaises(ValueError):histogram_pdf([1.,2.],bins)

    def test_nearly_constant_hydro_temperature_and_phase_diagram(self):
        from core.analysis import hydro_diagnostics
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot=root/'snapshot.hdf5'
            with h5py.File(snapshot,'w') as h5:
                h5.create_group('Header').attrs.update(Time=.5,HubbleParam=.7)
                gas=h5.create_group('PartType0');gas['Coordinates']=np.zeros((12,3))
                gas['Density']=np.ones(12);gas['Temperature']=np.full(12,1.e4)
                gas['Temperature'][1]=np.nextafter(1.e4,np.inf)
                for name in ('Density','Temperature'):
                    gas[name].attrs.update(to_cgs=1.,a_scaling=0.,h_scaling=0.)
                gas['Temperature'].attrs['unit']='K'
            with patch('core.analysis.savefig'),patch('core.analysis.hydro_field_maps',return_value={'status':'SKIPPED'}):
                result=hydro_diagnostics(root,{'hydro':{'star_formation':False}},0,[snapshot])
            self.assertEqual(result['temperature_status'],'COMPLETE')
            self.assertEqual(result['temperature_valid_sample_count'],12)
            self.assertEqual(int(result['phase_particle_count_2d'].sum()),12)
            self.assertAlmostEqual(float(np.sum(result['temperature_probability_density']*np.diff(result['temperature_log10_bin_edges']))),1.)
            with h5py.File(snapshot,'r+') as h5:h5['PartType0/Temperature'][:]=np.nan
            with patch('core.analysis.savefig'),patch('core.analysis.hydro_field_maps',return_value={}):
                missing=hydro_diagnostics(root,{'hydro':{'star_formation':False}},0,[snapshot])
            self.assertEqual(missing['temperature_status'],'NO_VALID_VALUES')
            self.assertNotIn('temperature_probability_density',missing)
            plt.close('all')

    def test_multishard_remainder_sampling_has_no_invalid_subtraction(self):
        from core.analysis import read_particle_fields
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i,count in enumerate([5,5,5,5]):
                path=Path(tmp)/f'shard{i}.hdf5';paths.append(path)
                with h5py.File(path,'w') as h5:
                    g=h5.create_group('PartType0');g['Coordinates']=np.zeros((count,3))
                    g['Temperature']=np.full(count,10.+i)
            with np.errstate(invalid='raise'):
                data,weights=read_particle_fields(paths,0,['Temperature'],max_points=6,return_sample_weights=True)
            self.assertEqual(len(data['Temperature']),6)
            self.assertAlmostEqual(float(weights.sum()),20.)
            second,second_weights=read_particle_fields(paths,0,['Temperature'],max_points=6,return_sample_weights=True)
            np.testing.assert_array_equal(data['Temperature'],second['Temperature'])
            np.testing.assert_array_equal(weights,second_weights)

    def test_grouped_input_validates_whole_rows_and_exact_integer_ids(self):
        from core.cli_prompts import required_values
        fields=[('seed','exact seed',int,0,2**64-1),('max','time',float,0,1),
                ('min','time',float,0,lambda row:row['max'])]
        with patch('builtins.input',side_effect=['2.5,.1,.05','1,.1,.2',f'{2**64-1},.1,.05']),patch('builtins.print'):
            answer=required_values('Controls',fields)
        self.assertEqual(answer,{'seed':2**64-1,'max':.1,'min':.05})

    def test_grouped_linear_prompt_has_no_preset_values(self):
        from core.linear import prompt_config
        with patch('builtins.input',side_effect=['fresh','lcdm','67.4,.049,.265','2.1e-9,.965,.054','99,200']),patch('builtins.print'):
            result=prompt_config()
        self.assertEqual(result['name'],'fresh');self.assertEqual(result['z_start'],99.)
        self.assertEqual(result['k_max_h_mpc'],200.)

    def test_available_records_unready_products_while_all_stays_strict(self):
        from commands import plan_saved_analysis
        inventory={'snapshot_epochs':[{'snapshot':0},{'snapshot':1}],
            'species_by_epoch':{'0':{'dm':10,'gas':10,'stars':5},'1':{'dm':10,'gas':10,'stars':5}},
            'catalogue_support':{'0':True,'1':True},'zoom_support':{'declared':False,'coarse_particles_present':False},
            'native_features':{},'field_support':{'0':{},'1':{}},'volume_selection':{'status':'COMPLETE_PERIODIC_VOLUME'}}
        # Use real classification: only the expensive native inventory is replaced.
        with patch('core.research.capabilities.inspect_run_capabilities',return_value=inventory), \
                patch('visualization._find_blender_executable',return_value=None):
            available=plan_saved_analysis('/unused','available')
            strict=plan_saved_analysis('/unused','all')
        self.assertIn('hydro',available['selected'])
        self.assertIn('stellar_components',available['omitted_missing_inputs'])
        self.assertIn('Blender executable',available['omitted_missing_inputs']['blender'])
        self.assertFalse(available['missing_inputs'])
        self.assertIn('stellar_components',strict['missing_inputs'])

    def test_public_full_analysis_command_avoids_python_shell_quoting(self):
        import commands
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv',['commands.py analysis',tmp,'--full-analysis','--overwrite']),patch('commands.analyze_run') as execute:
                commands.run_analysis_main()
            execute.assert_called_once_with(Path(tmp),overwrite=True,full_analysis=True)
            with patch('sys.argv',['commands.py analysis',tmp,'--overwrite']):
                with self.assertRaises(SystemExit):commands.run_analysis_main()

    def test_fresh_pipeline_hands_off_actual_linear_output(self):
        from core.cli import run_pipeline_main
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'class';binary.write_text('fixture; never executed')
            cfg={'name':'user_choice','model':'lcdm'};actual=root/'state/runs/linear/user_choice'
            seen=[];previous=os.environ.get('RESEARCH2_STATE_ROOT')
            with patch('sys.argv',['commands.py run','--count','1','--state-root',str(root/'state'),'--binary',str(binary)]), \
                    patch('core.linear.prompt_config',return_value=cfg),patch('builtins.input',return_value='yes'),patch('core.linear.validate'), \
                    patch('core.linear.run',return_value=actual) as solver, \
                    patch('commands.run_gadget_main',side_effect=lambda:seen.append(list(sys.argv))):
                run_pipeline_main()
            solver.assert_called_once_with(cfg,(root/'state/runs/linear').resolve(),binary.resolve())
            self.assertEqual(seen,[['commands.py gadget','--cosmology',str(actual)]])
            self.assertEqual(os.environ.get('RESEARCH2_STATE_ROOT'),previous)

TEST_GROUPS['test_workstation_regressions']=['WorkstationRegressionTests']


@unittest.skipUnless(shutil.which('c++'), 'A C++ compiler is required for native cooling verification')
class NativeCoolingTests(unittest.TestCase):
    def test_original_native_solver_reproduces_bad_cold_energy_closure(self):
        from core.native_cooling_control import run_control
        baseline=Path(__file__).parent/'core/software/upstream/gadget4/src/cooling_sfr/cooling.cc'
        with tempfile.TemporaryDirectory() as tmp:
            record=run_control(Path(tmp)/'original',mode='reproduce',source_path=baseline)
        self.assertEqual(record['status'],'FAIL')
        self.assertIn('Thermodynamic closure failed',record['failure'])

    def test_corrected_native_rate_temperature_and_energy_sweeps(self):
        from core.native_cooling_control import run_control
        with tempfile.TemporaryDirectory() as tmp:
            record=run_control(Path(tmp)/'corrected')
        self.assertEqual(record['status'],'PASS',record.get('failure'))
        result=record['measurements']
        self.assertEqual(result['cases'],1344)
        self.assertEqual(result['implicit_cases'],2400)
        # Recomputing the closure introduces roundoff beyond the solver's
        # 1e-8 stopping threshold; retain a small independent-check allowance.
        self.assertLess(result['max_energy_relative_error'],2.e-8)
        self.assertLess(result['max_implicit_residual_relative_error'],2.e-8)
        self.assertLess(result['max_charge_absolute_error'],1.e-12)

TEST_GROUPS['test_native_cooling']=['NativeCoolingTests']


def selected_test_suite(groups, loader=None):
    """Load retained domain groups without recreating separate test modules."""
    import unittest
    loader=loader or unittest.defaultTestLoader
    suite=unittest.TestSuite()
    classes=[]
    for group in groups:
        if group not in TEST_GROUPS:
            raise ValueError('Unknown verification group: '+group)
        for name in TEST_GROUPS[group]:
            value=globals().get(name)
            if isinstance(value,type) and issubclass(value,unittest.TestCase) and value not in classes:
                classes.append(value)
    for value in classes:suite.addTests(loader.loadTestsFromTestCase(value))
    return suite






def main(argv=None):
    import sys
    import unittest
    args=list(sys.argv[1:] if argv is None else argv)
    groups=[]
    while '--group' in args:
        index=args.index('--group')
        if index+1>=len(args):raise SystemExit('--group requires a retained domain group name')
        groups.append(args[index+1]);del args[index:index+2]
    if groups:
        suite=selected_test_suite(groups)
        if args and any(a not in {'-v','--verbose','-q','--quiet'} for a in args):
            raise SystemExit('Group verification accepts --group NAME plus -v or -q')
        result=unittest.TextTestRunner(verbosity=2 if '-v' in args or '--verbose' in args else 0 if '-q' in args or '--quiet' in args else 1).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main(module=__name__,argv=[str(Path(__file__).resolve()),*args])

class VisualWorkflowRegressionTests(unittest.TestCase):
    def test_layered_actions_respect_datablock_slot(self):
        from types import SimpleNamespace as N
        from blender import action_fcurves
        action=N(is_action_layered=True,layers=[N(strips=[N(channelbags=[
            N(slot_handle=1,fcurves=['own']),N(slot_handle=2,fcurves=['other'])])])])
        self.assertEqual(action_fcurves(N(action=action,action_slot=N(handle=1))),['own'])
        self.assertEqual(action_fcurves(N(action=N(fcurves=['legacy']))),['legacy'])
        self.assertEqual(action_fcurves(None),())

    def test_blender5_glare_uses_menu_sockets(self):
        from blender import configure_glare
        from types import SimpleNamespace as N
        class Node:
            __slots__=('inputs',)
        node=Node();node.inputs={'Type':N(default_value=None),'Quality':N(default_value=None)}
        configure_glare(node)
        self.assertEqual(node.inputs['Type'].default_value,'Fog Glow')
        self.assertEqual(node.inputs['Quality'].default_value,'High')
        old=N(inputs={});configure_glare(old)
        self.assertEqual(old.glare_type,'FOG_GLOW');self.assertEqual(old.quality,'HIGH')

    def test_blender5_format_switches_media_before_format(self):
        from blender import set_output_format
        from types import SimpleNamespace as N
        class Settings:
            media_type='IMAGE'
            @property
            def file_format(self):return self.format
            @file_format.setter
            def file_format(self,value):
                if value=='FFMPEG' and self.media_type!='VIDEO':raise ValueError('wrong media type')
                if value=='PNG' and self.media_type!='IMAGE':raise ValueError('wrong media type')
                self.format=value
        settings=Settings();scene=N(render=N(image_settings=settings))
        set_output_format(scene,'FFMPEG');self.assertEqual(settings.media_type,'VIDEO')
        set_output_format(scene,'PNG');self.assertEqual(settings.media_type,'IMAGE')

    def test_visual_failure_does_not_prevent_other_exports(self):
        from core.visual_pipeline import execute_visual_products
        status={'artifacts':{},'unavailable':{}};stages=[]
        cfg={'visualization':{'map_fields':['dm'],'map_projections':['xy']}}
        with patch('visualization.make_maps',side_effect=ValueError('bad map')), \
                patch('visualization.make_movie',return_value=Path('/saved/movie.mp4')), \
                patch('visualization.make_xdmf',return_value=[Path('/saved/data.xdmf')]):
            execute_visual_products('/unused',['maps','movie','xdmf'],cfg,status,stages.append)
        self.assertIn('maps',status['visual_failures'])
        self.assertEqual(status['artifacts']['movie'],'/saved/movie.mp4')
        self.assertIn('xdmf',status['artifacts'])
        self.assertIn('xdmf_finished',stages)
        json.dumps(status,allow_nan=False)

    def test_halo_movie_consumes_current_selected_bundle(self):
        from core.visual_pipeline import execute_visual_products
        status={'artifacts':{'analysis':'/run/results/selected_123'},'unavailable':{}}
        with patch('visualization.make_halo_growth_movie',return_value=Path('/saved/halo.mp4')) as movie:
            execute_visual_products('/run',['halo_movie'],{'visualization':{}},status,lambda name:None)
        movie.assert_called_once_with('/run',analysis_path=Path('/run/results/selected_123/analysis_products.h5'))

    def test_combined_blender_movie_is_explicitly_rendered(self):
        from core.visual_pipeline import execute_visual_products
        status={'artifacts':{},'unavailable':{}}
        cfg={'visualization':{'blender_sequence_max_points':10,'blender_render_quality':'preview','blender_frames_per_snapshot':2}}
        with patch('commands.export',return_value=Path('/export/blender_universe.json')),patch('commands.build_scene') as build:
            execute_visual_products('/run',['blender','blender_movie'],cfg,status,lambda name:None)
        build.assert_called_once_with(Path('/export/blender_universe.json'),preview=True,render=True)
        self.assertIn('blender_scene',status['artifacts']);self.assertIn('blender_movie',status['artifacts'])

    def test_hmf_tail_and_deviance_reference_including_missing_support(self):
        from core.hmf_fitting_functions import count_diagnostics
        result=count_diagnostics([0,2,3],[1,2,4],[1,2,4,8],10.)
        np.testing.assert_allclose(result['measured_cumulative_number_density'],[.5,.5,.3])
        np.testing.assert_allclose(result['fit_cumulative_number_density'],[.7,.6,.4])
        np.testing.assert_allclose(result['signed_poisson_deviance'],[-np.sqrt(2),0,-np.sqrt(2*(1+3*np.log(.75)))])
        result=count_diagnostics([0,2,3],[1,np.nan,4],[1,2,4,8],10.)
        self.assertTrue(np.isnan(result['fit_cumulative_number_density'][:2]).all())
        self.assertEqual(result['fit_cumulative_number_density'][2],.4)

    def test_planetarium_exports_frames_numeric_contracts_and_decodable_movie(self):
        from core.visual_pipeline import export_planetarium_timeline
        import imageio.v2 as imageio
        from PIL import Image
        def frame(run,snap,path,**options):
            Image.fromarray(np.full((16,16,3),snap*50,dtype=np.uint8)).save(path)
            Path(str(path)+'.data').mkdir()
            np.savez(Path(str(path)+'.data')/'arrays.npz',actual_values=np.array([snap]))
            return {'source':{'snapshot':snap,'redshift':float(1-snap)}}
        with tempfile.TemporaryDirectory() as tmp:
            cfg={'visualization':{'movie_fps':2,'planetarium':{'resolution':16,'max_particles':10}}}
            with patch('visualization.snapshot_ids',return_value=[0,1]),patch('visualization.export_domemaster',side_effect=frame):
                first=Path(export_planetarium_timeline(tmp,cfg))
                second=Path(export_planetarium_timeline(tmp,cfg))
            self.assertNotEqual(first,second)
            report=json.loads(first.read_text());self.assertEqual(report['status'],'COMPLETE')
            self.assertEqual([row['snapshot'] for row in report['frames']],[0,1])
            movie=first.parent/report['movie']['path']
            reader=imageio.get_reader(movie)
            try:self.assertEqual(len(list(reader.iter_data())),2)
            finally:reader.close()
            self.assertTrue((first.parent/'frame_00000.png.data/arrays.npz').is_file())

TEST_GROUPS['test_visual_workflow_regressions']=['VisualWorkflowRegressionTests']

class AllOutputConnectionTests(unittest.TestCase):
    def test_halo_movie_consumes_actual_selected_history_writer(self):
        from analysis import _write_product_hdf5
        from core.data import canonical_halo_history_payload,attach_time_series_contract
        from core.cosmology import run_proper_time_clock
        from visualization import _read_halo_movie_tracks,make_halo_growth_movie
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);test_blender_data_contract_fixture(root)
            folder=root/'results/analysis_products_control';folder.mkdir(parents=True)
            path=folder/'analysis_products.h5'
            payload={'status':'COMPLETE','histories':[{'final_halo_id':2**63+7,
                'track':[{'scale_factor':.5,'redshift':1.,'mass_msun_h':1.e10},
                         {'scale_factor':1.,'redshift':0.,'mass_msun_h':2.e10}],
                'merger_events':[{'redshift':0.,'mass_ratio':.5,'main_mass_msun_h':1.e10}]}]}
            with h5py.File(path,'w') as out:
                group=out.create_group('main_progenitor_histories')
                _write_product_hdf5(group,'data',canonical_halo_history_payload(payload))
                attach_time_series_contract(group['data'],clock=run_proper_time_clock(root))
            with h5py.File(path) as out:
                tracks=_read_halo_movie_tracks(out['main_progenitor_histories'])
            self.assertEqual(tracks[0]['id'],2**63+7)
            np.testing.assert_array_equal(tracks[0]['mass'],[1.e10,2.e10])
            self.assertEqual(tracks[0]['mergers'],[(1.,1.e10,.5)])
            with patch('visualization.imageio.get_writer',side_effect=CapturingEncoder):
                movie=make_halo_growth_movie(root,analysis_path=path)
            self.assertTrue(movie.is_file())
            receipt=json.loads((movie.parent/'halo_growth_contract.json').read_text())
            self.assertEqual(receipt['schema'],'halo-growth-publication-v1')

    def test_aliases_accept_supplied_request_files_before_evolution(self):
        from commands import validate_analysis_requests
        request={'snapshots':[0,1],'catalogue':'/actual/selection','candidate_ids':[1],'memory_gib':1}
        for name in ('all','available'):
            result=validate_analysis_requests([name],stellar_history=request,allow_incomplete=True)
            self.assertEqual(result['stellar_history'],request)
        with self.assertRaisesRegex(ValueError,'Complete analysis requires'):
            validate_analysis_requests(['all'])

    def test_halo_movie_selection_enables_configured_tree_prerequisite(self):
        from commands import configure_analysis_outputs
        cfg={'analysis':{},'halos':{'rockstar':True,'consistent_trees':False}}
        configure_analysis_outputs(cfg,['halo_movie'])
        self.assertTrue(cfg['halos']['consistent_trees'])

    def test_stellar_movie_handles_star_formation_after_first_output(self):
        from visualization import make_movie
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'config.yaml').write_text(yaml.safe_dump({
                'simulation':{'box_mpc_h':1.},'visualization':{'movie_fps':2,'slice_mode':'full_box','slice_thickness_mpc_h':None,'slice_center_mpc_h':None}}))
            native=[]
            for snap,count in [(0,0),(1,2),(2,3)]:
                path=root/f'snapshot_{snap:03d}.hdf5';native.append(path)
                with h5py.File(path,'w') as out:
                    if count:out.create_group('PartType4')['Coordinates']=np.zeros((count,3))
            maps=root/'maps';maps.mkdir();(maps/'map_contract.json').write_text('{}')
            paths=[]
            for snap in (1,2):
                path=maps/f'stars_xy_{snap:03d}.png';Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(path);paths.append(path)
            with patch('visualization.snapshot_ids',return_value=[0,1,2]), \
                    patch('visualization.read_snapshot_header',side_effect=lambda run,snap:({},[native[snap]])), \
                    patch('visualization.make_maps',return_value=paths), \
                    patch('visualization.imageio.get_writer',side_effect=CapturingEncoder):
                movie=make_movie(root,field='stars')
            report=json.loads((movie.parent/'movie_contract.json').read_text())
            self.assertEqual(report['snapshot_storage_indices'],[1,2])
            self.assertEqual(report['omitted_snapshots'][0]['snapshot'],0)
            paths.pop()
            with patch('visualization.snapshot_ids',return_value=[0,1,2]), \
                    patch('visualization.read_snapshot_header',side_effect=lambda run,snap:({},[native[snap]])), \
                    patch('visualization.make_maps',return_value=paths):
                with self.assertRaisesRegex(RuntimeError,'species-present'):
                    make_movie(root,field='stars')

TEST_GROUPS['test_all_output_connections']=['AllOutputConnectionTests']

if __name__=='__main__':main()
