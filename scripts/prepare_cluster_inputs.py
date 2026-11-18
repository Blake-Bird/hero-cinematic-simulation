"""Freeze one model's inputs from matched templates and the Slurm allocation."""
import json,math,os,re,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.resources import cpu_count,memory_limit_gb,estimate_run_resources
P=Path(__file__).resolve().parents[1]
model,n=sys.argv[1],int(sys.argv[2])
if model not in ('ede','lcdm') or n not in (32, 128, 512):raise SystemExit('Choose ede/lcdm and 32/128/512')
campaign=os.environ.get('CAMPAIGN','matched40')
if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,31}',campaign):raise SystemExit('CAMPAIGN must be a short safe identifier')
box=float(os.environ.get('BOX_MPC_H','40'));seed=int(os.environ.get('SEED','42'))
if not math.isfinite(box) or box<=0 or not 0<=seed<=2147483647:raise SystemExit('Invalid box or seed')
ranks=int(os.environ['SLURM_NTASKS']);threads=int(os.environ.get('SLURM_CPUS_PER_TASK','1'))
if ranks*threads>cpu_count() and not (os.environ.get('ALLOW_LOCAL') == '1' or sys.platform == 'darwin'):raise SystemExit('Python affinity/allocation exposes fewer CPUs than ranks x threads. Inspect Slurm task binding before launch.')
root=Path(os.environ['RESEARCH2_STATE_ROOT']).expanduser().resolve()/('campaigns/'+campaign+'/n'+str(n))
pair_mode = os.environ.get('COSMOLOGY_PAIR', 'bestfit')
config_dir = P / 'configs' if pair_mode == 'bestfit' else P / 'configs' / 'controlled_pair'
run = json.loads((P / f'configs/run_{n}.json').read_text())
linear = json.loads((config_dir / f'linear_{model}.json').read_text())

# Use generated schedule if requested or for production 512^3
is_benchmark = os.environ.get('BENCHMARK') == '1'
default_tier = 'benchmark_steps' if is_benchmark else ('production_120' if n >= 128 else 'smoke_32')
schedule_tier = os.environ.get('SCHEDULE_TIER', default_tier)
from scripts.generate_schedule import build_snapshot_schedule, format_schedule_string
sched_list = build_snapshot_schedule(schedule_tier)
run['schedule'] = format_schedule_string(sched_list)

# Match physical comoving volume by using LCDM as the reference physical scale
match_phys = os.environ.get('MATCH_PHYSICAL_VOLUME', '1') == '1'
h_ref = 0.6781  # LCDM reference hubble parameter
h_model = float(linear.get('H0', 67.81 if model == 'lcdm' else 74.83)) / 100.0
model_box = box * (h_model / h_ref) if match_phys else box

linear.update(name=f'{model}_{campaign}_n{n}',k_max_h_mpc=max(250.,3*math.pi*n/model_box))
run.update(name=f'{model}_{campaign}_n{n}',box_mpc_h=model_box,seed=seed,softening_kpc_h=1000*model_box/n/40,softening_max_physical_kpc_h=1000*model_box/n/80,mpi_ranks=ranks,omp_threads=threads,cpu_threads=ranks*threads,build_jobs=min(8,cpu_count()),num_files_per_snapshot=min(8,ranks),filament_smoothing_mpc_h=model_box/64,rockstar_wall_seconds=21600 if n>=128 else 3600)
# Compute a per-job budget from scheduler/cgroup memory, never from the whole cluster.
mem=memory_limit_gb({'hardware':{'memory_limit_gb':'auto'}})
run['memory_limit_gb']=mem

# Common physical mass bins across both models (reference Omega_m = 0.30)
ref_omega_m = 0.30
ref_particle_mass = 2.77536627245708e11 * ref_omega_m * box**3 / n**3
run['catalogue_min_particles'] = 32
run['hmf_min_particles'] = 100 if n >= 128 else 32
run['hmf_diagnostic_thresholds'] = [32, 100, 200, 400, 1000] if n >= 128 else [32, 100]
min_quarter = math.floor(4 * math.log10(ref_particle_mass * 32))
run['hmf_min_mass_msun_h'] = 10**(min_quarter / 4)
run['hmf_max_mass_msun_h'] = 1e15
run['hmf_mass_bins'] = 60 - min_quarter

# Model-specific particle mass and resolution thresholds
model_omega_m = float(linear.get('Omega_b', 0.049) + linear.get('Omega_cdm', 0.251))
particle_mass = 2.77536627245708e11 * model_omega_m * model_box**3 / n**3
run['particle_mass_msun_h'] = particle_mass
run['mass_threshold_100_particles_msun_h'] = 100.0 * particle_mass
run['mass_threshold_400_particles_msun_h'] = 400.0 * particle_mass
run['mass_threshold_1000_particles_msun_h'] = 1000.0 * particle_mass

if not 3<=run['hmf_mass_bins']<=200:raise SystemExit('Box/particle threshold leaves fewer than three supported quarter-dex HMF bins')
# Read Slurm's requested duration, expressed as D-HH:MM:SS or HH:MM:SS when supplied.
# The fixed upper native limit is intentionally long; Slurm controls job wall time.
# Periodic native checkpoints remain available when Slurm ends the allocation.
run['time_limit_cpu_seconds']=float(os.environ.get('NATIVE_WALL_SECONDS', '1200' if is_benchmark else ('160000' if n>=128 else '12000')))
run_dir=root/'runs'/model
if (run_dir / 'config.yaml').is_file():
 if os.environ.get('RESUME')!='1':raise SystemExit('Run exists; use a new CAMPAIGN or explicitly set RESUME=1')
 import yaml
 saved=yaml.safe_load((run_dir/'config.yaml').read_text())
 for key in ('box_mpc_h','particles_per_dimension','pm_grid','softening_kpc_h','softening_max_physical_kpc_h'):
  if saved['simulation'][key]!=run[key]:raise SystemExit('Resume changes frozen '+key)
 if saved['simulation']['random_seed']!=seed:raise SystemExit('Resume changes random seed')
 for key in ('mpi_ranks','omp_threads'):
  if saved['hardware'][key]!=run[key]:raise SystemExit('Resume must retain frozen '+key)
 if saved['cosmology']['model']!=model:raise SystemExit('Resume changes model')
resources=estimate_run_resources({'simulation':dict(run,physics='dm'),'hardware':{'memory_limit_gb':mem}},len(run['schedule'].split(',')))
n_snaps = len(run['schedule'].split(','))
raw_snap_gib = (n**3 * 32) / (1024**3)
total_storage_gib = raw_snap_gib * n_snaps
resources['storage_estimate_gib'] = round(total_storage_gib, 1)
resources['raw_snapshot_gib'] = round(raw_snap_gib, 2)
resources['num_snapshots'] = n_snaps
if resources['memory_gib']>mem:raise SystemExit(f"Planning RAM {resources['memory_gib']:.1f} GiB exceeds budget {mem:.1f}; request more memory")
print(json.dumps({'model':model,'n':n,'ranks':ranks,'threads':threads,'visible_cpu_slots':cpu_count(),'memory_budget_gib':mem,'resource_estimate':resources,'run_dir':str(run_dir)},indent=2),flush=True)
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:json.dump(value,f,indent=2);f.write('\n');tmp=f.name
 os.replace(tmp,path)
write(root/'inputs'/model/'linear.json',linear);write(root/'inputs'/model/'run.json',run)
write(root/'inputs'/model/'allocation.json',{'ranks':ranks,'threads':threads,'visible_slots':cpu_count(),'memory_budget_gib':mem,'estimate':resources,'job_id':os.environ.get('SLURM_JOB_ID')})
