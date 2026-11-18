"""Coupled MUSIC Poisson equations, independent references and failure intake.

The coupled operator keeps leaf source equations and eight-child restriction
rows distinct. Whole-box Fourier and rational failed-field calculations keep
independent arithmetic: sharing production stencils would hide disagreement.
All coordinates here use the declared normalized box, not implicit kpc units.
Linear residuals and model sensitivity are not physical accuracy certificates.
"""
import math
import struct
from fractions import Fraction
from pathlib import Path
import numpy as np
from core.foundation import file_sha256
from core.music.replay import _budget, finite_gradient
from core.music.boundary import NativeBoundaryOperator

_CORE_ROOT = Path(__file__).resolve().parent.parent


class NativePoissonCoupling:
    """Interior-potential operator with optional explicit zero-mean gauge.

    Equations are scaled by h**2 on leaf cells, and by one on restriction
    cells. With gauge=True, the first base leaf row is replaced by mean(Phi)
    on the base grid. The removed physical equation must still be checked
    against the original source after a solve; it is never fitted away.
    Four parent interior cells of padding are required around each child.
    """
    def __init__(self,hierarchy,build_receipt,*,order=6,memory_gib=16.,restart=32,max_iterations=128):
        self.boundaries=[]
        from core.music.resources import coupled_workspace_plan
        plan=coupled_workspace_plan(hierarchy,build_receipt,order=order,restart=restart,max_iterations=max_iterations)
        # Validate all geometry and complete declared solver allocation before
        # allocating even the covered-cell masks at production dimensions.
        _budget(memory_gib,plan['minimum_operator_workspace_bytes'])
        self.allocation_plan=plan
        self.order=order
        self.memory_gib=memory_gib
        self.levels=plan['levels']
        self.shapes=[tuple(s) for s in plan['shapes']]
        self.origins=[tuple(o) for o in plan['origins']]
        self.offsets=[tuple(o) for o in plan['parent_offsets']]
        self.covered=[np.zeros(s,dtype=bool) for s in self.shapes]
        for i,offset in enumerate(self.offsets,1):
            self.covered[i-1][tuple(slice(o,o+n//2) for o,n in zip(offset,self.shapes[i]))]=True
        self.sizes=[math.prod(s) for s in self.shapes]
        self.size=sum(self.sizes)
        self.starts=np.cumsum([0]+self.sizes).tolist()
        try:
            for i,offset in enumerate(self.offsets,1):
                allocation=plan['interfaces'][i-1]['allocated_workspace_bytes']/1024**3
                self.boundaries.append(NativeBoundaryOperator(build_receipt,order=order,
                    coarse_shape=list(self.shapes[i-1]),fine_shape=list(self.shapes[i]),
                    parent_offset=list(offset),memory_gib=allocation))
        except Exception:
            self.close()
            raise
        self.coefficients={2:(-2.,[1.]),4:(-2.5,[4/3,-1/12]),6:(-49/18,[1.5,-.15,1/90])}[order]
        self.source_binding={'schema':'native-poisson-coupling-operator-v1','order':order,
            'levels':self.levels,'shapes':[list(s) for s in self.shapes],
            'origins':[list(o) for o in self.origins],'interfaces':[b.source_binding for b in self.boundaries],
            'row_scaling':'h**2 leaf Poisson equations; unit covered restriction equations',
            'gauge':'Optional first base leaf row replaced by base potential mean',
            'memory_budget_gib':memory_gib,'allocation_plan':plan,'physical_boundary_accuracy':'NOT_ESTABLISHED'}

    def close(self):
        for boundary in self.boundaries:
            boundary.close()
    def __enter__(self):
        return self
    def __exit__(self,*args):
        self.close()
    def __del__(self):
        if hasattr(self,'boundaries'):
            self.close()

    def split(self,values):
        v=np.asarray(values)
        if v.shape!=(self.size,) or v.dtype.kind not in 'fiu' or not np.isfinite(v).all():
            raise ValueError('Finite aligned native interior potential/equation vector required')
        return [v[a:b].reshape(s) for a,b,s in zip(self.starts[:-1],self.starts[1:],self.shapes)]

    def pack(self,values):
        if len(values)!=len(self.shapes) or any(np.asarray(v).shape!=s for v,s in zip(values,self.shapes)):
            raise ValueError('Aligned native interior hierarchy required')
        result=np.concatenate([np.asarray(v).ravel() for v in values])
        self.split(result)
        return result

    def fill_potentials(self,values):
        interiors=self.split(values)
        padded=[np.pad(interiors[0],4,mode='wrap')]
        for i,boundary in enumerate(self.boundaries,1):
            fine=np.pad(interiors[i],4)
            fine.ravel()[boundary.output_indices]=boundary.apply(padded[i-1],fine)
            padded.append(fine)
        return padded

    @staticmethod
    def _interior(shape):
        return tuple(slice(4,4+n) for n in shape)

    def potential_adjoint(self,weights):
        """Transpose filled interior/periodic/stencil-face potential samples.

        Uncomputed edge/corner or unused halo samples reject nonzero weights.
        These values cannot be silently treated as independent boundary data.
        """
        if len(weights)!=len(self.shapes):
            raise ValueError('Aligned potential-weight hierarchy required')
        padded=[]
        for i,(w,shape) in enumerate(zip(weights,self.shapes)):
            w=np.asarray(w)
            if w.shape!=tuple(n+8 for n in shape) or w.dtype.kind not in 'fiu' or not np.isfinite(w).all():
                raise ValueError('Finite actual padded potential weights required')
            if i:
                support=np.zeros(w.shape,dtype=bool)
                support[self._interior(shape)]=True
                support.ravel()[self.boundaries[i-1].output_indices]=True
                if np.any(w[~support]!=0):
                    raise ValueError('Potential weights touch uncomputed native ghost samples')
            padded.append(w.astype(float,copy=True))
        result=[np.zeros(s) for s in self.shapes]
        for i in range(len(self.shapes)-1,0,-1):
            boundary=self.boundaries[i-1]
            weight=padded[i]
            coarse,fine=boundary.transpose(weight.ravel()[boundary.output_indices])
            # The adapter's inputs include old fine ghosts. This equation
            # defines them as zero before the canonical boundary traversal.
            result[i]+=weight[self._interior(self.shapes[i])]+fine[self._interior(self.shapes[i])]
            padded[i-1]+=coarse
        root=padded[0]
        shape=self.shapes[0]
        coords=np.ix_(*[(np.arange(n+8)-4)%n for n in shape])
        np.add.at(result[0],coords,root)
        return self.pack(result)

    def apply(self,values,*,gauge=True):
        if type(gauge) is not bool:
            raise ValueError('Explicit Boolean native gauge required')
        interiors=self.split(values)
        padded=self.fill_potentials(values)
        centre,coeff=self.coefficients
        rows=[]
        for i,(phi,shape) in enumerate(zip(padded,self.shapes)):
            out=-3*centre*interiors[i].astype(float)
            for axis in range(3):
                for distance,coefficient in enumerate(coeff,1):
                    for sign in (-1,1):
                        delta=[0,0,0]
                        delta[axis]=sign*distance
                        out-=coefficient*phi[tuple(slice(4+d,4+d+n) for d,n in zip(delta,shape))]
            if i<len(self.boundaries):
                child=interiors[i+1]
                s=child.shape
                average=child.reshape(s[0]//2,2,s[1]//2,2,s[2]//2,2).mean(axis=(1,3,5))
                selection=tuple(slice(o,o+n) for o,n in zip(self.offsets[i],average.shape))
                out[selection]=interiors[i][selection]-average
            rows.append(out)
        if gauge:
            rows[0].flat[0]=float(np.mean(interiors[0],dtype=np.longdouble))
        return self.pack(rows)

    def transpose(self,weights,*,gauge=True):
        if type(gauge) is not bool:
            raise ValueError('Explicit Boolean native gauge required')
        rows=[w.astype(float,copy=True) for w in self.split(weights)]
        gauge_weight=rows[0].flat[0] if gauge else 0.
        if gauge:
            rows[0].flat[0]=0.
        centre,coeff=self.coefficients
        padded=[np.zeros(tuple(n+8 for n in s)) for s in self.shapes]
        direct=[np.zeros(s) for s in self.shapes]
        for i,(weight,shape) in enumerate(zip(rows,self.shapes)):
            leaf=weight.copy()
            leaf[self.covered[i]]=0.
            padded[i][self._interior(shape)]-=3*centre*leaf
            for axis in range(3):
                for distance,coefficient in enumerate(coeff,1):
                    for sign in (-1,1):
                        delta=[0,0,0]
                        delta[axis]=sign*distance
                        padded[i][tuple(slice(4+d,4+d+n) for d,n in zip(delta,shape))]-=coefficient*leaf
            if i<len(self.boundaries):
                child_shape=self.shapes[i+1]
                selection=tuple(slice(o,o+n//2) for o,n in zip(self.offsets[i],child_shape))
                restriction=weight[selection]
                direct[i][selection]+=restriction
                direct[i+1]-=np.repeat(np.repeat(np.repeat(restriction,2,axis=0),2,axis=1),2,axis=2)/8
        direct[0]+=gauge_weight/self.sizes[0]
        return self.potential_adjoint(padded)+self.pack(direct)

    def source_rhs(self,sources,*,gauge=True):
        """Scale leaf sources by grid_spacing**2; restriction/gauge RHS is zero.

        grid_spacing = 2**(-level) in normalized box coordinates. This h is
        a cell width, not the cosmological reduced Hubble parameter.
        """
        if type(gauge) is not bool:
            raise ValueError('Explicit Boolean native gauge required')
        rows=self.split(self.pack(sources))
        result=[]
        for row,covered,level in zip(rows,self.covered,self.levels):
            row=row.astype(float)/float(2**level)**2
            row[covered]=0.
            result.append(row)
        if gauge:
            result[0].flat[0]=0.
        return self.pack(result)

    def as_linear_operator(self,*,gauge=True):
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator((self.size,self.size),matvec=lambda v:self.apply(v,gauge=gauge),
            rmatvec=lambda w:self.transpose(w,gauge=gauge),dtype=np.float64)

    def solve(self,rhs,*,transpose=False,tolerance=1e-10,restart=32,max_iterations=128):
        """Bounded gauge-fixed GMRES; report true residual, including failures.

        max_iterations counts inner Krylov iterations, not restart cycles.
        A converged linear solve alone supplies no physical boundary bound.
        All original source equations must be checked after a forward solve.
        """
        from scipy.sparse.linalg import gmres,LinearOperator
        self.split(rhs)
        if (type(transpose) is not bool or isinstance(tolerance,(bool,np.bool_))
                or not math.isfinite(tolerance) or not 0<tolerance<=1e-10
                or type(restart) is not int or not 2<=restart<=256
                or type(max_iterations) is not int or not 1<=max_iterations<=100000):
            raise ValueError('Bounded Krylov settings and unchanged or tighter numerical tolerance required')
        workspace=64*1024**2+320*sum(math.prod(tuple(n+8 for n in s)) for s in self.shapes)
        workspace+=8*self.size*(min(restart,max_iterations)+16)
        workspace+=sum(96*b.nodes+40*b.input_count+24*b.output_count for b in self.boundaries)
        _budget(self.memory_gib,workspace)
        rhs=np.asarray(rhs,dtype=float)
        norm=float(np.linalg.norm(rhs))
        apply=self.transpose if transpose else self.apply
        operator=LinearOperator((self.size,self.size),matvec=apply,dtype=np.float64)
        # Diagonal row scaling is a preconditioner only. The acceptance
        # residual below always uses the complete original coupled operator.
        diagonal=[]
        centre,_=self.coefficients
        for covered in self.covered:
            row=np.full(covered.shape,-3*centre)
            row[covered]=1.
            diagonal.append(row)
        diagonal=self.pack(diagonal)
        diagonal[0]=1/self.sizes[0]
        preconditioner=LinearOperator(operator.shape,matvec=lambda v:v/diagonal,dtype=np.float64)
        history=[]
        solution,info=gmres(operator,rhs,M=preconditioner,rtol=tolerance,atol=0.,restart=min(restart,max_iterations),
            maxiter=max_iterations,callback=lambda residual:history.append(float(residual)),callback_type='legacy')
        residual=apply(solution)-rhs
        relative=float(np.linalg.norm(residual))/norm if norm else float(np.linalg.norm(residual))
        record={'schema':'native-poisson-coupled-linear-solve-v1',
            'status':'PASS' if info==0 and math.isfinite(relative) and relative<=tolerance else 'UNRESOLVED_SOLVE',
            'transpose':transpose,'relative_true_residual':relative,'tolerance':tolerance,
            'inner_iterations':len(history),'max_inner_iterations':max_iterations,'restart':min(restart,max_iterations),
            'gmres_info':int(info),'preconditioned_residual_history':history,'gauge':'First base leaf equation replaced by base potential mean',
            'physical_boundary_accuracy':'NOT_ESTABLISHED','science_acceptance':'NOT_ACCEPTED','launch_authorized':False}
        return solution,record


def periodic_poisson_reference(source,*,level,order=6,tolerance=1e-10,memory_gib=1.):
    """Invert the declared negative finite Laplacian on a periodic cube.

    Coordinates and potential use normalized box units. The DC compatibility
    condition is checked against the unchanged declared relative tolerance
    before inversion; the recorded roundoff DC is the only source projection.
    The residual is evaluated in real space by an independent stencil.
    """
    from core.music.archive import poisson_source_from_potential
    values=np.asarray(source)
    if (type(level) is not int or not 2<=level<=20 or type(order) is not int or order not in (2,4,6)
            or values.shape!=(2**level,)*3 or values.dtype.kind not in 'fiu'
            or isinstance(tolerance,(bool,np.bool_)) or not math.isfinite(tolerance) or not 0<tolerance<1):
        raise ValueError('Full periodic cube and explicit finite Poisson operator/tolerance required')
    _budget(memory_gib,64*1024**2+256*values.size)
    if not np.isfinite(values).all():
        raise ValueError('Finite periodic source required')
    rms=float(np.sqrt(np.mean(values.astype(np.longdouble)**2)))
    mean=float(np.mean(values,dtype=np.longdouble))
    dc=abs(mean)/rms if rms else 0.
    if dc>tolerance:
        raise ValueError('Periodic source fails DC compatibility; do not fit away its mean')
    side=2**level
    axes=[2*np.pi*np.fft.fftfreq(side)[:,None,None],2*np.pi*np.fft.fftfreq(side)[None,:,None],
        2*np.pi*np.fft.rfftfreq(side)[None,None,:]]
    # For spacing 1/N and cell-phase theta, the negative-Laplacian symbol is
    # lambda = 4*N**2*sum_s c_s*sin(s*theta/2)**2. Inverting each nonzero
    # Fourier mode therefore uses the declared finite stencil, not k**2.
    eigen=np.zeros((side,side,side//2+1))
    weights={2:[1.],4:[4/3,-1/12],6:[1.5,-.15,1/90]}[order]
    for wave in axes:
        for step,weight in enumerate(weights,1):
            eigen+=4*weight*np.sin(step*wave/2)**2*side**2
    spectrum=np.fft.rfftn(values)
    spectrum=np.divide(spectrum,eigen,out=np.zeros_like(spectrum),where=eigen>0)
    phi=np.fft.irfftn(spectrum,s=values.shape,axes=(0,1,2))
    ghosts=np.pad(phi,4,mode='wrap')
    recovered=poisson_source_from_potential(ghosts,level=level,order=order,halo=4,memory_gib=memory_gib)
    residual=recovered-(values-mean)
    error=float(np.sqrt(np.mean(residual.astype(np.longdouble)**2)))/rms if rms else 0.
    return phi,{'schema':'periodic-finite-poisson-reference-v1','level':level,'laplace_order':order,
        'source_rms':rms,'source_mean':mean,'relative_dc':dc,'relative_stencil_residual':error,
        'tolerance':tolerance,'status':'PASS' if error<=tolerance else 'FAIL',
        'gauge':'Zero Fourier DC potential','source_projection':'Recorded compatible roundoff DC only',
        'science_acceptance':'NOT_ACCEPTED'}


def hierarchy_boundary_reference(hierarchy,potentials,*,degree=5,order=6,gradient_order=6,
        tolerance=1e-10,memory_gib=1.):
    """Compare finest captured potential gradients to a whole-box reference.

    Source reconstruction uses explicit conservative cell averages. Degree0
    and degree5 are different source models; their difference must remain an
    uncertainty, not an empirical acceptance allowance. No native potential
    or native boundary value enters the independent Poisson solve.
    """
    from core.zoom import compose_hierarchy_density_reference
    if not hierarchy or len(hierarchy)!=len(potentials):
        raise ValueError('Aligned source and potential hierarchy required')
    high=hierarchy[-1]['level']
    fine=hierarchy[-1]
    potential=potentials[-1]
    if type(high) is not int or not 2<=high<=20:
        raise ValueError('Supported reference level required')
    _budget(memory_gib,64*1024**2+384*(2**high)**3+128*sum(row['values'].size for row in potentials))
    for s,p in zip(hierarchy,potentials):
        if s['level']!=p['level'] or s['origin']!=p['origin'] or p['values'].shape!=tuple(n+2*p['halo'] for n in s['values'].shape):
            raise ValueError('Captured source/potential geometry differs')
    source,composition=compose_hierarchy_density_reference(hierarchy,degree=degree,memory_gib=memory_gib)
    phi,solver=periodic_poisson_reference(source,level=high,order=order,tolerance=tolerance,memory_gib=memory_gib)
    # Differentiating the independent solution with the actual native stencil
    # isolates potential/boundary differences from the hybrid correction.
    gradient=finite_gradient(np.pad(phi,4,mode='wrap'),level=high,order=gradient_order,memory_gib=memory_gib)
    selections=np.ix_(*[(int(o)+np.arange(n))%2**high for o,n in zip(fine['origin'],fine['values'].shape)])
    reference=gradient[selections]
    actual=finite_gradient(potential['values'],level=high,order=gradient_order,halo=potential['halo'],memory_gib=memory_gib)
    shell=np.zeros(fine['values'].shape,dtype=bool)
    width=gradient_order//2
    for axis in range(3):
        for edge in (slice(0,width),slice(-width,None)):
            selection=[slice(None)]*3
            selection[axis]=edge
            shell[tuple(selection)]=True
    difference=actual-reference
    comparisons=[]
    for label,mask in (('boundary_stencil_shell',shell),('interior',~shell),('whole_patch',np.ones(shell.shape,dtype=bool))):
        if not mask.any():
            continue
        scale=float(np.linalg.norm(reference[mask]))
        error=float(np.linalg.norm(difference[mask]))
        comparisons.append({'support':label,'cells':int(mask.sum()),'reference_vector_rms':scale/math.sqrt(int(mask.sum())),
            'difference_vector_rms':error/math.sqrt(int(mask.sum())),
            'relative_l2_difference':error/scale if scale else 0. if error==0 else None})
    return {'reference_gradient':reference,'captured_gradient':actual,'gradient_difference':difference,'boundary_shell':shell}, {
        'schema':'hierarchy-periodic-boundary-reference-v1','status':'REFERENCE_MEASURED' if solver['status']=='PASS' else 'FAIL',
        'source_reconstruction':composition,'solver':solver,'gradient_order':gradient_order,'shell_cells_thick':width,
        'comparisons':comparisons,'scope':'Independent global periodic solve of the explicitly reconstructed composite source; no captured potential/ghosts used in reference solve',
        'source_sampling_acceptance':'NOT_ESTABLISHED','source_reconstruction_error_budget':'NOT_ESTABLISHED',
        'boundary_accuracy_acceptance':'NOT_ESTABLISHED','launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}


def audit_music_archive_boundary(index,configuration,ic,destination,*,memory_gib=1.):
    """Publish measured boundary discrepancies for two explicit source models.

    A successful computation is REFERENCE_MEASURED, never launch acceptance.
    In particular the degree0/degree5 spread is not a bound on unrepresented
    physical source modes or a license to expand a numerical tolerance.
    """
    import json
    import h5py
    from core.data import export_numeric_audit,epoch_contract
    from core.zoom import _configuration
    from core.music.archive import load_music_source_hierarchy
    index,configuration,ic=map(lambda p:Path(p).resolve(),(index,configuration,ic))
    pins={str(p):file_sha256(p) for p in (index,configuration,ic)}
    if index.with_name(index.name+'.sha256').read_text().strip()!=pins[str(index)]:
        raise ValueError('Native boundary source index seal mismatch')
    record=json.loads(index.read_text())
    cfg=_configuration(configuration)
    low,high=cfg.getint('setup','levelmin'),cfg.getint('setup','levelmax')
    order=cfg.getint('poisson','laplace_order',fallback=6)
    grad=cfg.getint('poisson','grad_order',fallback=6)
    tolerance=cfg.getfloat('poisson','accuracy',fallback=1e-5)
    if not 2<=low<high<=20 or cfg.getboolean('poisson','kspace',fallback=False):
        raise ValueError('Nested finite-difference boundary reference required')
    # Reject an impractical whole-box reference before loading any archive
    # arrays. This validation route is deliberately distinct from production.
    count=sum(math.prod(row['stored_shape']) for field in record['fields'] for row in field['grids'])
    _budget(memory_gib,64*1024**2+384*(2**high)**3+128*count)
    with h5py.File(ic) as f:
        h=f['Header'].attrs
        a=float(h['Time'])
        if (h.get('Music_levelmin')!=low or h.get('Music_levelmax')!=high or h.get('Music_lpt_order')!=2
                or h.get('Music_mixed_fluid_schema')!=1 or h.get('Music_composite_source_normalization_schema')!=1
                or not math.isclose(a,1/(1+cfg.getfloat('setup','zstart')),rel_tol=1e-12)
                or not math.isclose(float(h['BoxSize']),1000*cfg.getfloat('setup','boxlength'),rel_tol=1e-12)):
            raise ValueError('Boundary reference configuration differs from actual normalized native IC')
    arrays={}
    specs={}
    comparisons=[]
    spread=[]
    for species in (0,1):
        for quantity,potential_code,label in ((1,2,'density'),(3,4,'quadratic'),(6,7,'velocity')):
            source,_=load_music_source_hierarchy(index,quantity,species,memory_gib=memory_gib)
            potential,_=load_music_source_hierarchy(index,potential_code,species,memory_gib=memory_gib,include_ghosts=True)
            if any([row['level'] for row in rows]!=list(range(low,high+1)) for rows in (source,potential)):
                raise ValueError('Captured boundary hierarchy differs from configured levels')
            previous=None
            for degree in (0,5):
                values,report=hierarchy_boundary_reference(source,potential,degree=degree,order=order,gradient_order=grad,
                    tolerance=tolerance,memory_gib=memory_gib)
                comparisons.append({'particle_type':species,'quantity':label,'source_reconstruction_degree':degree,**report})
                prefix=f'species{species}_{label}_degree{degree}_'
                for name,value in values.items():
                    arrays[prefix+name]=value
                    specs[prefix+name]={'units':'dimensionless boolean support' if name=='boundary_shell' else
                        'km/s per comoving Mpc/h' if quantity==6 else 'dimensionless normalized-box potential gradient',
                        'quantity':prefix+name}
                reference=values['reference_gradient']
                if previous is not None:
                    diff=float(np.linalg.norm(reference-previous))
                    norm=float(np.linalg.norm(reference))
                    spread.append({'particle_type':species,'quantity':label,'model_difference_vector_rms':diff/math.sqrt(reference.size//3),
                        'relative_l2_model_difference':diff/norm if norm else 0. if diff==0 else None,
                        'status':'MODEL_SENSITIVITY_ONLY; NOT_AN_ERROR_BOUND'})
                previous=reference
    from core.music.archive import archive_precision_sources
    pins.update(archive_precision_sources(record))
    pins[record['archive']['path']]=record['archive']['sha256']
    for path in (Path(__file__).resolve(),(_CORE_ROOT / 'music/replay.py'),
            (_CORE_ROOT / 'music/archive.py'),(_CORE_ROOT / 'zoom.py')):
        pins[str(path)]=file_sha256(path)
    report={'schema':'native-music-periodic-boundary-comparison-v1',
        'status':'FAIL' if any(row['status']=='FAIL' for row in comparisons) else 'REFERENCE_MEASURED',
        'comparisons':comparisons,'source_model_sensitivity':spread,
        'scope':'Bounded whole-box independent periodic finite-Laplacian references for two conservative source-cell models; finest original potential-gradient boundary and interior comparisons',
        'reference_solve_tolerance':tolerance,'physical_source_sampling_acceptance':'NOT_ESTABLISHED',
        'source_reconstruction_error_budget':'NOT_ESTABLISHED','boundary_accuracy_acceptance':'NOT_ESTABLISHED',
        'production_scalability':'Whole finest box reference is bounded validation only; not a production generation strategy',
        'launch_authorized':False,'science_acceptance':'NOT_ACCEPTED'}
    return export_numeric_audit(destination,report,arrays,specs,algorithm=__file__,context={'epoch':epoch_contract(a)},source_hashes=pins)


def read_failure_snapshot(path):
    path=Path(path).resolve()
    size=path.stat().st_size
    if not 36<=size<=64*1024**2:
        raise ValueError('Bounded native Poisson failure snapshot required')
    with path.open('rb') as stream:
        def read(fmt):
            count=struct.calcsize(fmt)
            data=stream.read(count)
            if len(data)!=count:
                raise ValueError('Truncated Poisson failure metadata')
            return struct.unpack(fmt,data)
        if stream.read(8)!=b'MUSICPS1':
            raise ValueError('Native Poisson failure magic differs')
        version,first,last=read('<III')
        error,tolerance=read('<dd')
        if version!=1 or not 1<=first<=last<=20 or not math.isfinite(tolerance) or tolerance<=0:
            raise ValueError('Unsupported native Poisson failure header')
        levels=[]
        for expected in range(first,last+1):
            level,halo,nx,ny,nz,ox,oy,oz=read('<8I')
            if level!=expected or not 1<=halo<=16 or min(nx,ny,nz)<1:
                raise ValueError('Invalid native failure geometry')
            shape=(nx,ny,nz)
            extended=tuple(n+2*halo for n in shape)
            count=math.prod(extended)
            offset=stream.tell()
            end=offset+count*24
            if end>size:
                raise ValueError('Truncated or excessive native failure fields')
            fields={name:np.memmap(path,dtype='<f8',mode='r',offset=offset+index*count*8,shape=extended)
                for index,name in enumerate(('original_source','returned_source','potential'))}
            levels.append({'level':level,'halo':halo,'shape':shape,'offset':(ox,oy,oz),**fields})
            stream.seek(end)
        if stream.tell()!=size:
            raise ValueError('Native failure snapshot has trailing data')
    return {'path':path,'reported_error':error,'tolerance':tolerance,'levels':levels}


def inspect_failure_snapshot(path,*,order,memory_gib=.25):
    """Measure R = finite_Laplacian(Phi) + source on the original leaf cells.

    Fraction arithmetic represents the saved binary64 samples exactly. It
    distinguishes changed source values from residual-evaluation roundoff;
    it cannot prove that a different representable solution is impossible.
    """
    if type(order) is not int or order not in (2,4,6):
        raise ValueError('Explicit native Laplacian order2/4/6 required')
    if type(memory_gib) not in (int,float) or not math.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Explicit positive inspection memory allocation required')
    snapshot=read_failure_snapshot(path)
    binding=file_sha256(snapshot['path'])
    rows=[]
    weights,denominator={2:((1,),1),4:((16,-1),12),6:((270,-27,2),180)}[order]
    for index,grid in enumerate(snapshot['levels']):
        shape=grid['shape']
        halo=grid['halo']
        cells=math.prod(shape)
        # Seven Python float lists plus their pointer storage, the coverage
        # mask, mapped input residency and fixed interpreter workspace.
        if halo<order//2 or cells*320+snapshot['path'].stat().st_size+16*1024**2>memory_gib*1024**3:
            raise MemoryError('Native failure inspection exceeds halo or allocated workspace')
        interior=tuple(slice(halo,halo+n) for n in shape)
        before=grid['original_source'][interior]
        after=grid['returned_source'][interior]
        u=grid['potential']
        if not np.isfinite(before).all() or not np.isfinite(after).all() or not np.isfinite(u).all():
            raise ValueError('Failure fields contain nonfinite values; rational residual is undefined')
        leaves=np.ones(shape,dtype=bool)
        if index+1<len(snapshot['levels']):
            child=snapshot['levels'][index+1]
            off=child['offset']
            child_shape=child['shape']
            if any(n%2 for n in child_shape) or any(o+n//2>s for o,n,s in zip(off,child_shape,shape)):
                raise ValueError('Invalid native failure refinement coverage')
            leaves[tuple(slice(o,o+n//2) for o,n in zip(off,child_shape))]=False
        count=int(np.count_nonzero(leaves))
        original=[]
        returned=[]
        floating=[]
        arithmetic=[]
        source0=[]
        source1=[]
        drift=[]
        bands={name:{'residual':[],'source':[],'rounded_point_update_unchanged':0}
            for name in ('boundary_stencil','near_boundary','interior')}
        h2=math.ldexp(1.,-2*grid['level'])
        for i,j,k in np.ndindex(shape):
            if not leaves[i,j,k]:
                continue
            cell=(i+halo,j+halo,k+halo)
            center=float(u[cell])
            exact=Fraction(0)
            rounded=0.
            for distance,weight in enumerate(weights,1):
                exact_difference=Fraction(0)
                difference=0.
                for axis in range(3):
                    for sign in (-1,1):
                        neighbor=list(cell)
                        neighbor[axis]+=sign*distance
                        value=float(u[tuple(neighbor)])
                        exact_difference+=Fraction(value)-Fraction(center)
                        difference+=value-center
                exact+=weight*exact_difference
                rounded+=weight*difference
            # Rational evaluation uses the exact stored binary64 samples and
            # exact integer coefficients, with one conversion after addition
            # of the physical source. It does not fit or change any field.
            laplacian=exact/denominator*2**(2*grid['level'])
            d0=float(before[i,j,k])
            d1=float(after[i,j,k])
            r0=float(laplacian+Fraction(d0))
            r1=float(laplacian+Fraction(d1))
            rf=rounded/denominator/h2+d1
            original.append(r0*r0)
            returned.append(r1*r1)
            floating.append(rf*rf)
            arithmetic.append((rf-r1)**2)
            source0.append(d0*d0)
            source1.append(d1*d1)
            drift.append((d1-d0)**2)
            distance=min(i,j,k,shape[0]-1-i,shape[1]-1-j,shape[2]-1-k)
            band=bands['boundary_stencil' if distance<order//2 else 'near_boundary' if distance<order else 'interior']
            band['residual'].append(r0*r0)
            band['source'].append(d0*d0)
            diagonal=6*sum(weights)/denominator
            # This describes only a local point correction with fixed ghosts;
            # it is not a lower bound for a coupled hierarchy update.
            if center+r0*h2/diagonal==center:
                band['rounded_point_update_unchanged']+=1
        def rms(values):
            return math.sqrt(math.fsum(values)/count) if count else 0.
        s0=rms(source0)
        s1=rms(source1)
        def relative(value,scale):
            return value/scale if scale else (0. if value==0 else None)
        band_report={}
        for name,band in bands.items():
            n=len(band['residual'])
            source_sum=math.fsum(band['source'])
            residual_sum=math.fsum(band['residual'])
            band_report[name]={'cells':n,'exact_residual_original_source_relative_rms':
                math.sqrt(residual_sum/source_sum) if source_sum else (0. if residual_sum==0 else None),
                'rounded_point_update_unchanged_cells':band['rounded_point_update_unchanged']}
        rows.append({'level':grid['level'],'physical_leaf_cells':count,'distance_bands':band_report,
            'original_source_rms':s0,'returned_source_rms':s1,
            'source_change_relative_rms':relative(rms(drift),s0),
            'exact_residual_original_source_relative_rms':relative(rms(original),s0),
            'exact_residual_returned_source_relative_rms':relative(rms(returned),s1),
            'binary64_recomputed_residual_relative_rms':relative(rms(floating),s1),
            'residual_arithmetic_difference_relative_rms':relative(rms(arithmetic),s1)})
    if file_sha256(snapshot['path'])!=binding:
        raise ValueError('Native failure snapshot changed during inspection')
    return {'schema':'native-poisson-failure-rational-inspection-v1','status':'INSPECTED_FAILURE',
        'snapshot':{'path':str(snapshot['path']),'sha256':binding},'order':order,'levels':rows,
        'reported_error':snapshot['reported_error'] if math.isfinite(snapshot['reported_error']) else None,
        'reported_error_finite':math.isfinite(snapshot['reported_error']),'required_tolerance':snapshot['tolerance'],
        'source_hashes':{str(Path(__file__).resolve()):file_sha256(__file__)},
        'scope':'Exact rational residual of the stored failed field, with original and returned physical-leaf sources. This does not establish a lower bound over other representable solutions.',
        'science_acceptance':'NOT_ACCEPTED','launch_authorized':False}
