"""Measured halo structure, splashback and cosmic-web products and figures.

Profiles retain proper-kpc shell radii, Msun masses and peculiar km/s velocities.
Density and potential shape measurements keep their distinct grids and support.
Splashback uses the original logarithmic slope and covariance, without refitting.
T-web voxel regions use comoving kpc; published skeleton arcs retain Mpc/h.
Neither voxel labels nor reconstructed display paths establish filament ancestry.
"""
from pathlib import Path
import numpy as np
from .foundation import file_sha256, atomic_json
from .data import export_numeric_audit, numeric_audit_source_bindings
from .stream_common import read_product
from .research.display import export_scientific_suite


def export_native_structure_figures(product,destination,*,memory_gib=1.):
    from .halos import verify_halo_structure
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Finite positive structure workspace required')
    path=Path(product).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native structure exports')
    if path.stat().st_size*12>memory_gib*1024**3/3:
        raise MemoryError('Native structure metadata exceeds declared workspace')
    # Bound archive intake before the native verifier reloads its coefficients.
    report,raw,sources=read_product(path,memory_gib=memory_gib/3)
    verified=verify_halo_structure(path)
    if report!=verified:raise ValueError('Structure record changed during verification')
    sources.update(report['source_hashes'])
    specs=report['numeric_data']['array_contract']['datasets']
    epoch=report['source_context']['epoch'];figures=[];products={};out.mkdir(parents=True)
    for component in [*report['request']['components'],'total']:
        prefix='profile_'+component+'_'
        def get(key,unit):
            name=prefix+key
            if specs[name]['units']!=unit:raise ValueError('Native structure units mismatch: '+name)
            return np.asarray(raw[name])
        r=get('radius_proper_kpc','proper kpc');outer=get('radial_outer_edge_proper_kpc','proper kpc')
        rho=get('density_msun_kpc3','Msun/(proper kpc)^3')
        shot=get('density_shot_noise_msun_kpc3','Msun/(proper kpc)^3')
        vc=get('circular_velocity_spherical_km_s','km/s')
        beta=get('anisotropy_beta','dimensionless')
        cov=get('velocity_covariance_km2_s2','(km/s)^2')
        kinematic=get('kinematics_supported','dimensionless')
        count=get('shell_count','dimensionless');mass=get('enclosed_mass_msun','Msun')
        n=len(r)
        if (r.ndim!=1 or not n or any(a.shape!=(n,) for a in (outer,rho,shot,vc,beta,kinematic,count,mass))
                or cov.shape!=(n,3,3) or kinematic.dtype.kind!='b' or count.dtype.kind not in 'iu'
                or not np.isfinite(r).all() or np.any(r<=0) or np.any(np.diff(r)<=0)
                or not np.isfinite(outer).all() or np.any(outer<r) or np.any(np.diff(outer)<=0)):
            raise ValueError('Aligned proper shell arrays and exact native support required')
        variance=np.trace(cov,axis1=1,axis2=2)/3
        sigma=np.sqrt(np.where(variance>=0,variance,np.nan))
        arrays={'radius':r,'outer_radius':outer,'density':rho,'density_error':shot,'circular_velocity':vc,
                'anisotropy':beta,'dispersion':sigma,'shell_count':count,'enclosed_mass':mass,
                'density_supported':np.isfinite(rho)&np.isfinite(shot)&(shot>=0)&(count>0),
                'circular_supported':np.isfinite(vc)&(mass>0),
                'kinematics_supported':kinematic,
                'dispersion_supported':kinematic&np.isfinite(sigma),
                'anisotropy_supported':kinematic&np.isfinite(beta)}
        units={key:'proper kpc' if key in ('radius','outer_radius') else 'Msun/(proper kpc)^3' if key in ('density','density_error')
               else 'km/s' if key in ('circular_velocity','dispersion') else 'Msun' if key=='enclosed_mass' else 'dimensionless' for key in arrays}
        definitions={'circular_velocity':'spherical-equivalent sqrt(G M(<r_outer)/r_outer), including nonspherical components',
            'dispersion':'sqrt(trace of mass-weighted Cartesian peculiar population covariance / 3); no isotropy assumption',
            'anisotropy':'native local spherical beta = 1 - (sigma_theta^2 + sigma_phi^2)/(2 sigma_r^2)',
            'density_error':'native shell mass shot noise; excludes simulation convergence and correlated selection error'}
        target=out/(component+'-profiles.json');products[component]=str(target)
        export_numeric_audit(target,{'schema':'flagship-native-structure-figures-v1','epoch':epoch,
            'native_structure':str(path),'component':component,'request':report['request'],
            'measurement_definitions':definitions,'field_validation_status':report['field_validation_status'],
            'science_acceptance':'NOT_ACCEPTED','resolution_convergence':report['simulation_resolution_convergence']},
            arrays,{key:{'units':unit,'quantity':definitions.get(key,key),'missing_policy':'native unsupported bins retained with explicit masks'} for key,unit in units.items()},
            algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
        for variant,x,y,support,error in (
            ('density_profile','radius','density','density_supported','density_error'),
            ('circular_velocity','outer_radius','circular_velocity','circular_supported',None),
            ('velocity_dispersion','radius','dispersion','dispersion_supported',None),
            ('anisotropy','radius','anisotropy','anisotropy_supported',None)):
            series={'source':component,'kind':'errorbar' if error else 'line','label':component,
                    'x':x,'y':y,'support':support}
            if error:series['error']=error
            figures.append({'figure_id':component+'-'+variant,'family':'structure','variant':variant,
                'title':component+' '+variant.replace('_',' '),'science_status':'NOT_ACCEPTED','epoch':epoch,
                'panels':[{'panel_id':'native','title':'Native complete component shells',
                    'quality_status':'PENDING_SIMULATION_CONVERGENCE','selection':report['source_context']['selection'],
                    'xlabel':'Radius ['+units[x]+']','ylabel':variant+' ['+units[y]+']',
                    'x_units':units[x],'y_units':units[y],'series':[series]}]})
    shape_products,shape_figures,pending=_shape_exports(report,out,sources)
    products.update(shape_products);figures.extend(shape_figures)
    concentration_products,concentration_figures,concentration_pending=_concentration_export(report,out,sources)
    products.update(concentration_products);figures.extend(concentration_figures);pending.extend(concentration_pending)
    for source,expected in sources.items():
        if file_sha256(source)!=expected:raise ValueError('Native structure input changed during export')
    atomic_json(out/'products.json',{'schema':'flagship-native-structure-index-v1','products':products,'pending':pending})
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,
        'maximum_panels_per_figure':2,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)



def shape_display_arrays(density,potential,*,field_validated):
    """Adapt original shape rows and masks without refitting or aligning grids."""
    arrays={};units={}
    for name,record,rows in (('density',density,density['shells']),('potential',potential,potential['levels'])):
        if not rows:raise ValueError('Native shape product must retain its requested bins')
        measured=np.array([r['status']=='MEASURED' for r in rows],bool)
        radius=np.array([(r['lower_radius_kpc']+r['upper_radius_kpc'])/2 if name=='density'
                         else r['reference_radius_proper_kpc'] for r in rows],float)
        if not np.isfinite(radius).all() or np.any(radius<=0) or np.any(np.diff(radius)<=0):
            raise ValueError('Original shape radii must be finite positive and increasing')
        for key in ('b_over_a','c_over_a','triaxiality'):
            value=np.array([r.get(key,np.nan) if r.get(key) is not None else np.nan for r in rows],float)
            arrays[name+'_'+key]=value
            arrays[name+'_'+key+'_supported']=measured&np.isfinite(value)&(field_validated if name=='potential' else True)
        arrays[name+'_radius']=radius
        axes=np.array([r.get('axes_columns',np.full((3,3),np.nan)) for r in rows],float)
        if axes.shape!=(len(rows),3,3):raise ValueError('Native shape axes must be columns of 3 by 3 matrices')
        arrays[name+'_axes_columns']=axes
        orientation=np.array([r.get('orientation_sampling_resolved' if name=='density' else 'orientation_resolved',[False]*3) for r in rows])
        if orientation.dtype.kind!='b' or orientation.shape!=(len(rows),3):raise ValueError('Exact native per-axis orientation masks required')
        arrays[name+'_orientation_supported']=orientation&measured[:,None]&(field_validated if name=='potential' else True)
        twists=record.get('radial_twists',[])
        if len(twists)!=len(rows)-1:raise ValueError('Native shape twist records must align adjacent original shells')
        angles=np.array([r['unsigned_axis_twist_deg'] for r in twists],float).reshape(-1,3)
        arrays[name+'_twist_deg']=angles
        arrays[name+'_twist_radius']=radius[1:]
        arrays[name+'_twist_supported']=np.isfinite(angles)&arrays[name+'_orientation_supported'][:-1]&arrays[name+'_orientation_supported'][1:]
        if name=='density':
            intervals=np.array([r.get('sampling_uncertainty',{}).get('axis_ratio_percentile_16_50_84')
                if r.get('sampling_uncertainty',{}).get('axis_ratio_percentile_16_50_84') is not None
                else np.full((3,2),np.nan) for r in rows],float)
            if intervals.shape!=(len(rows),3,2):raise ValueError('Native bootstrap axis-ratio percentiles must be bins x 3 x 2')
            arrays['density_ratio_percentiles']=intervals
            arrays['density_uncertainty_supported']=measured&np.array([r.get('sampling_uncertainty',{}).get('status')=='ESTIMATED' for r in rows])&np.isfinite(intervals).all(axis=(1,2))
        for key in arrays:
            if key.startswith(name+'_'):
                units[key]='proper kpc' if key.endswith('radius') else 'deg' if key.endswith('twist_deg') else 'dimensionless'
    return arrays,units



def _shape_exports(report,out,sources):
    products={};figures=[];pending=[]
    if not report.get('density_shapes') or not report.get('equipotential_shapes'):
        return products,figures,[{'family':'structure','variant':v,'status':'MISSING_NATIVE_SHAPE_PRODUCT'}
            for v in ('radial_axis_ratios','triaxiality','axis_orientation_twist','density_potential_shape')]
    epoch=report['source_context']['epoch']
    for component,density in report['density_shapes'].items():
        potential=report['equipotential_shapes']
        arrays,units=shape_display_arrays(density,potential,field_validated=report['field_validation_status']=='PASS')
        name=component+'-shapes';target=out/(name+'.json');products[name]=str(target)
        def line(x,y,support,label,**extra):
            return {'source':name,'kind':'line','x':x,'y':y,'support':support,'label':label,**extra}
        def panel(pid,title,yunit,series):
            return {'panel_id':pid,'title':title,'quality_status':'PENDING_SIMULATION_CONVERGENCE',
                'selection':density['window']+'; '+density['estimator']+'; unsigned axes; original native support',
                'xlabel':'Proper radius [kpc]','ylabel':title+' ['+yunit+']','x_units':'proper kpc','y_units':yunit,'series':series}
        ratios=[]
        for j,key in enumerate(('b_over_a','c_over_a')):
            series=[line('density_radius','density_'+key,'density_'+key+'_supported',key)]
            for q,label in ((0,'16th percentile'),(2,'84th percentile')):
                view='density_'+key+'_percentile_'+str(q);arrays[view]=arrays['density_ratio_percentiles'][:,q,j];units[view]='dimensionless'
                series.append(line('density_radius',view,'density_uncertainty_supported',label))
            ratios.append(panel('ratio_'+str(j),key,'dimensionless',series))
        tri=[panel('triaxiality','Density triaxiality','dimensionless',[
            line('density_radius','density_triaxiality','density_triaxiality_supported','native density tensor')])]
        twists=[panel('twists','Adjacent unsigned axis twist','deg',[
            line('density_twist_radius','density_twist_deg','density_twist_supported',label,y_column=j,support_column=j)
            for j,label in enumerate(('major','intermediate','minor'))])]
        comparison=[panel('comparison','Density and total-field shape','dimensionless',[
            line(kind+'_radius',kind+'_'+key,kind+'_'+key+'_supported',kind+' '+key)
            for kind in ('density','potential') for key in ('b_over_a','c_over_a')])]
        export_numeric_audit(target,{'schema':'flagship-native-shapes-v1','epoch':epoch,
            'component':component,'native_measurements':{'density':density,'potential':potential},
            'potential_scope':'total reconstructed field; not a separate component potential',
            'field_validation_status':report['field_validation_status'],'science_acceptance':'NOT_ACCEPTED'},
            arrays,{k:{'units':u,'quantity':k,'missing_policy':'original failed, degenerate and sampling-unresolved bins explicitly masked'} for k,u in units.items()},
            algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
        for variant,panels in (('radial_axis_ratios',ratios),('triaxiality',tri),
                ('axis_orientation_twist',twists),('density_potential_shape',comparison)):
            figures.append({'figure_id':name+'-'+variant,'family':'structure','variant':variant,
                'title':component+' '+variant.replace('_',' '),'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':panels})
    return products,figures,pending



def _concentration_export(report,out,sources):
    fit=report.get('concentration',{})
    if fit.get('status')!='MEASURED':
        return {},[],[{'family':'structure','variant':'concentration_mass',
                      'status':fit.get('status','MISSING_NATIVE_CONCENTRATION')}]
    definition=fit['definition'];radius=float(fit['r_delta_proper_kpc'])
    mass=4*np.pi/3*radius**3*definition['overdensity_delta']*definition['reference_density_msun_kpc3']
    if not np.isfinite(mass) or mass<=0:raise ValueError('Positive native declared-overdensity mass required')
    arrays={};units={};series=[]
    components=list(fit['components'])
    for i,component in enumerate(components):
        row=fit['components'][component];key='component_%03d_'%i
        c=row.get('concentration',np.nan);c=np.nan if c is None else float(c)
        covariance=row.get('covariance_log_parameters')
        error=np.nan
        if covariance is not None:
            covariance=np.asarray(covariance,float)
            if covariance.shape!=(2,2) or not np.isfinite(covariance).all() or covariance[1,1]<0:
                raise ValueError('Native NFW conditional covariance is invalid')
            error=c*np.sqrt(covariance[1,1])
        arrays.update({key+'mass':np.array([mass]),key+'concentration':np.array([c]),key+'error':np.array([error]),
                       key+'supported':np.array([row['status']=='MEASURED' and np.isfinite(c) and c>0]),
                       key+'error_supported':np.array([row['status']=='MEASURED' and np.isfinite(c) and c>0 and np.isfinite(error)])})
        # Only this component's five columns are new. Earlier components
        # already have units; rebuilding their entries adds no information.
        for quantity in ('mass','concentration','error','supported','error_supported'):
            units[key+quantity]='Msun' if quantity=='mass' else 'dimensionless'
        series.extend([{'source':'concentration','kind':'scatter','x':key+'mass','y':key+'concentration',
                        'support':key+'supported','label':component+' native NFW fit','marker_size':12},
                       {'source':'concentration','kind':'errorbar','x':key+'mass','y':key+'concentration',
                        'error':key+'error','support':key+'error_supported','label':component+' conditional fit error'}])
    if not components:raise ValueError('Measured native concentration must retain component fit records')
    epoch=report['source_context']['epoch'];path=out/'concentration-mass.json'
    export_numeric_audit(path,{'schema':'flagship-native-concentration-mass-v1','epoch':epoch,
        'native_fit':fit,'component_order':components,'mass_definition':definition,
        'mass_scope':'TOTAL spherical-overdensity mass derived from native interpolated R_delta; same abscissa for each component NFW fit',
        'uncertainty_scope':'delta-method conditional c error from native log-rs covariance; excludes R_delta, model, host selection and resolution uncertainty',
        'science_acceptance':'NOT_ACCEPTED'},arrays,
        {k:{'units':u,'quantity':k,'missing_policy':'unresolved native fits and unavailable covariance explicitly masked'} for k,u in units.items()},
        algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
    return {'concentration':str(path)},[{'figure_id':'native-concentration-mass','family':'structure','variant':'concentration_mass',
        'title':'Native conditional concentration fits','science_status':'NOT_ACCEPTED','epoch':epoch,
        'panels':[{'panel_id':'fits','title':definition['definition_label'],
            'quality_status':'PENDING_MODEL_AND_RESOLUTION_CONVERGENCE','selection':report['source_context']['selection'],
            'xlabel':'Total '+definition['definition_label']+' mass [Msun]','ylabel':'Concentration [dimensionless]',
            'x_units':'Msun','y_units':'dimensionless','series':series}]}],[]



def export_native_splashback_figures(product,destination,*,memory_gib=1.):
    from .analysis import verify_native_environment
    from .halos import verify_halo_structure
    if isinstance(memory_gib,(bool,np.bool_)) or not np.isfinite(memory_gib) or memory_gib<=0:
        raise ValueError('Finite positive environment workspace required')
    path=Path(product).resolve();out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous splashback exports')
    if path.stat().st_size*12>memory_gib*1024**3/4:raise MemoryError('Environment metadata exceeds allocation')
    report,raw,sources=read_product(path,memory_gib=memory_gib/4)
    if report!=verify_native_environment(path):raise ValueError('Environment changed during verification')
    specs=report['numeric_data']['array_contract']['datasets'];sources.update(report['source_hashes'])
    epoch=report['epoch'];products={};figures=[];out.mkdir(parents=True)
    paths=None
    for ordinal,row in enumerate(report['halo_measurements']):
        if 'splashback' not in row:continue
        index=row['smoothing_index'];structure_path=Path(row['halo_structure']).resolve()
        structure,profiles,pins=read_product(structure_path,memory_gib=memory_gib/4)
        if structure!=verify_halo_structure(structure_path):raise ValueError('Host structure changed during verification')
        if (structure['source_context']['epoch']['actual_scale_factor']!=epoch['actual_scale_factor']
                or Path(structure['request']['run_directory']).resolve()!=Path(report['request']['run_directory']).resolve()
                or structure['request']['snapshot']!=report['request']['snapshot']):
            raise ValueError('Splashback profile and environment must bind the same actual native run/epoch')
        sources.update(pins);sources.update(structure['source_hashes'])
        if paths is None:
            paths=[str(Path(p).resolve()) for p in report['request']['halo_structure_products']]
        if paths.count(str(structure_path))!=1:raise ValueError('Unique native host structure binding required')
        host=paths.index(str(structure_path));prefix='smoothing_%02d_halo_%02d_splashback_'%(index,host)
        component=report['request']['splashback_component'];radius_key='profile_'+component+'_radius_proper_kpc'
        if structure['numeric_data']['array_contract']['datasets'][radius_key]['units']!='proper kpc':raise ValueError('Splashback proper radius units differ')
        r=profiles[radius_key]
        values={}
        for key in ('logarithmic_slope','slope_covariance','supported'):
            name=prefix+key
            if specs[name]['units']!='dimensionless':raise ValueError('Native logarithmic slope units differ')
            values[key]=raw[name]
        slope,cov,support=(values[k] for k in ('logarithmic_slope','slope_covariance','supported'))
        if (r.ndim!=1 or slope.shape!=r.shape or support.shape!=r.shape or support.dtype.kind!='b'
                or cov.shape!=(len(r),len(r)) or not np.isfinite(cov).all() or np.any(np.diag(cov)<0)):
            raise ValueError('Native slope, covariance and boolean support arrays must align')
        feature=row['splashback'];candidates=feature['candidates'];n=len(candidates)
        arrays={'radius':r,'slope':slope,'slope_covariance':cov,'error':np.sqrt(np.diag(cov)),
            'slope_supported':support&np.isfinite(slope),
            'candidate_radius':np.array([c['radius_proper_kpc'] for c in candidates],float),
            'candidate_depth':np.array([c['depth'] for c in candidates],float),
            'candidate_width_ln_radius':np.array([c['half_depth_width_ln_radius'] for c in candidates],float),
            'candidate_inner_radius':np.array([c['half_depth_inner_radius_proper_kpc'] for c in candidates],float),
            'candidate_outer_radius':np.array([c['half_depth_outer_radius_proper_kpc'] for c in candidates],float),
            'candidate_supported':np.full(n,feature['status']=='MEASURED',bool)}
        uncertainty=feature.get('sampling_uncertainty',{})
        arrays['radius_depth_width_percentiles']=np.array(uncertainty.get('radius_depth_width_percentile_16_50_84',np.full((3,3),np.nan)),float)
        if arrays['radius_depth_width_percentiles'].shape!=(3,3):raise ValueError('Native radius/depth/width quantile columns differ')
        units={k:'proper kpc' if k in ('radius','candidate_radius','candidate_inner_radius','candidate_outer_radius') else
               'mixed; columns proper kpc, dimensionless depth, dimensionless ln-radius width' if k=='radius_depth_width_percentiles'
               else 'dimensionless' for k in arrays}
        name='halo_%03d_smoothing_%02d'%(ordinal,index);target=out/(name+'.json');products[name]=str(target)
        export_numeric_audit(target,{'schema':'flagship-native-splashback-v1','epoch':epoch,
            'native_environment':str(path),'halo_structure':str(structure_path),'component':component,
            'feature':feature,'slope_definition':row['slope_definition'],
            'slope_window_study':row['slope_window_study'],'covariance_definition':row['covariance_definition'],
            'science_acceptance':'NOT_ACCEPTED'},arrays,
            {k:{'units':u,'quantity':k,'missing_policy':'unsupported slopes and ambiguous/unresolved features remain masked; candidate arrays retained'} for k,u in units.items()},
            algorithm=__file__,context={'epoch':epoch},source_hashes=sources)
        panels=[]
        for pid,title,x,y,mask,yunit,error in (
            ('slope','Native logarithmic density slope','radius','slope','slope_supported','dimensionless','error'),
            ('depth','Supported valley prominence','candidate_radius','candidate_depth','candidate_supported','dimensionless',None),
            ('width','Supported half-prominence width in ln r','candidate_radius','candidate_width_ln_radius','candidate_supported','dimensionless',None)):
            series={'source':name,'kind':'errorbar' if error else 'scatter','x':x,'y':y,'support':mask,'label':component}
            if error:series['error']=error
            panels.append({'panel_id':pid,'title':title,'quality_status':feature['status'],
                'selection':'native component '+component+'; '+feature['definition'],
                'xlabel':'Proper radius [kpc]','ylabel':title+' ['+yunit+']','x_units':'proper kpc','y_units':yunit,'series':[series]})
        figures.append({'figure_id':name+'-splashback','family':'structure','variant':'splashback_slope_radius_depth_width',
            'title':'Native nonparametric splashback diagnostic','science_status':'NOT_ACCEPTED','epoch':epoch,'panels':panels})
    for source,expected in sources.items():
        if file_sha256(source)!=expected:raise ValueError('Native splashback input changed during export')
    atomic_json(out/'products.json',{'schema':'flagship-native-splashback-index-v1','products':products,
        'status':'EXPORTED' if products else 'NO_NATIVE_SPLASHBACK_MEASUREMENTS'})
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,
        'maximum_panels_per_figure':3,'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)



def web_inputs(product,*,memory_gib):
    from .analysis import verify_native_environment
    path=Path(product).resolve()
    report,arrays,pins=read_product(path,memory_gib=memory_gib/2)
    if report!=verify_native_environment(path):raise ValueError('Native web product changed during intake')
    return report,arrays,{**pins,**numeric_audit_source_bindings(report)}



def web_region_arrays(report,arrays,index):
    prefix=f'smoothing_{index:02d}_filament_';contracts=report['numeric_data']['array_contract']['datasets']
    positions=np.asarray(arrays[prefix+'voxel_positions_comoving_kpc']);ids=np.asarray(arrays[prefix+'component_ids'])
    if contracts[prefix+'voxel_positions_comoving_kpc']['units']!='comoving kpc' or contracts[prefix+'component_ids']['units']!='identity':
        raise ValueError('Original web region units differ')
    shape=report['smoothing_fields'][index]['filaments']['grid_shape'];n=report['request']['mesh_size']
    if shape!=[n,n,n] or ids.shape!=(n**3,) or ids.dtype!=np.dtype('uint64') or positions.shape!=(np.count_nonzero(ids),3) or not np.isfinite(positions).all():
        raise ValueError('Original periodic web region geometry differs')
    spacing=report['box_comoving_kpc']/n
    original=np.argwhere(ids.reshape(shape)>0)*spacing
    if not np.array_equal(positions,original):raise ValueError('Web voxel positions differ from original spectral nodes')
    return {'x':positions[:,0],'y':positions[:,1],'z':positions[:,2],
        'component_ids':ids[ids>0]},contracts[prefix+'component_ids']['identity_namespace']



def _publish(path,report,arrays,units,sources,*,identity=None):
    specs={k:{'units':u,'quantity':k,'missing_policy':'original measured support and nonfinite bins retained'} for k,u in units.items()}
    for key,namespace in (identity or {}).items():specs[key]['identity_namespace']=namespace
    export_numeric_audit(path,report,arrays,specs,algorithm=__file__,context={'epoch':report['epoch']},
        source_hashes={**sources,str(Path(__file__).resolve()):file_sha256(__file__)})



def web_alignment_arrays(report,arrays,row,structure,structure_arrays,kind):
    index=row['smoothing_index']
    order=[r['halo_structure'] for r in report['halo_measurements'] if r['smoothing_index']==0]
    if len(order)!=len(report['request']['halo_structure_products']) or len(set(order))!=len(order):
        raise ValueError('Original web halo selection/order is ambiguous')
    halo=order.index(row['halo_structure'])
    prefix=f'smoothing_{index:02d}_halo_{halo:02d}_{kind}_'
    contracts=report['numeric_data']['array_contract']['datasets']
    cosine=np.asarray(arrays[prefix+'cosine_unsigned']);support=np.asarray(arrays[prefix+'supported'])
    if contracts[prefix+'cosine_unsigned']['units']!='dimensionless' or contracts[prefix+'supported']['units']!='dimensionless':raise ValueError('Web alignment units differ')
    if kind=='spin' and structure['numeric_data']['array_contract']['datasets']['profile_total_radial_outer_edge_proper_kpc']['units']!='proper kpc':
        raise ValueError('Original native spin aperture unit differs')
    radius=np.asarray(structure_arrays['profile_total_radial_outer_edge_proper_kpc']) if kind=='spin' else np.asarray(structure['request']['radial_edges_proper_kpc'])[1:]
    if cosine.ndim!=1 or support.dtype!=np.dtype(bool) or support.shape!=cosine.shape or radius.shape!=cosine.shape or not np.isfinite(radius).all() or np.any(radius<=0):
        raise ValueError('Original native alignment aperture/support differs')
    if np.any(~np.isfinite(cosine[support])) or np.any((cosine[support]<0)|(cosine[support]>1)):
        raise ValueError('Supported unoriented-axis cosine invalid')
    return {'radius':radius,'cosine':cosine,'supported':support,
        'isotropic_null_mean_cosine':np.asarray(arrays[prefix+'isotropic_null_mean_cosine'])}



def topology_alignment_arrays(report,arrays,index,kind):
    prefix=f'topology_halo_{index:02d}_';contracts=report['numeric_data']['array_contract']['datasets']
    keys={'radius':prefix+kind+'_radius_proper_kpc','cosine':prefix+kind+'_unique_cosine_unsigned',
        'supported':prefix+kind+'_unique_supported','isotropic_null_mean_cosine':prefix+'candidate_00_'+kind+'_isotropic_null_mean_cosine'}
    for name,key in keys.items():
        if contracts[key]['units']!=('proper kpc' if name=='radius' else 'dimensionless'):
            raise ValueError('Original topology alignment unit differs')
    values={name:np.asarray(arrays[key]) for name,key in keys.items()}
    radius,cosine,support=[values[name] for name in ('radius','cosine','supported')]
    if radius.ndim!=1 or cosine.shape!=radius.shape or support.shape!=radius.shape or support.dtype!=np.dtype(bool) or not np.isfinite(radius).all() or np.any(radius<=0):
        raise ValueError('Original topology aperture/unique support differs')
    if np.any(~np.isfinite(cosine[support])) or np.any((cosine[support]<0)|(cosine[support]>1)):
        raise ValueError('Supported native arc-axis alignment is invalid')
    return values



def export_native_web_figures(product,destination,*,memory_gib=2.):
    from .halos import verify_halo_structure
    from .analysis import read_bound_filament_graph
    from .halos import _snapshot_files
    report,arrays,sources=web_inputs(product,memory_gib=memory_gib);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native web figures')
    out.mkdir(parents=True);products={};figures=[];epoch=report['epoch']
    title_prefix='Synthetic control: ' if report['source_origin']=='synthetic_control' else ''
    for index,field in enumerate(report['smoothing_fields']):
        values,namespace=web_region_arrays(report,arrays,index);name=f'web-regions-{index:02d}';target=out/(name+'.json');products[name]=str(target)
        _publish(target,{'schema':'native-web-regions-figure-v1','epoch':epoch,'original_definition':field,
            'geometry_scope':'periodic face-connected T-web class-2 voxel regions; no ridge or cross-epoch filament identity inferred',
            'science_acceptance':'NOT_ACCEPTED'},values,{'x':'comoving kpc','y':'comoving kpc','z':'comoving kpc','component_ids':'identity'},sources,identity={'component_ids':namespace})
        figures.append({'figure_id':name,'family':'web','variant':'filament_map','title':title_prefix+'Periodic T-web filament-class regions',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'regions','projection':'3d',
                'title':f'Original classified spectral nodes: {len(values["x"])}; voxel regions, not ridge skeletons',
                'quality_status':'PENDING_CONVERGENCE' if len(values['x']) else 'NO_CLASSIFIED_FILAMENT_REGIONS',
                'selection':'original class-2 cells with supported filtered density; product-local component labels retained',
                'xlabel':'x [comoving kpc]','ylabel':'y [comoving kpc]','zlabel':'z [comoving kpc]',
                'x_units':'comoving kpc','y_units':'comoving kpc','z_units':'comoving kpc',
                'xlim':[0.,report['box_comoving_kpc']],'ylim':[0.,report['box_comoving_kpc']],'zlim':[0.,report['box_comoving_kpc']],
                'series':[{'source':name,'kind':'scatter3d','x':'x','y':'y','z':'z','label':'T-web region nodes','marker_size':2}]}]})
    binding=report['request'].get('filament_skeleton')
    if binding is not None:
        run=Path(report['request']['run_directory']);files=_snapshot_files(run,report['request']['snapshot'])
        geometry,pins,original=read_bound_filament_graph(binding,run,report['request']['snapshot'],
            {str(p.resolve()):sources[str(p.resolve())] for p in files},memory_bytes=int(memory_gib*1024**3/3))
        sources.update(pins);paths=geometry['arc_paths_mpc_h']
        points=np.concatenate([np.vstack([path,np.full((1,3),np.nan)]) for path in paths]) if paths else np.empty((0,3))
        values={axis:points[:,i] for i,axis in enumerate('xyz')};target=out/'filament-skeleton.json';products['skeleton']=str(target)
        _publish(target,{'schema':'native-filament-skeleton-figure-v1','epoch':epoch,'original_publication':original,
            'geometry_scope':'original native arc topology with recorded integer spatial images; NaN breaks between distinct arcs',
            'science_acceptance':'NOT_ACCEPTED'},values,{k:'comoving Mpc/h' for k in values},sources)
        figures.append({'figure_id':'native-filament-skeleton','family':'web','variant':'filament_map','title':title_prefix+'Published native filament arcs',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'arcs','projection':'3d','title':'Original arcs and recorded spatial images',
                'quality_status':'PENDING_CONVERGENCE','selection':'sealed raw topology; no smoothing of display paths into measurements',
                'xlabel':'x [comoving Mpc/h]','ylabel':'y [comoving Mpc/h]','zlabel':'z [comoving Mpc/h]',
                'x_units':'comoving Mpc/h','y_units':'comoving Mpc/h','z_units':'comoving Mpc/h',
                'series':[{'source':'skeleton','kind':'line3d','x':'x','y':'y','z':'z','label':'original arcs'}]}]})
        for index,row in enumerate(report['filament_skeleton']['halo_alignments']):
            for kind in ('spin','shape'):
                values=topology_alignment_arrays(report,arrays,index,kind);name=f'web-topology-{kind}-{index:03d}';target=out/(name+'.json');products[name]=str(target)
                _publish(target,{'schema':'native-filament-topology-alignment-figure-v1','epoch':epoch,'original_halo_measurement':row,
                    'original_filament_binding':binding,'definition':'original unsigned cosine to nearest native arc tangent; unique support excludes junction ambiguity',
                    'null_scope':'retained original candidate-zero isotropic draws; no covariance/error estimate inferred for unique-selected sample',
                    'science_acceptance':'NOT_ACCEPTED'},values,{'radius':'proper kpc','cosine':'dimensionless','supported':'dimensionless','isotropic_null_mean_cosine':'dimensionless'},sources)
                figures.append({'figure_id':name,'family':'web','variant':kind+'_alignment','title':title_prefix+'Native '+kind+' / published arc alignment',
                    'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'alignment','title':'Original unique-assignment mask; junction alternatives retained in source',
                        'quality_status':'PENDING_CONVERGENCE','selection':'original native halo apertures and unambiguous published-arc assignments',
                        'xlabel':'Outer aperture radius [proper kpc]','ylabel':'Unsigned cosine [dimensionless]',
                        'x_units':'proper kpc','y_units':'dimensionless','series':[{'source':name,'kind':kind_plot,
                            'x':'radius','y':'cosine','support':'supported','label':label,'marker_size':8}
                            for kind_plot,label in [('line','unique native arc alignment'),('scatter','measured apertures')]]}]})
    for ordinal,row in enumerate(report['halo_measurements']):
        path=Path(row['halo_structure']);structure=verify_halo_structure(path)
        original,saved,pins=read_product(path,memory_gib=memory_gib/4)
        if original!=structure or file_sha256(path)!=row['source_product_sha256']:raise ValueError('Native web/halo join changed')
        sources.update(pins);sources.update(numeric_audit_source_bindings(structure))
        for kind in ('spin','shape'):
            values=web_alignment_arrays(report,arrays,row,structure,saved,kind);name=f'web-{kind}-{ordinal:03d}';target=out/(name+'.json');products[name]=str(target)
            _publish(target,{'schema':'native-web-alignment-figure-v1','epoch':epoch,'original_halo_measurement':row,
                'halo_structure':str(path),'definition':'original unsigned cosine between native '+kind+' and local least-compressive T-web filament axis',
                'null_scope':row[kind+'_alignment'],'science_acceptance':'NOT_ACCEPTED'},values,
                {'radius':'proper kpc','cosine':'dimensionless','supported':'dimensionless','isotropic_null_mean_cosine':'dimensionless'},sources)
            figures.append({'figure_id':name,'family':'web','variant':kind+'_alignment','title':title_prefix+'Native '+kind+' / T-web alignment',
                'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'alignment','title':'Original support masks; unoriented axes',
                    'quality_status':'PENDING_CONVERGENCE','selection':'source-selected native halo and proper radial apertures; unsupported vectors/axes retained',
                    'xlabel':'Outer aperture radius [proper kpc]','ylabel':'Unsigned cosine [dimensionless]',
                    'x_units':'proper kpc','y_units':'dimensionless','series':[
                        {'source':name,'kind':kind_plot,'x':'radius','y':'cosine','support':'supported','label':label,'marker_size':8}
                        for kind_plot,label in [('line','native conditional alignment'),('scatter','measured apertures')]]}]})
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Native web sources changed during export')
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)



def web_history_inputs(product,*,memory_gib):
    from .analysis import verify_environment_history
    report,arrays,pins=read_product(product,memory_gib=memory_gib/2)
    if report!=verify_environment_history(product):raise ValueError('Native web history changed during intake')
    return report,arrays,{**pins,**numeric_audit_source_bindings(report)}



def web_history_arrays(report,arrays,index):
    prefix=f'center_filament_{index:02d}_';contracts=report['numeric_data']['array_contract']['datasets']
    if contracts['proper_time_myr']['units']!='Myr':raise ValueError('Original saved web clock unit differs')
    for key,unit in [('change_angle_deg','degrees'),('actual_time_brackets_myr','Myr'),('clock_bounded_accepted_interval','dimensionless')]:
        if contracts[prefix+key]['units']!=unit:raise ValueError('Original web reorientation unit differs')
    times=np.asarray(arrays['proper_time_myr']);brackets=np.asarray(arrays[prefix+'actual_time_brackets_myr'])
    angles=np.asarray(arrays[prefix+'change_angle_deg']);support=np.asarray(arrays[prefix+'clock_bounded_accepted_interval'])
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or not np.array_equal(brackets,np.column_stack([times[:-1],times[1:]])):
        raise ValueError('Original web saved proper-time brackets differ')
    if angles.shape!=(len(times)-1,1) or support.shape!=angles.shape or support.dtype!=np.dtype(bool):
        raise ValueError('Original central web orientation/support differs')
    if np.any(~np.isfinite(angles[support])) or np.any((angles[support]<0)|(angles[support]>90.)):
        raise ValueError('Supported unoriented web-axis change is invalid')
    return {'saved_interval_end_myr':times[1:],'actual_time_brackets_myr':brackets,
        'axis_change_degrees':angles[:,0],'supported':support[:,0]}



def export_native_web_history(product,destination,*,memory_gib=2.):
    report,arrays,sources=web_history_inputs(product,memory_gib=memory_gib);out=Path(destination).resolve()
    if out.exists():raise FileExistsError('Keep previous native web history exports')
    out.mkdir(parents=True);products={};figures=[]
    epoch={'canonical_time':'actual saved proper-time intervals','row_epoch_contracts':report['epoch_contracts']}
    for index,_ in enumerate(report['field_definitions']['smoothing_comoving_kpc']):
        values=web_history_arrays(report,arrays,index);name=f'web-evolution-{index:02d}';target=out/(name+'.json');products[name]=str(target)
        _publish(target,{'schema':'native-web-evolution-figure-v1','epoch':epoch,'original_history_request':report['request'],
            'original_reorientation_definition':report['reorientations'][f'center_filament_{index:02d}'],
            'original_native_graph_nodes':report['graph_nodes'],
            'definition':'unoriented local T-web axis change along original persistent host graph observations; not cross-epoch voxel-component ancestry',
            'display_time':'original interval end; full saved bracket retained, no continuous event time inferred',
            'science_acceptance':'NOT_ACCEPTED'},values,{'saved_interval_end_myr':'Myr','actual_time_brackets_myr':'Myr',
                'axis_change_degrees':'degrees','supported':'dimensionless'},sources)
        figures.append({'figure_id':name,'family':'web','variant':'web_evolution',
            'title':('Synthetic control: ' if report['source_origin']=='synthetic_control' else '')+'Local web-axis evolution along native host history',
            'science_status':'NOT_ACCEPTED','epoch':epoch,'panels':[{'panel_id':'history',
                'title':'Saved intervals; original orientation, cadence and clock bounds','quality_status':'PENDING_CONVERGENCE',
                'selection':'original persistent host graph; only original clock-bounded accepted intervals plotted',
                'xlabel':'Saved interval end [Myr]','ylabel':'Unsigned axis change [degrees]',
                'x_units':'Myr','y_units':'degrees','series':[{'source':name,'kind':'scatter',
                    'x':'saved_interval_end_myr','y':'axis_change_degrees','support':'supported','label':'accepted saved intervals','marker_size':10}]}]})
    if any(file_sha256(p)!=h for p,h in sources.items()):raise ValueError('Native web history sources changed during export')
    return export_scientific_suite({'sources':products,'figures':figures,'dpi':160,'maximum_panels_per_figure':1,
        'organized_outputs':True},out/'gallery',base=out,memory_gib=memory_gib)

