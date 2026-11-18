"""Registered BH/extended-profile figures from the existing template engine.

These are matched imposed-field experiments, never native BH event histories
or a classification of a measured stream. Their original force models and
conditional support remain attached to the numerical exports.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
from .foundation import file_sha256,canonical_digest


def validate_matched_template_figures(parameters):
    for name,minimum in (('maximum_pairs',1),('dpi',32)):
        if type(parameters[name]) is not int or parameters[name]<minimum:raise ValueError('Bounded exact template '+name+' required')
    value=parameters['memory_gib']
    if parameters['dpi']>2400:raise ValueError('Bounded template figure resolution required')
    if type(value) not in (int,float) or not np.isfinite(value) or value<.1:raise ValueError('Bounded positive matched-template workspace required')


def matched_template_inputs(parameters):
    from core.stream_common import read_product
    from .data import numeric_audit_source_bindings
    validate_matched_template_figures(parameters);path=Path(parameters['product']).resolve();memory=parameters['memory_gib']
    if path.stat().st_size*16>memory*1024**3/4:raise MemoryError('Template metadata exceeds declared workspace')
    report,arrays,pins=read_product(path,memory_gib=memory/3);pins.update(numeric_audit_source_bindings(report))
    if report.get('schema')!='stream-perturber-templates-v1' or report.get('execution_status')!='COMPLETE' or report.get('request',{}).get('kind')!='perturber_templates':
        raise ValueError('Completed original matched perturber-template producer required')
    return report,arrays,pins


def matched_template_input_bindings(parameters):return matched_template_inputs(parameters)[2]


def matched_template_figures_runner(parameters,context):
    from core.gravity_fields import EvolvingPerturber
    from .data import export_numeric_audit
    from core.research.display import export_scientific_suite
    report,arrays,pins=matched_template_inputs(parameters)
    if parameters['memory_gib']>context.resources['memory_gib']:raise MemoryError('Template exports exceed workflow allocation')
    groups=defaultdict(list);ids=set();pairs=[]
    for row in report['templates']:
        if row['template_id'] in ids:raise ValueError('Duplicate original template identity')
        ids.add(row['template_id']);groups[(row['design_id'],row['realization_group'])].append(row)
    for rows in groups.values():
        count=sum(row['category']=='black_hole' for row in rows)*sum(row['category'] in {'cdm_subhalo','wdm_subhalo','sidm_subhalo'} for row in rows)
        if len(pairs)+count>parameters['maximum_pairs']:raise MemoryError('Matched profile pair count exceeds frozen allocation')
        for bh in (row for row in rows if row['category']=='black_hole'):
            for extended in (row for row in rows if row['category'] in {'cdm_subhalo','wdm_subhalo','sidm_subhalo'}):pairs.append((bh,extended))
    if not pairs:raise ValueError('Matched design/realization group has no BH and extended-subhalo pair')
    if len(pairs)>parameters['maximum_pairs']:raise MemoryError('Matched profile pair count exceeds frozen allocation')
    out=context.generation/'matched-template-figures';out.mkdir();sources={};figures=[];contracts=report['numeric_data']['array_contract']['datasets']
    pins[str(Path(__file__).resolve())]=file_sha256(__file__)
    for ordinal,(bh,extended) in enumerate(pairs):
        if bh['initial_state_sha256']!=extended['initial_state_sha256'] or canonical_digest(bh['matched_geometry'])!=canonical_digest(extended['matched_geometry']):
            raise ValueError('BH and extended templates change the common initial cohort or encounter geometry')
        for row in (bh,extended):
            if canonical_digest(row['component']['worldline'])!=canonical_digest(row['matched_geometry']):raise ValueError('Template modifies the supposedly matched trajectory')
            field=EvolvingPerturber(row['component'])
            if field.category!=row['category']:raise ValueError('Template category differs from its executed force model')
        values=[]
        for row in (bh,extended):
            prefix=row['template_id']+'__';value={}
            for key,unit in (('particle_ids','identity'),('times_myr','Myr'),('position_kpc','proper kpc'),('velocity_kms','km/s'),
                    ('mass_msun','Msun'),('bin_center_kpc','proper kpc'),('linear_density_msun_kpc','Msun / proper kpc'),
                    ('centroid_kpc','proper kpc'),('velocity_mean_kms','km/s'),('measurement_supported','dimensionless')):
                name=prefix+key
                if contracts[name]['units']!=unit:raise ValueError('Matched template quantity units differ: '+name)
                value[key]=arrays[name]
            values.append(value)
        left,right=values
        e=len(left['times_myr']);b=len(left['bin_center_kpc']);n=len(left['particle_ids'])
        for value in values:
            if (e<2 or b<1 or value['position_kpc'].shape!=(e,n,3) or value['velocity_kms'].shape!=(e,n,3)
                    or value['linear_density_msun_kpc'].shape!=(e,b) or value['centroid_kpc'].shape!=(e,b,2)
                    or value['velocity_mean_kms'].shape!=(e,b,3) or value['measurement_supported'].shape!=(e,b)):
                raise ValueError('Aligned original template particle/epoch/bin moment dimensions required')
        for key in ('particle_ids','times_myr','bin_center_kpc'):
            if not np.array_equal(left[key],right[key]):raise ValueError('Matched template identities, times or measurement bins differ')
        for key in ('position_kpc','velocity_kms'):
            if not np.array_equal(left[key][0],right[key][0]):raise ValueError('Matched template initial phase space differs')
        if not np.array_equal(left['mass_msun'],right['mass_msun']):raise ValueError('Matched template tracer mass selection differs')
        supported=left['measurement_supported']&right['measurement_supported']
        if supported.dtype.kind!='b':raise ValueError('Original boolean matched template support required')
        density=right['linear_density_msun_kpc']-left['linear_density_msun_kpc']
        centroid=right['centroid_kpc']-left['centroid_kpc'];velocity=right['velocity_mean_kms']-left['velocity_mean_kms']
        payload={'s':left['bin_center_kpc'],'times':left['times_myr'],'bh_density':left['linear_density_msun_kpc'],
            'extended_density':right['linear_density_msun_kpc'],'density_difference':density,'transverse_difference_kpc':centroid,
            'velocity_difference_kms':velocity,'paired_supported':supported,'final_supported':supported[-1],
            'final_bh_density':left['linear_density_msun_kpc'][-1],'final_extended_density':right['linear_density_msun_kpc'][-1],
            'final_transverse_difference_kpc':np.linalg.norm(centroid[-1],axis=1),'final_velocity_difference_kms':np.linalg.norm(velocity[-1],axis=1)}
        units={key:'proper kpc' if key=='s' or key.endswith('_kpc') else 'km/s' if key.endswith('_kms') else 'Myr' if key=='times'
            else 'dimensionless' if 'supported' in key else 'Msun / proper kpc' for key in payload}
        key='matched_'+str(ordinal);path=out/(key+'.json');sources[key]=str(path)
        export_numeric_audit(path,{'schema':'matched-bh-extended-template-figure-v1','bh':bh,'extended':extended,
            'scope':'Matched imposed-field forward predictions, not native BH/subhalo ancestry or a calibrated classification',
            'uncertainty':'Original conditional sampling support; no independent ensemble covariance inferred','science_acceptance':'NOT_ACCEPTED'},
            payload,{name:{'units':unit,'quantity':name,'missing_policy':'paired_supported masks govern original moment support'} for name,unit in units.items()},
            algorithm=__file__,source_hashes=pins,context={'epoch':{'actual_forward_output_times_myr':left['times_myr'].tolist()}})
        panels=[]
        for name,title,unit,ys in (('density','Matched final line density','Msun / proper kpc',('final_bh_density','final_extended_density')),
                ('spur','Final transverse centroid difference','proper kpc',('final_transverse_difference_kpc',)),
                ('velocity','Final local velocity difference','km/s',('final_velocity_difference_kms',))):
            panels.append({'panel_id':name,'title':title,'quality_status':'PENDING_CONVERGENCE','selection':'same initial stars, geometry, output clock and fixed track',
                'xlabel':'Frozen track arclength [proper kpc]','ylabel':title+' ['+unit+']','x_units':'proper kpc','y_units':unit,
                'series':[{'source':key,'kind':'line','label':y,'x':'s','y':y,'support':'final_supported'} for y in ys]})
        figures.append({'figure_id':'bh-extended-'+str(ordinal),'family':'encounters','variant':'bh_extended_subhalo_comparison',
            'title':'BH versus '+extended['category']+' — matched imposed fields','science_status':'IMPOSED_FIELD_FORWARD_NOT_ACCEPTED','panels':panels})
    return export_scientific_suite({'sources':sources,'figures':figures,'dpi':parameters['dpi'],'maximum_panels_per_figure':3,'organized_outputs':True},
        out/'gallery',base=out,memory_gib=parameters['memory_gib'])
