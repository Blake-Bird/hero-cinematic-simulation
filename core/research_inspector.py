"""Measured object inspector embedded in the saved scene, standard library only."""
import json
from pathlib import Path


def install_inspector(bpy):
    source = Path(__file__).read_text()
    text = bpy.data.texts.get('research_object_inspector.py') or bpy.data.texts.new('research_object_inspector.py')
    text.clear()
    text.write(source+'\n\nregister()\n')
    text.use_module = True
    register()


def register():
    import bpy
    class RESEARCH_PT_measured(bpy.types.Panel):
        bl_label = 'Measured Research Object'
        bl_idname = 'RESEARCH_PT_measured'
        bl_space_type = 'VIEW_3D'
        bl_region_type = 'UI'
        bl_category = 'Research'
        def draw(self, context):
            layout = self.layout
            scene = context.scene
            if 'research_current_snapshot' in scene:
                layout.label(text='Saved snapshot: '+scene['research_current_snapshot'])
                layout.label(text='Units: '+scene['research_coordinate_units'])
                clock = json.loads(scene.get('research_current_epoch','{}'))
                for key in ('scale_factor','native_time_code','redshift'):
                    if key in clock:
                        layout.label(text=key+': '+str(clock[key]))
            obj = context.active_object
            if obj is None or 'research_identity' not in obj or obj.get('research_display_owner') != scene.get('research_display_owner'):
                layout.label(text='Select a measured object')
                return
            layout.label(text='Identity: '+obj['research_identity'])
            layout.label(text='Kind: '+obj['research_kind'])
            object_epoch = int(obj.get('research_epoch_index',-1))
            current_epoch = int(scene.get('research_current_epoch_index',-2))
            if object_epoch != current_epoch:
                layout.label(text='Not detected at the current saved epoch',icon='INFO')
            if obj.get('research_geometry_supported') is False:
                layout.label(text='Unsupported geometry; symbolic marker only',icon='INFO')
            layout.prop(scene,'["research_follow_exact_identity"]',text='Follow exact identity')
            lifecycle = json.loads(scene.get('research_lifecycle_index','{}')).get(obj['research_identity'])
            if lifecycle:
                layout.label(text='Detection range: epochs '+str(lifecycle['first_detection_epoch_index'])+'–'+str(lifecycle['last_detection_epoch_index']))
                if current_epoch in lifecycle['missing_detection_epoch_indices']:
                    layout.label(text='Catalogue gap; physical death unresolved',icon='INFO')
                previous = [i for i,present in enumerate(lifecycle['presence']) if present and i < current_epoch]
                following = [i for i,present in enumerate(lifecycle['presence']) if present and i > current_epoch]
                row = layout.row(align=True)
                for label,epochs in (('Previous detection',previous[-1:]),('Next detection',following[:1])):
                    if epochs:
                        op = row.operator('research.jump_measured_epoch',text=label)
                        op.epoch_index = epochs[0]
                        op.owner = scene['research_display_owner']
                for entry in lifecycle['measured_events']:
                    if entry['epoch_index'] == current_epoch:
                        box = layout.box()
                        box.label(text='Measured event: '+entry['event']['event_type'])
                        event = entry['event']
                        if 'quantity_definition' in event:
                            box.label(text=event['quantity_definition'])
                        for name,value in event.get('measurements',{}).items():
                            box.label(text=name+': '+str(value)+' '+event.get('measurement_units',{}).get(name,''))
                        for source in entry['event']['source_links']:
                            op = box.operator('research.open_measured_source',text=source.get('label','Open event source'))
                            op.path = source['path']
                            op.object_name = obj.name
            record = json.loads(obj['research_measurement_json'])
            for key in ('density', 'interaction_exposure', 'interaction_count', 'core_radius', 'event_type', 'scientific_acceptance'):
                if key in record:
                    layout.label(text=key+': '+str(record[key]))
            layout.label(text='Selected layer: '+obj.get('research_geometry_layer','primary'))
            for layer in record.get('measured_geometry_layers',[]):
                box = layout.box()
                box.label(text='Measured geometry: '+layer['kind'])
                box.label(text='Producer support: '+str(layer['geometry_supported']))
            for diagnostic in record.get('measured_diagnostics',[]):
                box = layout.box()
                box.label(text='Measured SIDM macro diagnostics')
                box.label(text='Producer support: '+str(diagnostic['geometry_supported']))
                units = diagnostic.get('measurement_units',{})
                for key in ('density','interaction_exposure','interaction_count','core_radius'):
                    if key in diagnostic:
                        box.label(text=key+': '+str(diagnostic[key])+' '+units.get(key,''))
                box.label(text='Attached to exact identity; no collision trajectory inferred')
            for parent in record.get('parents', []):
                layout.label(text='Parent: '+parent['namespace']+':'+str(parent['object_id']))
                op = layout.operator('research.select_exact_identity',text='Select measured ancestor')
                op.identity = parent['namespace']+':'+str(parent['object_id'])
                op.owner = scene['research_display_owner']
            for source in record['source_links']:
                box = layout.box()
                box.label(text=source.get('label', 'Scientific source'))
                box.label(text=source['path'])
                box.label(text='SHA256 '+source['sha256'][:20])
                op = box.operator('research.open_measured_source', text='Open source file')
                op.path = source['path']
                op.object_name = obj.name
            if record['kind'] == 'sidm':
                layout.label(text='Macro diagnostics; symbolic marker')
    class RESEARCH_OT_epoch(bpy.types.Operator):
        bl_idname = 'research.jump_measured_epoch'
        bl_label = 'Jump to saved measured epoch'
        epoch_index: bpy.props.IntProperty(min=0)
        owner: bpy.props.StringProperty()
        def execute(self,context):
            scene = context.scene
            if scene.get('research_display_owner') != self.owner:
                self.report({'ERROR'},'Measured epoch belongs to another scene generation')
                return {'CANCELLED'}
            timeline = json.loads(scene['research_display_manifest'])['timeline']
            if not 0 <= self.epoch_index < len(timeline):
                self.report({'ERROR'},'Saved measured epoch is absent')
                return {'CANCELLED'}
            scene.frame_set(timeline[self.epoch_index]['frame'])
            return {'FINISHED'}
    class RESEARCH_OT_identity(bpy.types.Operator):
        bl_idname = 'research.select_exact_identity'
        bl_label = 'Select exact measured identity'
        identity: bpy.props.StringProperty()
        owner: bpy.props.StringProperty()
        def execute(self,context):
            scene = context.scene
            if scene.get('research_display_owner') != self.owner:
                self.report({'ERROR'},'Exact identity belongs to another scene generation')
                return {'CANCELLED'}
            candidates = [obj for obj in scene.objects if obj.get('research_display_owner') == self.owner
                and obj.get('research_identity') == self.identity and obj.get('research_geometry_layer') == 'primary']
            if not candidates:
                self.report({'INFO'},'Ancestor is outside this measured display selection')
                return {'CANCELLED'}
            current = int(scene.get('research_current_epoch_index',0))
            target = min(candidates,key=lambda obj:(abs(int(obj['research_epoch_index'])-current),int(obj['research_epoch_index'])))
            timeline = json.loads(scene['research_display_manifest'])['timeline']
            scene.frame_set(timeline[int(target['research_epoch_index'])]['frame'])
            for obj in context.selected_objects:
                obj.select_set(False)
            target.select_set(True)
            context.view_layer.objects.active = target
            return {'FINISHED'}
    class RESEARCH_OT_source(bpy.types.Operator):
        bl_idname = 'research.open_measured_source'
        bl_label = 'Open measured source'
        path: bpy.props.StringProperty()
        object_name: bpy.props.StringProperty()
        def execute(self, context):
            import hashlib
            try:
                source_object = bpy.data.objects.get(self.object_name)
                if source_object is None or 'research_measurement_json' not in source_object:
                    raise ValueError('Measured source object is no longer present')
                if source_object.get('research_display_owner') != context.scene.get('research_display_owner'):
                    raise ValueError('Measured source belongs to another scene generation')
                base = Path(bpy.path.abspath(context.scene['research_display_base'])).resolve()
                path = (base/self.path).resolve()
                if not path.is_relative_to(base):
                    raise ValueError('Source path escaped package')
                record = json.loads(source_object['research_measurement_json'])
                allowed = list(record['source_links'])
                lifecycle = json.loads(context.scene.get('research_lifecycle_index','{}')).get(source_object['research_identity'],{})
                for entry in lifecycle.get('measured_events',[]):
                    allowed.extend(entry['event']['source_links'])
                references = [q for q in allowed if q['path'] == self.path]
                hashes = {q['sha256'] for q in references}
                if len(hashes) != 1:
                    raise ValueError('Exact measured source link is absent or ambiguous')
                expected = next(iter(hashes))
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError('Scientific source changed')
                bpy.ops.wm.path_open(filepath=str(path))
            except (OSError,ValueError,KeyError) as error:
                self.report({'ERROR'},str(error))
                return {'CANCELLED'}
            return {'FINISHED'}
    for cls in (RESEARCH_OT_source,RESEARCH_OT_epoch,RESEARCH_OT_identity,RESEARCH_PT_measured):
        prefix,separator,name=cls.bl_idname.partition('.')
        identifier=prefix.upper()+'_OT_'+name if separator else cls.bl_idname
        old = getattr(bpy.types, identifier, None)
        if old:
            bpy.utils.unregister_class(old)
        bpy.utils.register_class(cls)
