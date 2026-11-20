"""Serialize native Oxford CS1 results into a copy of an H5OINA acquisition."""
from pathlib import Path
import json
import os
import tempfile
import posixpath
import re
import warnings
import numpy as np

OXFORD_PLANE_PC_WARNING = 'Orientations were indexed with a fitted source plane. Standard Oxford pattern-centre fields retain acquisition values. Fitted values are stored in /zebsdsolve/PatternCentres.'

def _h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError("HDF5 export requires h5py: pip install 'zebsdsolve[hdf5]'") from e
    return h5py

def _temporary(path):
    p = Path(path)
    fd, name = tempfile.mkstemp(prefix='.' + p.name + '.', suffix='.h5', dir=p.parent)
    os.close(fd)
    return name

def _string(group, name, value):
    return group.create_dataset(name, data=str(value), dtype=_h5py().string_dtype('utf-8'))

def _scalar(group, name):
    return np.asarray(group[name][()]).reshape(-1)[0]

def _unit(dataset):
    unit = dataset.attrs.get('Unit', '')
    if isinstance(unit, bytes): unit = unit.decode()
    return str(unit).strip().lower()

def _lattice(group):
    d = group['Lattice Dimensions']
    unit = d.attrs.get('Unit', '')
    if isinstance(unit, bytes): unit = unit.decode()
    factor = {'angstrom': 1, 'angstroms': 1, 'å': 1, 'nm': 10}.get(str(unit).lower())
    if factor is None:
        raise ValueError('Unknown Oxford lattice length unit')
    angles = group['Lattice Angles']
    unit = angles.attrs.get('Unit', '')
    if isinstance(unit, bytes): unit = unit.decode()
    if str(unit).lower() not in ('rad', 'radian', 'radians'):
        raise ValueError('Oxford lattice angles must be radians')
    return np.r_[np.asarray(d[()]).reshape(3)*factor, np.rad2deg(np.asarray(angles[()]).reshape(3))]

def _matches(group, phase):
    sg = int(_scalar(group, 'Space Group')) if 'Space Group' in group else 0
    return sg > 0 and sg == phase['space_group'] and np.allclose(_lattice(group), phase['lattice'], rtol=1e-5, atol=1e-4)


PATTERN_DATASETS = {'Processed Patterns', 'Unprocessed Patterns', 'Raw Patterns'}

def _raw_signal(path):
    return posixpath.basename(path) in PATTERN_DATASETS or re.search(r'(?:^|/)EDS\d*/Data/Spectrum$', path) is not None


def _copy_without_patterns(source, target, omit_roots=()):
    """Retain metadata and maps without reading raw EBSD or EDS signal payloads."""
    h5py = _h5py()
    def identity(obj):
        return h5py.h5o.get_info(obj.id).addr
    def references(dtype):
        if h5py.check_dtype(ref=dtype) is not None: return True
        if dtype.fields: return any(references(v[0]) for v in dtype.fields.values())
        if dtype.subdtype: return references(dtype.subdtype[0])
        base = h5py.check_dtype(vlen=dtype)
        if isinstance(base, np.dtype): return references(base)
        return False
    omitted_ids, omitted, checked, omitted_edges = set(), [], set(), set()
    def preflight(g):
        key = identity(g)
        if key in checked or key in omitted_ids: return
        checked.add(key)
        for name in g.attrs:
            if references(g.attrs.get_id(name).dtype):
                raise ValueError('Oxford metadata export does not support HDF5 reference attributes: '+g.name)
        if isinstance(g, h5py.Dataset):
            if g.external or g.is_virtual:
                raise ValueError('Oxford export requires self-contained acquisition data: ' + g.name)
            if references(g.dtype):
                raise ValueError('Oxford metadata export does not support HDF5 reference datasets: '+g.name)
            return
        for name in g:
            link = g.get(name, getlink=True)
            if isinstance(link,h5py.SoftLink) and '..' in link.path.split('/'):
                raise ValueError('Oxford metadata export does not support literal .. in HDF5 soft links: '+g.name+'/'+name)
            if _raw_signal(posixpath.join(g.name,name)):
                if isinstance(link, h5py.ExternalLink):
                    omitted.append(posixpath.join(g.name,name)); continue
                obj = local_target(posixpath.join(g.name,name))
                if isinstance(obj,h5py.Dataset):
                    omitted_ids.add(identity(obj)); omitted.append(posixpath.join(g.name,name)); continue
            if not isinstance(link, h5py.HardLink): continue
            obj = g[name]
            preflight(obj)
    def local_target(path, seen=None):
        # Resolve local links only; external files are never opened.
        seen = set() if seen is None else seen
        path = '/'+ '/'.join(p for p in path.split('/') if p and p != '.')
        if path in seen: return None
        seen.add(path)
        parts = path.strip('/').split('/') if path != '/' else []
        obj = source
        for i, name in enumerate(parts):
            if not isinstance(obj, h5py.Group): return None
            link = obj.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink) or link is None: return None
            if isinstance(link, h5py.SoftLink):
                target_path = link.path if link.path.startswith('/') else posixpath.join(obj.name, link.path)
                return local_target(posixpath.join(target_path, *parts[i+1:]), seen)
            obj = obj[name]
        return obj
    # Drop old processing solutions and every alias to their private objects.
    # Keep any objects shared with the acquisition, including phase definitions.
    protected = set()
    def protect_tree(obj):
        key = identity(obj)
        if key in protected: return
        protected.add(key)
        if isinstance(obj, h5py.Group):
            for key in obj:
                if isinstance(obj.get(key, getlink=True), h5py.HardLink): protect_tree(obj[key])
    for path in omit_roots:
        acquisition = local_target(path.rsplit('/', 1)[0]+'/EBSD')
        if acquisition is not None: protect_tree(acquisition)
    def omit_tree(obj):
        key = identity(obj)
        if key in omitted_ids or key in protected: return
        omitted_ids.add(key)
        if isinstance(obj, h5py.Group):
            for key in obj:
                if isinstance(obj.get(key, getlink=True), h5py.HardLink): omit_tree(obj[key])
    for path in omit_roots:
        obj = local_target(path)
        if obj is not None: omit_tree(obj)
    # An earlier alias to an EDS Data group must not hide its canonical raw signal.
    for scan_name in source:
        scan = local_target('/'+scan_name)
        if not isinstance(scan, h5py.Group): continue
        for detector in scan:
            if not re.fullmatch(r'EDS\d*', detector): continue
            path = '/'+scan_name+'/'+detector+'/Data/Spectrum'
            data = local_target(path.rsplit('/',1)[0])
            if isinstance(data, h5py.Group) and data.get('Spectrum', getlink=True) is not None:
                omitted_edges.add((identity(data), 'Spectrum')); omitted.append(path)
            obj = local_target(path)
            if isinstance(obj, h5py.Dataset):
                omitted_ids.add(identity(obj))
    preflight(source)
    copied = {identity(source): target.name}
    def copy_attrs(a,b):
        for name in a.attrs:
            attr = a.attrs.get_id(name)
            b.attrs.create(name, a.attrs[name], shape=attr.shape, dtype=attr.dtype)
    def copy_group(a,b):
        copy_attrs(a,b)
        for name in a:
            path = posixpath.join(a.name, name)
            if any(path == root or path.startswith(root+'/') for root in omit_roots): continue
            if _raw_signal(path) or (identity(a), name) in omitted_edges:
                omitted.append(path); continue
            link = a.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                if _raw_signal(link.path):
                    omitted.append(path)
                    continue
                raise ValueError('Oxford export requires self-contained acquisition metadata: ' + path)
            if isinstance(link, h5py.SoftLink):
                path = link.path if link.path.startswith('/') else posixpath.join(a.name,link.path)
                if _raw_signal(path):
                    omitted.append(posixpath.join(a.name,name)); continue
                obj = local_target(path)
                if any(path == root or path.startswith(root+'/') for root in omit_roots): continue
                if obj is not None and identity(obj) in omitted_ids:
                    omitted.append(posixpath.join(a.name,name)); continue
                b[name] = link
                continue
            obj = a[name]; key = identity(obj)
            if key in omitted_ids:
                omitted.append(obj.name); continue
            if key in copied:
                b[name] = target[copied[key]]
            elif isinstance(obj, h5py.Group):
                group = b.create_group(name)
                copied[key] = group.name
                copy_group(obj,group)
            else:
                source.copy(obj,b,name=name)
                copied[key] = b[name].name
    copy_group(source,target)
    return sorted(set(omitted))

def _oxford_group(f, s):
    found = []
    f.visititems(lambda name, obj: found.append(name) if isinstance(obj, _h5py().Group) and name.endswith('/EBSD') and 'Data/Euler' in obj else None)
    if len(found) != 1:
        raise ValueError('Oxford export requires one unambiguous EBSD acquisition')
    g = f[found[0]]
    header, data = g['Header'], g['Data']
    n = s['scan']['rows']*s['scan']['cols']
    if (int(_scalar(header, 'X Cells')), int(_scalar(header, 'Y Cells'))) != (s['scan']['cols'], s['scan']['rows']):
        raise ValueError('Oxford baseline map dimensions differ from indexed scan')
    if data['Euler'].shape != (n, 3) or data['Phase'].shape != (n,):
        raise ValueError('Oxford baseline result layout differs from indexed scan')
    if data['Euler'].dtype.kind != 'f' or data['Phase'].dtype.kind not in 'ui':
        raise ValueError('Unsupported Oxford orientation/phase datatype')
    for key, expected in [('Pattern Width',s['scan']['pat_w']),('Pattern Height',s['scan']['pat_h'])]:
        if key in header and int(_scalar(header,key)) != expected:
            raise ValueError('Oxford baseline detector dimensions differ from indexed patterns')
    euler_unit = data['Euler'].attrs.get('Unit', '')
    if isinstance(euler_unit, bytes): euler_unit = euler_unit.decode()
    if str(euler_unit).lower() not in ('rad', 'radian', 'radians'):
        raise ValueError('Oxford baseline Euler unit must be radians')
    for axis, expected in [('X', np.tile(np.arange(s['scan']['cols']), s['scan']['rows'])), ('Y', np.repeat(np.arange(s['scan']['rows']), s['scan']['cols']))]:
        if axis not in data:
            raise ValueError('Oxford baseline requires per-pixel '+axis+' for a verified pixel join')
        microns = ('um','µm','μm','micrometres','micrometers')
        if _unit(data[axis]) not in microns or _unit(header[axis+' Step']) not in microns:
            raise ValueError('Oxford positions and raster steps must declare micrometre units')
        step = float(_scalar(header, axis+' Step'))
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Invalid Oxford raster step')
        if not np.isclose(step, s['geometry']['step_'+axis.lower()+'_um'], rtol=1e-5, atol=1e-7):
            raise ValueError('Oxford baseline and indexed map steps differ')
        positions = np.asarray(data[axis][()]).reshape(n)
        if not np.allclose(positions, positions[0]+expected*step, rtol=0, atol=max(1e-6, abs(step)*1e-3)):
            raise ValueError('Oxford positions do not match engine row-major pixel IDs')
    return g.name

def write_oxford(path, reference, s):
    frame = s.get('oxford_frame', {})
    preserve_source_pc = bool(frame.get('preserve_source_pattern_centres', False)) or s.get('geometry', {}).get('plane_kind', 0) > 0
    if preserve_source_pc:
        if 'pattern_centres' not in s:
            raise ValueError('Native fitted-plane snapshot requires pattern_centres')
        frame = {**frame, 'preserve_source_pattern_centres': True, 'warnings': [OXFORD_PLANE_PC_WARNING]}
        s = {**s, 'oxford_frame': frame}
        # A warnings-as-errors policy must stop before any output work or replacement.
        warnings.warn(OXFORD_PLANE_PC_WARNING, UserWarning, stacklevel=2)
    h5py = _h5py()
    if Path(path).resolve() == Path(reference).resolve() or (Path(path).exists() and os.path.samefile(path, reference)):
        raise ValueError('Oxford export must not overwrite its original baseline')
    with h5py.File(reference, 'r') as f:
        if 'zebsdsolve' in f:
            raise ValueError('Use the original acquisition as baseline, not a prior zebsdsolve export')
        name = _oxford_group(f, s)
        phases = f[name+'/Header/Phases']
        ids = sorted(int(k) for k in phases.keys())
        mapping, additions = [], []
        next_id = max(ids, default=0)+1
        for phase in s['phases']:
            matches = [pid for pid in ids if _matches(phases[str(pid)], phase)]
            if len(matches) == 1:
                pid = matches[0]
            elif len(matches) > 1:
                raise ValueError('Ambiguous Oxford phase identities')
            else:
                pid = next_id; next_id += 1; additions.append((pid, phase))
            mapping.append(pid)
        if len(set(mapping)) != len(mapping):
            raise ValueError('Each indexed phase must map to a distinct Oxford phase')
        dtype = f[name+'/Data/Phase'].dtype
        if dtype.kind not in 'ui' or max(mapping, default=0) > np.iinfo(dtype).max:
            raise ValueError('New phase IDs exceed the original Oxford Phase datatype')
    temporary = _temporary(path)
    try:
        with h5py.File(reference, 'r') as source, h5py.File(temporary, 'w') as output:
            processing = name.rsplit('/',1)[0]+'/Data Processing'
            omitted_signals = _copy_without_patterns(source, output, (processing,))
        with h5py.File(temporary, 'r+') as f:
            root = f.create_group('zebsdsolve')
            _string(root, 'Version', s['version'])
            _string(root, 'GeometryJSON', json.dumps(s['geometry']))
            _string(root, 'ProvenanceJSON', json.dumps({
                'frame': s['oxford_frame'], 'calibration': s['calibration'],
                'omitted_signal_paths': omitted_signals,
            }))
            centres = np.asarray(s['pattern_centres'], dtype=float).reshape(-1, 3)
            if not np.isfinite(centres).all():
                raise ValueError('Invalid native pattern centres')
            centre_dataset = root.create_dataset('PatternCentres', data=centres)
            centre_dataset.attrs['Convention'] = 'Bruker: x from left / width; y from top / height; z / height'
            calibration = s['calibration']
            header = f[name+'/Header']
            for key, field, count in [('Detector Orientation Euler', 'effective_detector_euler_radians', 3),
                                      ('Tilt Angle', 'effective_tilt_radians', 1)]:
                if field not in calibration:
                    continue
                values = np.asarray(calibration[field], dtype=float).reshape(-1)
                if values.size != count or not np.isfinite(values).all():
                    raise ValueError('Invalid native effective Oxford calibration: '+field)
                if key not in header:
                    raise ValueError('Oxford source calibration header is missing: '+key)
                dataset = header[key]
                if dataset.size != count or dataset.dtype.kind != 'f' or _unit(dataset) not in ('rad', 'radians'):
                    raise ValueError('Unsupported Oxford calibration layout or units: '+key)
                original = root.require_group('OriginalCalibration')
                f.copy(dataset, original, name=key)
                dataset[...] = values.reshape(dataset.shape)
            data = f[name+'/Data']; phases = f[name+'/Header/Phases']
            for pid, phase in additions:
                g = phases.create_group(str(pid))
                _string(g, 'Phase Name', phase['name'])
                _string(g, 'Reference', 'zebsdsolve master pattern: '+phase['name'])
                g.create_dataset('Space Group', data=[phase['space_group']], dtype='i4')
                laue = g.create_dataset('Laue Group', data=[phase['laue']], dtype='i4')
                laue.attrs['Symbol'] = phase['laue_symbol']
                g.create_dataset('Lattice Dimensions', data=[phase['lattice'][:3]]).attrs['Unit'] = 'angstrom'
                g.create_dataset('Lattice Angles', data=[np.deg2rad(phase['lattice'][3:])]).attrs['Unit'] = 'rad'
            r = s['results']
            phase = np.asarray(r['phase'], dtype=int)
            euler = np.asarray(r['euler'], dtype=float).reshape(-1, 3)
            valid = (phase >= 0) & (phase < len(mapping)) & np.isfinite(euler).all(axis=1)
            root.create_dataset('IndexedMask', data=valid.astype('u1'))
            if 'Hit Rate' in header:
                header['Hit Rate'][...] = 100 * valid.mean()
            root.create_dataset('PhaseIDMap', data=np.asarray(mapping, dtype='u4'))
            # Keep the full raster but clear all original indexing values.
            metrics = ['Mean Angular Deviation', 'MAD', 'Mean Angular Error', 'Error', 'Bands', 'Band Count', 'Number of Bands', 'Bands Detected', 'Bands Indexed', 'Confidence Index', 'Fit', 'Pattern Fit']
            for key in ['Euler', 'Phase', *metrics]:
                if key not in data: continue
                values = data[key][()]
                if values.shape[0] != len(valid): raise ValueError('Oxford indexing layout differs from indexed pixels: '+key)
                values[...] = np.nan if values.dtype.kind == 'f' and key not in ('Euler', 'Phase') else 0
                if key == 'Euler': values[valid] = euler[valid]
                elif key == 'Phase': values[valid] = np.asarray(mapping)[phase[valid]]
                elif key == 'Error':
                    values[...] = 0  # Oxford NotAnalyzed
                    values[valid] = 1  # Oxford Success
                data[key][...] = values
            if not s.get('oxford_frame', {}).get('preserve_source_pattern_centres', False):
                pc = centres
                ratio = s['scan']['pat_h']/s['scan']['pat_w']
                for key, values in [('Pattern Center X', pc[:, 0]), ('Pattern Center Y', (1-pc[:, 1])*ratio), ('Detector Distance', pc[:, 2]*ratio)]:
                    out = np.full(len(valid), np.nan)
                    out[valid] = values[valid]
                    if key in data:
                        if data[key].shape != valid.shape: raise ValueError('Oxford calibration layout differs from indexed pixels')
                        data[key][...] = out
                    else: data.create_dataset(key, data=out).attrs['Unit'] = 'fraction'
            for key in ('ncc', 'nmi', 'z', 'sweep'):
                if key in r:
                    root.create_dataset(key.upper(), data=np.where(valid, np.asarray(r[key], dtype=float), np.nan))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
