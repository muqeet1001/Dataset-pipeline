"""Build an interactive full-resolution mesh preview with embedded geometry."""
import argparse
import base64
import json
from pathlib import Path
import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    mesh = trimesh.load_mesh(args.mesh, process=False)
    if len(mesh.vertices) > 65535:
        raise ValueError('Preview supports up to 65,535 vertices')
    data = {
        'subject': args.mesh.parent.name,
        'height': float(mesh.extents[1]),
        'vertices': len(mesh.vertices),
        'positions': base64.b64encode(np.asarray(mesh.vertices, dtype='<f4').tobytes()).decode('ascii'),
        'indices': base64.b64encode(np.asarray(mesh.faces, dtype='<u2').tobytes()).decode('ascii'),
    }
    template = Path(__file__).with_name('mesh_preview_template.html').read_text(encoding='utf-8')
    content = '<script type="application/json" data-mesh>' + json.dumps(data, separators=(',', ':')) + '</script>'
    preview = template.replace('<!--MESH_DATA-->', content)
    if len(preview.encode('utf-8')) >= 1_000_000:
        raise ValueError('Preview exceeds inline size limit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(preview, encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
