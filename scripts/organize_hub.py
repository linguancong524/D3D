"""Group the six released draft checkpoints in a public D3D Hub collection."""

import argparse
import json
from pathlib import Path


CHECKPOINTS = (
    ('D3D-1.7B-to-8B', 'Qwen3-8B · 1.7B-family draft · 28 layers'),
    ('D3D-1.7B-to-32B', 'Qwen3-32B · 1.7B-family draft · 28 layers'),
    ('D3D-8B-to-8B', 'Qwen3-8B · 8B-family draft · 36 layers'),
    ('D3D-8B-to-32B', 'Qwen3-32B · 8B-family draft · 36 layers'),
    ('D3D-24D-p1.7B-to-32B', 'Qwen3-32B · pruned 1.7B-family draft · 24 layers'),
    ('D3D-31D-p8B-to-8B', 'Qwen3-8B · pruned 8B-family draft · 31 layers'),
)
DESCRIPTION = (
    'Decoupled Diffusion Drafting for Asynchronous Speculative Decoding. '
    'Six Qwen3-derived diffusion draft checkpoints for Qwen3-8B and Qwen3-32B.'
)


def organize_collection(api, namespace='lingc8'):
    models = [(f'{namespace}/{name}', note) for name, note in CHECKPOINTS]
    for model_id, _ in models:
        if api.model_info(model_id).private:
            raise ValueError(f'Expected a public checkpoint: {model_id}')
    matches = [collection for collection in api.list_collections(owner=namespace)
               if collection.title == 'D3D']
    if len(matches) > 1:
        raise ValueError('Multiple D3D collections exist; resolve duplicates before updating.')
    collection = matches[0] if matches else api.create_collection(
        title='D3D', namespace=namespace, description=DESCRIPTION,
        private=False, exists_ok=True,
    )
    api.update_collection_metadata(
        collection.slug, description=DESCRIPTION, private=False, theme='purple', position=0,
    )
    for model_id, note in models:
        api.add_collection_item(
            collection.slug, item_id=model_id, item_type='model', note=note, exists_ok=True,
        )
    verified = api.get_collection(collection.slug)
    actual = {item.item_id for item in verified.items if item.item_type == 'model'}
    missing = {model_id for model_id, _ in models} - actual
    if missing or verified.private:
        raise RuntimeError(f'Collection verification failed: missing={sorted(missing)}')
    return {
        'title': 'D3D', 'namespace': namespace, 'slug': collection.slug,
        'url': f'https://huggingface.co/collections/{collection.slug}',
        'models': [model_id for model_id, _ in models],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='lingc8')
    parser.add_argument('--output', type=Path, help='Write verified collection metadata as JSON.')
    parser.add_argument('--dry-run', action='store_true', help='List models without Hub changes.')
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({'title': 'D3D', 'namespace': args.namespace,
                          'models': [f'{args.namespace}/{name}' for name, _ in CHECKPOINTS]},
                         indent=2))
        return
    from huggingface_hub import HfApi

    result = organize_collection(HfApi(), args.namespace)
    content = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(content)
    print(content, end='')


if __name__ == '__main__':
    main()
