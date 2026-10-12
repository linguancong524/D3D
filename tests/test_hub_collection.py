import unittest
from types import SimpleNamespace

from scripts.organize_hub import CHECKPOINTS, organize_collection


class FakeHub:
    def __init__(self, existing=False, private_model=False):
        self.collection = SimpleNamespace(
            title='D3D', slug='lingc8/d3d-test', items=[], private=False,
        )
        self.collections = [self.collection] if existing else []
        self.private_model = private_model
        self.created = 0

    def model_info(self, model_id):
        return SimpleNamespace(private=self.private_model)

    def list_collections(self, **kwargs):
        return self.collections

    def create_collection(self, **kwargs):
        self.created += 1
        self.collections.append(self.collection)
        return self.collection

    def update_collection_metadata(self, slug, **kwargs):
        self.collection.private = kwargs['private']

    def add_collection_item(self, slug, *, item_id, item_type, **kwargs):
        if item_id not in {item.item_id for item in self.collection.items}:
            self.collection.items.append(SimpleNamespace(item_id=item_id, item_type=item_type))

    def get_collection(self, slug):
        return self.collection


class CollectionTests(unittest.TestCase):
    def test_creates_public_collection_with_all_six_models(self):
        api = FakeHub()
        result = organize_collection(api)
        self.assertEqual(api.created, 1)
        self.assertEqual(result['models'], [f'lingc8/{name}' for name, _ in CHECKPOINTS])
        self.assertEqual(len(api.collection.items), 6)

    def test_reuses_collection_without_duplicate_models(self):
        api = FakeHub(existing=True)
        self.assertEqual(organize_collection(api), organize_collection(api))
        self.assertEqual(api.created, 0)
        self.assertEqual(len(api.collection.items), 6)

    def test_refuses_private_models_before_creating_collection(self):
        api = FakeHub(private_model=True)
        with self.assertRaisesRegex(ValueError, 'public checkpoint'):
            organize_collection(api)
        self.assertEqual(api.created, 0)

    def test_refuses_ambiguous_duplicate_collections(self):
        api = FakeHub(existing=True)
        api.collections.append(api.collection)
        with self.assertRaisesRegex(ValueError, 'Multiple'):
            organize_collection(api)

    def test_verifies_items_after_upload(self):
        api = FakeHub()
        api.add_collection_item = lambda *args, **kwargs: None
        with self.assertRaisesRegex(RuntimeError, 'verification failed'):
            organize_collection(api)


if __name__ == '__main__':
    unittest.main()
