"""Feature Manifest guard — the machine-enforced Iron Rule (Flask).

Fails if any endpoint or nav item recorded in `feature_manifest.json` has
disappeared. Adding features is fine (subset check); removing one requires
`python feature_manifest.py` + committing the changed JSON. Uses the canary_app
fixture (boots the real app) so the endpoint set is the real url_map.
"""
from feature_manifest import build_manifest, load_manifest


def test_no_manifested_feature_was_removed(canary_app):
    current = build_manifest(canary_app)
    manifest = load_manifest()
    missing_routes = sorted(set(manifest['routes']) - set(current['routes']))
    missing_nav = sorted(set(manifest['nav']) - set(current['nav']))
    problems = []
    if missing_routes:
        problems.append(f'ROUTES removed ({len(missing_routes)}): {missing_routes}')
    if missing_nav:
        problems.append(f'MENU items removed ({len(missing_nav)}): {missing_nav}')
    assert not problems, (
        'FEATURE MANIFEST GUARD FAILED — features that existed before are gone:\n'
        + '\n'.join(problems)
        + '\n\nIf intentional AND approved by Festus, run '
        '`python feature_manifest.py` and commit the updated feature_manifest.json. '
        'Otherwise restore them (Iron Rule).')


def test_manifest_is_non_trivial():
    manifest = load_manifest()
    assert len(manifest['routes']) > 40, 'Feature manifest looks truncated.'
