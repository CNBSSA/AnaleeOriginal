"""Feature Manifest — the machine-enforced Iron Rule (Flask edition).

No route, menu, or feature that existed before may disappear as a side effect of
another change. `feature_manifest.json` is the committed baseline: every Flask
url_map endpoint the app exposes, plus every endpoint the navigation template
links to via url_for(). The guard test (`tests/test_feature_manifest.py`) fails
if anything in the manifest is missing now. Adding features is always fine
(subset check); REMOVING one is only allowed as a deliberate, approved act — run
`python feature_manifest.py` to regenerate and commit the changed JSON (that
commit is the audit trail). `python feature_manifest.py --check` exits non-zero
if a manifested feature is missing (CI / pre-deploy gate).

NOTE: this repo is under a maintenance freeze; this guard is a protective,
read-only safety net (it enumerates the url_map and reads the nav template only)
— it changes no product behaviour.
"""
import json
import os
import re
import sys

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_manifest.json')
NAV_TEMPLATES = [os.path.join('templates', 'base.html')]
_URL_RE = re.compile(r"url_for\(\s*['\"]([a-zA-Z0-9_.]+)['\"]")


def build_app():
    os.environ.setdefault('FLASK_SECRET_KEY', 'feature-manifest')
    os.environ.setdefault('DATABASE_URL', 'sqlite:///:memory:')
    from app import create_app
    return create_app()


def collect_endpoints(app=None):
    app = app or build_app()
    return {r.endpoint for r in app.url_map.iter_rules() if r.endpoint != 'static'}


def collect_nav_refs():
    refs = set()
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in NAV_TEMPLATES:
        path = os.path.join(here, rel)
        try:
            with open(path) as fh:
                refs.update(_URL_RE.findall(fh.read()))
        except OSError:
            continue
    return refs


def build_manifest(app=None):
    return {
        'routes': sorted(collect_endpoints(app)),
        'nav': sorted(collect_nav_refs()),
    }


def load_manifest():
    with open(MANIFEST_PATH) as fh:
        return json.load(fh)


def save_manifest(manifest):
    with open(MANIFEST_PATH, 'w') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write('\n')


def diff_against_current(app=None):
    current = build_manifest(app)
    try:
        committed = load_manifest()
    except FileNotFoundError:
        committed = {'routes': [], 'nav': []}
    result = {}
    for key in ('routes', 'nav'):
        cur, old = set(current.get(key, [])), set(committed.get(key, []))
        result[key] = {'missing': sorted(old - cur), 'added': sorted(cur - old)}
    return result


def _main(argv):
    check = '--check' in argv
    diff = diff_against_current()
    mr, mn = diff['routes']['missing'], diff['nav']['missing']
    ar, an = diff['routes']['added'], diff['nav']['added']
    for r in ar:
        print(f'[+] route  {r}')
    for n in an:
        print(f'[+] menu   {n}')
    for r in mr:
        print(f'[-] route  {r}  (REMOVED)')
    for n in mn:
        print(f'[-] menu   {n}  (REMOVED)')
    if check:
        if mr or mn:
            print('FEATURE MANIFEST CHECK FAILED — features that existed before are gone. '
                  'If intentional AND approved, run `python feature_manifest.py` and commit; '
                  'else restore them (Iron Rule).')
            return 1
        print('Feature Manifest OK — nothing removed.')
        return 0
    save_manifest(build_manifest())
    print(f'Feature Manifest written: +{len(ar)+len(an)} -{len(mr)+len(mn)} '
          '(commit feature_manifest.json).')
    return 0


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
