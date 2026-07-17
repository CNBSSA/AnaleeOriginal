"""Keep Analee's chart reconciled with BooksXperts' — the trial-balance handoff.

Analee exports a trial balance that BooksXperts (and The Accountants) import via
`dataimports`, keyed on each account's LINK (see reports/trial_balance_service.py:
columns Link, Account Name, Amount). On import, a row's link maps to a BooksXperts
CompanyAccount whose SUBCATEGORY drives AFS classification. So for the handoff to
be correct, every link Analee can emit must:
  • exist in BooksXperts' chart (otherwise the row lands in SUSPENSE), and
  • carry the SAME subcategory there (otherwise the row is MISCLASSIFIED).

`reconciliation/booksxperts_chart_reference.json` is a committed snapshot of
BooksXperts' transmission contract (link -> subcategory). This module compares
Analee's own chart against it; the guard test fails on any drift. Regenerate the
reference with `scripts/refresh_booksxperts_chart_reference.py` when BooksXperts'
chart legitimately changes (a Festus-approved act) and commit the new JSON.
"""
import ast
import json
import os
from dataclasses import dataclass, field

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_PATH = os.path.join(_ROOT, 'reconciliation', 'booksxperts_chart_reference.json')
SEED_REL = os.path.join('app', 'management', 'commands', 'seed_chart_of_accounts.py')

# BooksXperts' subcategory PK -> the sub_category NAME string Analee's own
# chart_seed_data.py speaks (same vocabulary as the committed reference JSON —
# see scripts/refresh_booksxperts_chart_reference.py's SUBCATEGORY_NAMES).
_SUBCATEGORY_NAMES = {
    1: 'Current Asset', 2: 'Current Liability', 3: 'Cost of Sales', 4: 'Expenses',
    5: 'Sales', 6: 'Non-Current Asset', 7: 'Non-Current Liabilities', 8: 'Equity',
    9: 'Tax', 10: 'Income',
}
# The coarser `category` bucket Analee's tuples also carry, derived from
# sub_category (matches the existing rows in chart_seed_data.py, e.g. Tax and
# Cost of Sales both file under category 'Expenses').
_CATEGORY_FOR_SUBCATEGORY = {
    'Current Asset': 'Assets', 'Non-Current Asset': 'Assets',
    'Current Liability': 'Liabilities', 'Non-Current Liabilities': 'Liabilities',
    'Sales': 'Income', 'Income': 'Income',
    'Cost of Sales': 'Expenses', 'Expenses': 'Expenses', 'Tax': 'Expenses',
    'Equity': 'Equity',
}
# Starting decade for a freshly-allocated account_number, by sub_category —
# matches chart_seed_data.py's existing numbering convention.
_NUMBER_BLOCK_START = {
    'Current Asset': 1000, 'Non-Current Asset': 1300,
    'Current Liability': 2000, 'Non-Current Liabilities': 2500,
    'Sales': 4000, 'Income': 4100,
    'Cost of Sales': 5000, 'Expenses': 6000,
    'Equity': 7000, 'Tax': 8000,
}


def analee_chart():
    """Analee's own chart as {link: subcategory}, from services/chart_seed_data.py.

    Tuple shape: (account_number, account_name, account_link, category, sub_category).
    Covers the common accounts plus every entity-specific account (a link should
    classify identically across entities; the last wins if ever duplicated)."""
    from services import chart_seed_data as seed
    out = {}
    for t in seed.COMMON_ACCOUNTS:
        out[t[2]] = t[4]
    for _entity, rows in seed.ENTITY_SPECIFIC.items():
        for t in rows:
            out[t[2]] = t[4]
    return out


def booksxperts_reference():
    """BooksXperts' transmission contract as {link: subcategory}."""
    with open(REFERENCE_PATH) as fh:
        ref = json.load(fh)
    return {link: meta['subcategory'] for link, meta in ref['accounts'].items()}


def reconcile():
    """Return the drift that would break the TB handoff.

    {
      'missing_in_booksxperts': [link, ...],   # Analee emits it, BX has no such link -> suspense
      'subcategory_mismatch':   [(link, analee_subcat, bx_subcat), ...],  # -> misclassified
    }
    An empty result on both keys == fully reconciled.
    """
    an = analee_chart()
    bx = booksxperts_reference()
    missing = sorted(link for link in an if link not in bx)
    mismatch = sorted(
        (link, an[link], bx[link]) for link in an if link in bx and an[link] != bx[link]
    )
    return {'missing_in_booksxperts': missing, 'subcategory_mismatch': mismatch}


# --- propose + apply sync (Festus, 2026-07-17 — extends the detect-only guard
# above into an additive-only, conflict-safe sync from BooksXperts' live seed).
#
#   ADD      — a link BooksXperts has that Analee does not have at all (within
#              the relevant scope). Safe to propose and apply automatically.
#   CONFLICT — a link both sides already have, classified under a DIFFERENT
#              sub_category. Never auto-applied — surfaced for a human.
#   STALE    — a link Analee has that BooksXperts no longer carries anywhere.
#              Surfaced only; never auto-removed.
#   NO-OP    — link + sub_category already match. Not reported.
#
# Scope: BooksXperts' COMMON_ACCOUNTS is compared against Analee's own
# COMMON_ACCOUNTS (global). ENTITY_SPECIFIC rows are compared per-entity
# against the union of COMMON_ACCOUNTS and that entity's own ENTITY_SPECIFIC
# bucket. BooksXperts entities with no Analee counterpart (Personal Liability
# Company, Trust — Analee's ENTITY_NAMES only covers five) are out-of-scope.


def default_booksxpert_root() -> str:
    """Sibling checkout, matching scripts/refresh_booksxperts_chart_reference.py."""
    return os.path.join(os.path.dirname(_ROOT), 'booksxpert')


def _parse_seed_tables(seed_path: str):
    """Extract COMMON_ACCOUNTS and ENTITY_SPECIFIC via `ast` — the seed module
    imports Django models at module level, so exec'ing it needs a configured
    BooksXperts environment; parsing does not. Same technique as
    scripts/refresh_booksxperts_chart_reference.py."""
    with open(seed_path) as fh:
        tree = ast.parse(fh.read(), filename=seed_path)

    constants = {}
    tables = {}

    class _Resolver(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in constants:
                return ast.copy_location(ast.Constant(constants[node.id]), node)
            return node

    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        value = node.value
        if isinstance(value, ast.Constant):
            constants[name] = value.value
        elif name in ('COMMON_ACCOUNTS', 'ENTITY_SPECIFIC'):
            resolved = _Resolver().visit(value)
            ast.fix_missing_locations(resolved)
            tables[name] = ast.literal_eval(resolved)

    missing = {'COMMON_ACCOUNTS', 'ENTITY_SPECIFIC'} - set(tables)
    if missing:
        raise ValueError(
            f'Seed tables not found in {seed_path}: {sorted(missing)} — '
            'has the seed command been restructured?')
    return tables['COMMON_ACCOUNTS'], tables['ENTITY_SPECIFIC']


@dataclass
class SyncDiff:
    adds_common: list = field(default_factory=list)       # [(number, name, link, category, sub_category)]
    adds_by_entity: dict = field(default_factory=dict)     # {entity_name: [row, ...]}
    conflicts: list = field(default_factory=list)          # [(link, analee_subcat, bx_subcat)]
    stale: list = field(default_factory=list)              # [link, ...]
    out_of_scope: list = field(default_factory=list)       # [(bx_entity_name, link, name)]

    @property
    def is_clean(self) -> bool:
        return not (self.adds_common or self.adds_by_entity or self.conflicts or self.stale)

    @property
    def add_count(self) -> int:
        return len(self.adds_common) + sum(len(v) for v in self.adds_by_entity.values())


def _bx_subcat_name(subcat_pk: int) -> str:
    if subcat_pk not in _SUBCATEGORY_NAMES:
        raise ValueError(
            f'Unknown BooksXperts subcategory pk {subcat_pk!r} — update '
            '_SUBCATEGORY_NAMES only if BooksXperts has legitimately '
            'extended its pinned taxonomy.')
    return _SUBCATEGORY_NAMES[subcat_pk]


def _current_common_links() -> dict:
    from services import chart_seed_data as seed
    return {t[2]: t[4] for t in seed.COMMON_ACCOUNTS}


def _current_entity_links(entity_name: str) -> dict:
    from services import chart_seed_data as seed
    return {t[2]: t[4] for t in seed.ENTITY_SPECIFIC.get(entity_name, [])}


def _next_account_number(sub_category_name: str, used_numbers: set) -> int:
    n = _NUMBER_BLOCK_START.get(sub_category_name, 9000)
    while n in used_numbers:
        n += 1
    used_numbers.add(n)
    return n


def compute_sync_diff(booksxpert_root: str = None) -> SyncDiff:
    from services import chart_seed_data as seed

    root = booksxpert_root or default_booksxpert_root()
    seed_path = os.path.join(root, SEED_REL)
    if not os.path.exists(seed_path):
        raise FileNotFoundError(
            f'BooksXperts seed not found at {seed_path} — pass booksxpert_root=<checkout>.')
    common, entity_specific = _parse_seed_tables(seed_path)

    current_common = _current_common_links()
    diff = SyncDiff()
    bx_links_anywhere = set()

    used_numbers = {t[0] for t in seed.COMMON_ACCOUNTS}
    for rows in seed.ENTITY_SPECIFIC.values():
        used_numbers.update(t[0] for t in rows)

    for bx_row in common:
        _num, name, link, subcat_pk, _main, _default_for = bx_row
        bx_links_anywhere.add(link)
        sub_category = _bx_subcat_name(subcat_pk)
        if link not in current_common:
            number = _next_account_number(sub_category, used_numbers)
            diff.adds_common.append(
                (number, name, link, _CATEGORY_FOR_SUBCATEGORY[sub_category], sub_category))
        elif current_common[link] != sub_category:
            diff.conflicts.append((link, current_common[link], sub_category))

    for entity_name, rows in entity_specific.items():
        in_scope = entity_name in seed.ENTITY_NAMES
        entity_scope = dict(current_common)
        if in_scope:
            entity_scope.update(_current_entity_links(entity_name))
        for bx_row in rows:
            _num, name, link, subcat_pk, _main, _default_for = bx_row
            bx_links_anywhere.add(link)
            if not in_scope:
                diff.out_of_scope.append((entity_name, link, name))
                continue
            sub_category = _bx_subcat_name(subcat_pk)
            if link not in entity_scope:
                number = _next_account_number(sub_category, used_numbers)
                diff.adds_by_entity.setdefault(entity_name, []).append(
                    (number, name, link, _CATEGORY_FOR_SUBCATEGORY[sub_category], sub_category))
            elif entity_scope[link] != sub_category:
                diff.conflicts.append((link, entity_scope[link], sub_category))

    all_analee_links = set(current_common)
    for rows in seed.ENTITY_SPECIFIC.values():
        all_analee_links.update(t[2] for t in rows)
    diff.stale = sorted(all_analee_links - bx_links_anywhere)

    return diff


def _format_row(row: tuple) -> str:
    number, name, link, category, sub_category = row
    esc = lambda s: s.replace('\\', '\\\\').replace("'", "\\'")
    return f"({number}, '{esc(name)}', '{esc(link)}', '{esc(category)}', '{esc(sub_category)}')"


def apply_sync_diff(diff: SyncDiff, target_path: str) -> int:
    """Write ADD-only rows into services/chart_seed_data.py: appended to the
    COMMON_ACCOUNTS list literal, or into the correct ENTITY_SPECIFIC[entity]
    list literal — insertion points located via `ast`, never guessed by hand.
    Never touches CONFLICT or STALE entries."""
    if not diff.adds_common and not diff.adds_by_entity:
        return 0

    with open(target_path) as fh:
        source = fh.read()
    tree = ast.parse(source, filename=target_path)

    edits = []  # [(start_offset, end_offset, replacement_text)]

    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        target_name = node.targets[0].id

        if target_name == 'COMMON_ACCOUNTS' and diff.adds_common and isinstance(node.value, ast.List):
            insert_text = ''.join(', ' + _format_row(r) for r in diff.adds_common)
            edits.append((node.value.end_col_offset - 1, node.value.end_lineno, insert_text))

        if target_name == 'ENTITY_SPECIFIC' and diff.adds_by_entity and isinstance(node.value, ast.Dict):
            for key_node, val_node in zip(node.value.keys, node.value.values):
                if not (isinstance(key_node, ast.Constant) and isinstance(val_node, ast.List)):
                    continue
                rows = diff.adds_by_entity.get(key_node.value)
                if not rows:
                    continue
                insert_text = ''.join(', ' + _format_row(r) for r in rows)
                edits.append((val_node.end_col_offset - 1, val_node.end_lineno, insert_text))

    if not edits:
        return 0

    lines = source.splitlines(keepends=True)
    # Apply bottom-up (by line, then by column) so earlier offsets stay valid.
    for col_offset, line_no, insert_text in sorted(edits, key=lambda e: (-e[1], -e[0])):
        line = lines[line_no - 1]
        lines[line_no - 1] = line[:col_offset] + insert_text + line[col_offset:]

    with open(target_path, 'w') as fh:
        fh.writelines(lines)

    return diff.add_count
