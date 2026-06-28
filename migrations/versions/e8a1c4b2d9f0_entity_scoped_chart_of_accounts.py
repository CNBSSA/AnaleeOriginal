"""Entity-scoped master chart of accounts (BooksXperts parity)

Revision ID: e8a1c4b2d9f0
Revises: c1f5b2a4d8e9
Create Date: 2026-06-28
"""
from alembic import op
import sqlalchemy as sa


revision = 'e8a1c4b2d9f0'
down_revision = 'c1f5b2a4d8e9'
branch_labels = None
depends_on = None

ENTITY_NAMES = (
    'Sole Proprietor',
    'Close Corporation',
    'Private Company',
    'NPO',
    'Partnership',
)


def upgrade():
    op.create_table(
        'entity',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )

    with op.batch_alter_table('admin_chart_of_accounts', schema=None) as batch_op:
        batch_op.add_column(sa.Column('entity_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            'fk_admin_coa_entity_id', 'entity', ['entity_id'], ['id'], ondelete='CASCADE'
        )

    conn = op.get_bind()
    for name in ENTITY_NAMES:
        conn.execute(sa.text('INSERT INTO entity (name) VALUES (:name)'), {'name': name})

    private_id = conn.execute(
        sa.text("SELECT id FROM entity WHERE name = 'Private Company'")
    ).scalar()
    conn.execute(
        sa.text('UPDATE admin_chart_of_accounts SET entity_id = :eid WHERE entity_id IS NULL'),
        {'eid': private_id},
    )

    with op.batch_alter_table('admin_chart_of_accounts', schema=None) as batch_op:
        batch_op.alter_column('entity_id', existing_type=sa.Integer(), nullable=False)
        batch_op.drop_constraint('admin_chart_of_accounts_link_key', type_='unique')
        batch_op.create_unique_constraint('uq_admin_coa_entity_link', ['entity_id', 'link'])

    with op.batch_alter_table('company_settings', schema=None) as batch_op:
        batch_op.add_column(sa.Column('entity_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            'fk_company_settings_entity_id', 'entity', ['entity_id'], ['id'], ondelete='SET NULL'
        )


def downgrade():
    with op.batch_alter_table('company_settings', schema=None) as batch_op:
        batch_op.drop_constraint('fk_company_settings_entity_id', type_='foreignkey')
        batch_op.drop_column('entity_id')

    with op.batch_alter_table('admin_chart_of_accounts', schema=None) as batch_op:
        batch_op.drop_constraint('uq_admin_coa_entity_link', type_='unique')
        batch_op.create_unique_constraint('admin_chart_of_accounts_link_key', ['link'])
        batch_op.drop_constraint('fk_admin_coa_entity_id', type_='foreignkey')
        batch_op.drop_column('entity_id')

    op.drop_table('entity')
