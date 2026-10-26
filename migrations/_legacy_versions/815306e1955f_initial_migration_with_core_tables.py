"""Initial migration with core tables

Revision ID: 815306e1955f
Revises: 
Create Date: 2024-12-25 23:19:45.842987

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '815306e1955f'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create user table first
    op.create_table('user',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=64), nullable=False),
        sa.Column('email', sa.String(length=120), nullable=False),
        sa.Column('password_hash', sa.String(length=256), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('is_admin', sa.Boolean(), nullable=True),
        sa.Column('subscription_status', sa.String(length=20), nullable=True),
        sa.Column('mfa_secret', sa.String(length=32), nullable=True),
        sa.Column('mfa_enabled', sa.Boolean(), nullable=True),
        sa.Column('reset_token', sa.String(length=100), nullable=True),
        sa.Column('reset_token_expires', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username')
    )

    # Create account table
    op.create_table('account',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('link', sa.String(length=20), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('sub_category', sa.String(length=100), nullable=True),
        sa.Column('account_code', sa.String(length=20), nullable=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create company_settings table
    op.create_table('company_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('company_name', sa.String(length=200), nullable=False),
        sa.Column('registration_number', sa.String(length=50), nullable=True),
        sa.Column('tax_number', sa.String(length=50), nullable=True),
        sa.Column('vat_number', sa.String(length=50), nullable=True),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('financial_year_end', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Other tables that were in the downgrade
    op.create_table('admin_chart_of_accounts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('link', sa.String(length=20), nullable=False),
        sa.Column('code', sa.String(length=20), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('sub_category', sa.String(length=50), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('link')
    )
    op.drop_table('admin_chart_of_accounts_backup')
    with op.batch_alter_table('historical_data', schema=None) as batch_op:
        batch_op.create_foreign_key(None, 'account', ['account_id'], ['id'])
        batch_op.create_foreign_key(None, 'user', ['user_id'], ['id'])

    with op.batch_alter_table('keyword_rule', schema=None) as batch_op:
        batch_op.drop_column('user_id')

    with op.batch_alter_table('risk_assessment', schema=None) as batch_op:
        batch_op.add_column(sa.Column('created_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('updated_at', sa.DateTime(), nullable=True))
        batch_op.alter_column('assessment_date',
               existing_type=postgresql.TIMESTAMP(),
               nullable=True)
        batch_op.alter_column('risk_level',
               existing_type=sa.VARCHAR(length=32),
               type_=sa.String(length=20),
               existing_nullable=False)
        batch_op.alter_column('assessment_type',
               existing_type=sa.VARCHAR(length=64),
               type_=sa.String(length=50),
               existing_nullable=False)
        batch_op.create_foreign_key(None, 'user', ['user_id'], ['id'])

    with op.batch_alter_table('risk_indicator', schema=None) as batch_op:
        batch_op.alter_column('indicator_name',
               existing_type=sa.VARCHAR(length=64),
               type_=sa.String(length=100),
               existing_nullable=False)
        batch_op.alter_column('created_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=True)


    # ### end Alembic commands ###


def downgrade():
    # Drop tables in reverse order of creation (respecting foreign key constraints)
    op.drop_table('company_settings')
    op.drop_table('account')
    op.drop_table('admin_chart_of_accounts')
    op.drop_table('user')
    with op.batch_alter_table('risk_indicator', schema=None) as batch_op:
        batch_op.alter_column('created_at',
               existing_type=postgresql.TIMESTAMP(),
               nullable=False)
        batch_op.alter_column('indicator_name',
               existing_type=sa.String(length=100),
               type_=sa.VARCHAR(length=64),
               existing_nullable=False)

    with op.batch_alter_table('risk_assessment', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.alter_column('assessment_type',
               existing_type=sa.String(length=50),
               type_=sa.VARCHAR(length=64),
               existing_nullable=False)
        batch_op.alter_column('risk_level',
               existing_type=sa.String(length=20),
               type_=sa.VARCHAR(length=32),
               existing_nullable=False)
        batch_op.alter_column('assessment_date',
               existing_type=postgresql.TIMESTAMP(),
               nullable=False)
        batch_op.drop_column('updated_at')
        batch_op.drop_column('created_at')

    with op.batch_alter_table('keyword_rule', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.INTEGER(), autoincrement=False, nullable=False))

    with op.batch_alter_table('historical_data', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.drop_constraint(None, type_='foreignkey')

    op.create_table('admin_chart_of_accounts_backup',
    sa.Column('id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('account_code', sa.VARCHAR(length=20), autoincrement=False, nullable=True),
    sa.Column('name', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.Column('category', sa.VARCHAR(length=50), autoincrement=False, nullable=True),
    sa.Column('sub_category', sa.VARCHAR(length=50), autoincrement=False, nullable=True),
    sa.Column('description', sa.TEXT(), autoincrement=False, nullable=True),
    sa.Column('created_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True),
    sa.Column('updated_at', postgresql.TIMESTAMP(), autoincrement=False, nullable=True)
    )
    # ### end Alembic commands ###