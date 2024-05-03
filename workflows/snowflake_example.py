# Flyte
import flytekit
from flytekit import Secret, task, workflow, ImageSpec

# Snowpark
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
from snowflake.snowpark.functions import col, lit
from snowflake.snowpark.types import FloatType

# Snowpark ML
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.preprocessing import OrdinalEncoder, OneHotEncoder
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.model_selection import GridSearchCV
from snowflake.ml.modeling.metrics import mean_absolute_percentage_error
from snowflake.ml.registry import Registry

# Snowflake Task API
from snowflake.core import Root
from snowflake.core.database import Database
from snowflake.core.schema import Schema
from snowflake.core.warehouse import Warehouse
from snowflake.core import Root

# Other
import numpy as np

# Custom Image for Tasks that includes Snowpark ML library
snowpark_image_spec = ImageSpec(
    base_image="ghcr.io/flyteorg/flytekit:py3.10-latest",
    packages=["snowflake-ml-python==1.3.0", "snowflake-snowpark-python[pandas]==1.14.0","scikit-learn==1.3.0","xgboost==1.7.3","snowflake==0.5.1"],
    python_version="3.11",
    env={"Debug": "True"},
    registry="localhost:30000", 
)

# Snowflake Credentials
SECRET_GROUP = "snowflake-creds"
SF_ACCOUNT = "SF_ACCOUNT"
SF_USER = "SF_USER"
SF_PASSWORD = "SF_PASSWORD"

# Snowflake Context
SF_ROLE = "ACCOUNTADMIN" # --> must be a role that can create a new Database
SF_DATABASE = "FLYTE_DEMO_DB"
SF_SCHEMA = "FLYTE_DEMO_SCHEMA"
SF_WAREHOUSE = "FLYTE_WH"

def connect_to_snowflake(S_SF_ACCOUNT, S_SF_USER, S_SF_PASSWORD, SF_ROLE, SF_DATABASE, SF_SCHEMA, SF_WAREHOUSE):
    snowflake_connection_cfg = {
        "ACCOUNT": S_SF_ACCOUNT,
        "USER": S_SF_USER,
        "PASSWORD": S_SF_PASSWORD,
        "ROLE": SF_ROLE,
        "DATABASE": SF_DATABASE,
        "SCHEMA": SF_SCHEMA,
        "WAREHOUSE": SF_WAREHOUSE
    }

    # Creating Snowpark Session
    session = Session.builder.configs(snowflake_connection_cfg).create()
    return session

@task(
    container_image=snowpark_image_spec,
    secret_requests=[
        Secret(key=SF_ACCOUNT, group=SECRET_GROUP),
        Secret(key=SF_USER, group=SECRET_GROUP),
        Secret(key=SF_PASSWORD, group=SECRET_GROUP)
    ]
)
def setup_environment() -> str:
    # Retrieve secrets for Snowflake Access
    context = flytekit.current_context()
    S_SF_ACCOUNT = context.secrets.get(SECRET_GROUP, SF_ACCOUNT)
    S_SF_USER = context.secrets.get(SECRET_GROUP, SF_USER)
    S_SF_PASSWORD = context.secrets.get(SECRET_GROUP, SF_PASSWORD)
    session = connect_to_snowflake(S_SF_ACCOUNT, S_SF_USER, S_SF_PASSWORD, SF_ROLE, SF_DATABASE, SF_SCHEMA, SF_WAREHOUSE)
   
    # Create Database & Schema
    root = Root(session)
    ml_demo_db = Database(name="FLYTE_DEMO_DB")
    ml_demo_db = root.databases.create(ml_demo_db, mode='or_replace')
    ml_demo_schema = Schema(name="FLYTE_DEMO_SCHEMA")
    ml_demo_schema = ml_demo_db.schemas.create(ml_demo_schema, mode='or_replace')

    # Create warehouse
    ml_wh = Warehouse(name="FLYTE_WH", warehouse_size="XSMALL", auto_suspend=600, auto_resume='true')
    warehouses = root.warehouses
    ml_wh = warehouses.create(ml_wh)
    return 'Successfully created Demo Environment!'

@task(
    container_image=snowpark_image_spec,
    secret_requests=[
        Secret(key=SF_ACCOUNT, group=SECRET_GROUP),
        Secret(key=SF_USER, group=SECRET_GROUP),
        Secret(key=SF_PASSWORD, group=SECRET_GROUP)
    ]
)
def generate_data() -> str:
    # Retrieve secrets for Snowflake Access
    context = flytekit.current_context()
    S_SF_ACCOUNT = context.secrets.get(SECRET_GROUP, SF_ACCOUNT)
    S_SF_USER = context.secrets.get(SECRET_GROUP, SF_USER)
    S_SF_PASSWORD = context.secrets.get(SECRET_GROUP, SF_PASSWORD)
    session = connect_to_snowflake(S_SF_ACCOUNT, S_SF_USER, S_SF_PASSWORD, SF_ROLE, SF_DATABASE, SF_SCHEMA, SF_WAREHOUSE)

    num_rows = 1000
    random_id = F.uniform(0,5, F.random()).as_('RAND_ID')
    email = F.concat(F.call_builtin('RANDSTR', 10, F.random()), F.lit('@'), F.call_builtin('RANDSTR', 5, F.random()), F.lit('.com')).as_('EMAIL')
    gender = F.when(F.uniform(1,10,F.random())<=7, F.lit('MALE')).otherwise('FEMALE').as_('GENDER')
    yearly_spent = (F.round(F.uniform(100,75000,F.random()) / 100, 2)).as_('YEARLY_SPENT')
    membership_status = F.when(col('YEARLY_SPENT') < 150, F.lit('BASIC'))\
        .when(col('YEARLY_SPENT') < 250, F.lit('BRONZE'))\
            .when(col('YEARLY_SPENT') < 350, F.lit('SILVER'))\
                .when(col('YEARLY_SPENT') < 550, F.lit('GOLD'))\
                    .when(col('YEARLY_SPENT') < 650, F.lit('PLATIN'))\
                        .when(col('YEARLY_SPENT') >= 650, F.lit('DIAMOND')).as_('MEMBERSHIP_STATUS')
    membership_length = (col('YEARLY_SPENT') / 100 + F.uniform(0,100, F.random())).cast(FloatType()).as_('MEMBERSHIP_LENGTH_DAYS')
    avg_session_length = (col('YEARLY_SPENT') / 100 + F.uniform(0,5, F.random())).cast(FloatType()).as_('AVG_SESSION_LENGTH_MIN')
    avg_time_on_app = (col('YEARLY_SPENT') / 100 + F.uniform(1,7, F.random())).cast(FloatType()).as_('AVG_TIME_ON_APP_MIN')
    avg_time_on_website = (col('YEARLY_SPENT') / 100 + F.uniform(3,7, F.random())).cast(FloatType()).as_('AVG_TIME_ON_WEBSITE_MIN')

    df = session.generator(random_id, email, yearly_spent, gender,membership_status, membership_length, avg_session_length, avg_time_on_app, avg_time_on_website, rowcount=num_rows)

    # Add some missing data
    df = df.with_column('MEMBERSHIP_STATUS', F.when(col('RAND_ID') == 0, None).otherwise(col('MEMBERSHIP_STATUS')))
    df = df.with_column('MEMBERSHIP_LENGTH_DAYS', F.when(col('RAND_ID') == 1, None).otherwise(col('MEMBERSHIP_LENGTH_DAYS')))
    df = df.with_column('AVG_SESSION_LENGTH_MIN', F.when(col('RAND_ID') == 2, None).otherwise(col('AVG_SESSION_LENGTH_MIN')))
    df = df.with_column('AVG_TIME_ON_APP_MIN', F.when(col('RAND_ID') == 3, None).otherwise(col('AVG_TIME_ON_APP_MIN')))
    df = df.with_column('AVG_TIME_ON_WEBSITE_MIN', F.when(col('RAND_ID') == 4, None).otherwise(col('AVG_TIME_ON_WEBSITE_MIN')))
    df = df.drop('RAND_ID')
    df.write.save_as_table('ECOMMERCE_DATA', mode='overwrite')
    return f'Successfully created dataset with {num_rows} rows.'

@task(
    container_image=snowpark_image_spec,
    secret_requests=[
        Secret(key=SF_ACCOUNT, group=SECRET_GROUP),
        Secret(key=SF_USER, group=SECRET_GROUP),
        Secret(key=SF_PASSWORD, group=SECRET_GROUP)
    ]
)
def preprocess_data() -> str:
    # Retrieve secrets for Snowflake Access
    context = flytekit.current_context()
    S_SF_ACCOUNT = context.secrets.get(SECRET_GROUP, SF_ACCOUNT)
    S_SF_USER = context.secrets.get(SECRET_GROUP, SF_USER)
    S_SF_PASSWORD = context.secrets.get(SECRET_GROUP, SF_PASSWORD)
    session = connect_to_snowflake(S_SF_ACCOUNT, S_SF_USER, S_SF_PASSWORD, SF_ROLE, SF_DATABASE, SF_SCHEMA, SF_WAREHOUSE)
    
    # Create a Snowpark DataFrame
    df = session.table('ECOMMERCE_DATA')

    # Split the data into train and test sets
    train_df, test_df = df.random_split(weights=[0.9, 0.1], seed=0)
    train_df.count(), test_df.count()

    # Define sklearn-like Imputers and Encoders
    si_numeric =  SimpleImputer(
        input_cols=['MEMBERSHIP_LENGTH_DAYS','AVG_SESSION_LENGTH_MIN','AVG_TIME_ON_APP_MIN','AVG_TIME_ON_WEBSITE_MIN'], 
        output_cols=['MEMBERSHIP_LENGTH_DAYS_IMP','AVG_SESSION_LENGTH_MIN_IMP','AVG_TIME_ON_APP_MIN_IMP','AVG_TIME_ON_WEBSITE_MIN_IMP'],
        strategy='mean',
        drop_input_cols=True
    )

    si_cat =  SimpleImputer(
        input_cols=['MEMBERSHIP_STATUS'], 
        output_cols=['MEMBERSHIP_STATUS_IMP'],
        strategy='most_frequent',
        drop_input_cols=True
    )

    # Define sklearn-like Encoders
    categories = {
        "MEMBERSHIP_STATUS_IMP": np.array(["BASIC", "BRONZE", "SILVER", "GOLD", "PLATIN", "DIAMOND"]),
    }
    oe_categorical = OrdinalEncoder(
        input_cols=["MEMBERSHIP_STATUS_IMP"], 
        output_cols=["MEMBERSHIP_STATUS_IMP_OE"], 
        categories=categories,
        drop_input_cols=True
    )

    ohe_categorical = OneHotEncoder(
        input_cols=["GENDER"], 
        output_cols=["GENDER_OHE"],
        drop_input_cols=True
    )

    # Build the pipeline
    preprocessing_pipeline = Pipeline(
        steps=[
            ('SI_CAT',si_cat),
            ("SI_NUMERIC",si_numeric),
            ("OE_CATEGORICAL",oe_categorical),
            ("OHE_CATEGORICAL",ohe_categorical),
        ]
    )

    # Fit the pipeline and transform data
    transformed_train_df = preprocessing_pipeline.fit(train_df).transform(train_df)
    transformed_train_df.write.save_as_table('ECOMMERCE_DATA_TRAIN_PREPARED', mode='overwrite')
    num_rows_train = session.table('ECOMMERCE_DATA_TRAIN_PREPARED').count()
    transformed_test_df = preprocessing_pipeline.transform(test_df)
    transformed_test_df.write.save_as_table('ECOMMERCE_DATA_TEST_PREPARED', mode='overwrite')
    num_rows_test = session.table('ECOMMERCE_DATA_TEST_PREPARED').count()
    return f"Prepared new training table with {num_rows_train} customers and new test table with {num_rows_test} customers."


@task(
    container_image=snowpark_image_spec,
    secret_requests=[
        Secret(key=SF_ACCOUNT, group=SECRET_GROUP),
        Secret(key=SF_USER, group=SECRET_GROUP),
        Secret(key=SF_PASSWORD, group=SECRET_GROUP)
    ]
)
def train_model() -> str:
    # Retrieve secrets for Snowflake Access
    context = flytekit.current_context()
    S_SF_ACCOUNT = context.secrets.get(SECRET_GROUP, SF_ACCOUNT)
    S_SF_USER = context.secrets.get(SECRET_GROUP, SF_USER)
    S_SF_PASSWORD = context.secrets.get(SECRET_GROUP, SF_PASSWORD)
    session = connect_to_snowflake(S_SF_ACCOUNT, S_SF_USER, S_SF_PASSWORD, SF_ROLE, SF_DATABASE, SF_SCHEMA, SF_WAREHOUSE)
    reg = Registry(session=session, database_name=SF_DATABASE, schema_name=SF_SCHEMA)
    df = session.table('ECOMMERCE_DATA_TRAIN_PREPARED')
    feature_cols = [
        'GENDER_OHE_FEMALE',
        'GENDER_OHE_MALE',
        'MEMBERSHIP_STATUS_IMP_OE',
        'MEMBERSHIP_LENGTH_DAYS_IMP',
        'AVG_SESSION_LENGTH_MIN_IMP',
        'AVG_TIME_ON_APP_MIN_IMP',
        'AVG_TIME_ON_WEBSITE_MIN_IMP'
    ]
    label_cols = ['YEARLY_SPENT']
    output_cols = ['YEARLY_SPENT_PREDICTION']

    # Define parameter tuning
    grid_search = GridSearchCV(
        estimator=XGBRegressor(),
        param_grid={
            "n_estimators":[100, 200, 300, 400],
            "learning_rate":[0.1, 0.2, 0.3],
        },
        n_jobs = -1,
        scoring="neg_mean_absolute_percentage_error",
        input_cols=feature_cols,
        label_cols=label_cols,
        output_cols=output_cols
    )

    # Train
    grid_search.fit(df)

    # Get latest model versiont
    try:
        model_versions = reg.get_model("ECOMMERCE_SPENT_MODEL").show_versions()
        idx = model_versions['created_on'].idxmax()
        most_recent_version = model_versions.loc[idx]
        new_version = 'V'+str(int(most_recent_version['name'][1:])+1)
    except:
        # If no version exists, create version 0
        new_version = 'V0'

    # Register new model version
    registered_model = reg.log_model(
        grid_search,
        model_name="ECOMMERCE_SPENT_MODEL",
        version_name=new_version,
        comment="Model trained using GridsearchCV in Snowpark to predict customer's yearly spending.",
        sample_input_data=df.select(feature_cols).limit(100)
    )
    return f"Registered new model with version: {new_version}"

@workflow
def full_pipeline() -> str:
    print('Creating Environment before Running ML.')
    setup_environment_result = setup_environment()
    generate_data_result = generate_data()
    preprocess_data_result = preprocess_data()
    train_model_result = train_model()
    setup_environment_result >> generate_data_result
    generate_data_result >> preprocess_data_result
    preprocess_data_result >> train_model_result
    return train_model_result

@workflow
def training_only_pipeline() -> str:
    print('Assuming existing Environment. Running ML.')
    preprocess_data_result = preprocess_data()
    train_model_result = train_model()
    preprocess_data_result >> train_model_result
    return train_model_result
