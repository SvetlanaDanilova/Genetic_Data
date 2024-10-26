import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import gc
import utils
import os

def initialize_model(params):
    """Initializes a LightGBM model."""
    return lgb.LGBMRegressor(**params)

def train_and_save_model(model, X_batch, y_batch, X_val, y_val, model_name, is_initial_training):
    """
    Updates the LightGBM model with new batch data and evaluates it.

    Args:
        model: Initialized LightGBM model.
        X_batch (pd.DataFrame): Batch of training data.
        y_batch (pd.Series): Batch of training target.
        X_val (pd.DataFrame): Validation data.
        y_val (pd.Series): Validation target.
        model_name (str): Name for saving the model.
        is_initial_training (bool): Flag indicating if this is the first training iteration.

    Returns:
        float: MAE score for the validation data.
    """
    model.fit(
        X_batch, y_batch,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        verbose=10,
        early_stopping_rounds=50,
        init_model=model if not is_initial_training else None  # Load previous model if not initial training
    )
    
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f'Current MAE for {model_name}: {mae}')

    return mae

def data_generator(X, y, batch_size):
    """
    A generator that yields batches of data.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
        batch_size (int): Size of each batch.

    Yields:
        Tuple of (X_batch, y_batch)
    """
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        yield X[start:end], y[start:end]

def train(batch_size=100000000):
    print("Loading train data...")
    # Load data
    train_data = utils.preprocessing('train')
    
    # Prepare features and target variables
    X = train_data.drop(['Allele1', 'Allele2'], axis=1)
    y_allele1 = train_data['Allele1']
    y_allele2 = train_data['Allele2']

    print("Splitting data into training and validation sets...")
    X_train, X_val, y_allele1_train, y_allele1_val = train_test_split(X, y_allele1, test_size=0.2, random_state=42)
    _, _, y_allele2_train, y_allele2_val = train_test_split(X, y_allele2, test_size=0.2, random_state=42)

    # Model parameters
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbose': -1,
        'n_estimators': 1000
    }

    print("Initializing model for Allele1...")
    model_allele1 = initialize_model(params)
    is_initial_training_allele1 = True
    
    print("Training model for Allele1 in batches...")
    for X_batch, y_batch in data_generator(X_train, y_allele1_train, batch_size):
        mae_allele1 = train_and_save_model(
            model_allele1, X_batch, y_batch, X_val, y_allele1_val, 'model_allele1', is_initial_training_allele1
        )
        is_initial_training_allele1 = False

    # Save the model for Allele1 after training on all batches
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_allele1, 'models/model_allele1.pkl')
    print(f'Model for Allele1 saved as model_allele1.pkl')

    print("Initializing model for Allele2...")
    model_allele2 = initialize_model(params)
    is_initial_training_allele2 = True

    print("Training model for Allele2 in batches...")
    for X_batch, y_batch in data_generator(X_train, y_allele2_train, batch_size):
        mae_allele2 = train_and_save_model(
            model_allele2, X_batch, y_batch, X_val, y_allele2_val, 'model_allele2', is_initial_training_allele2
        )
        is_initial_training_allele2 = False

    # Save the model for Allele2 after training on all batches
    joblib.dump(model_allele2, 'models/model_allele2.pkl')
    print(f'Model for Allele2 saved as model_allele2.pkl')

    # Cleanup
    del X_train, X_val, y_allele1_train, y_allele1_val, y_allele2_train, y_allele2_val
    gc.collect()

    print(f'Model training complete. MAE for Allele1: {mae_allele1}, MAE for Allele2: {mae_allele2}')