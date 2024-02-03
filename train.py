import pandas as pd
import mlflow
from dotenv import load_dotenv
from mlflow.models import infer_signature
import argparse
import logging
import tensorflow
import tensorflow as tf
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from sklearn.metrics import precision_score
load_dotenv()


with mlflow.start_run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type = int)
    parser.add_argument('--drop_out', type = float)
    parser.add_argument("--batch_size", type = int)
    args = parser.parse_args()

    print(args)
    # Read data
    df = pd.read_csv('data/drug200.csv', sep=",")
    
    
    o_en = OrdinalEncoder(categories=[["LOW","NORMAL","HIGH"]])
    df['BP'] = o_en.fit_transform(df[['BP']])
    df['Cholesterol'] = o_en.fit_transform(df[['Cholesterol']])

    on_encode = OrdinalEncoder()
    df['Sex'] = on_encode.fit_transform(df[['Sex']])

    l_encode = LabelEncoder()
    df['Drug'] = l_encode.fit_transform(df[['Drug']])

    
    joblib.dump(o_en,"Ordinal_encode.pkl")
    mlflow.log_artifact("Ordinal_encode.pkl", artifact_path="artifact")

    
    joblib.dump(on_encode,"Onehot_encode.pkl")
    mlflow.log_artifact("Onehot_encode.pkl", artifact_path="artifact")


    joblib.dump(l_encode,"Label_encode.pkl")
    mlflow.log_artifact("Label_encode.pkl", artifact_path="artifact")


    y_data = df['Drug']
    y_data = to_categorical(y_data)
    

    X_train, X_test, y_train, y_test = train_test_split(df[df.columns[:-1]], y_data, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


    std_value = StandardScaler()
    std_value = std_value.fit(X_train)

    joblib.dump(std_value,"Std.pkl")
    mlflow.log_artifact("Std.pkl", artifact_path="artifact")


    X_train = std_value.transform(X_train)
    X_val = std_value.transform(X_val)


    batch_size = args.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
    train_dataset = train_dataset.shuffle(buffer_size = len(X_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val,y_val))
    val_dataset = val_dataset.shuffle(buffer_size = len(X_val)).batch(batch_size)


    model = Sequential(
        layers=[tensorflow.keras.layers.InputLayer(input_shape = (5,)),
        tensorflow.keras.layers.Dense(128),
        tensorflow.keras.layers.Dropout(args.drop_out),
        tensorflow.keras.layers.Dense(16),
        tensorflow.keras.layers.Dense(5, activation = 'softmax'),
    ])
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



    for epoch in range(args.num_epoch):

        # Training loop
        for batch_x, batch_y in train_dataset:
            model.train_on_batch(batch_x, batch_y)

        # Validation loop
        for val_batch_x, val_batch_y in val_dataset:
            model.test_on_batch(val_batch_x, val_batch_y)
        
        # Optionally, print or log training/validation metrics
        train_loss, train_accuracy = model.evaluate(train_dataset, verbose=0)
        val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)

        mlflow.log_metric("Train Accuracy", train_accuracy)
        mlflow.log_metric("Validation Accuracy", val_accuracy)
        print(f'Epoch {epoch + 1}/{args.num_epoch}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


    if tracking_url_type_store != "file":
        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow

        mlflow.tensorflow.log_model(
            lr, "model", registered_model_name = "ElasticnetWineModel"
        )
    else:
        mlflow.tensorflow.log_model(model, "model")


    predictions = model.predict(train_dataset)
    signature = infer_signature(train_dataset, predictions)

    
    X_test = std_value.transform(X_test)
    x = []
    for i in model.predict(X_test):
        x.append(np.argmax(i))

    y = []
    for j in y_test:
        y.append(np.argmax(j))
    

    precision_sc = precision_score(x, y, average='weighted')
    mlflow.log_metric("Test Precision", precision_sc)


