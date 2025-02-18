import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error


# a function to convert the various currencies prices into standard USD prices
def convert_price(price):
    # Check if the price is a valid string with at least 4 characters (currency code + space + value)
    if isinstance(price, str) and len(price) > 3:
        # Extract the currency code (first 3 characters)
        c = price[:3]
        # Extract the price string (everything after the currency code) and remove commas
        price_str = price[4:].replace(',', '').strip()

        # Check if price_str is not empty and is numeric
        if price_str and price_str.replace('.', '', 1).isdigit():
            p = float(price_str)  # Convert to float

            # Currency conversion
            conversion_rates = {
                'AED': 0.27,
                'KWD': 3.25,
                'OMR': 2.60,
                'BHD': 2.65,
                'QAR': 0.27,
                'SAR': 0.27,
                'EGP': 0.02
            }

            # Convert to USD if currency is known
            return p * conversion_rates[c]  # Default to 1 if currency code not found

    # If format is incorrect or conversion fails, return NaN
    return np.nan


def standardize_price(df):
    df['price'] = df['price'].apply(convert_price)


def convert_to_numeric(df):
    standardize_price(df)
    standardize_seat(df)
    # convert all numeric features to approbriate data type with converting any illegal value to "NaN"
    for feature in ["engine_capacity", "cylinder", "horse_power", "top_speed", "seats", "price"]:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')


def convert_seat(seat_value):
    if seat_value.isdigit() or '.' in seat_value:  # if the value is numeric
        return float(seat_value)
    elif 'Seater' in seat_value:  # if the value like '5 Seater'
        return float(seat_value.split(' ')[0])
    else:  # for illegal values
        return np.nan


def standardize_seat(df):
    df['seats'] = df['seats'].apply(convert_seat)


def fill_missing_values(df):
    # if the feature is numeric, then replace the missing value with mean, else replace it with mode
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])


def encode_categorical_values(df):
    # Extract categorical columns from the dataframe
    # Here we extract the columns with object datatype as they are the categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)

    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    # Create a DataFrame with the one-hot encoded columns
    # We use get_feature_names_out() to get the column names for the encoded data
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded = pd.concat([df, one_hot_df], axis=1)

    # Drop the original categorical columns
    df_encoded = df_encoded.drop(categorical_columns, axis=1)  # axis = 1 for columns
    return df_encoded


def label_encoding(df):
    # initialize label encoder
    label_encoder = LabelEncoder()
    # apply the label encoder to categorical data
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = label_encoder.fit_transform(df[col])


def z_score_normalization(df):
    # define the scaler
    scaler = StandardScaler()
    # remove the price feature temporarily to avoid normalizing it
    new_df = df.drop(columns=['price'])
    # fit and transform features
    standardized_data = scaler.fit_transform(new_df)
    # convert the normalized numpy array to a data frame
    standardized_df = pd.DataFrame(standardized_data, columns=new_df.columns)
    # put the price column back in the data frame
    standardized_df['price'] = df['price']
    # return
    return standardized_df


def MinMax_normalization(df):
    # define the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # remove the price feature temporarily to avoid normalizing it
    new_df = df.drop(columns=['price'])
    # fit and transform features
    standardized_data = scaler.fit_transform(new_df)
    # convert the normalized numpy array to a data frame
    standardized_df = pd.DataFrame(standardized_data, columns=new_df.columns)
    # put the price column back in the data frame
    standardized_df['price'] = df['price']
    # return
    return standardized_df


def randomization_dataset(df):
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def splitDataSet(df):
    # Calculate the split indices
    n = len(df)
    split_60 = int(n * 0.6)
    split_20 = int(n * 0.2)

    # Split the DataFrame into 60%, 20%, and 20% subsets
    df_Training = df.iloc[:split_60]
    df_Validation = df.iloc[split_60:split_60 + split_20]
    df_Testing = df.iloc[split_60 + split_20:]

    return [df_Training, df_Validation, df_Testing]


# -------------------------------Building Regression Models------------------------------#
# this function adds a new column of ones to the data set to manipulate the bias weight
def add_bias_vector(df):
    # Ensure X is a 2D NumPy array
    df_array = np.array(df)
    # add column of ones
    df_array = np.hstack([np.ones((df_array.shape[0], 1)), df_array])
    return df_array


def Linear_Regression(X, Y):
    # Reshape Y to be a column vector
    Y = np.array(Y).reshape(-1, 1)

    # Compute (X^T X)^(-1) X^T Y
    # X matrix transpose
    XT = X.T
    XTX = np.dot(XT, X)
    XTX_inv = np.linalg.inv(XTX)
    XTY = np.dot(XT, Y)

    # Compute the weights
    W = np.dot(XTX_inv, XTY)

    return W


# this function takes the features matrix X and model weights w and the actual output vector Y and returns the MSE of the model
def find_MSE(X, W, Y_actual):
    # Reshape Y to be a column vector
    Y_actual = np.array(Y_actual).reshape(-1, 1)
    # Compute Y_predicted as a column vector of predictions
    Y_predicted = np.dot(X, W).reshape(-1, 1)

    return mean_squared_error(Y_actual, Y_predicted)


def findR2(X, W, Y_actual):
    # Reshape Y to be a column vector
    Y_actual = np.array(Y_actual).reshape(-1, 1)
    # Compute Y_predicted as a column vector of predictions
    Y_predicted = np.dot(X, W).reshape(-1, 1)

    return r2_score(Y_actual, Y_predicted)


def find_MAE(X, W, Y_actual):
    # Reshape Y to be a column vector
    Y_actual = np.array(Y_actual).reshape(-1, 1)
    # Compute Y_predicted as a column vector of predictions
    Y_predicted = np.dot(X, W).reshape(-1, 1)

    return mean_absolute_error(Y_actual, Y_predicted)


def lasso_regression(data, target, alpha):
    # Fit the model
    lassoreg = Lasso(alpha=alpha, max_iter=10000)
    lassoreg.fit(data, target)
    return lassoreg.coef_


def ridge_regression(X, y, alpha):
    m, n = X.shape
    I = np.eye(n)
    I[0, 0] = 0
    theta = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
    return theta


def gradient_descent_linear_regression(X, Y, learning_rate):
    # Reshape Y to be a column vector
    Y = np.array(Y).reshape(-1, 1)
    # define the random weights
    W = np.random.uniform(-5e4, 5e4, size=(
        X.shape[1], 1))  # X.shape[1] is the number of columns in X, while X.shape[0] is the number of rows
    n = X.shape[0]
    iterations = 1000
    for i in range(iterations):
        # Compute predicted values
        y_predicted = X @ W
        # Compute the gradient
        gradient = (2 / n) * X.T @ (y_predicted - Y)
        gradient = np.clip(gradient, -1e7, 1e7)  # Limit gradient values
        # Update weights
        W = W - learning_rate * gradient

    return W


def polynomial_regression(X, y, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    features_poly = poly.fit_transform(X)
    q_model = LinearRegression()
    q_model.fit(features_poly, y)
    predictions = q_model.predict(features_poly)

    R2 = r2_score(y, predictions)
    MSE = mean_squared_error(y, predictions)
    MAE = mean_absolute_error(y, predictions)

    return [MSE, MAE, R2]


def gaussian_rbf(x, center, sigma):
     return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

def build_design_matrix(X,sigma):
    # Define the number of centers
    n_centers = 10

    # Use k-means to find cluster centers
    kmeans = KMeans(n_clusters=n_centers, random_state=42)
    kmeans.fit(features_matrix)

    # Centers are the cluster centroids
    centers = kmeans.cluster_centers_

    design_matrix = np.zeros((X.shape[0], len(centers)))
    for i, center in enumerate(centers):
        for j, x in enumerate(X):
            design_matrix[j, i] = gaussian_rbf(x, center, sigma)

    return design_matrix


def weights_rbf(X, y, sigma):

    phi = build_design_matrix(X,sigma)

    weights = np.linalg.lstsq(phi, y, rcond=None)[0]

    return phi, weights


def models_as_dictionary(X, Y, learning_rate, alpha, alpha2, sigma):
    # define a dictionary that stores weights of each model
    models = {}
    # linear regression using closed form solution
    models['linear_regression_closed_form'] = Linear_Regression(X, Y)
    models['linear_regression_gradient_descent'] = gradient_descent_linear_regression(X, Y, learning_rate)
    models['LASSO_regression'] = lasso_regression(X, Y, alpha)
    models['Ridge_regression'] = ridge_regression(X, Y, alpha2)
    phi, W_RBF = weights_rbf(X,Y, sigma)
    models['Gaussian_regression'] = W_RBF

    return models


def model_selection(models, set_features, set_output, polynomial_degree, sigma):
    features_matrix_validation = add_bias_vector(set_features)
    # define a dictionary contains the 3 metrics for each mode
    metrics = {}
    for key, value in models.items():
        if key != 'Gaussian_regression':
            MSE = find_MSE(features_matrix_validation, value, set_output)
            MAE = find_MAE(features_matrix_validation, value, set_output)
            R2 = findR2(features_matrix_validation, value, set_output)
            metrics[key] = [MSE, MAE, R2]

    # adding the polynomial regression
    metrics['polynomial_regression'] = polynomial_regression(set_features, set_output, polynomial_degree)

    # adding the rbf model
    phi_validation = build_design_matrix(features_matrix_validation, sigma)
    model_rbf = phi_validation @ models['Gaussian_regression']
    metrics['Gaussian_regression']= [mean_squared_error(set_output, model_rbf),
                                     mean_absolute_error(set_output, model_rbf),
                                     r2_score(set_output, model_rbf)]

    # print results
    for key, value in metrics.items():
        print("\nfor ", key)
        print("MSE is: ", value[0], ",MAE is: ", value[1], ",R2 is: ", value[2])



def forward_feature_selection(X_training, Y_training, X_validation, Y_validation, model_type='none'):
    chosen_features = []  # These are the best features for the model
    remaining_features = list(X_training.columns)
    max_R2 = 0

    while remaining_features:
        R2_values = {}
        for feature in remaining_features:
            # building the model
            X_train_model = X_training.loc[:, chosen_features + [feature]]
            model_weights = Linear_Regression(X_train_model, Y_training)

            # Compute the MSE on the validation set
            model_R2 = findR2(X_validation.loc[:, chosen_features + [feature]], model_weights, Y_validation)
            R2_values[feature] = model_R2

        # Choosing the feature that results in the minimum MSE for the model
        max_R2_feature = max(R2_values, key=R2_values.get)
        new_max_R2 = R2_values[max_R2_feature]

        # If the new MSE is better than the current minimum MSE, update the chosen features
        if new_max_R2 > max_R2:
            max_R2 = new_max_R2
            chosen_features.append(max_R2_feature)
            remaining_features.remove(max_R2_feature)
        else:
            break  # Stop if no feature improves the model

    return chosen_features


def grid_search_for_L12(x_vald, y_vald):
    lasso = Lasso(max_iter=10000)
    lasso_params = {"alpha": np.logspace(-3, 3, 50)}
    GS = GridSearchCV(
        estimator=lasso,
        param_grid=lasso_params,
        scoring=["r2", "neg_root_mean_squared_error"],
        refit="r2",
        cv=5,
        verbose=0
    )

    GS.fit(x_vald, y_vald)

    ridge = Ridge(max_iter=10000)

    ridge_params = {"alpha": np.logspace(-3, 3, 50)}  # Test alpha values from 0.001 to 1000

    # Set up GridSearchCV
    ridge_gs = GridSearchCV(
        estimator=ridge,
        param_grid=ridge_params,
        scoring=["r2", "neg_root_mean_squared_error"],
        refit="r2",
        cv=5,
        verbose=0
    )

    ridge_gs.fit(x_vald, y_vald)

    # Output best lambda and alpha and its performance
    # print("Best Ridge Alpha:", ridge_gs.best_params_["alpha"])
    # print("Best Ridge  R2 Score:", ridge_gs.best_score_)
    # print("Best lasso (Lambda):", GS.best_params_["alpha"])
    # print("Best lasso R2 Score:", GS.best_score_)

    return [GS.best_params_["alpha"], ridge_gs.best_params_["alpha"]]


def learning_rate_tuning(X, Y):
    learning_rate = np.logspace(-2, 2, 30)
    n = len(learning_rate)
    dict = {}
    for i in range(n):
        W = gradient_descent_linear_regression(X, Y, learning_rate[i])
        Y_predicted = np.dot(X, W).reshape(-1, 1)
        r2 = r2_score(Y,Y_predicted)
        dict[learning_rate[i]] = r2

    #print(max(dict.values()))
    max_key = max(dict, key=dict.get)
    #print(max_key)
    return max_key



def polynomial_regression_tuning(X, y):
    degree = np.arange(10)
    n = len(degree)
    dict = {}
    for i in range(n):
        [r,r,r2] = polynomial_regression(X, y, degree[i])
        dict[degree[i]] = r2

    max_key = max(dict, key=dict.get)
    return max_key


def rbf_tuning(X,y):
    sigma = np.logspace(-2, 2, 30)
    n = len(sigma)
    dict = {}
    for i in range(n):
        phi,W = weights_rbf(X, y,sigma[i])
        Y_predicted = np.dot(phi, W).reshape(-1, 1)
        r2 = r2_score(y, Y_predicted)
        dict[sigma[i]] = r2

    max_key = max(dict, key=dict.get)
    return max_key


def plot_feature_importances(df, coefficients):
    feature_names = df.columns
    feature_importances = np.abs(coefficients)

    # Create a DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title("Feature importances")
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--')
    plt.show()


def plot_error_distribution(y_actual, y_predicted):
    # Calculate residuals
    residuals = y_actual - y_predicted

    # Plot the histogram of residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=10, edgecolor='black', alpha=0.7, density=True)
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals (y_actual - y_predicted)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def plot_actual_vs_predicted_values(y_actual, y_predicted):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_predicted, color='blue', label='Predictions')

    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color='red', linestyle='--',
             label='Ideal Line')

    plt.title("Model Predictions vs. Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    # store the data set as a data frame
    df = pd.read_csv('cars.csv')
    # convert numeric data to numeric type.
    convert_to_numeric(df)
    # filling missing values
    fill_missing_values(df)
    # encode categorical data
    # df_encoded = encode_categorical_values(df)
    label_encoding(df)
    # print(df['brand'].value_counts())
    #print(df.info())
    df_normalized = MinMax_normalization(df)

    df_random = randomization_dataset(df_normalized)

    [df_Training, df_Validation, df_Testing] = splitDataSet(df_random)

    df_without_price = df_Training.drop(columns=['price'])

    df_validation_without_price = df_Validation.drop(columns=['price'])

    df_test_without_price = df_Testing.drop(columns=['price'])

    y_training = df_Training['price']
    y_validation = df_Validation['price']
    y_testing = df_Testing['price']

    features =  forward_feature_selection(df_without_price, y_training, df_validation_without_price, y_validation, model_type='none')
    df_without_price = df_without_price[features]
    features_matrix = add_bias_vector(df_without_price)

    df_validation_without_price = df_validation_without_price[features]
    features_matrix_validation = add_bias_vector(df_validation_without_price)

    df_test_without_price = df_test_without_price[features]
    features_matrix_test = add_bias_vector(df_test_without_price)

    learning_rate = learning_rate_tuning(features_matrix_validation, y_validation)
    [alpha, alpha2] = grid_search_for_L12(features_matrix_validation, y_validation)
    sigma = rbf_tuning(features_matrix_validation, y_validation)
    models = models_as_dictionary(features_matrix, df_Training['price'], learning_rate, alpha, alpha2, sigma)
    #degree = polynomial_regression_tuning(features_matrix_validation, y_validation) == 9
    print("Testing the model on the validation set:\n")
    model_selection(models, df_validation_without_price, df_Validation['price'], 9, sigma)

    print("\nTesting the model on the testing set:\n")
    model_selection(models, df_test_without_price, df_Testing['price'], 9, sigma)

    # plot_feature_importances(df_without_price, Linear_Regression(features_matrix, df_Training['price'])[1:].ravel())

    # W = lasso_regression(features_matrix, df_Training['price'], alpha)
    # # Reshape Y to be a column vector
    # Y_actual = np.array(df_Training['price']).reshape(-1, 1)
    # # Compute Y_predicted as a column vector of predictions
    # Y_predicted = np.dot(features_matrix, W).reshape(-1, 1)
    # plot_error_distribution(Y_actual, Y_predicted)
    # plot_actual_vs_predicted_values(Y_actual, Y_predicted)
    # plot_feature_importances(df_without_price, lasso_regression(features_matrix, df_Training['price'],alpha)[1:].ravel())