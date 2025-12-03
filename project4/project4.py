import numpy as np
from astropy import units as u
from mw_plot import MWFaceOn, MWSkyMap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def reproduce_milky_way(centre=(266.41683 * u.deg, -29.00781 * u.deg), radius=(90 * u.deg, 45 * u.deg)): 

    """
    Reproduces the Milky Way face-on map with the given center coordinates.

    params:
     centre: tuple
         A tuple representing the center coordinates in degrees (longitude, latitude).
    radius: tuple
            A tuple representing the radius in degrees (longitude radius, latitude radius).
    """  

    # test
    assert isinstance(centre, tuple), "Centre must be a tuple"
    assert len(centre) == 2, "Centre must have two elements (longitude, latitude)"
    assert all(isinstance(c, u.Quantity) for c in centre), "Centre elements must be astropy Quantity"
    assert isinstance(radius, tuple), "Radius must be a tuple"
    assert len(radius) == 2, "Radius must have two elements (longitude radius, latitude radius)"
    assert all(isinstance(r, u.Quantity) for r in radius), "Radius elements must be astropy Quantity"
 

    mw1 = MWSkyMap(
        projection="aitoff",
        center=centre,
        radius=radius,    
        background="Mellinger color optical survey",
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="aitoff")

    mw1.transform(ax)
    mw1.savefig("galaxy.png")
    plt.show()


def sky_map_view(
    centre=(-93.58317 * u.deg, -29.00781 * u.deg),
    radius=(30 * u.deg, 30 * u.deg),
    figure_title="Milky Way Face-On Map",

):
    
    """
        uses Skymap to plot the milky way with given center and radius.
    """
    mw1 = MWSkyMap(
        center=centre,
        radius=radius,
        background="Mellinger color optical survey",
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(figure_title)
    mw1.transform(ax)
    mw1.savefig(f'galaxy_skymap_radius_{radius[0]}_{radius[1]}_centre_{centre[0]}_{centre[1]}.png')
    plt.show()
    return fig




def plt2rgbarr(fig):
    """
    A function to transform a matplotlib to a 3d rgb np.array 

    Input
    -----
    fig: matplotlib.figure.Figure
        The plot that we want to encode.        

    Output
    ------
    np.array(ndim, ndim, 3): A 3d map of each pixel in a rgb encoding (the three dimensions are x, y, and rgb)
    
    """
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr[:, :, :3]



def kmeans_rgb_clusters(rgb_img, k=3, random_state=0):
    """
    Apply K-means clustering to the RGB image to cluster pixels into k clusters.
    Parameters
    ----------
    rgb_img : np.ndarray
        Input RGB image as a 3D numpy array (height, width, 3).
    k : int
        Number of clusters.
    random_state : int
        Random state for reproducibility.
    Returns
    -------
    kmeans : KMeans
        Trained KMeans model.
    label_img : np.ndarray
        2D array of cluster labels for each pixel.

    """

    # Reshape image to a 2D array of pixels
    h, w, _ = rgb_img.shape

    pixels = rgb_img.reshape(-1, 3).astype(np.float32) / 255.0

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(pixels)         
    # Reshape labels back to image dimensions
    label_img = labels.reshape(h, w)

    return kmeans, label_img

def categorize_pixels_brightness(rgb_array, categories):
    """
    Categorizes each pixel in the rgb_array based on the provided categories.
    """

    height, width, _ = rgb_array.shape
    categorized_array = np.empty((height, width), dtype=object)

    for i in range(height):
        for j in range(width):
            r, g, b = rgb_array[i, j]
            pixel_sum = r + g + b

            for category, props in categories.items():
                low, high = props["range"]
                if low <= pixel_sum <= high:
                    categorized_array[i, j] = category
                    break

    return categorized_array


def categorize_pixels_color(rgb_array, grey_threshold=20):
    """
    Categorize each pixel as 'red', 'green', 'blue', or 'grey'
    based on its RGB values.
    """
    height, width, _ = rgb_array.shape
    categorized_array = np.empty((height, width), dtype=object)

    for i in range(height):
        for j in range(width):
            r, g, b = rgb_array[i, j]

            # from RGB to Grey
            if (abs(int(r) - int(g)) < grey_threshold and
                abs(int(g) - int(b)) < grey_threshold and
                abs(int(r) - int(b)) < grey_threshold):
                categorized_array[i, j] = "grey"
            else:
                # otherwise categorize based on dominant color
                if r >= g and r >= b:
                    categorized_array[i, j] = "red"
                elif g >= r and g >= b:
                    categorized_array[i, j] = "green"
                else:
                    categorized_array[i, j] = "blue"

    return categorized_array


def overlay_clusters_on_image(rgb_img, label_img, alpha=0.4, title=None):
    """
    Overlay a cluster label map on top of the original RGB image.

    rgb_img  : np.ndarray, shape (H, W, 3)  
    label_img: np.ndarray, shape (H, W)      
    alpha    : float                       
    title    : str or None 
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_img)   
    plt.imshow(label_img, cmap="tab10", alpha=alpha) 
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def categorize_pixels_greyness(rgb_array, threshold=20):
    """
       Function that categorizes each pixel as "grey" or "colored" based on RGB array
    """
    H, W, _ = rgb_array.shape
    out = np.empty((H, W), dtype=object)

    for i in range(H):
        for j in range(W):
            r, g, b = map(int, rgb_array[i, j])
            if abs(r-g) < threshold and abs(g-b) < threshold and abs(r-b) < threshold:
                out[i, j] = "grey"
            else:
                out[i, j] = "colored"
    return out



def plot_ebola_cases(filename, country):
    """
    Plot the Ebola outbreak data for a given country, showing
    both new and cumulative cases versus time.

    The dataset is assumed to be tab-separated (.dat format),
    where the 2nd column represents time (days since first outbreak)
    and the 3rd column contains the number of new cases per day.

    Parameters
    ----------
    filename : str
        Path to the data file (tab-separated values).
    country : str
        Country name for use in the plot title.

    Returns
    -------
    None
        Displays the plot directly.
    """
    # Load data
    df = pd.read_csv(filename, sep="\t")
    time = df.iloc[:, 1].to_numpy()    
    new_cases = df.iloc[:, 2].to_numpy() 
    cumulative = np.cumsum(new_cases)    

    # Create figure with twin y-axes
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # title
    plt.title(f"Ebola cases in {country}")

    # --- Left axis: new cases (red circles) ---
    ax1.plot(time, new_cases, "ro", label="New cases")
    ax1.set_xlabel("Days since first outbreak")
    ax1.set_ylabel("New cases", color="r")
    ax1.tick_params(axis="y", labelcolor="r")

    # --- Right axis: cumulative cases (black squares) ---
    ax2 = ax1.twinx()
    ax2.plot(time, cumulative, "ks", label="Cumulative cases")
    ax2.set_ylabel("Cumulative cases", color="k")
    ax2.tick_params(axis="y", labelcolor="k")


def linear_regression_ebola(filename, country):
    """
    Train a linear regression line on Days vs NumOutbreaks.
    """

 

    # Load data
    df = pd.read_csv(filename, sep="\t")

    # Extract features
    X = df["Days"].to_numpy().reshape(-1, 1)
    y = df["NumOutbreaks"].to_numpy().reshape(-1, 1)

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    plt.scatter(X, y, label="Data")
    plt.plot(X, model.predict(X), color="red", label="Linear fit")
    plt.xlabel("Days")
    plt.ylabel("NumOutbreaks")
    plt.title(f"Linear Regression for {country}")
    plt.legend()
    plt.show()

    return model



def polynomial_regression_ebola(filename, country, degree=2):
    """
    Train a better fitting model than a straight line by using polynomial 
    regression (still linear regression under the hood).

    Parameters
    ----------
    filename : str
        Path to the data file.
    country : str
        Name of the country (for labeling).
    degree : int
        Degree of the polynomial model.

    Returns
    -------
    model : sklearn.linear_model.LinearRegression
        The trained model.
    poly : sklearn.preprocessing.PolynomialFeatures
        The polynomial transformer.
    """


    # Load data
    df = pd.read_csv(filename, sep="\t")

    X = df["Days"].to_numpy().reshape(-1, 1)
    y = df["NumOutbreaks"].to_numpy().reshape(-1, 1)

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Train model
    model = LinearRegression()
    model.fit(X_poly, y)

    #plotting
    plt.scatter(X, y, label="Data")
    plt.plot(X, model.predict(X_poly), color="green", label=f"Polynomial fit (degree {degree})")
    plt.xlabel("Days")
    plt.ylabel("NumOutbreaks")
    plt.title(f"Polynomial Regression for {country} (degree {degree})")
    plt.legend()
    plt.show()

    return model, poly



def load_and_combine_ebola_data():
    """
    Load Ebola data from all three countries, align by calendar date,
    sum outbreak numbers across countries, and create a global time index.

    Returns:
        X: np.ndarray of days since global start
        y: np.ndarray of total outbreaks per reporting date
        df: combined dataframe
    """

    # Load all countries
    df1 = pd.read_csv("./data/ebola_cases_guinea.dat", sep='\t')
    df2 = pd.read_csv("./data/ebola_cases_liberia.dat", sep='\t')
    df3 = pd.read_csv("./data/ebola_cases_sierra_leone.dat", sep='\t')

    # Parse dates
    for df in (df1, df2, df3):
        df["Date"] = pd.to_datetime(df["Date"])

    # Combine everything (all rows)
    df_all = pd.concat([df1, df2, df3], ignore_index=True)

    # Group by calendar date: sum outbreaks across countries
    df_grouped = df_all.groupby("Date", as_index=False)["NumOutbreaks"].sum()

    # Sort by date
    df_grouped = df_grouped.sort_values("Date").reset_index(drop=True)

    # Create global time index
    df_grouped["Days_since_start"] = (df_grouped["Date"] - df_grouped["Date"].min()).dt.days

    X = df_grouped["Days_since_start"].values.reshape(-1, 1)
    y = df_grouped["NumOutbreaks"].values.reshape(-1, 1)

    return X, y, df_grouped


def ebola_outbreaks_dense(X, y, train_split=0.8, epochs=100, batch_size=16):
    """
    Train a simple feed-forward neural network (Task 3).

    Parameters
    ----------
    X : np.ndarray
        Days since first outbreak, shape (N, 1)
    y : np.ndarray
        Outbreak counts, shape (N, 1)
    train_split : float
        Fraction of samples to use for training.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.

    Returns
    -------
    model : keras.Model
        The trained feed-forward neural network.
    X_train, y_train, X_test, y_test : np.ndarray
        Train/test splits (scaled).
    x_scaler, y_scaler : MinMaxScaler
        Scalers for X and y.
    history : History
        Training history.
    """

    n_samples = len(X)
    split_idx = int(train_split * n_samples)

    # Raw train/test split
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

    # Scale X and y to [0, 1]
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train_raw)
    X_test  = x_scaler.transform(X_test_raw)

    y_train = y_scaler.fit_transform(y_train_raw)
    y_test  = y_scaler.transform(y_test_raw)

    # Build a simple feed-forward NN
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train 
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=False,
        verbose=1
    )

    return model, X_train_raw, y_train_raw, X_test_raw, y_test_raw, X_train, y_train, X_test, y_test, x_scaler, y_scaler, history





def plot_ebola_predictions(model, X_test, y_test_scaled, X_axis_raw, y_scaler,
                           title="Model prediction vs actual"):
    """
    Plot predicted and actual Ebola outbreak values (original scale).
    """

    # Predict on scaled test inputs
    y_pred_scaled = model.predict(X_test)

    # Inverse scale both predictions and true values
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test_scaled)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(X_axis_raw, y_true, 'o', label='Actual values')
    plt.plot(X_axis_raw, y_pred, 'x-', label='Predicted')
    plt.xlabel('Days since first outbreak')
    plt.ylabel('Number of outbreaks')
    plt.title(title)
    plt.legend()
    plt.show()



def series_to_X_y(series_2d, window_size):
    """
    series_2d: shape (N, 1) – scaled outbreak values
    returns:
      X: (num_samples, window_size, 1)
      y: (num_samples, 1)
    """
    values = series_2d.reshape(-1)  # (N,) 1D
    X, y = [], []
    # 
    for i in range(len(values) - window_size):
        X.append(values[i : i + window_size])      
        y.append(values[i + window_size])   
    X = np.array(X)        
    y = np.array(y).reshape(-1, 1)  
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def ebola_outbreaks_lstm(y_outbreaks, window_size=5, train_split=0.8,
                         epochs=50, batch_size=16):
    """
    Train an LSTM model to predict Ebola outbreaks from past outbreak values.

    Parameters
    ----------
    y_outbreaks : np.ndarray
        Array of outbreak counts, shape (N, 1).
    window_size : int
        Number of past time steps used as input.
    train_split : float
        Fraction of sequence samples used for training.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.

    Returns
    -------
    lstm_model : keras.Model
        Trained LSTM model.
    y_scaler : MinMaxScaler
        Scaler fitted on y_outbreaks.
    X_train, y_train, X_test, y_test : np.ndarray
        Train/test sequence data (scaled).
    history : History
        Training history object.
    """

    # Scale target series to [0, 1]
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y_outbreaks)

    # Create LSTM sequences
    X_seq, y_seq = series_to_X_y(y_scaled, window_size)

    # Time-aware train–test split
    num_seq_samples = X_seq.shape[0]
    split_idx = int(train_split * num_seq_samples)

    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Build LSTM model
    lstm_model = Sequential([
        LSTM(64, input_shape=(window_size, 1)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model (no shuffling for time series)
    history = lstm_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        shuffle=False,
        verbose=1
    )

    return lstm_model, y_scaler, X_train, y_train, X_test, y_test, history

