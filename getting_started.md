# Solar Forecasting Volunteer Onboarding

Welcome to the Solar Forecasting project! This document will introduce you to the key concepts and knowledge needed to contribute effectively.

## Table of Contents
1. [Introduction to Solar Forecasting](#introduction-to-solar-forecasting)
2. [Coding Style](#coding-style)
3. [What is NWP Data?](#what-is-nwp-data)
4. [Understanding Zarr Format](#understanding-zarr-format)
5. [Target Data: What is UK PVlive?](#target-data-what-is-uk-pvlive)
6. [Basics of Machine Learning for Solar Forecasting](#basics-of-machine-learning-for-solar-forecasting)
7. [APIs and Data Retrieval](#apis-and-data-retrieval)
8. [Data Pipelines for Solar Forecasting](#data-pipelines-for-solar-forecasting)
9. [Benchmarks and Comparisons](#benchmarks-and-comparisons)
10. [Geographical Adaptability](#geographical-adaptability)
11. [Key Tools and Technologies](#key-tools-and-technologies)
12. [Common Terminology](#common-terminology)
13. [Expected Knowledge and Skills](#expected-knowledge-and-skills)
14. [How This Project Fits into Renewable Energy](#how-this-project-fits-into-renewable-energy)

---

## Introduction to Solar Forecasting
Solar forecasting is the process of predicting the amount of solar energy that will be generated over a specific period. Understanding this helps optimize renewable energy systems and integrate them with the grid.

---

## Coding Style
To ensure code quality and maintain consistency, we follow the [Open Climate Fix (OCF) Coding Style](https://github.com/openclimatefix/.github/blob/main/coding_style.md). All contributors are expected to adhere to these standards.

### Key Points:
- **Formatting**: Use `black` with a line length of 100 characters.
- **Linting**: Run `ruff` to catch and fix style violations.
- **Docstrings**: Write Google-style docstrings for all functions, classes, and modules.
- **Pre-commit Hooks**: Install pre-commit hooks to automatically format and lint your code before commits.

Refer to the full [OCF Coding Style Guide](https://github.com/openclimatefix/.github/blob/main/coding_style.md) for detailed instructions.

---

## What is NWP Data?
**Numerical Weather Prediction (NWP)** data uses mathematical models of the atmosphere and oceans to forecast weather. It predicts various atmospheric conditions such as temperature, pressure, wind speed, humidity, precipitation type and amount, cloud cover, and sometimes even surface conditions and air quality—all of which are crucial for solar forecasting.

https://en.wikipedia.org/wiki/Numerical_weather_prediction

---

## Understanding Zarr Format

**zarr** is a relatively new, cloud-based data format designed to improve access to N-dimensional arrays. It provides an effective way to store large N-dimensional data in the cloud, with access facilitated through predefined chunks. Zarr can be viewed as the cloud-based counterpart to HDF5/NetCDF files, as it follows a similar data model. However, unlike NetCDF or HDF5, which store data in a single file, Zarr organizes data as a directory containing compressed binary files for chunks of data, alongside metadata stored in external JSON files.

The semantic mapping from the NetCDF Data Model to the Zarr Data Model is as follows:

| NetCDF Data Model | Zarr V2 Data Model                |
|-------------------|-----------------------------------|
| File              | Store                             |
| Group             | Group                             |
| Variable          | Array                             |
| Attribute         | User Attribute                    |
| Dimension         | Not supported as a native feature |

A Zarr array can be stored in any storage system that supports a key/value interface. In this system:

A key is an ASCII string.
A value is an arbitrary sequence of bytes.
Supported operations include:
Read: Retrieve the sequence of bytes associated with a key.
Write: Set the sequence of bytes associated with a key.
Delete: Remove a key/value pair.
Currently, Zarr V2 is the stable version, while Zarr V3 is considered experimental.
https://wiki.earthdata.nasa.gov/display/ESO/Zarr+Format

---

## Target Data: What is UK PVlive?
UK PVlive provides national solar generation data, accessible via API. This data serves as a "ground truth" for training and evaluating solar forecasting models.

---

## Basics of Machine Learning for Solar Forecasting
Discover key ML concepts such as data splitting, feature engineering, and model evaluation, all tailored to the solar forecasting domain.

---

## APIs and Data Retrieval
Learn how to use APIs to fetch solar generation data and capacity information, critical for building datasets.

---

## Data Pipelines for Solar Forecasting
Explore how pipelines prepare and batch data for machine learning models, making training and testing efficient.

---

## Benchmarks and Comparisons
Understand the importance of benchmarking and how our models compare to existing solutions.

---

## Geographical Adaptability
This project isn't limited to the UK. Learn how it can be adapted to other regions and data sources.

---

## Key Tools and Technologies
Familiarize yourself with tools like Python, pandas, and open-source libraries like `ocf-datasample`.

---

## Common Terminology

Below is a glossary of key terms that might be useful when working on this project:

### General Solar Energy Terms
- **Solar Irradiance**: The power per unit area received from the Sun in the form of electromagnetic radiation, measured in watts per square meter (W/m²).
- **Photovoltaic (PV)**: A method of generating electricity by converting sunlight directly into electricity using solar panels.
- **Solar Panel Efficiency**: The ratio of the electrical output of a solar panel to the incident sunlight energy, expressed as a percentage.
- **Capacity Factor**: The ratio of actual energy produced by a solar PV system to the maximum possible energy it could produce over a given period.
- **Global Horizontal Irradiance (GHI)**: Total solar radiation received on a horizontal surface.
- **Direct Normal Irradiance (DNI)**: Solar radiation received directly from the Sun at a perpendicular angle.
- **Diffuse radiation**: Refers to solar radiation that reaches the Earth's surface after being scattered by molecules, aerosols, or clouds in the atmosphere.
- **Diffuse Horizontal Irradiance (DHI)**: Solar radiation received indirectly due to scattering in the atmosphere.
- **Solar Zenith Angle**: The angle between the Sun and a line perpendicular to the Earth's surface; influences solar irradiance.

---

### Weather Forecasting and Numerical Weather Prediction (NWP) Terms
- **Numerical Weather Prediction (NWP)**: The use of mathematical models to simulate atmospheric processes and predict future weather conditions.
- **Global Forecast System (GFS)**: A global NWP model produced by the National Weather Service that provides weather forecasts up to 16 days in advance.
- **European Centre for Medium-Range Weather Forecasts (ECMWF)**: An independent intergovernmental organization that produces highly accurate medium-range weather forecasts.
- **Model Resolution**: The spatial and temporal granularity of an NWP model, usually measured in kilometers or degrees.
- **Initialization**: The process of incorporating current observational data into a model to start a forecast.
- **Boundary Conditions**: Data input to a weather model defining conditions at the edges of the modeled area.

---

### Climate and Atmosphere Terms

- **Albedo**: The reflectivity of a surface, important for understanding how much sunlight is absorbed or reflected by the Earth. Surfaces with high albedo, such as snow and ice, reflect more sunlight, while darker surfaces absorb more.
- **Aerosols**: Tiny particles or liquid droplets suspended in the atmosphere that affect solar radiation by scattering or absorbing sunlight. They play a significant role in cloud formation and can influence local and global temperatures.
- **Greenhouse Gases**: Atmospheric gases, such as carbon dioxide (CO₂), methane (CH₄), and water vapor (H₂O), that trap heat in the Earth's atmosphere and contribute to global warming.
- **Atmospheric Pressure**: The force exerted by the weight of the atmosphere above a given point, measured in hectopascals (hPa) or millibars (mb). It affects weather patterns and the movement of air masses.
- **Relative Humidity**: The amount of water vapor in the air compared to the maximum amount the air can hold at a given temperature, expressed as a percentage. It influences cloud formation and precipitation.
- **Dew Point**: The temperature at which air becomes saturated with moisture and water vapor condenses into dew, clouds, or fog.
- **Radiative Forcing**: The change in the energy balance of the Earth’s atmosphere due to factors like greenhouse gases, aerosols, and changes in solar irradiance. It is a key concept in climate change studies.
- **Turbidity**: A measure of the atmosphere's clarity, influenced by aerosols, dust, and pollution. High turbidity reduces the amount of solar radiation reaching the Earth's surface.
- **Ozone Layer**: A layer of ozone (O₃) in the stratosphere that absorbs the majority of the Sun’s harmful ultraviolet radiation. Changes in the ozone layer can impact solar irradiance measurements.
- **Wind Shear**: A change in wind speed or direction over a short distance in the atmosphere. It can influence cloud formation, storm development, and the dispersal of aerosols.
- **Thermal Inversion**: A phenomenon where a layer of warm air traps cooler air near the Earth's surface, preventing vertical mixing. It can lead to increased pollution and reduced solar irradiance at the surface.

---

### Cloud and Sky Observation Terms
- **Cloud Cover**: The fraction of the sky obscured by clouds, typically expressed as a percentage.
- **Spatial Homogeneity** in cloud classification refers to the uniformity or consistency in the structure and appearance of clouds over a given spatial area. When clouds exhibit spatial homogeneity, they appear relatively uniform in terms of their texture, brightness, and structure across the entire observed area.
- The **World Meteorological Organization (WMO)** classifies clouds into the following types based on their appearance and altitude. These are grouped into **three main altitude categories**: high, middle, and low clouds.
#### High Clouds (Above 20,000 feet / 6,000 meters)
- **Cirrus (Ci)**: Wispy, hair-like clouds composed of ice crystals, often indicating fair weather.
- **Cirrostratus (Cs)**: Thin, veil-like clouds covering the sky, often producing a halo around the Sun or Moon.
- **Cirrocumulus (Cc)**: Small, white, patchy clouds without shading, often arranged in rows or ripples.

#### Middle Clouds (6,500–20,000 feet / 2,000–6,000 meters)
- **Altostratus (As)**: Gray or blue-gray clouds covering the sky, usually associated with continuous rain or snow.
- **Altocumulus (Ac)**: White or gray clouds in patches or layers, often with shading, signaling changing weather.

#### Low Clouds (Below 6,500 feet / 2,000 meters)
- **Stratus (St)**: Uniform gray clouds covering the entire sky, often producing drizzle or mist.
- **Stratocumulus (Sc)**: Low, lumpy clouds, typically covering most of the sky with breaks of blue.
- **Nimbostratus (Ns)**: Thick, dark clouds producing steady precipitation and obscuring the Sun.

#### Clouds with Vertical Development
- **Cumulus (Cu)**: Fluffy, white clouds with a flat base, often indicating fair weather when small.
- **Cumulonimbus (Cb)**: Towering thunderstorm clouds with an anvil-shaped top, capable of producing heavy rain, lightning, and severe weather.

#### Special Types (Optional to include)
- **Contrails (Ct)**: Man-made clouds formed by aircraft exhaust.
- **Pyrocumulus**: Clouds formed by intense heat, such as from wildfires or volcanic eruptions.

For more details, refer to the [WMO Cloud Identification Guide](https://cloudatlas.wmo.int/en/cloud-identification-guide.html).

---

### Data and Measurement Terms
- **Ground Truth**: Data collected on-site (e.g., solar generation data from PV systems) used to validate predictions.
- **Weather Station**: A facility with instruments and equipment to measure atmospheric conditions such as temperature, humidity, and wind speed.
- **Time Series Data**: Data points collected or recorded at time-ordered intervals, often used in forecasting.
- **Power Units**:
  - **Watt (W)**: The basic unit of power in the International System of Units (SI), representing one joule per second.
  - **Kilowatt (kW)**: Equal to 1,000 watts, commonly used to measure the capacity of small solar systems.
  - **Megawatt (MW)**: Equal to 1,000 kilowatts or one million watts, used for larger solar farms and power plants.
  - **Gigawatt (GW)**: Equal to 1,000 megawatts or one billion watts, used to represent national or regional energy capacities.
  - **Terawatt (TW)**: Equal to 1,000 gigawatts or one trillion watts, often used for global energy capacity.
  - **Petawatt (PW)**: Equal to 1,000 terawatts or one quadrillion watts, applicable for global-scale discussions.
- **Energy vs. Power**:
  - **Power**: The rate at which energy is produced or consumed, typically measured in watts (W).
  - **Energy**: The total amount of work performed over time, measured in watt-hours (Wh), kilowatt-hours (kWh), etc.
- **Capacity Factor**: The ratio of actual energy produced by a system to the maximum possible energy it could produce over a given period.

- **Irradiance vs. Insolation**:
  - **Irradiance**: The power of solar radiation received per unit area (W/m²).
  - **Insolation**: The total energy received over a given time, typically measured in kilowatt-hours per square meter (kWh/m²).
- **PV Capacity**: The maximum amount of power that a photovoltaic system can generate under ideal conditions, often measured in megawatts (MW).

---

### Grid and Power Systems Terms

- **Grid Supply Point (GSP)**: A location where electricity is transferred from the transmission network to the distribution network, serving as a critical node in the power grid.
- **Peak Load**: The maximum power demand in a given period, often used to assess system capacity and ensure reliability during high-demand times.
- **Base Load**: The minimum level of demand on an electrical grid over a 24-hour period. It is typically supplied by reliable, continuous sources like nuclear or coal-fired power plants.
- **Load Factor**: The ratio of the average load over a given period to the peak load during that same period, indicating the efficiency of grid usage.
- **Frequency Regulation**: The process of maintaining the grid's operating frequency (e.g., 50 Hz in Europe or 60 Hz in the US) within acceptable limits to ensure stable power delivery.
- **Distributed Energy Resources (DERs)**: Small-scale power generation or storage units, such as rooftop solar panels and batteries, connected to the distribution network.
- **Interconnection**: The linkage of two or more electricity systems to enable power exchange and enhance reliability.
- **Curtailment**: The reduction of power output from renewable energy sources (e.g., solar or wind) due to oversupply or grid limitations.
- **Net Metering**: A billing mechanism that allows consumers who generate their own electricity (e.g., via rooftop solar panels) to send excess power back to the grid in exchange for credits on their electricity bill.
- **Smart Grid**: An advanced electricity network that uses digital technology to monitor and manage the flow of electricity, improving efficiency and reliability.
- **Reactive Power**: Power that oscillates between the source and the load, necessary for maintaining voltage levels in the grid.
- **Transformer**: A device used in the grid to step up or step down voltage levels for efficient power transmission and distribution.
- **Grid Resilience**: The ability of the power grid to recover quickly from disruptions, such as natural disasters or cyberattacks.
- **Voltage Drop**: A reduction in voltage as electricity travels through transmission and distribution lines, influenced by the resistance of the lines and the distance from the source.
- **Power Factor**: A measure of how effectively electrical power is converted into useful work output. A power factor of 1 indicates maximum efficiency.
- **Black Start**: The process of restoring the grid after a complete shutdown, using backup power sources to start key components.

---

### Machine Learning Terms
- **Feature Engineering**: The process of selecting, modifying, and transforming raw data into features suitable for machine learning models.
- **Train-Test Split**: Dividing data into training and testing subsets to evaluate model performance.
- **Validation**: Using a subset of data to tune model hyperparameters and prevent overfitting.
- **Cross-Validation**: A technique for assessing model performance by splitting data into multiple training and testing subsets.

#### **Model Evaluation Metrics**
- **Mean Absolute Error (MAE)**: A common metric for evaluating forecasting accuracy by measuring the average magnitude of errors. Lower values indicate better model performance.
- **Root Mean Square Error (RMSE)**: Measures the standard deviation of prediction errors, giving higher weight to large errors. Lower RMSE values indicate better performance.
- **R² Score (Coefficient of Determination)**: Indicates how well the model predictions approximate the real data. An R² of 1 means perfect prediction; values closer to 0 indicate weaker predictive performance.
- **Mean Bias Error (MBE)**: Measures the average bias in model predictions, indicating whether predictions are systematically overestimating or underestimating.
- **Precision**: In classification tasks, the ratio of true positive predictions to the total predicted positives. Indicates the accuracy of positive predictions.
- **Recall (Sensitivity)**: The ratio of true positives to the actual positives in the dataset. Measures the ability to capture all relevant instances.
- **F1 Score**: The harmonic mean of precision and recall, providing a balanced metric for evaluating model performance in classification.

#### **Optimization and Cost Functions**
- **Cost Function**: A function that measures the difference between the predicted and actual values, guiding the optimization process. Examples include:
  - **Mean Squared Error (MSE)**: Commonly used in regression tasks.
  - **Cross-Entropy Loss**: Frequently used in classification tasks.
- **Gradient Descent**: An optimization algorithm used to minimize the cost function by iteratively adjusting the model parameters in the direction of the steepest descent.
  - **Learning Rate**: A hyperparameter that determines the step size during gradient descent. A smaller rate ensures convergence but may slow down training.
- **Stochastic Gradient Descent (SGD)**: A variation of gradient descent where a single data point or a small batch is used to compute gradients, making the optimization process faster.
- **Adam Optimizer**: An advanced optimization algorithm combining the benefits of momentum and adaptive learning rates for efficient gradient descent.
- **Regularization**: Techniques to prevent overfitting by adding a penalty to the cost function. Examples:
  - **L1 Regularization (Lasso)**: Encourages sparsity in the model by shrinking less important coefficients to zero.
  - **L2 Regularization (Ridge)**: Penalizes large coefficients to make the model more generalizable.

#### **Training and Model Behavior**
- **Masking**: A technique used to ignore certain parts of input data during model training or inference. For example, in transformers, masking ensures that certain tokens or parts of the input sequence are not considered.
- **Overfitting**: A situation where the model performs well on the training data but fails to generalize to unseen data due to excessive complexity.
- **Underfitting**: Occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and testing datasets.
- **Early Stopping**: A technique to prevent overfitting by halting training when the model's performance on the validation set stops improving.
- **Batch Size**: The number of samples processed before the model updates its parameters during training.

#### **Model Architectures and Learning Paradigms**
- **Convolutional Neural Network (CNN)**: A type of neural network designed to process data with a grid-like structure, such as images. CNNs are particularly useful for image classification, segmentation, and pattern detection.
- **Transformer**: A deep learning architecture that relies on self-attention mechanisms, widely used in natural language processing and increasingly in computer vision tasks.
- **Supervised Learning**: A machine learning paradigm where the model learns from labeled data, meaning each input has a corresponding output.
- **Self-Supervised Learning**: A learning paradigm where the model generates labels or tasks from the data itself, enabling training without explicit human-labeled data.
- **Reinforcement Learning (RL)**: A learning paradigm where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties based on its actions.

#### **Visualization and Interpretability**
- **Confusion Matrix**: A table used in classification tasks to visualize the performance of a model by showing true positives, true negatives, false positives, and false negatives.
- **Loss Curve**: A plot showing the value of the cost function during training to monitor convergence and detect overfitting.
- **Learning Curve**: A plot showing model performance (e.g., accuracy, loss) against training progress, often used to assess whether a model is underfitting or overfitting.

#### **Hyperparameter Tuning**
- **Grid Search**: An exhaustive search over a manually specified set of hyperparameter values to find the optimal configuration.
- **Random Search**: A more efficient method that randomly samples hyperparameter values to find good configurations.
- **Bayesian Optimization**: A probabilistic approach to hyperparameter tuning that models the relationship between hyperparameters and model performance to find optimal settings.

---

### Tools and Software
- **Black**: A Python code formatter used for maintaining consistent code style.
- **Ruff**: A fast Python linter that helps enforce coding standards.
- **Pre-commit Hooks**: Tools that automatically check or modify code before it is committed to a repository.

---

## Expected Knowledge and Skills
An overview of the skills contributors should have or be willing to learn, such as Python programming and data analysis.

---

## How This Project Fits into Renewable Energy
Understand the broader impact of this work and its contribution to a sustainable future.

---

Thank you for joining us on this journey to advance solar forecasting and renewable energy solutions!
