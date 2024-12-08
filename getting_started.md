# Solar Forecasting Volunteer Onboarding

Welcome to the Solar Forecasting project! This document will introduce you to the key concepts and knowledge needed to contribute effectively.

## Table of Contents
1. [Introduction to Solar Forecasting](#introduction-to-solar-forecasting)
2. [What is NWP Data?](#what-is-nwp-data)
3. [Understanding Zarr Format](#understanding-zarr-format)
4. [Target Data: What is UK PVlive?](#target-data-what-is-uk-pvlive)
5. [Basics of Machine Learning for Solar Forecasting](#basics-of-machine-learning-for-solar-forecasting)
6. [APIs and Data Retrieval](#apis-and-data-retrieval)
7. [Data Pipelines for Solar Forecasting](#data-pipelines-for-solar-forecasting)
8. [Benchmarks and Comparisons](#benchmarks-and-comparisons)
9. [Geographical Adaptability](#geographical-adaptability)
10. [Key Tools and Technologies](#key-tools-and-technologies)
11. [Common Terminology](#common-terminology)
12. [Expected Knowledge and Skills](#expected-knowledge-and-skills)
13. [How This Project Fits into Renewable Energy](#how-this-project-fits-into-renewable-energy)

---

## Introduction to Solar Forecasting
Solar forecasting is the process of predicting the amount of solar energy that will be generated over a specific period. Understanding this helps optimize renewable energy systems and integrate them with the grid.

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
Learn the meanings of key terms like Grid Supply Point (GSP), solar irradiance, and capacity factors.

---

## Expected Knowledge and Skills
An overview of the skills contributors should have or be willing to learn, such as Python programming and data analysis.

---

## How This Project Fits into Renewable Energy
Understand the broader impact of this work and its contribution to a sustainable future.

---

Thank you for joining us on this journey to advance solar forecasting and renewable energy solutions!
