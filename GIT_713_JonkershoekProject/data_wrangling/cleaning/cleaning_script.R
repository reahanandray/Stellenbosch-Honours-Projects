# libraries

library(dplyr)
library(ggplot2)
library(patchwork)

# Importing Data

# Weather Station data from 2021-2024
swartbos_data <- read.csv(
  "C:/Users/25950495/Desktop/GIT713/Project/Cleaning/CSV_Files/swartbos_data.csv",
  sep = ";"
)
elsenburg_data <- read.csv(
  "C:/Users/25950495/Desktop/GIT713/Project/Cleaning/CSV_Files/elsenburg_data.csv",
  sep = ","
)

# Fire Observations by different sensors
j1v_data <- read.csv(
  "C:/Users/25950495/Desktop/GIT713/Project/Cleaning/CSV_Files/j1v_data.csv",
  sep = ";"
)
j2v_data <- read.csv(
  "C:/Users/25950495/Desktop/GIT713/Project/Cleaning/CSV_Files/j2v_data.csv",
  sep = ";"
)
m_data <- read.csv(
  "C:/Users/25950495/Desktop/GIT713/Project/Cleaning/CSV_Files/m_data.csv",
  sep = ";"
)
sv_data <- read.csv(
  "C:/Users/25950495/Desktop/GIT713/Project/Cleaning/CSV_Files/sv_data.csv",
  sep = ";"
)

# Cleaning fire data

j1v_data$TYPE <- NULL
sv_data$TYPE <- NULL
m_data$TYPE <- NULL

# merge the datasets into new variable
firedata <- rbind(j1v_data, j2v_data, m_data, sv_data) |>
  # remove unnecessary columns from dataset
  select(-c(2, 6:7, 12:15, 17:18, 20:21, 23, 25:28)) |>
  # change data types to numeric and date to permit analysis
  mutate(
    LATITUDE = as.numeric(gsub(",", ".", LATITUDE)),
    LONGITUDE = as.numeric(gsub(",", ".", LONGITUDE)),
    ACQ_DATE = as.Date(ACQ_DATE, format = "%Y-%m-%d %H:%M:%S"),
    BRIGHTNESS = as.numeric(gsub(",", ".", BRIGHTNESS)),
    MeanElevation = as.numeric(gsub(",", ".", MeanElevation)),
    MeanSlope = as.numeric(gsub(",", ".", MeanSlope)),
    MeanAspect = as.numeric(gsub(",", ".", MeanAspect))
  ) |>
  # sort the data in ascending order by date and time of capture
  arrange(ACQ_DATE, ACQ_TIME)

# change column names to be more readable
colnames(firedata) <- c('object_id', 'latitude', 'longitude', 'brightness',
                        'date_of_fire', 'time_of_fire', 'satellite', 'instrument',
                        'daynight','mean_elevation','mean_slope', 'mean_aspect')

# remove seperate datasets from memory
rm(j1v_data, j2v_data, m_data, sv_data)

# Cleaning Weather data

# function to determine whether a year is a leap year
is_leap_yr <- function(year) {
  (year %% 4 == 0 & year %% 100 != 0) | (year %% 400 == 0)
}

elsenburg_data |>
  select(LogDate, MinTemp, MaxTemp, Windspeed, Rainfall, Humidity) |>
  mutate(
    LogDate = as.Date(LogDate, format = "%Y-%m-%dT%H:%M:%S")
  ) -> elsenburg_data

colnames(elsenburg_data) <- c('date_of_fire', 'min_temp', 'max_temp', 
                              'avg_windspeed', 'rain', 'avg_hum')

# rename columns for easier understanding
colnames(swartbos_data) <- c('year', 'month', 'day','max_temp', 'min_temp',
                             'max_hum','min_hum', 'avg_windspeed', 'rain')

swartbos_data |>
  # remove "extra" days from the table
  mutate(
    max_day = case_when(
      month %in% c(1, 3, 5, 7, 8, 10, 12) ~ 31,
      month %in% c(4, 6, 9, 11) ~ 30,
      month == 2 & is_leap_yr(year) ~ 29,
      month == 2 ~ 28,
      TRUE ~ NA_real_),
    max_temp = as.numeric(gsub(",", ".", max_temp)),
    min_temp = as.numeric(gsub(",", ".", min_temp)),
    max_hum = as.numeric(gsub(",", ".", max_hum)),
    min_hum = as.numeric(gsub(",", ".", min_hum)),
    avg_hum = (min_hum + max_hum) / 2,
    avg_windspeed = as.numeric(gsub(",", ".", avg_windspeed)),
    rain = as.numeric(gsub(",", ".", rain))
  ) |>
  filter(day <= max_day) |>
  select(-max_day) |>
  na.omit() |>
  # Create a new date column and remove the seperate year, month, and day
  mutate(
    date_of_fire = as.Date(paste(year, month, day, sep = "-"), 
                           format = "%Y-%m-%d")
  ) |>
  # move the new date column to the beginning
  relocate(date_of_fire, .before = year) |>
  select(-c(2:4, 7:8)) |>
  filter(date_of_fire > '2014-12-31') -> swartbos_data


weatherdata <- bind_rows(swartbos_data, elsenburg_data) |>
  arrange(date_of_fire)

# Merging fire and weather data

final_data <- firedata |>
  left_join(
    weatherdata,
    join_by(date_of_fire),
    relationship = "many-to-many") |>
  relocate(date_of_fire, .before = latitude) |>
  mutate(
    avg_temp = (min_temp + max_temp) / 2
  ) |>
  select(-c(1, 5, 13:14)) -> final_data

write.csv(final_data, "final_data.csv")