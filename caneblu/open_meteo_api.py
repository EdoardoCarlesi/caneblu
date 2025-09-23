import os
import time
from functools import lru_cache

import arrow
import numpy as np
import pandas as pd
import requests
import sklearn.linear_model as lm
import sklearn.metrics as met
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split

import utils

# How many hours should we wait before updating the temporary files storing the meteo data for the different areas


class OpenMeteoAPI:
    """
    This class handles the OpenMeteo APIs
    https://open-meteo.com/en/docs
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        n_days=30,
        date_end=None,
        logger=None,
        prov="CT",
        index=0,
        side=3600,
    ):
        """Initialize inner variables"""

        self.data = dict()
        self.hourly_values = [
            "temperature_2m",
            "windspeed_10m",
            "apparent_temperature",
            "et0_fao_evapotranspiration",
            "relativehumidity_2m",
        ]
        self.daily_values = [
            "temperature_2m_max",
            "windspeed_10m_max",
            "apparent_temperature_max",
            "et0_fao_evapotranspiration",
        ]
        self.lat = lat
        self.lon = lon
        self.prov = prov
        self.index = index
        self.side = side
        self.n_days = n_days
        self.logger = logger

        if type(date_end) == str:
            self.date_end = arrow.get(date_end)
        elif date_end == None:
            self.date_end = arrow.utcnow()

        self.date_start = self.date_end.shift(days=-self.n_days)

    def info(self) -> None:
        """What does this object contain"""

        print(f"At date:{self.date_end}, lat: {self.lat}, lon: {self.lon}")

    def latitude(self) -> float:
        """Self explanatory"""

        return self.lat

    def longitude(self) -> float:
        """Self explanatory"""

        return self.lon

    @lru_cache(maxsize=1024)
    def call_api(
        self,
        frequency="daily",
        call_type="archive",
        date_start=None,
        date_end=None,
        forecast_days=0,
        pause=30,
        timeout=5,
    ) -> dict:
        """
        General api call, all the details need to be specified
        call_type can be archive or forecast
        frequency can be daily or hourly
        """

        if date_start == None:
            date_start = self.date_start.format("YYYY-MM-DD")

        if date_end == None:
            date_end = self.date_end.format("YYYY-MM-DD")

        if call_type == "archive":
            api_call = f"https://archive-api.open-meteo.com/v1/{call_type}?latitude={self.lat}&longitude={self.lon}&start_date={date_start}&end_date={date_end}&{frequency}="
        elif call_type == "forecast":
            api_call = f"https://api.open-meteo.com/v1/{call_type}?latitude={self.lat}&longitude={self.lon}&forecast_days={forecast_days}&{frequency}="

        if frequency == "hourly":
            for val in self.hourly_values:
                api_call += f"{val},"
            self.data["hourly"] = dict()

        elif frequency == "daily":
            for val in self.daily_values:
                api_call += f"{val},"
            self.data["daily"] = dict()

        try:
            api_data = requests.get(api_call, timeout=timeout)
            self.data[frequency] = api_data.json()[frequency]

        except requests.exceptions.Timeout:

            if self.logger is None:
                print(
                    f"Error: The request to {api_call} timed out after {timeout} seconds."
                )
                print(f"Restarting after a {pause}s pause...")
            else:
                logger.info(
                    f"Error: The request to {api_call} timed out after {timeout} seconds."
                )
                logger.info(f"Restarting after a {pause}s pause...")
                
            time.sleep(pause)
            self.call_api(
                frequency=frequency,
                call_type=call_type,
                date_start=date_start,
                date_end=date_end,
                forecast_days=forecast_days,
            )

        return self.data[frequency]

    def history(self):
        """Recover the hourly weather history"""

        n_features = len(self.daily_values)
        n_days_train = self.n_days

        date_train_end = self.date_end
        date_train_start = self.date_start
        date_train_end = date_train_end.format("YYYY-MM-DD")
        date_train_start = date_train_start.format("YYYY-MM-DD")

        hourly_values = self.call_api(
            frequency="hourly", date_start=date_train_start, date_end=date_train_end
        )

        self.data["hourly"] = hourly_values

        return hourly_values

    def recover_humidity(self):
        """
        Get the hourly parameters for n_days_train, train a model and use that to extrapolate humidity for n_days_apply
        Humidity hourly data is available ONLY for recent times so we train it with data obtained TODAY
        """

        n_features = len(self.daily_values)
        n_days_train = self.n_days

        date_train_end = self.date_end
        date_train_start = self.date_start
        date_train_end = date_train_end.format("YYYY-MM-DD")
        date_train_start = date_train_start.format("YYYY-MM-DD")

        hourly_values = self.call_api(
            frequency="hourly", date_start=date_train_start, date_end=date_train_end
        )

        hourly_values["time"] = utils.fix_time(hourly_values["time"])
        hourly_df = pd.DataFrame(hourly_values)
        hourly_df_max = hourly_df.groupby("time").max()

        features = np.zeros((n_days_train + 1, n_features), dtype=float)

        # Prepare for the training
        for i, val in enumerate(self.hourly_values[:-1]):
            features[:, i] = hourly_df_max[val].values

        target = hourly_df_max[self.hourly_values[-1]].values

        # Fit a model on the fly
        model = RandomForestRegressor(max_depth=4, n_estimators=60)

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.33, random_state=69
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        daily_values = self.call_api(frequency="daily")
        daily_features = np.zeros((self.n_days + 1, n_features), dtype=float)

        # Prepare for the training
        for i, val in enumerate(self.daily_values):
            daily_features[:, i] = daily_values[val]

        # Now predict for our values
        self.data["daily"]["humidity"] = model.predict(daily_features)

        return self.data["daily"]["humidity"]

    def forecast(self, forecast_days=1):
        """Simple wrapper for the weather forecast API call"""

        api_tmp = f"{utils.TMP_PATH}{self.prov}_{self.index}_{self.side}_forecast_cache.json"

        if os.path.exists(api_tmp):
            if utils.is_file_older_than(api_tmp, hours=utils.HOURS_CACHE):
                # Call the apis
                json_apis = self.call_api(
                    call_type="forecast",
                    forecast_days=forecast_days,
                    frequency="hourly",
                )
                self.data["forecast"] = json_apis
                utils.dump_geojson(json_apis, api_tmp, logger=self.logger)
                time.sleep(0.5)
            else:
                # Read the tmp cache file
                self.data["forecast"] = utils.read_json_file(api_tmp)
        else:
            # Call the apis
            json_apis = self.call_api(
                call_type="forecast", forecast_days=forecast_days, frequency="hourly"
            )
            self.data["forecast"] = json_apis
            utils.dump_geojson(json_apis, api_tmp, logger=self.logger)
            time.sleep(0.5)

        return self.data["forecast"]


def test():
    """Simple test function"""

    lat = 14.63
    lon = 38.06
    date_end = "2022-08-24"
    n_days = 20
    oma = OpenMeteoAPI(lat=lat, lon=lon, date_end=date_end, n_days=n_days)

# Another simple test function
def apply_dataset():
    """Get the full dataset of burned locations and get the hourly data for the days before"""
    
    locations = "/home/edoardo/DATA/SEFORE/dataset_info.csv"
    loc_df = pd.read_csv(locations)
    omas = []

    for i, row in loc_df.iterrows():
        omas.append(
            OpenMeteoAPI(
                lat=row["lat"], lon=row["lon"], date_end=row["date"], n_days=10
            )
        )

    # print(omas)
    print(omas[4].history())


def main():
    """Main function, used for testing"""

    apply_dataset()
    # test()


if __name__ == "__main__":
    """Exec wrapper"""

    main()
