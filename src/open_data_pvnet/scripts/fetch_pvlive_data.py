from datetime import datetime
from pvlive_api import PVLive
import pytz

class PVLiveData:
    def __init__(self):
        self.pvl = PVLive()

    def get_latest_data(self, period, entity_type="gsp", entity_id=0, extra_fields=""):
        """
        Get the latest data from PVlive
        """
        try:
            df = self.pvl.latest(entity_type=entity_type, entity_id=entity_id, extra_fields=extra_fields, period=period, dataframe=True)
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_data_between(self, start, end, entity_type="gsp", entity_id=0, extra_fields=""):
        """
        Get the data between two dates
        """
        try:
            df = self.pvl.between(start=start, end=end, entity_type=entity_type, entity_id=entity_id, extra_fields=extra_fields, dataframe=True)
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_data_at_time(self, dt):
        """
        Get data at a specific time
        """
        try:
            df = self.pvl.at_time(dt, entity_type="pes", entity_id=0, extra_fields="", period=30, dataframe=True)
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None


# Example usage
pvl = PVLiveData()

# Get the latest data
latest_data = pvl.get_latest_data(period=30)
print(latest_data)

# Get the data between two dates
start = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
end = datetime(2021, 1, 2, 12, 0, tzinfo=pytz.utc)
data_between = pvl.get_data_between(start, end)
print(data_between)

# Get the data at a specific time
dt = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
data_at_time = pvl.get_data_at_time(dt)
print(data_at_time)
