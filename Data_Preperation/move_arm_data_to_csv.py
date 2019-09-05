# conversions to datetime, moving data into a csv, sorting out files
import pandas as pd
# imports data from netcdf to python, removes 2-d and other uneeded data in the import
import xarray as xr
# gets a list of files from the location stored in the file directory
import os


def hourly_data_to_csv(file_path):
    """
    When this runs it is going to be really slow. Keep this in mind. Running only one
    side of these files takes 20 minutes for 4 files.

    :param file_path:
    :return:
    """
    # sort out the non-hourly data, then between the two data types
    the_files = os.listdir(file_path)
    the_files = pd.Series(the_files)
    needed_files = the_files[the_files.str.startswith("nsaarmbecldradC1")]
    nc_files = needed_files[needed_files.str.endswith("nc")]
    nc_files = nc_files.sort_values()  # Sort files into correct timeseries order
    cdf_files = needed_files[needed_files.str.endswith("cdf")]
    cdf_files = cdf_files.sort_values()  # Sort files into correct timeseries order
    for item in nc_files:
        # Used for naming the file
        print("nc cycles")
        temp_name = item.split(".")

        # Opens up the netcdf data file, drops the below variables
        ds = xr.open_dataset(file_path + "/" + item, drop_variables=[
            "cld_frac", "cld_frac_MPL", "cld_base_source_status",
                                             "swdif", "lw_net_TOA", "lat",
                                             "time_bounds", "time_frac", "lon",
                                             "source_cld_frac", "height", "stdev_sedif"
                                             "cld_frac_radar", "time", "qc_swdif",
                                             "swdir", "stdev_swdir", "qc_swdir",
                                             "swup", "stdev_swup", "qc_swup",
                                             "lwdn", "stdev_lwdn", "qc_lwdn",
                                             "lwup", "stdev_lwup", "qc_lwup",
                                             "sw_net_TOA", "sw_dn_TOA", "totswfluxdn",
                                              "qc_totswfluxdn", "stdev_totswfluxdn", "alt"], decode_times=False)
        df = ds.to_dataframe()
        # Add base time (seconds since 1970-1-1 0:00:00 0:00) which in 2006 Jan 1st in this case
        # and the time offset. This will get me a useable timestamp in time_offset that
        # I can make into a datetime
        df["time_offset"] = df["time_offset"] + df["base_time"]
        df["time_offset"] = pd.to_datetime(df["time_offset"], unit='s')
        df.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/" + temp_name[0] + temp_name[2]
                  + "nc.csv")

    for item in cdf_files:
        print("cdf cycles")
        # Used for naming the file
        temp_name = item.split(".")

        # Opens up the netcdf data file, drops the below variables
        ds = xr.open_dataset(file_path + "/" + item, drop_variables=["cld_frac", "cld_frac_MMCR", "cld_frac_MPL",
                                                                     "qc_cld_frac", "time_bounds", "time_offset",
                                                                     "time_bounds"])
        df = ds.to_dataframe()
        df.to_csv("/home/nelson/PycharmProjects/Solar Forecasting Thesis Project/Data/" + temp_name[0] + temp_name[2]
                  + "cdf.csv")


if __name__ == "__main__":
    ARM_files = "/hdd/ARM files for thesis"
    hourly_data_to_csv(ARM_files)


